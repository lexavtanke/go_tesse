from pathlib import Path
import os

from gym import spaces
from stable_baselines.common.policies import CnnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import PPO2
from tesse.msgs import *

from tesse_gym import get_network_config
from tesse_gym.tasks.goseek import GoSeekFullPerception

from tesse.msgs import Camera, Channels, Compression, DataRequest, DataResponse, ObjectsRequest, RemoveObjectsRequest
from tesse_gym.tasks.goseek.goseek import GoSeek

import numpy as np
from typing import Dict, Tuple, Union

import tensorflow as tf
from stable_baselines.common.policies import nature_cnn

from stable_baselines.common.callbacks import CheckpointCallback

from tesse_gym.core.utils import set_all_camera_params
from tesse_gym.tasks.goseek import GoSeekFullPerception

#for square
from shapely.geometry import Point, Polygon
import math
from shapely.ops import unary_union


# update expected observation shape
class GoSeekUpdatedResolution(GoSeekFullPerception):
    shape = (120, 160, 5)

    @property
    def observation_space(self) -> spaces.Box:
        """ Define an observation space for RGB, depth, segmentation, and pose.

       Because Stables Baselines (the baseline PPO library) does not support dictionary spaces,
       the observation images and pose vector will be combined into a vector. The RGB image
       is of shape (240, 320, 3), depth and segmentation are both (240, 320), ose is (3,), thus
       the total shape is (240 * 320 * 5 + 3).
       """
        return spaces.Box(np.Inf, np.Inf, shape=(120 * 160 * 5 + 3,))

    def compute_reward(
            self, observation: DataResponse, action: int
    ) -> Tuple[float, Dict[str, Union[int, bool]]]:
        targets = self.env.request(ObjectsRequest())
        """ Compute reward.

        Reward consists of:
            - Small time penalty
            - # penalty for too near objects 
            - n_targets_found * `target_found_reward` if `action` == 3.
                n_targets_found is the number of targets that are
                (1) within `success_dist` of agent and (2) within
                a bearing of `CAMERA_FOV` degrees.

        Args:
            observation (DataResponse): TESSE DataResponse object containing images
                and metadata.
            action (int): Action taken by agent.

        Returns:
            Tuple[float, dict[str, [bool, int]]
                Reward
                Dictionary with the following keys
                    - env_changed: True if agent changed the environment.
                    - collision: True if there was a collision
                    - n_found_targets: Number of targets found during step.
        """
        # If not in ground truth mode, metadata will only provide position estimates
        # In that case, get ground truth metadata from the controller
        agent_metadata = (
            observation.metadata
            if self.ground_truth_mode
            else self.continuous_controller.get_broadcast_metadata()
        )
        reward_info = {"env_changed": False, "collision": False, "n_found_targets": 0}

        # compute agent's distance from targets
        agent_position = self._get_agent_position(agent_metadata)
        target_ids, target_position = self._get_target_id_and_positions(
            targets.metadata
        )

        reward = -0.01 * self.target_found_reward  # small time penalty
        # decode data by types
        rgb, segmentation, depth, pose = self.extract_img(self.form_agent_observation(observation))
        # penalty for too near objects
        far_clip_plane = 50
        # agent_observ = self.form_agent_observation(observation)
        depth *= far_clip_plane  # convert depth to meters
        # binary mask for obj nearly 0.7 m
        masked_depth = np.ma.masked_values(depth <= 1.0, depth)
        if np.count_nonzero(masked_depth) > 7000:
            reward -= self.target_found_reward * 0.01
        # get masked fruit from segmentation
        masked_fruit = np.ma.masked_values(segmentation == 1.0, segmentation)
        # penalty for get action without fruit in FOV
        size_masked_fruit = np.count_nonzero(masked_fruit)
        print(f"fruit consists of {size_masked_fruit} points")
        if action == 3 and np.count_nonzero(masked_fruit) < 100:
            print(f"sorry, you can't get it cause you don't see it")
            reward -= self.target_found_reward * 0.02

        # check for found targets
        if target_position.shape[0] > 0 and action == 3:
            found_targets = self.get_found_targets(
                agent_position, target_position, target_ids, agent_metadata
            )

            # if targets are found, update reward and related episode info
            if len(found_targets):
                self.n_found_targets += len(found_targets)
                reward += self.target_found_reward * len(found_targets) +\
                          self.n_found_targets * self.target_found_reward * 0.02
                self.env.request(RemoveObjectsRequest(ids=found_targets))
                reward_info["env_changed"] = True
                reward_info["n_found_targets"] += len(found_targets)

                # if all targets have been found, restart the episode
                if self.n_found_targets == self.n_targets:
                    self.done = True
            else:
                reward -= self.target_found_reward * 0.02

        self.steps += 1
        if self.steps > self.episode_length:
            square = self.getSquare()
            # reward for search new square
            if square < 340.0:
                reward += 0.1 * square * self.target_found_reward
            self.positions.clear()
            self.done = True

        # collision information isn't provided by the controller metadata
        if self._collision(observation.metadata):
            reward_info["collision"] = True
            reward -= self.target_found_reward * 0.02

            if self.restart_on_collision:
                self.done = True
        # else:
        #     reward += self.target_found_reward * 0.005
        print(f"reward for action {action} is {reward}")
        return reward, reward_info

    def extract_img(
            self, observation: np.ndarray, img_shape: Tuple[int, int, int, int] = (-1, 120, 160, 5)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Decode observation vector into images and poses.

        Args:
            observation (np.ndarray): Shape (N,) observation array of flattened
                images concatenated with a pose vector. Thus, N is equal to N*H*W*C + N*3.
            img_shape (Tuple[int, int, int, int]): Shapes of all observed images stacked across
                the channel dimension, resulting in a shape of (N, H, W, C).
                 Default value is (-1, 240, 320, 5).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Arrays with the following information
                - RGB image(s) of shape (N, H, W, 3)
                - Segmentation image(s) of shape (N, H, W), in range [0, C) where C is the number of classes.
                - Depth image(s) of shape (N, H, W), in range [0, 1]. To get true depth, multiply by the
                    Unity far clipping plane (default 50).
                - Pose array of shape (N, 3) containing (x, y, heading) relative to starting point.
                    (x, y) are in meters, heading is given in degrees in the range [-180, 180].
        """
        observation = np.expand_dims(observation, axis=0)
        imgs = observation[:, :-3].reshape(img_shape)
        rgb = imgs[..., :3]
        segmentation = imgs[..., 3]
        depth = imgs[..., 4]

        pose = observation[:, -3:]

        return rgb, segmentation, depth, pose

    def getTriangle(self, x, y, tetha):
        alpha = self.CAMERA_HFOV
        radius = self.success_dist
        x1 = x + (math.sin(math.radians(tetha + (alpha / 2))) * radius)
        y1 = y + (math.cos(math.radians(tetha + (alpha / 2))) * radius)
        x2 = x + (math.sin(math.radians(tetha - (alpha / 2))) * radius)
        y2 = y + (math.cos(math.radians(tetha - (alpha / 2))) * radius)
        return Polygon([(x, y), (x1, y1), (x2, y2)])

    def getSquare(self):
        polygons = []
        for ar in self.positions:
            poly = self.getTriangle(ar.item(0), ar.item(1), ar.item(2))
            polygons.append(poly)

        result = unary_union(polygons)
        return result.area


# update simulator cameras on init
def set_resolution(tesse_gym):
    set_all_camera_params(tesse_gym, height_in_pixels=120, width_in_pixels=160)


def decode_tensor_observations(observation, img_shape=(-1, 120, 160, 5)):
    """ Decode observation vector into images and poses.

    Args:
        observation (np.ndarray): Shape (N,) observation array of flattened
            images concatenated with a pose vector. Thus, N is equal to N*H*W*C + N*3.
        img_shape (Tuple[int, int, int, int]): Shapes of all images stacked in (N, H, W, C).
            Default value is (-1, 240, 320, 5).
    
    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Tensors with the following information
            - Tensor of shape (N, `img_shape[1:]`) containing RGB,
                segmentation, and depth images stacked across the channel dimension.
            - Tensor of shape (N, 3) containing (x, y, heading) relative to starting point.
                (x, y) are in meters, heading is given in degrees in the range [-180, 180].
    """
    imgs = tf.reshape(observation[:, :-3], img_shape)
    pose = observation[:, -3:]

    return imgs, pose


def image_and_pose_network(observation, **kwargs):
    """ Network to process image and pose data.
    
    Use the stable baselines nature_cnn to process images. The resulting
    feature vector is then combined with the pose estimate and given to an
    LSTM (LSTM defined in PPO2 below).
    
    Args:
        raw_observations (tf.Tensor): 1D tensor containing image and 
            pose data.
        
    Returns:
        tf.Tensor: Feature vector. 
    """
    imgs, pose = decode_tensor_observations(observation)
    image_features = nature_cnn(imgs)
    return tf.concat((image_features, pose), axis=-1)


def main():
    n_environments = 5  # number of environments to train over
    total_timesteps = 1200000  # number of training timesteps
    scene_id = [5, 2, 3, 4, 5, 5]  # list all available scenes
    n_targets = 30  # number of targets spawned in each scene
    target_found_reward = 3  # reward per found target
    episode_length = 400
    model_name = "fruit_sem_real_softstalin2_low_res"

    # Create log dir
    log_dir = "./result"
    os.makedirs(log_dir, exist_ok=True)

    def make_unity_env(filename, num_env):
        """ Create a wrapped Unity environment. """

        def make_env(rank):
            def _thunk():
                env = GoSeekUpdatedResolution(
                    str(filename),
                    network_config=get_network_config(worker_id=rank),
                    n_targets=n_targets,
                    episode_length=episode_length,
                    scene_id=scene_id[rank],
                    target_found_reward=target_found_reward,
                    init_hook=set_resolution
                )
                return env

            return _thunk

        return SubprocVecEnv([make_env(i) for i in range(num_env)])

    filename = Path("../../goseek-challenge/simulator/goseek-v0.1.4.x86_64")
    assert filename.exists(), f"Must set a valid path!"

    env = make_unity_env(filename, n_environments)

    policy_kwargs = {'cnn_extractor': image_and_pose_network}

    model = PPO2(
       CnnLstmPolicy,
       env,
       verbose=1,
       tensorboard_log="./tensorboard/",
       nminibatches=5,
       n_steps=256,
       gamma=0.995,
       learning_rate=0.0003,
       policy_kwargs=policy_kwargs,
    )

    # model = PPO2.load(
    #     'real_softstalin2_low_res_800000_steps.zip',
    #     env,
    #     verbose=1,
    #     tensorboard_log="./tensorboard/",
    #     nminibatches=5,
    #     n_steps=256,
    #     gamma=0.995,
    #     learning_rate=0.0003,
    #     )

    # Create the callback: check every 1000 steps
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir,
                                             name_prefix=model_name)

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    model.save(model_name+"_final")

    return True


if __name__ == "__main__":
    if main():
        print("learning ended")
    else:
        print("learning failed")
