###################################################################################################
# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions
# or recommendations expressed in this material are those of the author(s) and do not necessarily
# reflect the views of the Under Secretary of Defense for Research and Engineering.
#
# (c) 2020 Massachusetts Institute of Technology.
#
# MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013
# or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work
# are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other
# than as specifically authorized by the U.S. Government may violate any copyrights that exist in
# this work.
###################################################################################################

from typing import Callable, Dict, List, Optional, Tuple, Union

import defusedxml.ElementTree as ET
import numpy as np
from gym import spaces

from tesse.msgs import (
    Camera,
    Channels,
    Compression,
    DataRequest,
    DataResponse,
    MetadataMessage,
    ObjectSpawnMethod,
    ObjectsRequest,
    RemoveObjectsRequest,
    SpawnObjectRequest,
)
from tesse_gym.core.tesse_gym import TesseGym
from tesse_gym.core.utils import NetworkConfig, set_all_camera_params

#for square
from shapely.geometry import Point, Polygon
import math
from shapely.ops import unary_union


# define custom message to signal episode reset
# Used for resetting external perception pipelines
class EpisodeResetSignal(MetadataMessage):
    __tag__ = "sRES"


class GoSeek(TesseGym):
    TARGET_COLOR = (10, 138, 80)
    CAMERA_HFOV = 80
    CAMERA_REL_AGENT = np.array([-0.05, 0])

    def __init__(
        self,
        build_path: str,
        network_config: Optional[NetworkConfig] = NetworkConfig(),
        scene_id: Optional[int] = None,
        episode_length: Optional[int] = 400,
        step_rate: Optional[int] = 20,
        n_targets: Optional[int] = 30,
        success_dist: Optional[float] = 2,
        restart_on_collision: Optional[bool] = False,
        init_hook: Optional[Callable[[TesseGym], None]] = set_all_camera_params,
        target_found_reward: Optional[int] = 1,
        ground_truth_mode: Optional[bool] = True,
        n_target_types: Optional[int] = 5,
    ):
        """ Initialize the TESSE treasure hunt environment.

        Args:
            build_path (str): Path to TESSE executable.
            network_config (NetworkConfig): Network configuration parameters.
            scene_id (int): Scene id to load.
            episode_length (int): Maximum number of steps in the episode.
            step_rate (int): If specified, game time is fixed to
                `step_rate` FPS.
            n_targets (int): Number of targets to spawn in the scene.
            success_dist (float): Distance target must be from agent to
                be considered found. Target must also be in agent's
                field of view.
            init_hook (callable): Method to adjust any experiment specific parameters
                upon startup (e.g. camera parameters).
            ground_truth_mode (bool): Assumes gym is consuming ground truth data. Otherwise,
                assumes an external perception pipeline is running. In the latter mode, discrete
                steps will be translated to continuous control commands and observations will be
                explicitly synced with sim time.
            n_target_types (int): Number of target types available to spawn. GOSEEK challenge 
                has 5 target types by default. 
        """
        super().__init__(
            build_path,
            network_config,
            scene_id,
            episode_length,
            step_rate,
            init_hook=init_hook,
            ground_truth_mode=ground_truth_mode,
        )
        self.n_targets = n_targets
        self.success_dist = success_dist
        self.restart_on_collision = restart_on_collision
        self.target_found_reward = target_found_reward
        self.n_found_targets = 0
        self.n_target_types = n_target_types
        self.positions = []
        self.n_predictions = 0
        self.n_collisions = 0
        self.n_successful_predictions = 0
        self.episode_num = 0
        self.resultFigure = None
        self.prev_action = -1


    @property
    def action_space(self) -> spaces.Discrete:
        """ Actions available to agent. """
        return spaces.Discrete(4)

    @property
    def observation_space(self) -> spaces.Box:
        """ Space observed by the agent """
        return spaces.Box(-np.Inf, np.Inf, dtype=np.uint8, shape=self.shape)

    def observe(self) -> DataResponse:
        """ Observe the state.

        Returns:
            DataResponse: The `DataResponse` object. """
        cameras = [(Camera.RGB_LEFT, Compression.OFF, Channels.THREE)]
        return self._data_request(DataRequest(metadata=True, cameras=cameras))

    def reset(
        self, scene_id: Optional[int] = None, random_seed: Optional[int] = None
    ) -> np.ndarray:
        """ Reset environment and respawn agent.

        Args:
            scene_id (int): If given, change to this scene.
            random_seed (int): If give, set simulator random seed.

        Returns:
            np.ndarray: Agent's observation. """
        self.env.send(EpisodeResetSignal())
        super().reset(scene_id, random_seed)

        self.env.request(RemoveObjectsRequest())
        self.n_found_targets = 0
        self.n_successful_predictions  = 0
        self.n_collisions = 0
        self.n_predictions = 0
        self.positions.clear()
        self.resultFigure = None
        self.episode_num += 1
        self.prev_action = -1

        for i in range(self.n_targets):
            self.env.request(
                SpawnObjectRequest(i % self.n_target_types, ObjectSpawnMethod.RANDOM)
            )

        # respawn doesn't advance game time
        # if running an external perception server, advance game time to refresh
        if not self.ground_truth_mode:
            self.advance_game_time(1)

        observation = self.get_synced_observation()
        return self.form_agent_observation(observation)

    def apply_action(self, action: int) -> None:
        """ Make agent take the specified action.

        Args:
            action (int): Make agent take `action`.
        """
        if action == 0:  # move forward 0.5m
            self.transform(0, 0.5, 0)
        elif action == 1:  # turn right 8 degrees
            self.transform(0, 0, 8)
        elif action == 2:  # turn left 8 degrees
            self.transform(0, 0, -8)
        elif action != 3:
            raise ValueError(f"Unexpected action {action}")

    def getTriangle(self, x, y, tetha):
        alpha = self.CAMERA_HFOV
        radius = self.success_dist
        x1 = x + (math.sin(math.radians( tetha + (alpha / 2))) * radius)
        y1 = y + (math.cos(math.radians( tetha + (alpha / 2))) * radius)
        x2 = x + (math.sin(math.radians( tetha - (alpha / 2))) * radius)
        y2 = y + (math.cos(math.radians( tetha - (alpha / 2))) * radius)
        return Polygon([(x, y), (x1, y1), (x2, y2)])

    def getSquare(self):
        polygons = []
        for ar in self.positions:
            poly = self.getTriangle(ar.item(0),ar.item(1),ar.item(2))
            polygons.append(poly)
            
        result = unary_union(polygons)
        return result.area

    def getSquareStep(self, position):
        
        polygons = []
        poly = self.getTriangle(position.item(0),position.item(1),position.item(2))
        polygons.append(poly)
    
        if self.resultFigure is None:
            polygons.append(poly)
        else:
            polygons.append(self.resultFigure)
            
        self.resultFigure = unary_union(polygons)
        return self.resultFigure.area


    def decode_observations2(
        self, observation: np.ndarray, img_shape: Tuple[int, int, int, int] = (-1, 240, 320, 5)
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


    def compute_reward(
        self, observation: DataResponse, action: int
    ) -> Tuple[float, Dict[str, Union[int, bool]]]:
        """ Compute reward.

        Reward consists of:
            - Small time penalty
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
        targets = self.env.request(ObjectsRequest())
#        print("reward!")

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

        reward = -0.01  # small time penalty

        if self.prev_action == 3 and action == 3:
            reward -= 0.1 * self.target_found_reward
        if self.prev_action == 1 and action == 2:
            reward -= 0.1 * self.target_found_reward
        if self.prev_action == 2 and action == 1:
            reward -= 0.1 * self.target_found_reward

        self.prev_action = action    

        rgb, segmentation, depth, pose = self.decode_observations2(self.form_agent_observation(observation))
 #       depth *= 50  # convert depth to meters
        # binary mask for obj nearly 0.7 m
#        masked_depth = np.ma.masked_values(depth <= 1.0, depth)
 #       if np.count_nonzero(masked_depth) > 25000:
#            reward -= self.target_found_reward * 0.01

    #    mask = np.all(segmentation == (10,138,80), axis=-1)
    #    print(mask)
#        if action == 3:
  #      masked_fruit = np.ma.masked_values(segmentation == 1.0, segmentation)
  #      if np.count_nonzero(masked_fruit) > 0:
  #          reward += self.target_found_reward * 0.01 

  #      if action == 0:
  #          masked_chair = np.ma.masked_values(segmentation == 0.6, segmentation)
  #          if np.count_nonzero(masked_chair) > 300:
  #              reward -= self.target_found_reward * 0.01  *  np.count_nonzero(masked_chair) / 76800
        if action == 3:
            self.n_predictions += 1

        # check for found targets
        if target_position.shape[0] > 0 and action == 3:
            found_targets = self.get_found_targets(
                agent_position, target_position, target_ids, agent_metadata
            )
            
            # if targets are found, update reward and related episode info
            if len(found_targets):
                self.n_found_targets += len(found_targets)
                reward += self.target_found_reward * len(found_targets) * 2
                self.n_successful_predictions += 1
                self.env.request(RemoveObjectsRequest(ids=found_targets))
                reward_info["env_changed"] = True
                reward_info["n_found_targets"] += len(found_targets)

                # if all targets have been found, restart the episode
                if self.n_found_targets == self.n_targets:
                    reward += 30.0
                    self.done = True
            else:
                masked_fruit = np.ma.masked_values(segmentation == 1.0, segmentation)
                count_fruitPix = np.count_nonzero(masked_fruit)
                if count_fruitPix > 0:
                    reward -= 0.01 * self.target_found_reward * (100 / count_fruitPix)
                else:
                    reward -= 0.1 * self.target_found_reward
            #    reward -= 0.01 * self.target_found_reward   
       

        # self.positions.append(agent_position) 
        depth *= 50  # convert depth to meters
        masked_depth = np.ma.masked_values(depth > 45.0, depth)
        if np.count_nonzero(masked_depth) < 35000:
            square = self.getSquareStep(agent_position)
            if square < 340.0:
                reward += 0.001 * square * self.target_found_reward    

        self.steps += 1
        if self.steps > self.episode_length:
        #    self.positions.clear()
            self.done = True
            ##print results for episode
            precision = (
                1 if self.n_predictions == 0 else self.n_successful_predictions / self.n_predictions
            )
            recall = self.n_found_targets / self.n_targets
            weight_total = recall + 0.1 * precision - 0.1 * self.n_collisions / self.episode_length - 0.1
    
            results = {
                "episode_num" : self.episode_num,
                "weight_total" : weight_total,
                "precision": precision,
                "recall": recall,
                "collisions": self.n_collisions,
            }
            print (results)

        # collision information isn't provided by the controller metadata
        if self._collision(observation.metadata):
            self.n_collisions += 1
            reward_info["collision"] = True
            reward -= self.target_found_reward * 0.1
            if self.restart_on_collision:
                self.done = True
       # else:
       #     reward += 0.0125 * self.target_found_reward



        return reward, reward_info

    def get_found_targets(
        self,
        agent_position: np.ndarray,
        target_position: np.ndarray,
        target_ids: np.ndarray,
        agent_metadata: str,
    ) -> List[int]:
        """ Get IDs of all found targets

        Targets are considered found when they are:
            (1) within `success_dist` of the agent.
            (2) Within a bearing of `CAMERA_FOV` degrees.

        Args:
            agent_position (np.ndarray): Agent position in (x, y, z) as a shape (3,) array.
            target_position (np.ndarray): Target positions in (x, y, z) as a shape (n, 3) array.
            target_ids (np.ndarray): Target IDS corresponding to position
            agent_metadata (str): Agent metadata from TESSE.

        Returns:
            List[int]: IDs of found targets.
        """
        found_targets = []

        # only compare (x, z) coordinates
        agent_position = agent_position[np.newaxis, (0, 2)]

        # get bearing and distance of targets w.r.t the left camera
        # get left camera position in world coordinates
        agent_orientation = self._get_agent_rotation(agent_metadata)[-1]
        left_camera_position = agent_position + np.matmul(
            self.get_2d_rotation_mtrx(agent_orientation), self.CAMERA_REL_AGENT
        )

        target_position = target_position[:, (0, 2)]
        dists = np.linalg.norm(target_position - left_camera_position, axis=-1)

        if dists.min() < self.success_dist:
            # get positions of targets
            targets_in_range = target_ids[dists < self.success_dist]
            found_target_positions = target_position[dists < self.success_dist]

            target_bearing = self.get_target_bearing(
                agent_orientation, found_target_positions, left_camera_position
            )

            # targets that meet distance and bearing requirements
            found_targets = targets_in_range[np.where(target_bearing < self.CAMERA_HFOV / 2)]

        return found_targets

    @staticmethod
    def get_target_bearing(
        agent_orientation: float,
        target_positions: np.ndarray,
        agent_position: np.ndarray,
    ) -> np.ndarray:
        """ Get orientation of targets relative to agents given the agent position, orientation,
        and target positions.

        Args:
            agent_orientation (float): Orientation of agent (y rotation) in radians.
            target_positions (np.ndarray): Array of target (x, z) positions of shape (n, 2)
                where n is the number of targets.
            agent_position (np.ndarray): Array of agent (x, z) position of shape (1, 2)

        Returns:
            np.ndarray: Array of target orientations relative to agent.
        """
        heading = np.array([[np.sin(agent_orientation), np.cos(agent_orientation)]])
        target_relative_to_agent = target_positions - agent_position
        target_orientation = np.arccos(
            np.dot(heading, target_relative_to_agent.T)
            / (
                np.linalg.norm(target_relative_to_agent, axis=-1)
                * np.linalg.norm(heading)
            )
        )
        return np.rad2deg(target_orientation).reshape(-1)

    @staticmethod
    def _collision(metadata: str) -> bool:
        """ Check for collision with environment.

        Args:
            metadata (str): Metadata string.

        Returns:
            bool: True if agent has collided with the environment. Otherwise, false.
        """
        return (
            ET.fromstring(metadata).find("collision").attrib["status"].lower() == "true"
        )

    def _get_target_id_and_positions(
        self, target_metadata: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ Get target positions from metadata.

        Args:
            target_metadata (str): Metadata string.

        Returns:
            Tuple[np.ndarray, np.array]: shape (n, 3) of (x, y, z) target positions and
                shape (n,) target ids.
        """
        position, obj_ids = [], []
        for obj in ET.fromstring(target_metadata).findall("object"):
            position.append(self._read_position(obj.find("position")))
            obj_ids.append(obj.find("id").text)
        return np.array(obj_ids, dtype=np.uint32), np.array(position, dtype=np.float32)
