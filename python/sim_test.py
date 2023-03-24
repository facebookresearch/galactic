#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch  # isort:skip # noqa: F401  must import torch before importing bps_pytorch

from collections import OrderedDict

import habitat_sim
import magnum as mn
import numpy as np
from habitat_sim._ext.habitat_sim_bindings import (
    BatchedSimulator,
    BatchedSimulatorConfig,
    EpisodeGeneratorConfig,
)
from habitat_sim.utils import viz_utils as vut
from rl import agent

import bps_pytorch  # see https://github.com/shacklettbp/bps-nav#building

next_episode = 0


def get_next_episode(bsim):
    global next_episode
    ret_val = next_episode
    next_episode = (next_episode + 1) % bsim.get_num_episodes()
    return ret_val


def reset_next_episode():
    global next_episode
    next_episode = 0


class Camera:
    def __init__(self, attach_link_name, pos, rotation, hfov):
        self._attach_link_name = attach_link_name
        self._pos = pos
        self._rotation = rotation
        self._hfov = hfov


class BatchedEnvTester:
    r"""Todo"""

    def __init__(self):

        self._num_envs = 4
        sensor0_width = 512
        sensor0_height = 512
        self._include_rgb = True
        self._include_depth = True
        self._include_debug_sensor = True
        debug_width = 768
        debug_height = 768

        generator_config = EpisodeGeneratorConfig()
        # see habitat-sim/src/esp/batched_sim/EpisodeGenerator.h for generator params and defaults
        generator_config.num_episodes = 84
        # generator_config.seed
        # generator_config.num_stage_variations
        # generator_config.num_object_variations
        # generator_config.min_nontarget_objects
        # generator_config.max_nontarget_objects
        # generator_config.use_fixed_robot_start_pos
        # generator_config.use_fixed_robot_start_yaw
        # generator_config.use_fixed_robot_joint_start_positions

        assert torch.cuda.is_available() and torch.version.cuda.startswith("11")
        bsim_config = BatchedSimulatorConfig()
        bsim_config.gpu_id = 0
        bsim_config.include_depth = True
        bsim_config.include_color = True
        bsim_config.num_envs = self._num_envs
        bsim_config.sensor0.width = sensor0_width
        bsim_config.sensor0.height = sensor0_height
        bsim_config.num_debug_envs = self._num_envs if self._include_debug_sensor else 0
        bsim_config.debug_sensor.width = debug_width
        bsim_config.debug_sensor.height = debug_height
        bsim_config.force_random_actions = True
        bsim_config.do_async_physics_step = True
        bsim_config.num_physics_substeps = 1
        bsim_config.enable_robot_collision = True
        bsim_config.enable_held_object_collision = True
        bsim_config.do_procedural_episode_set = True
        bsim_config.episode_generator_config = generator_config
        # bsim_config.episode_set_filepath = "../data/episode_sets/train.episode_set.json"
        bsim_config.enable_sliding = True

        self._bsim = BatchedSimulator(bsim_config)

        self._action_dim = self._bsim.get_num_actions()
        self._num_episodes = self._bsim.get_num_episodes()
        self._next_episode_idx = 0

        self._bsim.enable_debug_sensor(True)

        self._sensor0_camera = Camera(
            "torso_lift_link",
            mn.Vector3(-0.536559, 1.16173, 0.568379),
            mn.Quaternion(mn.Vector3(-0.26714, -0.541109, -0.186449), 0.775289),
            60,
        )
        self.set_camera("sensor0", self._sensor0_camera)

        if self._include_debug_sensor:
            self._debug_camera = Camera(
                "base_link",
                mn.Vector3(-0.8, 3.5, -0.8),  # place behind, above, and to the left of the base
                mn.Quaternion.rotation(mn.Deg(-120.0), mn.Vector3(0.0, 1.0, 0.0))  # face 30 degs to the right
                * mn.Quaternion.rotation(mn.Deg(-45.0), mn.Vector3(1.0, 0.0, 0.0)),  # tilt down
                60,
            )
            self.set_camera("debug", self._debug_camera)

        buffer_index = 0
        SIMULATOR_GPU_ID = 0

        self._observations = OrderedDict()
        import bps_pytorch  # see https://github.com/shacklettbp/bps-nav#building

        self._obs_types_to_save = []

        if self._include_rgb:
            self._observations["rgb"] = bps_pytorch.make_color_tensor(
                self._bsim.rgba(buffer_index),
                SIMULATOR_GPU_ID,
                self._num_envs,
                [sensor0_height, sensor0_width],
            )[..., 0:3]
            self._obs_types_to_save.append("rgb")

        if self._include_depth:
            self._observations["depth"] = bps_pytorch.make_depth_tensor(
                self._bsim.depth(buffer_index),
                SIMULATOR_GPU_ID,
                self._num_envs,
                [sensor0_height, sensor0_width],
            ).unsqueeze(3)
            self._obs_types_to_save.append("depth")

        if self._include_debug_sensor:
            self._observations["debug_rgb"] = bps_pytorch.make_color_tensor(
                self._bsim.debug_rgba(buffer_index),
                SIMULATOR_GPU_ID,
                self._num_envs,
                [debug_height, debug_width],
            )[..., 0:3]
            self._obs_types_to_save.append("debug_rgb")

        self._envs_to_save = [i for i in range(self._num_envs)]
        self._saved_pixel_observations = {}
        for env_index in self._envs_to_save:
            self._saved_pixel_observations[env_index] = []

    def set_camera(self, sensor_name, camera):
        self._bsim.set_camera(
            sensor_name, camera._pos, camera._rotation, camera._hfov, camera._attach_link_name
        )

    def get_next_episode(self):
        assert self._num_episodes > 0
        retval = self._next_episode_idx
        self._next_episode_idx = (self._next_episode_idx + 1) % self._num_episodes
        return retval

    def run(self, num_batch_steps):

        reset_next_episode()
        resets = [self.get_next_episode() for _ in range(self._num_envs)]
        self._bsim.reset(resets)

        default_action = 0.0
        actions = [default_action] * (self._action_dim * self._num_envs)
        resets = [-1] * self._num_envs

        episode_len = 10

        for step_index in range(num_batch_steps):
            self._bsim.wait_step_physics_or_reset()
            self._bsim.start_render()
            batch_env_state = self._bsim.get_batch_environment_state()
            if step_index % episode_len == (episode_len - 1):
                resets = [self.get_next_episode() for _ in range(self._num_envs)]
            else:
                resets = [-1] * self._num_envs
            self._bsim.start_step_physics_or_reset(actions, resets)
            self._bsim.wait_render()
            self.save_pixel_observations(step_index, "default")

    def save_pixel_observations(self, step_index, set_name):

        if self._include_rgb:
            rgb_obs = self._observations["rgb"]
            assert len(rgb_obs) == self._num_envs
            if not isinstance(rgb_obs, np.ndarray):
                rgb_obs = rgb_obs.cpu().numpy()
        if self._include_depth:
            depth_obs = self._observations["depth"]
            assert len(depth_obs) == self._num_envs
            if not isinstance(depth_obs, np.ndarray):
                depth_obs = depth_obs.cpu().numpy()
        if self._include_debug_sensor:
            debug_rgb_obs = self._observations["debug_rgb"]
            assert len(debug_rgb_obs) == self._num_envs
            if not isinstance(debug_rgb_obs, np.ndarray):
                debug_rgb_obs = debug_rgb_obs.cpu().numpy()

        for env_index in self._envs_to_save:
            env_saved_observations = self._saved_pixel_observations[env_index]
            if len(env_saved_observations) < step_index + 1:
                assert len(env_saved_observations) == step_index
                env_saved_observations.append({})
            step_dict = env_saved_observations[step_index]
            if self._include_rgb:
                step_dict[set_name + "_rgb"] = rgb_obs[env_index, ...]
            if self._include_depth:
                step_dict[set_name + "_depth"] = depth_obs[env_index, ...]
            if self._include_debug_sensor:
                step_dict[set_name + "_debug_rgb"] = debug_rgb_obs[env_index, ...]

    def write_videos(self):

        video_folder = "../videos"
        print("saving videos to ", video_folder)
        rank = 0

        for env_index in self._envs_to_save:
            # todo: switch to self._obs_names_to_save
            env_saved_observations = self._saved_pixel_observations[env_index]
            for set_name in ["default"]:
                for obs_type in self._obs_types_to_save:
                    obs_name = set_name + "_" + obs_type
                    vut.make_video(
                        env_saved_observations,
                        obs_name,
                        "color" if obs_name.find("rgb") != -1 else "depth",
                        video_folder
                        + "/sim_test_env"
                        + str(env_index)
                        + "_"
                        + obs_name,
                        fps=10,  # very slow fps
                        open_vid=False,
                        include_frame_number=True
                    )


if __name__ == "__main__":
    tester = BatchedEnvTester()
    tester.run(220)
    tester.write_videos()
