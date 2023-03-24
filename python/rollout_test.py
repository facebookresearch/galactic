#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch  # isort:skip # noqa: F401  must import torch before importing bps_pytorch

from collections import OrderedDict
from typing import Dict, List
import time

import bps_pytorch  # see https://github.com/shacklettbp/bps-nav#building
import numpy as np

from habitat_sim._ext.habitat_sim_bindings import BatchedSimulator, BatchedSimulatorConfig
from habitat_sim.utils import viz_utils as vut

from rl import agent

if __name__ == "__main__":

    SIMULATOR_GPU_ID = 0
    num_envs = 1
    sensor_width = 128
    sensor_height = 128
    double_buffered = False
    num_joint_degrees = 15  # hard-coded to match Fetch
    num_base_degrees = 2  # rotate and move-forward/back
    action_dim = num_joint_degrees + num_base_degrees
    log_interval = 50
    do_debug_with_observations = True
    if do_debug_with_observations:
        num_observations_to_save = 10
        num_batch_steps = 10 # 2000
        num_warmup_batch_steps = 0  # 200
    else:
        num_observations_to_save = 0
        num_batch_steps = 2000
        num_warmup_batch_steps = 200

    print("creating bsim...", flush=True)

    bsim_config = BatchedSimulatorConfig()
    bsim_config.num_envs = num_envs
    bsim_config.sensor0.width = sensor_width
    bsim_config.sensor0.height = sensor_height
    bsim_config.sensor0.hfov = 45.0
    bsim = BatchedSimulator(bsim_config)

    # map the simulator's render buffer to observation tensor
    observations = []
    for buffer_index in range(2 if double_buffered else 1):
        observation = OrderedDict()
        observation["rgb"] = bps_pytorch.make_color_tensor(
            bsim.rgba(buffer_index),
            SIMULATOR_GPU_ID,
            num_envs // (2 if double_buffered else 1),
            [sensor_height, sensor_width],
        )[..., 0:3].permute(0, 1, 2, 3)
        observations.append(observation)
    print(observations[0]["rgb"].shape)  # [num_envs, 3, out_dim.x, out_dim.y]

    if num_observations_to_save:
        batch_saved_observations: List[List[Dict]] = []
        for _ in range(num_envs):
            batch_saved_observations.append([])

    agent = agent.GalaAgent(num_envs, action_dim, sensor_width, sensor_height)
    agent.reset()
    bsim.auto_reset_or_step_physics()  # sloppy: this should be an explicit reset
    

    primary_obs_name = "rgba_camera"
    buffer_index = 0
    print("start warmup steps...", flush=True)    

    for warmup_step_index in range(num_batch_steps + num_warmup_batch_steps):

        if warmup_step_index == num_warmup_batch_steps:
            print("done with warmup steps.")    
            t_recent = time.time()

        bsim.start_render()
        # this updates observations tensor
        bsim.wait_for_frame()

        # todo: skip last act and step_physics or otherwise beware off-by-one
        with torch.inference_mode():  # no_grad():
            # this updates agent.prev_actions
            agent.act(observations[buffer_index])
            # perf todo: avoid dynamic list allocation here
            actions_flat_list = agent.prev_actions.flatten().tolist()

        bsim.set_actions(actions_flat_list)  # note possible discarded action
        bsim.auto_reset_or_step_physics()

        dummy_rewards = bsim.get_rewards()
        dummy_dones = bsim.get_dones()

        step_index = warmup_step_index - num_warmup_batch_steps
        if step_index > 0 and step_index % log_interval == 0:
            t_curr = time.time()
            batch_sps = log_interval / (t_curr - t_recent)
            print(
                "step: {}\tbatch sps: {:.3f}\tenv sps: {:.3f}".format(
                    step_index,
                    batch_sps,
                    batch_sps * num_envs,
                )
            )
            t_recent = t_curr

        if num_observations_to_save and step_index < num_observations_to_save:
            # convert obs to numpy
            rgb = observations[buffer_index]["rgb"]
            if not isinstance(rgb, np.ndarray):
                rgb = rgb.cpu().numpy()
            for env_index in range(num_envs):
                env_rgb = rgb[env_index, ...]
                # [3, width, height] => [width, height, 3]
                # env_rgb = np.swapaxes(env_rgb, 0, 2)
                # env_rgb = np.swapaxes(env_rgb, 0, 1)
                env_saved_observations = batch_saved_observations[env_index]
                env_saved_observations.append({primary_obs_name: env_rgb})

    if num_observations_to_save:
        print("saving videos...", flush=True)
        video_folder = "../videos"
        for env_index in range(num_envs):

            env_saved_observations = batch_saved_observations[env_index]
            vut.make_video(
                env_saved_observations,
                primary_obs_name,
                "color",
                video_folder + "/env_" + str(env_index) + "_rgba_camera",
                fps=10,  # very slow fps
                open_vid=False,
            )
