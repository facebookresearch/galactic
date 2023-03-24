#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import random
from typing import Dict, Optional

import numpy as np
import torch
from gym.spaces import Box
from gym.spaces import Dict as SpaceDict
from gym.spaces import Discrete

import habitat
from habitat.config import Config
from habitat.core.agent import Agent
from habitat.core.simulator import Observations
from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy
from habitat_baselines.utils.common import batch_obs

class GalaAgent:
    def __init__(self, num_envs, action_dim, sensor_width, sensor_height) -> None:

        # todo: hook up to config
        self.num_envs = num_envs
        do_depth = False
        do_rgb = True
        gpu_id = 0
        hidden_size = 512
        random_seed = 0
        model_path = None
        do_discrete = False

        self.action_dim = 1 if do_discrete else action_dim

        spaces = {}

        if do_depth:
            spaces["depth"] = Box(
                low=0,
                high=1,
                shape=(sensor_height, sensor_width, 1),
                dtype=np.float32,
            )

        if do_rgb:
            spaces["rgb"] = Box(
                low=0,
                high=255,
                shape=(sensor_height, sensor_width, 3),
                dtype=np.uint8,
            )
        observation_spaces = SpaceDict(spaces)

        # action_spaces = Discrete(4)  # todo continuous
        policy_config = Config()
        if True:
            action_spaces = Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
            policy_config.action_distribution_type = "gaussian"
            # copied from default.py
            policy_config.ACTION_DIST = Config()
            policy_config.ACTION_DIST.use_log_std = False
            policy_config.ACTION_DIST.use_softplus = False
            policy_config.ACTION_DIST.min_std = 1e-6
            policy_config.ACTION_DIST.max_std = 1
            policy_config.ACTION_DIST.min_log_std = -5
            policy_config.ACTION_DIST.max_log_std = 2
            policy_config.ACTION_DIST.action_activation = "tanh"
        else:        
            action_spaces = Discrete(action_dim)
            policy_config.action_distribution_type = "categorical"

        self.device = (
            torch.device("cuda:{}".format(gpu_id))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.hidden_size = hidden_size

        random.seed(random_seed)
        torch.random.manual_seed(random_seed)
        # todo: investigate; probably will hurt perf
        # if torch.cuda.is_available():
        #     torch.backends.cudnn.deterministic = True  # type: ignore


            


        self.actor_critic = PointNavResNetPolicy(
            observation_space=observation_spaces,
            action_space=action_spaces,
            hidden_size=self.hidden_size,
            normalize_visual_inputs="rgb" in spaces,
            policy_config=policy_config
        )
        self.actor_critic.to(self.device)

        if model_path:
            ckpt = torch.load(model_path, map_location=self.device)
            #  Filter only actor_critic weights
            self.actor_critic.load_state_dict(
                {
                    k[len("actor_critic.") :]: v
                    for k, v in ckpt["state_dict"].items()
                    if "actor_critic" in k
                }
            )

        self.test_recurrent_hidden_states: Optional[torch.Tensor] = None
        self.not_done_masks: Optional[torch.Tensor] = None
        self.prev_actions: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.test_recurrent_hidden_states = torch.zeros(
            self.num_envs,
            self.actor_critic.net.num_recurrent_layers,
            self.hidden_size,
            device=self.device,
        )
        self.not_done_masks = torch.zeros(
            self.num_envs, 1, device=self.device, dtype=torch.bool
        )
        self.prev_actions = torch.zeros(
            self.num_envs, self.action_dim, dtype=torch.float, device=self.device
        )

    def act(self, observations: Observations):
        # batch = batch_obs([observations], device=self.device)
        batch = observations  # observations area already batched!
        with torch.no_grad():
            (
                _,
                actions,
                _,
                self.test_recurrent_hidden_states,
            ) = self.actor_critic.act(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=False,
            )

            # temp hack
            scale = 16.0
            actions = torch.mul(actions, scale)

            #  Make masks not done till reset (end of episode) will be called
            self.not_done_masks.fill_(True)
            self.prev_actions.copy_(actions)  # type: ignore

        # return {"action": actions[0][0].item()}
        return actions


# I don't know why this is here
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-type",
        default="rgb",
        choices=["blind", "rgb", "depth", "rgbd"],
    )
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument(
        "--task-config", type=str, default="configs/tasks/pointnav.yaml"
    )
    args = parser.parse_args()

    agent_config = get_default_config()
    agent_config.INPUT_TYPE = args.input_type
    if args.model_path is not None:
        agent_config.MODEL_PATH = args.model_path

    agent = PPOAgent(agent_config)
    benchmark = habitat.Benchmark(config_paths=args.task_config)
    metrics = benchmark.evaluate(agent)

    for k, v in metrics.items():
        habitat.logger.info("{}: {:.3f}".format(k, v))


if __name__ == "__main__":
    main()
