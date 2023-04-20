#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# import torch  # isort:skip # noqa: F401  must import torch before importing bps_pytorch
# import bps_pytorch  # see https://github.com/shacklettbp/bps-nav#building

from collections import OrderedDict
from typing import Dict, List
import time
from timeit import Timer

import numpy as np

import habitat_sim
from habitat_sim._ext.habitat_sim_bindings import BatchedSimulator, BatchedSimulatorConfig, EpisodeGeneratorConfig
from habitat_sim.utils import viz_utils as vut

from multiprocessing import Process, Value, Array

import time

# from rl import agent

next_episode = 0
def get_next_episode(bsim):
  global next_episode
  ret_val = next_episode
  next_episode = (next_episode + 1) % bsim.get_num_episodes()
  return ret_val

def reset_next_episode():
  global next_episode
  next_episode = 0

bsim = None
num_envs = 1024
action_dim = -1
obs_res = 128


def init_bsim():

  # generator_config = EpisodeGeneratorConfig()
  # generator_config.seed = 3
  # generator_config.num_stage_variations = -1
  # generator_config.num_object_variations = -1
  # generator_config.min_nontarget_objects = -1
  # generator_config.max_nontarget_objects = -1
  # generator_config.use_fixed_robot_start_pos = False
  # generator_config.use_fixed_robot_start_yaw = False
  # generator_config.use_fixed_robot_joint_start_positions = True

  # generator_config.reference_episode_set_filepath = "../data/val/tidy_house_10k_1k_v1.episode_set.json"
  # generator_config.save_filepath = "../data/val/tidy_house_10k_1k_v1.pick_distance.episode_set.json"  
  # generator_config.num_episodes = 1000

  # generator_config.pick_start_position = True # if true, then the dataset will have the robot be spawen in front of object

  global bsim
  # assert torch.cuda.is_available() and torch.version.cuda.startswith("11")
  bsim_config = BatchedSimulatorConfig()
  bsim_config.gpu_id = 0
  bsim_config.include_depth = True
  bsim_config.include_color = True
  bsim_config.num_envs = num_envs
  bsim_config.sensor0.width = obs_res
  bsim_config.sensor0.height = obs_res
  bsim_config.force_random_actions = True
  bsim_config.do_async_physics_step = True
  bsim_config.num_physics_substeps = 1
  bsim_config.do_procedural_episode_set = False
  # bsim_config.episode_generator_config = generator_config
  bsim_config.episode_set_filepath =  "../data/tidy_house_100ep.episode_set.json"
  
  bsim_config.enable_sliding = True
  bsim = BatchedSimulator(bsim_config)  

  global action_dim
  action_dim = bsim.get_num_actions()


def physics_benchmark(num_batch_steps):

  global bsim
  assert bsim
  
  reset_next_episode()
  resets = [get_next_episode(bsim) for _ in range(num_envs)]
  bsim.reset(resets)

  actions = [0.0] * (action_dim * num_envs)
  resets = [-1] * num_envs

  for _ in range(num_batch_steps):
    bsim.start_step_physics_or_reset(actions, resets)
    bsim.wait_step_physics_or_reset()

def physics_render_benchmark(num_batch_steps):

  global bsim
  assert bsim
  
  reset_next_episode()
  resets = [get_next_episode(bsim) for _ in range(num_envs)]
  bsim.reset(resets)

  actions = [0.0] * (action_dim * num_envs)
  resets = [-1] * num_envs

  for _ in range(num_batch_steps):
    bsim.start_step_physics_or_reset(actions, resets)
    bsim.wait_step_physics_or_reset()
    bsim.start_render()
    bsim.wait_render()
  
def render_benchmark(num_batch_steps):

  global bsim
  assert bsim
  
  reset_next_episode()
  resets = [get_next_episode(bsim) for _ in range(num_envs)]
  bsim.reset(resets)

  actions = [0.0] * (action_dim * num_envs)
  resets = [-1] * num_envs

  for _ in range(num_batch_steps):
    bsim.start_render()
    bsim.wait_render()

def overlapped_physics_render_benchmark(num_batch_steps):

  global bsim
  assert bsim
  
  reset_next_episode()
  resets = [get_next_episode(bsim) for _ in range(num_envs)]
  bsim.reset(resets)

  actions = [0.0] * (action_dim * num_envs)
  resets = [-1] * num_envs

  for _ in range(num_batch_steps):
    bsim.wait_step_physics_or_reset()
    bsim.start_render()
    bsim.start_step_physics_or_reset(actions, resets)
    bsim.wait_render()
    
def physics_reset_benchmark(num_batch_steps):

  global bsim
  assert bsim
  
  reset_next_episode()
  resets = [get_next_episode(bsim) for _ in range(num_envs)]
  bsim.reset(resets)

  actions = [0.0] * (action_dim * num_envs)
  resets = [-1] * num_envs
  num_env_resets = num_envs // 10
  next_env_to_reset = 0

  bsim.get_recent_stats_and_reset()

  for _ in range(num_batch_steps):
    bsim.start_step_physics_or_reset(actions, resets)
    bsim.wait_step_physics_or_reset()
    resets = [-1] * num_envs
    for _ in range(num_env_resets):
      resets[next_env_to_reset] = get_next_episode(bsim)
      next_env_to_reset = (next_env_to_reset + 1) % num_envs

  print("stats:", bsim.get_recent_stats_and_reset())


class Counter(object):
    def __init__(self):
        self.val = Value('i', 0)

    def increment(self, n=1):
        with self.val.get_lock():
            self.val.value += n

    @property
    def value(self):
        return self.val.value

def physics_reset_benchmark_standalone(num_init, do_start, num_done, num_batch_steps, process_id, gpu_id):

  print("process {} with gpu {} initializing...".format(process_id, gpu_id))

  num_envs = 1024
  action_dim = -1
  obs_res = 128

  # assert torch.cuda.is_available() and torch.version.cuda.startswith("11")
  bsim_config = BatchedSimulatorConfig()
  assert gpu_id < 8
  bsim_config.gpu_id = gpu_id
  bsim_config.include_depth = True
  bsim_config.include_color = True
  bsim_config.num_envs = num_envs
  bsim_config.sensor0.width = obs_res
  bsim_config.sensor0.height = obs_res
  bsim_config.force_random_actions = True
  bsim_config.do_async_physics_step = True
  bsim_config.num_physics_substeps = 1
  bsim_config.do_procedural_episode_set = False
  # bsim_config.episode_generator_config = generator_config
  bsim_config.episode_set_filepath =  "../data/tidy_house_100ep.episode_set.json"
  
  bsim_config.enable_sliding = True
  bsim = BatchedSimulator(bsim_config)    

  resets = [i % bsim.get_num_episodes() for i in range(num_envs)]
  bsim.reset(resets)

  actions = [0.0] * (action_dim * num_envs)
  resets = [-1] * num_envs

  num_init.increment()

  print("process {} with gpu {} waiting to start...".format(process_id, gpu_id), flush=True)
  while not do_start.value:
    time.sleep(0.1)

  print("process {} with gpu {} starting...".format(process_id, gpu_id), flush=True)
  for _ in range(num_batch_steps):
    bsim.wait_step_physics_or_reset()
    bsim.start_render()
    bsim.start_step_physics_or_reset(actions, resets)
    bsim.wait_render()

  num_done.increment()

def multiprocessing_stub_benchmark(num_batch_steps):

  num_init = Counter()
  do_start = Counter()
  do_start.increment()
  num_done = Counter()

  physics_reset_benchmark_standalone(num_init, do_start, num_done, num_batch_steps, 0, 0)


def multiprocessing_benchmark(num_batch_steps):

  num_init = Counter()
  do_start = Counter()
  num_done = Counter()

  num_processes = 24
  max_gpus = 8
  processes = []
  for process_id in range(num_processes):
    gpu_id = process_id * max_gpus // num_processes
    p = Process(target=physics_reset_benchmark_standalone, args=(num_init, do_start, num_done, num_batch_steps, process_id, gpu_id))
    p.start()
    processes.append(p)

  print("waiting for {} remaining processes to init...".format(num_processes - num_init.value), flush=True)
  while num_init.value < num_processes:
    time.sleep(0.1)

  print("starting all...")
  start_time = time.perf_counter()
  do_start.increment()

  print("waiting for {} remaining processes to finish...".format(num_processes - num_done.value), flush=True)
  while num_done.value < num_processes:
    time.sleep(0.1)

  end_time = time.perf_counter()
  secs = end_time - start_time
  steps_per_sec = num_processes * num_batch_steps * num_envs / secs
  desc = "multiprocessing_benchmark"
  print("num_processes: ", num_processes)
  print("max_gpus: ", max_gpus)
  print("{}: {:.0f} steps/sec\n".format(desc, steps_per_sec), flush=True)

  for p in processes:
    p.join()  

  print("all done")



def test_episode_set():

  global bsim
  assert bsim
  
  reset_next_episode()
  resets = [get_next_episode(bsim) for _ in range(num_envs)]
  bsim.reset(resets)
  num_resets = num_envs

  actions = [0.0] * (action_dim * num_envs)
  resets = [-1] * num_envs
  num_steps_per_episode = 3

  bsim.get_recent_stats_and_reset()

  print("test_episode_set: num_episodes: ", bsim.get_num_episodes())

  while True:
    for _ in range(num_steps_per_episode):
      bsim.start_step_physics_or_reset(actions, resets)
      bsim.wait_step_physics_or_reset()
      resets = [-1] * num_envs

    if num_resets >= bsim.get_num_episodes():
      break

    for b in range(num_envs):
      resets[b] = get_next_episode(bsim)
      num_resets += 1
      if num_resets % 50000 == 0 and num_resets > 0:
        print("{}/{} ({:.1f}%)".format(num_resets, bsim.get_num_episodes(), num_resets * 100 / bsim.get_num_episodes()))

  print("stats:", bsim.get_recent_stats_and_reset())




def timeit_helper(desc, lambda_expr):
  print(desc, "...", flush=True)
  secs = Timer(lambda_expr).timeit(number=1)
  print("{:.3f}s".format(secs))
  return secs

def step_timeit_helper(desc, func, num_batch_steps):
  num_reps = 1
  secs = 0
  for _ in range(num_reps):
    secs += timeit_helper(desc, lambda: func(num_batch_steps=num_batch_steps))
  steps_per_sec = num_reps * num_batch_steps * num_envs / secs
  print("{}: {:.0f} steps/sec\n".format(desc, steps_per_sec))


if __name__ == "__main__":
  print("build_type: ", habitat_sim.build_type)
  print("num_envs: ", num_envs)
  print("obs_res: ", obs_res)

  timeit_helper("init_bsim", lambda: init_bsim())

  timeit_helper("test_episode_set", lambda: test_episode_set())

  # step_timeit_helper("physics_benchmark", physics_benchmark, num_batch_steps=1000)

  # step_timeit_helper("render_benchmark", render_benchmark, num_batch_steps=1000)

  # step_timeit_helper("physics_render_benchmark", physics_render_benchmark, num_batch_steps=100)

  # step_timeit_helper("overlapped_physics_render_benchmark", overlapped_physics_render_benchmark, num_batch_steps=100)

  # step_timeit_helper("multiprocessing_benchmark", multiprocessing_benchmark, num_batch_steps=100)
  # multiprocessing_benchmark(400)

  # multiprocessing_stub_benchmark(100)

  # step_timeit_helper("physics_benchmark", physics_benchmark, num_batch_steps=2000)

  # step_timeit_helper("physics_reset_benchmark", physics_reset_benchmark, num_batch_steps=2000)

  # step_timeit_helper("physics_render_benchmark", physics_render_benchmark, num_batch_steps=500)

