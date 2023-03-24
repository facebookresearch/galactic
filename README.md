![pretty_robot_plus_debug_wide_header](https://user-images.githubusercontent.com/6557808/152262691-8e95c12a-5fb1-44e6-b8f3-4b921c3557ec.png)

# Overview
Galactic is a large-scale simulation and reinforcement-learning (RL) framework for robotic mobile manipulation in indoor environments.
* todo: link to paper

Some feature highlights:
* Galactic training is fast! Over **100,000 samples/second** (simulation+rendering+RL) on an 8-GPU node.
* RGB and depth-sensor rendering for photorealistic scenes using [bps3D](https://github.com/shacklettbp/bps3D).
* Kinematic physics for articulated robots and movable rigid objects, including fast approximations for collision-detection, sliding, grasping, and dropping.

# Extending Galactic

You can use this repo to reproduce the Galactic paper results. You can also extend Galactic data to enable new experiments, e.g. new objects, scenes, robots, and episodes. See [DATA.md](./DATA.md). However, beware we aren't actively developing or supporting this codebase at this time.

# Repo Structure
* The top-level `galactic` repo contains runtime data, a few scripts, and two git submodules: `habitat-sim` and `habitat-lab`.
* Galactic C++ simulator code is in our branch of `habitat-sim`, e.g. `habitat-sim/src/esp/batched_sim/BatchedSimulator.h`.
* Galactic Python RL code is in our branch of `habitat-lab`, e.g. `habitat-lab/habitat/core/batched_env.py`.

# License

The majority of Galactic is licensed under CC-BY-NC, however portions of the project are available under separate license terms: [habitat-lab](./habitat-lab/), [habitat-sim](./habitat-sim/) and [bps3D](https://github.com/eundersander/bps3D/tree/ce7e28f76f31f302f03903c09f99d899b575365e) are licensed under the MIT license.

# Requirements
* NVIDIA GPU
* CUDA 11.0 or higher
* NVIDIA driver with Vulkan 1.1 support (440.33 or later confirmed to work)
* Fedora or Ubuntu (recommended)

# Installation and Testing

```
# clone the repo and and create a conda env
git clone --recurse-submodules https://github.com/facebookresearch/galactic.git
cd galactic
conda create -n gala python=3.9 cmake=3.22
source activate gala

# download Vulkan SDK and call setup-env.sh (see also https://vulkan.lunarg.com/sdk/home)
mkdir vulkansdk
cd vulkansdk
wget https://sdk.lunarg.com/sdk/download/1.2.198.1/linux/vulkansdk-linux-x86_64-1.2.198.1.tar.gz
tar -xf vulkansdk-linux-x86_64-1.2.198.1.tar.gz
source 1.2.198.1/setup-env.sh
cd ../

# install and verify PyTorch with CUDA 11.0+
# see also latest instructions at https://pytorch.org/get-started/locally/
conda install pytorch=1.11 pytorch-cuda=11.7 -c pytorch -c nvidia
python -c "import torch; print(torch.version.cuda)"

# unzip compressed runtime data
cd data
unzip columngrids.zip
cd ../

# make videos output directory
mkdir videos

# build habitat-sim python module and tests
cd habitat-sim
./build.sh --build-type RelWithDebInfo --build-temp "build_relwithdebinfo" --bullet --build-tests

# run BatchedSimTest interactive viewer (hit ESC twice to exit, or see DATA.md for instructions)
./build_relwithdebinfo/tests/BatchedSimTest

# ensure we can import habitat_sim python module
export PYTHONPATH=`pwd`

# install bps_pytorch module
cd ../bps_pytorch
pip install -e .

# install habitat-lab
cd ../habitat-lab
pip install -r requirements.txt
python setup.py develop --all

# Run sim test to produce test videos, e.g sim_test_env0_default_rgb.mp4
cd ../python
python sim_test.py
```

# Training

The core training command shared for all tasks below :
```
srun --nodes=1 --ntasks-per-node=8 --gpus-per-task=1 --time=10 --cpus-per-task=10 python ./habitat_baselines/run.py --exp-config habitat_baselines/config/rearrange/gala_kinematic_ddppo.yaml --run-type train WRITER_TYPE tb ENABLE_SLIDING True [additional task-specific arguments]
```

### Mobile Pick
Add the following command line arguments :
```
WRITER_TYPE tb TRAIN_DATASET ../data/train/pick_all_recep_10k_1k.mobile_pick.episode_set.json TOTAL_NUM_STEPS 1e9   TASK_NO_END_ACTION True PREVENT_STOP_ACTION True TASK_IS_SIMPLE_PICK True
```

### Rearrange No Distractors
Add the following command line arguments :
```
TRAIN_DATASET ../data/train/tidy_house_10k_1k_v1.episode_set.removed_distractors.json  TOTAL_NUM_STEPS 5e9  TASK_HAS_SIMPLE_PLACE True  DO_NOT_END_IF_DROP_WRONG True TASK_NO_END_ACTION True PREVENT_STOP_ACTION True NPNP_SUCCESS_THRESH 0.15
```

### Rearrange Safe Drop/Stop
Add the following command line arguments :
```
TRAIN_DATASET ../data/train/tidy_house_10k_1k_v1.episode_set.json  TOTAL_NUM_STEPS 5e9  TASK_HAS_SIMPLE_PLACE True  DO_NOT_END_IF_DROP_WRONG True TASK_NO_END_ACTION True PREVENT_STOP_ACTION True NPNP_SUCCESS_THRESH 0.15
```

### Rearrange
Add the following command line arguments :
```
TRAIN_DATASET ../data/train/tidy_house_10k_1k_v1.episode_set.json TOTAL_NUM_STEPS 5e9   TASK_HAS_SIMPLE_PLACE True NPNP_SUCCESS_THRESH 0.15
```


# Evaluating a trained model
See [this PR](https://github.com/facebookresearch/habitat-lab/pull/850) for documentation for evaluating in Galactic.

To transfer the trained model to Habitat 2.0, follow these steps:
* Install a custom version of Habitat-Lab: `git clone -b hab2_gala_integration https://github.com/facebookresearch/habitat-lab.git`
* Checkout custom Habitat-Lab commit: `git checkout 1947475901256c2134f9bd245d3720d4d8a28761`
* Install Habitat-Lab following the [instructions on the README](https://github.com/facebookresearch/habitat-lab/tree/hab2_gala_integration#installation). Be sure to also follow the Habitat-Sim installation instructions in the README.
* Evaluate in Habitat 2.0: `python habitat_baselines/run.py --exp-config habitat_baselines/config/rearrange/hab/galadyn_nav_pick_nav_place.yaml --run-type eval EVAL_CKPT_PATH_DIR path_to_gala_model.pth`

# Troubleshooting

## GCC 12 compile error
When building Habitat-sim with GCC 12 or newer, you may hit this error:
```
galactic/habitat-sim/src/deps/magnum/src/Magnum/GL/Implementation/TransformFeedbackState.cpp:33:58: error: no matches converting function ‘attachImplementationFallback’ to type ‘void (class Magnum::GL::TransformFeedback::*)(GLuint, int)’ {aka ‘void (class Magnum::GL::TransformFeedback::*)(unsigned int, int)’}
```
Here's a workaround:
```
cd src/deps/corrade/
git cherry-pick c726965baa5b4663c5420f3e8a3c17f3e4bdf096
cd -
git add src/deps/corrade
```

## Link error "archive has no index"
You may hit this error if you try to build `Release`:
```
/usr/bin/ld: ../batched_sim/batched_sim.cpython-39-x86_64-linux-gnu.so: error adding symbols: archive has no index; run ranlib to add one
```
Our workaround is to build `RelWithDebInfo` instead of `Release`, as instructed above:
```
./build.sh --build-type RelWithDebInfo --build-temp "build_relwithdebinfo" --bullet --build-tests
```

## No module named 'google.protobuf'
You may hit this error:
```
ModuleNotFoundError: No module named 'google.protobuf'
```
In this case, one solution is:
```
conda install protobuf
```
