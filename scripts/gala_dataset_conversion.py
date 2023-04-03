# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import gzip
import json
import math
import os
import os.path as osp
import pickle

import habitat
import magnum as mn
import matplotlib.pyplot as plt
import numpy as np
from habitat.tasks.utils import get_angle
from scipy.spatial.transform.rotation import Rotation
from tqdm import tqdm


def orthogonalize(T: mn.Matrix4):
    T_np = np.array(T, dtype=np.float64)
    T_np[:3, :3] = Rotation.from_matrix(T_np[:3, :3]).as_matrix()
    return mn.Matrix4(T_np)


def fix_object_transform(name, T: mn.Matrix4):
    fixed_T = T.__matmul__(
        mn.Matrix4.rotation(mn.Deg(-90.0), mn.Vector3(1, 0, 0))
    )
    com = get_object_center_of_mass(name)
    fixed_T = fixed_T.__matmul__(mn.Matrix4.translation(-com))
    return fixed_T


def get_object_center_of_mass(name):

    # hack scraped from Galakhtic data/replicacad_composite.collection.json
    object_coms = {
        "024_bowl": (
            (
                -0.09557099640369415,
                -0.12427099794149399,
                -0.0005300004268065095,
            ),
            (0.06589200347661972, 0.03689299523830414, 0.05447899550199509),
        ),
        "003_cracker_box": (
            (
                -0.048785001039505005,
                -0.09616000950336456,
                -0.0032430035062134266,
            ),
            (0.02301499992609024, 0.06787599623203278, 0.21019400656223297),
        ),
        "010_potted_meat_can": (
            (
                -0.08382699638605118,
                -0.05660400539636612,
                -0.0031880023889243603,
            ),
            (0.018257999792695045, 0.0034989966079592705, 0.08035500347614288),
        ),
        "002_master_chef_can": (
            (
                -0.06831300258636475,
                -0.06094900891184807,
                -0.00018700220971368253,
            ),
            (0.03421600162982941, 0.04142799228429794, 0.13999000191688538),
        ),
        "004_sugar_box": (
            (
                -0.032214999198913574,
                -0.06379300355911255,
                3.0998555303085595e-05,
            ),
            (0.017280999571084976, 0.030368993058800697, 0.1760459989309311),
        ),
        "005_tomato_soup_can": (
            (-0.0431240014731884, 0.05014599487185478, 7.90045305620879e-05),
            (0.024786999449133873, 0.11788899451494217, 0.10193400084972382),
        ),
        "009_gelatin_box": (
            (
                -0.06747700273990631,
                -0.05879899859428406,
                -0.0005450012977235019,
            ),
            (0.02192699909210205, 0.042309001088142395, 0.02952899970114231),
        ),
        "008_pudding_box": (
            (
                -0.0684640035033226,
                -0.04525500163435936,
                -0.0004969995934516191,
            ),
            (0.069473996758461, 0.08350100368261337, 0.038391999900341034),
        ),
        "007_tuna_fish_can": (
            (
                -0.06882800161838531,
                -0.06490200012922287,
                -0.003218000056222081,
            ),
            (0.01673099957406521, 0.0206379983574152, 0.030319999903440475),
        ),
    }

    min_tuple, max_tuple = object_coms[name]
    com = (mn.Vector3(min_tuple) + mn.Vector3(max_tuple)) * 0.5
    return com


def parse_matrix(T):
    return mn.Matrix4([[T[j][i] for j in range(4)] for i in range(4)])


def get_quaternion_as_list(quat):
    return [quat.scalar, quat.vector[0], quat.vector[1], quat.vector[2]]


def get_rotation_quaternion_list(T: mn.Matrix4):
    quat = mn.Quaternion.from_matrix(T.rotation())
    return get_quaternion_as_list(quat)


def quaternion_list_to_yaw(quat_list):
    quat = mn.Quaternion(
        mn.Vector3(quat_list[1], quat_list[2], quat_list[3]), quat_list[0]
    )

    rotated_vec = quat.transform_vector(mn.Vector3(1.0, 0.0, 0.0))
    assert abs(rotated_vec.y) < 1.0e-6  # expect rotation about y-axis

    yaw = math.atan2(rotated_vec[2], rotated_vec[0])
    return yaw


def generate_inits(cfg_path, in_path, take_count):
    config = habitat.get_config(
        cfg_path,
        [
            "DATASET.DATA_PATH",
            in_path,
            "SIMULATOR.AGENT_0.SENSORS",
            "('HEAD_DEPTH_SENSOR', 'THIRD_RGB_SENSOR')",
        ],
    )

    ep_id_to_robo_start = {}
    cur_i = 0
    with habitat.Env(config=config) as env:
        if take_count != -1:
            num_eps = min(env.number_of_episodes, take_count)
        else:
            num_eps = env.number_of_episodes
        for i in tqdm(range(num_eps)):
            obs = env.reset()
            obs = env.step(env.action_space.sample())

            robot_y = env.sim.robot.base_pos[1]
            while robot_y > 0.2:
                env.sim.set_robot_base_to_random_point()
                robot_y = env.sim.robot.base_pos[1]

            T = env.sim.robot.base_transformation

            # quat = mn.Quaternion.from_matrix(T.rotation())
            # rotated_vec = quat.transform_vector(mn.Vector3(1.0, 0.0, 0.0))
            # yaw = math.atan2(rotated_vec[0], rotated_vec[2])

            forward = np.array([1.0, 0, 0])
            heading = np.array(T.transform_vector(forward))
            forward = forward[[0, 2]]
            heading = heading[[0, 2]]
            v1 = forward / np.linalg.norm(forward)
            v2 = heading / np.linalg.norm(heading)

            yaw = get_angle(v1, v2)
            c = np.cross(v1, v2) < 0
            if not c:
                yaw *= -1.0

            print(
                cur_i,
                env.current_episode.episode_id,
                yaw,
                np.dot(v1, v2),
                np.cross(v1, v2),
            )

            ep_id_to_robo_start[env.current_episode.episode_id] = (
                yaw,
                env.sim.robot.base_pos,
                env.sim.robot.arm_joint_pos,
                obs["robot_third_rgb"],
            )
            cur_i += 1

    return ep_id_to_robo_start


def convert_scene_name(name):
    name = name.split(".")[0]
    name = name.split("/")[-1]
    name = name.split("v3_")[-1]
    return "Baked_" + name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-path", required=True, type=str)
    parser.add_argument("--in-path", required=True, type=str)
    parser.add_argument("--ref-gala-episode-set-path", required=True, type=str)
    parser.add_argument("--take-count", type=int, default=-1)
    parser.add_argument("--cfg-path", required=True, type=str)

    args = parser.parse_args()

    with open(args.ref_gala_episode_set_path, "r") as f:
        ref_episode_set = json.loads(f.read())

    ref_static_scenes = ref_episode_set["episodeSet"]["staticScenes"]
    ref_render_assets = ref_episode_set["episodeSet"]["renderAssets"]
    ref_free_objects = ref_episode_set["episodeSet"]["freeObjects"]

    def get_static_scene_index(name):
        for i, obj in enumerate(ref_static_scenes):
            if obj["name"] == name:
                return i
        raise RuntimeError(
            "couldn't find static scene in ref episode set with name {}".format(
                name
            )
        )

    def get_free_object_index(name):
        for free_obj_index, obj in enumerate(ref_free_objects):
            renderAssetIndex = obj["renderAssetIndex"]
            ref_render_asset = ref_render_assets[renderAssetIndex]
            if ref_render_asset["name"] == name:
                return free_obj_index
        raise RuntimeError(
            "couldn't find free object in ref episode set with name {}".format(
                name
            )
        )

    with gzip.open(args.in_path, "r") as f:
        data = json.loads(f.read().decode("utf-8"))

    episodes = []
    # fixedObjects = []
    freeObjectSpawns = []
    freeObjects = []

    free_obj_name_to_idx = {}
    # all_fixed_objs = []
    all_static_scenes = []

    use_eps = data["episodes"]

    allowed_scenes = list(range(21))
    allowed_scs = [0, 1, 2, 3]
    allowed_objs = [
        "024_bowl",
        "003_cracker_box",
        "010_potted_meat_can",
        "002_master_chef_can",
        "004_sugar_box",
        "005_tomato_soup_can",
        "009_gelatin_box",
        "008_pudding_box",
        "007_tuna_fish_can",
    ]
    excluded_obj_names = []

    ep_inits = generate_inits(args.cfg_path, args.in_path, args.take_count)
    ep_counter = 0

    for ep in tqdm(use_eps):
        scene_name = convert_scene_name(ep["scene_id"])
        scene_idx = int(scene_name.split("_")[-1])
        sc_idx = int(scene_name.split("_")[1][len("sc") :])

        epid = ep["episode_id"]
        if epid not in ep_inits:
            continue
        if scene_idx not in allowed_scenes or sc_idx not in allowed_scs:
            continue

        # if scene_name in fixedObjects:
        #     stageFixedObjIndex = fixedObjects.index(scene_name)
        # else:
        #     fixedObjects.append({scene_name})
        #     all_fixed_objs.append(scene_name)
        #     stageFixedObjIndex = len(fixedObjects) - 1

        staticSceneIndex = get_static_scene_index(scene_name)

        firstFreeObjectSpawnIndex = len(freeObjectSpawns)
        for name, T in ep["rigid_objs"]:
            name = name.split(".")[0]
            freeObjIndex = get_free_object_index(name)

            T = fix_object_transform(name, orthogonalize(parse_matrix(T)))
            flat_rot = get_rotation_quaternion_list(T)

            freeObjectSpawns.append(
                {
                    "freeObjIndex": freeObjIndex,
                    "startRotation": flat_rot,
                    "startPos": list(T.translation),
                }
            )

        def find_target_object_index(target_instance, rigid_objs):
            target_rigid_obj_name = target_instance.split("_:")[0]
            target_count = int(target_instance.split("_:")[1])
            count = 0
            for obj_index, rigid_obj in enumerate(rigid_objs):
                rigid_obj_name = rigid_obj[0].split(".object_config.json")[0]
                if rigid_obj_name == target_rigid_obj_name:
                    if count == target_count:
                        return obj_index
                    count += 1
            raise RuntimeError(
                "couldn't find target {} in episode rigid_objs".format(
                    target_instance
                )
            )

        goal_name = next(iter(ep["targets"].keys()))

        targetObjIndex = find_target_object_index(goal_name, ep["rigid_objs"])

        goal_T = ep["targets"][goal_name]
        goal_object_name = goal_name.split("_:")[0]
        goal_T = fix_object_transform(goal_object_name, parse_matrix(goal_T))
        goal_flat_rot = get_rotation_quaternion_list(goal_T)

        targetObjGoalPos = list(goal_T.translation)
        start_pos = ep_inits[epid][1]
        start_yaw = float(ep_inits[epid][0])

        img = ep_inits[epid][3]
        # Uncomment if you want to visualize the starting frames of episodes.
        plt.imshow(img)
        plt.savefig(f"data/test/{ep_counter}.jpg")
        plt.clf()
        ep_counter += 1

        episodes.append(
            {
                "staticSceneIndex": staticSceneIndex,
                "firstFreeObjectSpawnIndex": firstFreeObjectSpawnIndex,
                "targetObjGoalPos": targetObjGoalPos,
                "targetObjGoalRotation": goal_flat_rot,
                "agentStartPos": [start_pos[0], start_pos[2]],
                "agentStartYaw": start_yaw,
                "targetObjIndex": targetObjIndex,
                "numFreeObjectSpawns": len(ep["rigid_objs"]),
                "robotStartJointPositions": ep_inits[epid][2].tolist(),
            }
        )

    out_dir = osp.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out_path, "w") as f:
        json.dump(
            {
                "episodeSet": {
                    "renderAssets": ref_render_assets,
                    "staticScenes": ref_static_scenes,
                    "freeObjects": ref_free_objects,
                    "freeObjectSpawns": freeObjectSpawns,
                    "episodes": episodes,
                }
            },
            f,
        )
        print(f"Wrote {len(episodes)} to {args.out_path}")
        print(f"Excluded {excluded_obj_names}")
