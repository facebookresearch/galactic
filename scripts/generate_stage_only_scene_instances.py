#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import glob
import os
from argparse import ArgumentParser, Namespace
from PIL import Image
import numpy as np
from habitat_sim.utils import viz_utils as vut

def get_filepaths_from_directory(directory, filepath_glob):
    """Returns a list of filepaths."""
    filepaths = []
    for filepath in glob.glob(directory + "/*" + filepath_glob):
        filepaths.append(filepath)
    return filepaths


def create_arg_parser() -> ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage-dir",
        default="./",
    )
    parser.add_argument(
        "--template-filepath",
        default="./",
    )
    parser.add_argument(
        "--scene-instance-dir",
        default="./",
    )
    
    return parser


def main():
    args = create_arg_parser().parse_args()

    f = open(args.template_filepath, "r")
    template_str = f.read()
    f.close()

    stage_filepaths = get_filepaths_from_directory(args.stage_dir, "*.glb")

    if len(stage_filepaths) == 0:
        print("No GLB files were found at {}".format(args.stage_dir))
        return

    stage_filepaths.sort()  # sort alphabetically

    for (i, stage_filepath) in enumerate(stage_filepaths):

        stage_name = os.path.splitext(os.path.basename(stage_filepath))[0]
        print("processing stage {}...".format(stage_name))
        scene_instance_str = template_str.replace("{stage_name}", stage_name)
        scene_instance_filepath = args.scene_instance_dir + "/" + stage_name + "_stage_only.scene_instance.json"
        f = open(scene_instance_filepath, "x")
        f.write(scene_instance_str)
        f.close()

if __name__ == "__main__":
    main()