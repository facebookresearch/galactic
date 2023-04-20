#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.


# note gala_kinematic branch has vital fixes
PREPROCESS="./habitat-sim/build_relwithdebinfo/deps/bps3D/bin/preprocess"

# see build instructions: https://github.com/shacklettbp/bps-nav#texture-compression
TOKTX="./KTX-Software/build/tools/toktx/toktx"

INPUT_BASE="temp/composite"
OUTPUT_PATH="data/bps_data/composite/"
OUTPUT_BASE="${OUTPUT_PATH}composite"
TEXTURE_OUTPUT_PATH="${OUTPUT_PATH}textures"

mkdir ${OUTPUT_PATH}
mkdir ${TEXTURE_OUTPUT_PATH}

echo "preprocessing ${INPUT_BASE}.glb..."
${PREPROCESS} "${INPUT_BASE}.glb" "${OUTPUT_BASE}.bps" right up forward ${TEXTURE_OUTPUT_PATH} --texture-dump --no-write-instances

# note it is unsafe to re-use existing ktx2 files (they could be from outdated source textures)
for x in `find ${TEXTURE_OUTPUT_PATH} \( -name "*.png" -o -name "*.jpeg" -o -name "*.jpg" \)`; do
    # change extension to .ktx2
    output_filepath=${x%.*}.ktx2
    # echo "output_filepath: ${output_filepath}"
    if test ! -f "$output_filepath"; then
        echo "compressing $x..."
        ${TOKTX} --uastc --uastc_rdo_d 1024 --zcmp 20 --genmipmap $output_filepath "${TEXTURE_OUTPUT_PATH}/`basename $x`"
    fi
    rm "${TEXTURE_OUTPUT_PATH}/`basename $x`"
done