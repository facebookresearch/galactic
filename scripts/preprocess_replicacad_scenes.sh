#!/bin/bash

cd habitat-sim

LEGACY_VIEWER="build_relwithdebinfo/utils/viewer/viewer"
SOURCE_SCENE_PATH="../source_data/replica_cad_baked_lighting/configs/scenes"

if ! [ -d ${SOURCE_SCENE_PATH} ]; then
  echo "Couldn't find ${SOURCE_SCENE_PATH} directory from ./habitat-sim ."
  exit 1
fi

for x in `find ${SOURCE_SCENE_PATH} -name '*.scene_instance.json'`; do
  echo "preprocessing $x..."
  ${LEGACY_VIEWER} $x --dataset "../source_data/replica_cad_baked_lighting/replicaCAD_baked.scene_dataset_config.json" --enable-physics --gala-write-scene-gfx-replay --gala-generate-column-grids
  RESULT=$?
  if [ $RESULT != 0 ]; then
    exit 1
  fi
done
