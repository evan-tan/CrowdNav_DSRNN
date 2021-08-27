#!/usr/bin/env bash

# space separated integers e.g. (0 1 2)
declare -a arr=()
# path to model
MODEL_DIR="data/fov_360_fps_scaled_reward_NOT_norm/"
# .pt file to load
TEST_MODEL=""

for i in "${arr[@]}"
do
    python3 test.py --model_dir="$MODEL_DIR" \
    --test_model="$TEST_MODEL" --num_threads=2 --test_case="$i" --visualize
done
