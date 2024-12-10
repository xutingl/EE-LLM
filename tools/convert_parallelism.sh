#!/bin/bash

# load/save dir
LOAD_DIR=/home/xutingl/ee/EE-LLM/models/EE-LLM-1B-dj-refine-300B/
SAVE_DIR=/home/xutingl/ee/EE-LLM/models/EE-LLM-1B-TP4-PP1/

# target parallelism
TP=4
PP=1

CUR_DIR=$(cd $(dirname "$0") && pwd)
MEGATRON_ROOT_PATH=$(cd "$CUR_DIR/.." && pwd)
cd $MEGATRON_ROOT_PATH

python $MEGATRON_ROOT_PATH/tools/checkpoint/util.py --model-type EarlyExitGPT --load-dir $LOAD_DIR --save-dir $SAVE_DIR --target-tensor-parallel-size $TP --target-pipeline-parallel-size $PP --megatron-path $MEGATRON_ROOT_PATH