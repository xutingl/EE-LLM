#!/bin/bash

PROJECT_NAME=EE-LLM

export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Tokenizer
TOKENIZER_PATH=/home/xutingl/ee/EE-LLM/models/EE-LLM-1B-dj-refine-300B/tokenizer.model
# Checkpoint
CHECKPOINT_PATH=/home/xutingl/ee/EE-LLM/models/EE-LLM-1B-TP1-PP1/
# Parallelism
TP=1
PP=1
# Server port
PORT=5000

MASTER_ADDR=127.0.0.1
MASTER_PORT=5950
NPROC_PER_NODE=$(( $TP * $PP ))
LOAD_ITERATION=0

DIST_ARGS="
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes 1 \
    --node_rank 0 \
    "

SERVER_ARGS="
  --use-checkpoint-args \
  --tokenizer-type SentencePieceTokenizer \
  --tokenizer-model $TOKENIZER_PATH \
  --load $CHECKPOINT_PATH \
  --load-iteration $LOAD_ITERATION \
  --port $PORT
"

CUR_DIR=$(cd $(dirname "$0") && pwd)
MEGATRON_ROOT_PATH=$(cd "$CUR_DIR/../.." && pwd)
cd $MEGATRON_ROOT_PATH

torchrun $DIST_ARGS \
    tools/run_early_exit_text_generation_server.py \
    $SERVER_ARGS
