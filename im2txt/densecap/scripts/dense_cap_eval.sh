#!/usr/bin/env bash

set -x
set -e

GPU_ID=0
CKPT=$1
TEST_IMDB=$2


if [ -d '/content/' ]; then
    CKPT="/content/im2txt/densecap/output/dc_tune_context/vg_1.2_train"
    TEST_IMDB="vg_1.2_test"
fi

LOG="densecap/logs/test_log.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python densecap/op/test_net.py  \
  --ckpt ${CKPT} \
  --imdb ${TEST_IMDB} \
  --cfg densecap/scripts/dense_cap_config.yml \
  --set ALL_TEST True
