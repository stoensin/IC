#!/usr/bin/env bash

set -x
set -e

ckpt=$1
vocab=$2

if [ -d '/content' ]; then
    ckpt='/content/im2txt/output/im2p_finetune/vg_1.2_train'
    vocab='/content/output/vocabulary.txt'
    TEST_IMDB="vg_1.2_test"
fi

LOG="../logs/test_log.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python ../test_net.py  \
  --ckpt ${ckpt} \
  --imdb ${TEST_IMDB} \
  --cfg densecap/scripts/dense_cap_config.yml \
  --set ALL_TEST True
