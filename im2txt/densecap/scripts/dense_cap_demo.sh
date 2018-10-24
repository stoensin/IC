#!/usr/bin/env bash

set -x
set -e

ckpt=$1
vocab=$2

if [ -d '/content' ]; then
    ckpt='/content/im2txt/densecap/output/dc_context/vg_1.2_train'
    vocab='/content/visual_genome/1.2/vocabulary.txt'
fi

time python ./op/inference.py \
    --ckpt ${ckpt} \
    --cfg  scripts/dense_cap_config.yml \
    --vocab ${vocab} \
    --set TEST.USE_BEAM_SEARCH True EMBED_DIM 512 TEST.LN_FACTOR 1. TEST.RPN_NMS_THRESH 0.7 TEST.NMS 0.3
