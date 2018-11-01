#!/usr/bin/env bash

set -x
set -e

ckpt=$1
vocab=$2

if [ -d '/content' ]; then
    ckpt='/content/im2txt/densecap/output/im2p_finetune/vg_1.2_train'
    vocab='/content/output/vocabulary.txt'
fi

time python densecap/op/inference.py \
      --ckpt ${ckpt} \
      --cfg  densecap/scripts/dense_cap_config.yml \
      --vocab ${vocab} \
      --set TEST.USE_BEAM_SEARCH False EMBED_DIM 512
