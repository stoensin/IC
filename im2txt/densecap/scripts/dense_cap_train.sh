#!/usr/bin/env bash

set -x
set -e

export PYTHONUNBUFFERED='True'

DATASET=$1
NET=$2
ckpt_path=$3
data_dir=$4
step=$5

if [ -d '/content' ]; then
    DATASET='visual_genome_1.2'
    NET='res50'
    ckpt_path="res101/res101.ckpt"
    data_dir='/content/visual_genome'
fi

case $DATASET in
   visual_genome)
    TRAIN_IMDB="vg_1.0_train"
    TEST_IMDB="vg_1.0_val"
    PT_DIR="dense_cap"
    FINETUNE_AFTER1=200000
    FINETUNE_AFTER2=100000
    ITERS1=400000
    ITERS2=300000
    ;;
  visual_genome_1.2)
    TRAIN_IMDB="vg_1.2_train"
    TEST_IMDB="vg_1.2_val"
    PT_DIR="dense_cap"
    FINETUNE_AFTER1=200000
    FINETUNE_AFTER2=100000
    ITERS1=400000
    ITERS2=300000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac


LOG="logs/s${step}_${NET}_${TRAIN_IMDB}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"

exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# First step, freeze conv nets weights
if [ ${step} -lt '2' ]
then
time python ./op/train_net.py \
    --weights ${ckpt_path} \
    --imdb ${TRAIN_IMDB} \
    --imdbval ${TEST_IMDB} \
    --iters ${FINETUNE_AFTER1}\
    --cfg scripts/dense_cap_config.yml \
    --data_dir ${data_dir} \
    --net ${NET} \
    --set EXP_DIR dc_conv_fixed CONTEXT_FUSION False RESNET.FIXED_BLOCKS 3
fi

# Step2: Finetune convnets
NEW_WIGHTS=output/dc_conv_fixed/${TRAIN_IMDB}
if [ ${step} -lt '3' ]
then
time python ./op/train_net.py \
    --weights ${NEW_WIGHTS} \
    --imdb ${TRAIN_IMDB} \
    --iters `expr ${ITERS1} - ${FINETUNE_AFTER1}` \
    --imdbval ${TEST_IMDB} \
    --cfg scripts/dense_cap_config.yml \
    --data_dir ${data_dir} \
    --net ${NET} \
    --set EXP_DIR dc_tune_conv CONTEXT_FUSION False RESNET.FIXED_BLOCKS 1 TRAIN.LEARNING_RATE 0.00025
fi

# Step3: train with contex fusion
NEW_WIGHTS=output/dc_tune_conv/${TRAIN_IMDB}
if [ ${step} -lt '4' ]
then
time python ./op/train_net.py \
    --weights ${NEW_WIGHTS} \
    --imdb ${TRAIN_IMDB} \
    --imdbval ${TEST_IMDB} \
    --iters ${FINETUNE_AFTER2} \
    --cfg scripts/dense_cap_config.yml \
    --data_dir ${data_dir} \
    --net ${NET} \
    --set EXP_DIR dc_context CONTEXT_FUSION True RESNET.FIXED_BLOCKS 3 TRAIN.LEARNING_RATE 0.000125
fi

# Step4: finetune context fusion
NEW_WIGHTS=output/dc_context/${TRAIN_IMDB}
if [ ${step} -lt '5' ]
then
time python ./op/train_net.py \
    --weights ${NEW_WIGHTS} \
    --imdb ${TRAIN_IMDB} \
    --imdbval ${TEST_IMDB} \
    --iters `expr ${ITERS2} - ${FINETUNE_AFTER2}` \
    --cfg scripts/dense_cap_config.yml \
    --data_dir ${data_dir} \
    --net ${NET} \
    --set EXP_DIR dc_tune_context CONTEXT_FUSION True RESNET.FIXED_BLOCKS 1 TRAIN.LEARNING_RATE 0.0000625
fi
