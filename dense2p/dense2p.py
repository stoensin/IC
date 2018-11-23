#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import cv2
import shutil
import itertools
import tqdm
import numpy as np
import json
import six
import pickle
import pandas as pd
import random
import h5py
import pdb

import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
try:
    import horovod.tensorflow as hvd
except ImportError:
    pass

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils import optimizer
from tensorpack.tfutils.common import get_tf_version_tuple
from tensorpack.models import (
    Conv2D, FullyConnected, layer_register)
import tensorpack.utils.viz as tpviz

from region_detector.basemodel import (
    image_preprocess, resnet_c4_backbone, resnet_conv5,
    resnet_fpn_backbone)
import region_detector.model_frcnn as model_frcnn
import region_detector.model_mrcnn as model_mrcnn
from region_detector.model_frcnn import (
    sample_fast_rcnn_targets, fastrcnn_outputs,
    fastrcnn_predictions, BoxProposals, FastRCNNHead)
from region_detector.model_mrcnn import maskrcnn_upXconv_head, maskrcnn_loss
from region_detector.model_rpn import rpn_head, rpn_losses, generate_rpn_proposals
from region_detector.model_fpn import (
    fpn_model, multilevel_roi_align,
    multilevel_rpn_losses, generate_fpn_proposals)
from region_detector.model_cascade import CascadeRCNNHead
from region_detector.model_box import (
    clip_boxes, crop_and_resize, roi_align, RPNAnchors)
from visual_genome.dataset import (
    get_train_dataflow, get_eval_dataflow,
    get_all_anchors, get_all_anchors_fpn)

from region_detector.config import finalize_configs, config as cfg


class DetectionModel(ModelDesc):
    def preprocess(self, image):
        image = tf.expand_dims(image, 0)
        image = image_preprocess(image, bgr=True)
        return tf.transpose(image, [0, 3, 1, 2])

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.003, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)

        # The learning rate is set for 8 GPUs, and we use trainers with average=False.
        lr = lr / 8.
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        if cfg.TRAIN.NUM_GPUS < 8:
            opt = optimizer.AccumGradOptimizer(opt, 8 // cfg.TRAIN.NUM_GPUS)
        return opt

    def get_inference_tensor_names(self):
        """
        Returns two lists of tensor names to be used to create an inference callable.
        Returns:
            [str]: input names
            [str]: output names
        """
        out = ['output/boxes', 'output/scores', 'output/labels']
        if cfg.MODE_MASK:
            out.append('output/masks')
        return ['image'], out


class ResNetC4Model(DetectionModel):

    def inputs(self):
        ret = [
            tf.placeholder(tf.float32, (None, None, 3), 'image'),
            tf.placeholder(tf.int32, (None, None, cfg.RPN.NUM_ANCHOR), 'anchor_labels'),
            tf.placeholder(tf.float32, (None, None, cfg.RPN.NUM_ANCHOR, 4), 'anchor_boxes'),
            tf.placeholder(tf.float32, (None, 4), 'gt_boxes'),
            tf.placeholder(tf.int64, (None,), 'gt_labels'),  # all > 0
            tf.placeholder(tf.int32, (None, 6), 'num_distribution'),
            tf.placeholder(tf.float32, (None, 6, 51), 'captions_masks'),
            tf.placeholder(tf.int32, (None, 6, 51), 'captions')]  # sentence_labels
        if cfg.MODE_MASK:
            ret.append(
                tf.placeholder(tf.uint8, (None, None, None), 'gt_masks')
            )   # NR_GT x height x width
        return ret

    def build_graph(self, *inputs):

        inputs = dict(zip(self.input_names, inputs))
        is_training = get_current_tower_context().is_training
        image = self.preprocess(inputs['image'])     # 1CHW

        featuremap = resnet_c4_backbone(image, cfg.BACKBONE.RESNET_NUM_BLOCK[:3])
        rpn_label_logits, rpn_box_logits = rpn_head('rpn', featuremap, cfg.RPN.HEAD_DIM, cfg.RPN.NUM_ANCHOR)

        anchors = RPNAnchors(get_all_anchors(), inputs['anchor_labels'], inputs['anchor_boxes'])
        anchors = anchors.narrow_to(featuremap)

        image_shape2d = tf.shape(image)[2:]     # h,w
        pred_boxes_decoded = anchors.decode_logits(rpn_box_logits)  # fHxfWxNAx4, floatbox
        proposal_boxes, proposal_scores = generate_rpn_proposals(
            tf.reshape(pred_boxes_decoded, [-1, 4]),
            tf.reshape(rpn_label_logits, [-1]),
            image_shape2d,
            cfg.RPN.TRAIN_PRE_NMS_TOPK if is_training else cfg.RPN.TEST_PRE_NMS_TOPK,
            cfg.RPN.TRAIN_POST_NMS_TOPK if is_training else cfg.RPN.TEST_POST_NMS_TOPK)

        gt_boxes, gt_labels = inputs['gt_boxes'], inputs['gt_labels']
        if is_training:
            # sample proposal boxes in training
            proposals = sample_fast_rcnn_targets(proposal_boxes, gt_boxes, gt_labels)
        else:
            # The boxes to be used to crop RoIs.
            # Use all proposal boxes in inference
            proposals = BoxProposals(proposal_boxes)

        boxes_on_featuremap = proposals.boxes * (1.0 / cfg.RPN.ANCHOR_STRIDE)
        roi_resized = roi_align(featuremap, boxes_on_featuremap, 14)

        feature_fastrcnn = resnet_conv5(roi_resized, cfg.BACKBONE.RESNET_NUM_BLOCK[-1])    # nxcx7x7
        # Keep C5 feature to be shared with mask branch
        feature_gap = GlobalAvgPooling('gap', feature_fastrcnn, data_format='channels_first')
        rois, fastrcnn_label_logits, fastrcnn_box_logits = fastrcnn_outputs('fastrcnn', feature_gap, cfg.DATA.NUM_CLASS)

        fastrcnn_head = FastRCNNHead(proposals, fastrcnn_box_logits, fastrcnn_label_logits,
                                     tf.constant(cfg.FRCNN.BBOX_REG_WEIGHTS, dtype=tf.float32))

        if is_training:
            all_losses = []
            # rpn loss
            all_losses.extend(rpn_losses(
                anchors.gt_labels, anchors.encoded_gt_boxes(), rpn_label_logits, rpn_box_logits))

            # fastrcnn loss
            all_losses.extend(fastrcnn_head.losses())

            if cfg.MODE_MASK:
                # maskrcnn loss
                # In training, mask branch shares the same C5 feature.
                fg_feature = tf.gather(feature_fastrcnn, proposals.fg_inds())
                mask_logits = maskrcnn_upXconv_head(
                    'maskrcnn', fg_feature, cfg.DATA.NUM_CATEGORY, num_convs=0)   # #fg x #cat x 14x14

                target_masks_for_fg = crop_and_resize(
                    tf.expand_dims(inputs['gt_masks'], 1),
                    proposals.fg_boxes(),
                    proposals.fg_inds_wrt_gt, 14,
                    pad_border=False)  # nfg x 1x14x14
                target_masks_for_fg = tf.squeeze(target_masks_for_fg, 1, 'sampled_fg_mask_targets')
                all_losses.append(maskrcnn_loss(mask_logits, proposals.fg_labels(), target_masks_for_fg))

            wd_cost = regularize_cost(
                '.*/W', l2_regularizer(cfg.TRAIN.WEIGHT_DECAY), name='wd_cost')
            all_losses.append(wd_cost)

            total_cost = tf.add_n(all_losses, 'total_cost')
            # add_moving_summary(total_cost, wd_cost)

            dense2p_loss = Dense2pModel()._hierarchicalRNN_layer(rois, inputs)
            final_loss = tf.identity(dense2p_loss, name="dense2p_loss")

            add_moving_summary(total_cost, final_loss)
            return final_loss

        else:

            decoded_boxes = fastrcnn_head.decoded_output_boxes()
            decoded_boxes = clip_boxes(decoded_boxes, image_shape2d, name='fastrcnn_all_boxes')
            label_scores = fastrcnn_head.output_scores(name='fastrcnn_all_scores')
            final_boxes, final_scores, final_labels = fastrcnn_predictions(
                decoded_boxes, label_scores, name_scope='output')

            if cfg.MODE_MASK:
                roi_resized = roi_align(featuremap, final_boxes * (1.0 / cfg.RPN.ANCHOR_STRIDE), 14)
                feature_maskrcnn = resnet_conv5(roi_resized, cfg.BACKBONE.RESNET_NUM_BLOCK[-1])
                mask_logits = maskrcnn_upXconv_head(
                    'maskrcnn', feature_maskrcnn, cfg.DATA.NUM_CATEGORY, 0)   # #result x #cat x 14x14
                indices = tf.stack([tf.range(tf.size(final_labels)), tf.to_int32(final_labels) - 1], axis=1)
                final_mask_logits = tf.gather_nd(mask_logits, indices)   # #resultx14x14
                tf.sigmoid(final_mask_logits, name='output/masks')


class ResNetFPNModel(DetectionModel):

    def inputs(self):
        ret = [
            tf.placeholder(tf.float32, (None, None, 3), 'image'),
            tf.placeholder(tf.int32, (None, 6), 'num_distribution'),
            tf.placeholder(tf.float32, (None, 6, 51), 'captions_masks'),
            tf.placeholder(tf.int32, (None, 6, 51), 'captions')]  # sentence_labels
        num_anchors = len(cfg.RPN.ANCHOR_RATIOS)
        for k in range(len(cfg.FPN.ANCHOR_STRIDES)):
            ret.extend([
                tf.placeholder(tf.int32, (None, None, num_anchors),
                               'anchor_labels_lvl{}'.format(k + 2)),
                tf.placeholder(tf.float32, (None, None, num_anchors, 4),
                               'anchor_boxes_lvl{}'.format(k + 2))])
        ret.extend([
            tf.placeholder(tf.float32, (None, 4), 'gt_boxes'),
            tf.placeholder(tf.int64, (None,), 'gt_labels')])
        if cfg.MODE_MASK:
            ret.append(
                tf.placeholder(tf.uint8, (None, None, None), 'gt_masks')
            )   # NR_GT x height x width
        return ret

    def slice_feature_and_anchors(self, image_shape2d, p23456, anchors):
        for i, stride in enumerate(cfg.FPN.ANCHOR_STRIDES):
            with tf.name_scope('FPN_slice_lvl{}'.format(i)):
                if i < 3:
                    # Images are padded for p5, which are too large for p2-p4.
                    # This seems to have no effect on mAP.
                    pi = p23456[i]
                    target_shape = tf.to_int32(tf.ceil(tf.to_float(image_shape2d) * (1.0 / stride)))
                    p23456[i] = tf.slice(pi, [0, 0, 0, 0],
                                         tf.concat([[-1, -1], target_shape], axis=0))
                    p23456[i].set_shape([1, pi.shape[1], None, None])

                anchors[i] = anchors[i].narrow_to(p23456[i])

    def build_graph(self, *inputs):
        inputs = dict(zip(self.input_names, inputs))
        num_fpn_level = len(cfg.FPN.ANCHOR_STRIDES)
        assert len(cfg.RPN.ANCHOR_SIZES) == num_fpn_level
        is_training = get_current_tower_context().is_training

        all_anchors_fpn = get_all_anchors_fpn()
        multilevel_anchors = [RPNAnchors(
            all_anchors_fpn[i],
            inputs['anchor_labels_lvl{}'.format(i + 2)],
            inputs['anchor_boxes_lvl{}'.format(i + 2)]) for i in range(len(all_anchors_fpn))]

        image = self.preprocess(inputs['image'])     # 1CHW
        image_shape2d = tf.shape(image)[2:]     # h,w

        c2345 = resnet_fpn_backbone(image, cfg.BACKBONE.RESNET_NUM_BLOCK)
        p23456 = fpn_model('fpn', c2345)
        self.slice_feature_and_anchors(image_shape2d, p23456, multilevel_anchors)

        # Multi-Level RPN Proposals
        rpn_outputs = [rpn_head('rpn', pi, cfg.FPN.NUM_CHANNEL, len(cfg.RPN.ANCHOR_RATIOS))
                       for pi in p23456]
        multilevel_label_logits = [k[0] for k in rpn_outputs]
        multilevel_box_logits = [k[1] for k in rpn_outputs]

        proposal_boxes, proposal_scores = generate_fpn_proposals(
            multilevel_anchors, multilevel_label_logits,
            multilevel_box_logits, image_shape2d)

        gt_boxes, gt_labels = inputs['gt_boxes'], inputs['gt_labels']
        if is_training:
            proposals = sample_fast_rcnn_targets(proposal_boxes, gt_boxes, gt_labels)
        else:
            proposals = BoxProposals(proposal_boxes)

        fastrcnn_head_func = getattr(model_frcnn, cfg.FPN.FRCNN_HEAD_FUNC)
        if not cfg.FPN.CASCADE:
            roi_feature_fastrcnn = multilevel_roi_align(p23456[:4], proposals.boxes, 7)

            head_feature = fastrcnn_head_func('fastrcnn', roi_feature_fastrcnn)
            rois, fastrcnn_label_logits, fastrcnn_box_logits = fastrcnn_outputs(
                'fastrcnn/outputs', head_feature, cfg.DATA.NUM_CLASS)
            fastrcnn_head = FastRCNNHead(proposals, fastrcnn_box_logits, fastrcnn_label_logits,
                                         tf.constant(cfg.FRCNN.BBOX_REG_WEIGHTS, dtype=tf.float32))
        else:
            def roi_func(boxes):
                return multilevel_roi_align(p23456[:4], boxes, 7)

            fastrcnn_head = CascadeRCNNHead(
                proposals, roi_func, fastrcnn_head_func, image_shape2d, cfg.DATA.NUM_CLASS)

        if is_training:
            all_losses = []
            all_losses.extend(multilevel_rpn_losses(
                multilevel_anchors, multilevel_label_logits, multilevel_box_logits))

            all_losses.extend(fastrcnn_head.losses())

            if cfg.MODE_MASK:
                # maskrcnn loss
                roi_feature_maskrcnn = multilevel_roi_align(
                    p23456[:4], proposals.fg_boxes(), 14,
                    name_scope='multilevel_roi_align_mask')
                maskrcnn_head_func = getattr(model_mrcnn, cfg.FPN.MRCNN_HEAD_FUNC)
                mask_logits = maskrcnn_head_func(
                    'maskrcnn', roi_feature_maskrcnn, cfg.DATA.NUM_CATEGORY)   # #fg x #cat x 28 x 28

                target_masks_for_fg = crop_and_resize(
                    tf.expand_dims(inputs['gt_masks'], 1),
                    proposals.fg_boxes(),
                    proposals.fg_inds_wrt_gt, 28,
                    pad_border=False)  # fg x 1x28x28
                target_masks_for_fg = tf.squeeze(target_masks_for_fg, 1, 'sampled_fg_mask_targets')
                all_losses.append(maskrcnn_loss(mask_logits, proposals.fg_labels(), target_masks_for_fg))

            wd_cost = regularize_cost(
                '.*/W', l2_regularizer(cfg.TRAIN.WEIGHT_DECAY), name='wd_cost')
            all_losses.append(wd_cost)

            total_cost = tf.add_n(all_losses, 'total_cost')
            add_moving_summary(total_cost, wd_cost)

            dense2p_loss = Dense2pModel()._hierarchicalRNN_layer(rois, inputs)
            final_loss = tf.identity(dense2p_loss, name="dense2p_loss")

            add_moving_summary(total_cost, final_loss)
            return final_loss

        else:
            decoded_boxes = fastrcnn_head.decoded_output_boxes()
            decoded_boxes = clip_boxes(decoded_boxes, image_shape2d, name='fastrcnn_all_boxes')
            label_scores = fastrcnn_head.output_scores(name='fastrcnn_all_scores')
            final_boxes, final_scores, final_labels = fastrcnn_predictions(
                decoded_boxes, label_scores, name_scope='output')
            if cfg.MODE_MASK:
                # Cascade inference needs roi transform with refined boxes.
                roi_feature_maskrcnn = multilevel_roi_align(p23456[:4], final_boxes, 14)
                maskrcnn_head_func = getattr(model_mrcnn, cfg.FPN.MRCNN_HEAD_FUNC)
                mask_logits = maskrcnn_head_func(
                    'maskrcnn', roi_feature_maskrcnn, cfg.DATA.NUM_CATEGORY)   # #fg x #cat x 28 x 28
                indices = tf.stack([tf.range(tf.size(final_labels)), tf.to_int32(final_labels) - 1], axis=1)
                final_mask_logits = tf.gather_nd(mask_logits, indices)   # #resultx28x28
                tf.sigmoid(final_mask_logits, name='output/masks')


class Dense2pModel(ModelDesc):

    def __init__(self):

        # FOR RegionPooling_HierarchicalRNN
        self.n_words = 9904
        self.batch_size = 1
        self.num_boxes = 32  # 50
        self.feats_dim = 2048  # 4096
        self.project_dim = 1024  # 1024
        self.S_max = 6  # 6
        self.N_max = 50  # 50
        self.word_embed_dim = 1024  # 1024

        self.sentRNN_lstm_dim = 512  # 512 hidden size
        self.sentRNN_FC_dim = 1024  # 1024 in fully connected layer
        self.wordRNN_lstm_dim = 512  # 512 hidden size
        self.bias_init_vector = None

        # embedding shape: n_words x wordRNN_lstm_dim
        with tf.device('/cpu:0'):
            with tf.variable_scope('Wemb'):
                self.Wemb = tf.random_uniform([self.n_words, self.word_embed_dim], -0.1, 0.1)

        # regionPooling_W shape: 4096 x 1024
        # regionPooling_b shape: 1024
        with tf.variable_scope('regionPooling_W'):
            self.regionPooling_W = tf.random_uniform([self.feats_dim, self.project_dim], -0.1, 0.1)
        with tf.variable_scope('regionPooling_b'):
            self.regionPooling_b = tf.zeros([self.project_dim])

        # sentence LSTM
        self.sentence_LSTM = tf.nn.rnn_cell.BasicLSTMCell(self.sentRNN_lstm_dim, state_is_tuple=True)

        # logistic classifier
        with tf.variable_scope('logistic_Theta_W'):
            self.logistic_Theta_W = tf.random_uniform([self.sentRNN_lstm_dim, 2], -0.1, 0.1)
        with tf.variable_scope('logistic_Theta_b'):
            self.logistic_Theta_b = tf.zeros(2)

        # fc1_W: 512 x 1024, fc1_b: 1024
        # fc2_W: 1024 x 1024, fc2_b: 1024
        with tf.variable_scope('sentlstm_fc1'):
            self.fc1_W = tf.random_uniform([self.sentRNN_lstm_dim, self.sentRNN_FC_dim], -0.1, 0.1)
            self.fc1_b = tf.zeros(self.sentRNN_FC_dim)
        with tf.variable_scope('sentlstm_fc2'):
            self.fc2_W = tf.random_uniform([self.sentRNN_FC_dim, 1024], -0.1, 0.1)
            self.fc2_b = tf.zeros(1024)

        # word LSTM
        def wordLSTM():
            lstm = tf.nn.rnn_cell.BasicLSTMCell(self.wordRNN_lstm_dim, state_is_tuple=True)
            return lstm

        self.word_LSTM = tf.nn.rnn_cell.MultiRNNCell([wordLSTM() for _ in range(2)], state_is_tuple=True)
        with tf.variable_scope('embed_word_W'):
            self.embed_word_W = tf.random_uniform([self.wordRNN_lstm_dim, self.n_words], -0.1, 0.1)

        if self.bias_init_vector is not None:
            self.embed_word_b = self.bias_init_vector.astype(np.float32)
        else:
            self.embed_word_b = tf.zeros([self.n_words])

    def _hierarchicalRNN_layer(self, region_featurs, inputs):
        # region_featurs's shape is 50 x 4096
        # tmp_feats: 50 x 4096
        # feats = tf.placeholder(tf.float32, [self.batch_size, self.num_boxes, self.feats_dim])
        feats = region_featurs

        # project_vec_all: 50 x 4096 * 4096 x 1024 --> 50 x 1024 ; project_vec: 10 x 1024
        with tf.name_scope('project_vec_all'):
            project_vec_all = tf.matmul(feats, self.regionPooling_W) + self.regionPooling_b

        # project_vec_all.set_shape([None,1024])
        project_vec_alls = tf.expand_dims(project_vec_all, 0)  # tf.reshape(project_vec_all, [1, 16, 1024])
        # project_vec_alls.set_shape([1,None,1024])

        with tf.name_scope('project_vec'):
            project_vec = tf.reduce_max(project_vec_alls, reduction_indices=1)

        # receive the [continue:0, stop:1] lists  example: [0, 0, 0, 0, 1, 1], it means this paragraph has five sentences
        num_distribution = inputs['num_distribution']  # tf.placeholder(tf.int32, [self.batch_size, self.S_max])
        # receive the ground truth words, which has been changed to idx use word2idx function
        captions = inputs['captions']  # tf.placeholder(tf.int32, [self.batch_size, self.S_max, self.N_max+1])
        captions_masks = inputs['captions_masks']  # tf.placeholder(tf.float32, [self.batch_size, self.S_max, self.N_max+1])

        sentence_state = self.sentence_LSTM.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        probs = []
        loss = 0.0
        loss_sent = 0.0
        loss_word = 0.0
        lambda_sent = 5.0
        lambda_word = 1.0

        # ----------------------------------------------------------------------------------------------
        # Hierarchical RNN: sentence RNN and words RNN
        # The word RNN has the max number, N_max = 50, the number in the papar is 50
        # ----------------------------------------------------------------------------------------------
        for i in range(0, self.S_max):

            with tf.variable_scope('sentence_LSTM', reuse=tf.AUTO_REUSE):
                sentence_output, sentence_state = self.sentence_LSTM(project_vec, sentence_state)

            with tf.name_scope('fc1'):
                hidden1 = tf.nn.relu(tf.matmul(sentence_output, self.fc1_W) + self.fc1_b)
            with tf.name_scope('fc2'):
                sentence_topic_vec = tf.nn.relu(tf.matmul(hidden1, self.fc2_W) + self.fc2_b)

            # sentence_state is a tuple, sentence_state = (c, h)
            # 'c': shape=(1, 512) dtype=float32, 'h': shape=(1, 512) dtype=float32
            sentRNN_logistic_mu = tf.nn.xw_plus_b(sentence_output, self.logistic_Theta_W, self.logistic_Theta_b)
            sentRNN_label = tf.stack([1 - num_distribution[:, i], num_distribution[:, i]])
            sentRNN_label = tf.transpose(sentRNN_label)

            sentRNN_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=sentRNN_label, logits=sentRNN_logistic_mu)
            sentRNN_loss = tf.reduce_sum(sentRNN_loss)/self.batch_size

            loss += sentRNN_loss * lambda_sent
            loss_sent += sentRNN_loss

            # the begining input of word_LSTM is topic vector, and DON'T compute the loss
            topic = tf.nn.rnn_cell.LSTMStateTuple(sentence_topic_vec[:, 0:512], sentence_topic_vec[:, 512:])
            word_state = (topic, topic)

            for j in range(0, self.N_max):

                with tf.device('/cpu:0'):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, captions[:, i, j])

                with tf.variable_scope('word_LSTM', reuse=tf.AUTO_REUSE):
                    word_output, word_state = self.word_LSTM(current_embed, word_state)

                labels = tf.reshape(captions[:, i, j+1], [-1, 1])
                indices = tf.reshape(tf.range(0, self.batch_size, 1), [-1, 1])

                concated = tf.concat([indices, labels], 1)
                onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)

                # At each timestep the hidden state of the last LSTM layer is used to predict a distribution over the words in the vocbulary
                logit_words = tf.nn.xw_plus_b(word_output[:], self.embed_word_W, self.embed_word_b)
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit_words, labels=onehot_labels)
                cross_entropy = cross_entropy * captions_masks[:, i, j]
                loss_wordRNN = tf.reduce_sum(cross_entropy) / self.batch_size
                loss += loss_wordRNN * lambda_word
                loss_word += loss_wordRNN

        return feats, num_distribution, captions, captions_masks, loss, loss_sent, loss_word

    def _hierarchicalRNN_generate_layer(self, region_featurs):
        # feats: 1 x 50 x 4096
        # feats = tf.placeholder(tf.float32, [1, self.num_boxes, self.feats_dim])
        feats = region_featurs
        # tmp_feats: 50 x 4096
        tmp_feats = tf.reshape(feats, [-1, self.feats_dim])

        # project_vec_all: 50 x 4096 * 4096 x 1024 + 1024 --> 50 x 1024
        project_vec_all = tf.matmul(tmp_feats, self.regionPooling_W) + self.regionPooling_b
        project_vec_all = tf.reshape(project_vec_all, [1, 50, self.project_dim])
        project_vec = tf.reduce_max(project_vec_all, reduction_indices=1)

        # initialize the sentence_LSTM state
        sentence_state = self.sentence_LSTM.zero_state(batch_size=1, dtype=tf.float32)
        # save the generated paragraph to list, here I named generated_sents
        generated_paragraph = []

        # pred
        pred_re = []

        # T_stop: run the sentence RNN forward until the stopping probability p_i (STOP) exceeds a threshold T_stop
        T_stop = tf.constant(0.5)

        # sentence RNN
        for i in range(0, self.S_max):

            # sentence_state:
            # LSTMStateTuple(c=<tf.Tensor 'sentence_LSTM/BasicLSTMCell/add_2:0' shape=(1, 512) dtype=float32>,
            #                h=<tf.Tensor 'sentence_LSTM/BasicLSTMCell/mul_2:0' shape=(1, 512) dtype=float32>)
            with tf.variable_scope('sentence_LSTM', reuse=tf.AUTO_REUSE):
                sentence_output, sentence_state = self.sentence_LSTM(project_vec, sentence_state)

            # self.fc1_W: 512 x 1024, self.fc1_b: 1024
            # hidden1: 1 x 1024
            # sentence_topic_vec: 1 x 1024
            with tf.name_scope('fc1'):
                hidden1 = tf.nn.relu(tf.matmul(sentence_output, self.fc1_W) + self.fc1_b)
            with tf.name_scope('fc2'):
                sentence_topic_vec = tf.nn.relu(tf.matmul(hidden1, self.fc2_W) + self.fc2_b)

            sentRNN_logistic_mu = tf.nn.xw_plus_b(sentence_output, self.logistic_Theta_W, self.logistic_Theta_b)
            pred = tf.nn.softmax(sentRNN_logistic_mu)
            pred_re.append(pred)

            # save the generated sentence to list, named generated_sent
            generated_sent = []

            # initialize the word LSTM state
            topic = tf.nn.rnn_cell.LSTMStateTuple(sentence_topic_vec[:, 0:512], sentence_topic_vec[:, 512:])
            word_state = (topic, topic)
            # word RNN, unrolled to N_max time steps
            for j in range(0, self.N_max):

                if j == 0:
                    with tf.device('/cpu:0'):
                        # get word embedding of SOS (index = 0)
                        current_embed = tf.nn.embedding_lookup(self.Wemb, tf.zeros([1], dtype=tf.int64))

                with tf.variable_scope('word_LSTM', reuse=tf.AUTO_REUSE):
                    word_output, word_state = self.word_LSTM(current_embed, word_state)

                # word_state (1x512):
                logit_words = tf.nn.xw_plus_b(word_output, self.embed_word_W, self.embed_word_b)
                max_prob_index = tf.argmax(logit_words, 1)[0]
                generated_sent.append(max_prob_index)

                with tf.device('/cpu:0'):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
                    current_embed = tf.expand_dims(current_embed, 0)

            generated_paragraph.append(generated_sent)

        return feats, generated_paragraph, pred_re, generated_sent

    def create_architecture(self, mode, region_featurs, inputs):
        # 1. Pooling the visual features into a single dense feature
        # 2. Then, build sentence LSTM, word LSTM

        if mode == 'TRAIN':
            tf_feats, tf_num_distribution, tf_captions_matrix, tf_captions_masks, tf_loss, tf_loss_sent, tf_loss_word = self._hierarchicalRNN_layer(region_featurs, inputs)

            return tf_loss

        elif mode == 'TEST':
            tf_feats, tf_generated_paragraph, tf_pred_re, tf_sentence_topic_vectors = self._hierarchicalRNN_generate_layer(region_featurs)

    def train_net(args):

        MODEL = ResNetFPNModel() if cfg.MODE_FPN else ResNetC4Model()

        if args.visualize or args.evaluate or args.predict:
            pass
        else:
            is_horovod = cfg.TRAINER == 'horovod'
            if is_horovod:
                hvd.init()
                logger.info("Horovod Rank={}, Size={}".format(hvd.rank(), hvd.size()))

            if not is_horovod or hvd.rank() == 0:
                logger.set_logger_dir(args.logdir, 'd')

            finalize_configs(is_training=True)
            stepnum = cfg.TRAIN.STEPS_PER_EPOCH

            # warmup is step based, lr is epoch based
            init_lr = cfg.TRAIN.BASE_LR * 0.33 * min(8. / cfg.TRAIN.NUM_GPUS, 1.)
            warmup_schedule = [(0, init_lr), (cfg.TRAIN.WARMUP, cfg.TRAIN.BASE_LR)]
            warmup_end_epoch = cfg.TRAIN.WARMUP * 1. / stepnum
            lr_schedule = [(int(warmup_end_epoch + 0.5), cfg.TRAIN.BASE_LR)]

            factor = 8. / cfg.TRAIN.NUM_GPUS
            for idx, steps in enumerate(cfg.TRAIN.LR_SCHEDULE[:-1]):
                mult = 0.1 ** (idx + 1)
                lr_schedule.append(
                    (steps * factor // stepnum, cfg.TRAIN.BASE_LR * mult))
            logger.info("Warm Up Schedule (steps, value): " + str(warmup_schedule))
            logger.info("LR Schedule (epochs, value): " + str(lr_schedule))
            train_dataflow = get_train_dataflow()
            # This is what's commonly referred to as "epochs"
            total_passes = cfg.TRAIN.LR_SCHEDULE[-1] * 8 / train_dataflow.size()
            logger.info("Total passes of the training set is: {}".format(total_passes))

            callbacks = [
                PeriodicCallback(
                    ModelSaver(max_to_keep=10, keep_checkpoint_every_n_hours=1),
                    every_k_epochs=20),
                ScheduledHyperParamSetter(
                    'learning_rate', warmup_schedule, interp='linear', step_based=True),
                ScheduledHyperParamSetter('learning_rate', lr_schedule),
                # EvalCallback(*MODEL.get_inference_tensor_names()),
                PeakMemoryTracker(),
                EstimatedTimeLeft(median=True),
                SessionRunTimeout(60000).set_chief_only(True),   # 1 minute timeout
            ]
            if not is_horovod:
                callbacks.append(GPUUtilizationTracker())
            if is_horovod and hvd.rank() > 0:
                session_init = None
            else:
                if args.load:
                    session_init = get_model_loader(args.load)
                else:
                    session_init = get_model_loader(cfg.BACKBONE.WEIGHTS) if cfg.BACKBONE.WEIGHTS else None

            traincfg = TrainConfig(
                model=MODEL,
                data=QueueInput(train_dataflow),
                callbacks=callbacks,
                steps_per_epoch=stepnum,
                max_epoch=cfg.TRAIN.LR_SCHEDULE[-1] * factor // stepnum,
                session_init=session_init,
            )
            if is_horovod:
                trainer = HorovodTrainer(average=False)
            else:
                # nccl mode has better speed than cpu mode
                trainer = SyncMultiGPUTrainerReplicated(cfg.TRAIN.NUM_GPUS, average=False, mode='nccl')
            launch_train_with_config(traincfg, trainer)
