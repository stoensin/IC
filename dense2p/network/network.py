from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pdb
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
import tensorflow.contrib.rnn as rnn



from im2txt.faster_rcnn.rpn.snippets import generate_anchors_pre
from im2txt.faster_rcnn.rpn.proposal_layer import proposal_layer
from im2txt.faster_rcnn.rpn.proposal_top_layer import proposal_top_layer

from .regions_layer import regions_project
from im2txt.densecap.config import cfg


class Network(object):
    def __init__(self):
        self._predictions = {}
        self._losses = {}
        self._anchor_targets = {}
        self._proposal_targets = {}
        self._layers = {}
        self._gt_image = None
        self._act_summaries = []
        self._score_summaries = {}
        self._train_summaries = []
        self._event_summaries = {}
        self._variables_to_fix = {}



        self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
        self._im_info = tf.placeholder(tf.float32, shape=[3])

        self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        # add 2 for: <SOS> and <EOS>
        self._gt_phrases = tf.placeholder(tf.int32, shape=[None, cfg.MAX_WORDS])

        self._gt_ptokens = tf.placeholder(tf.int32, shape=[None, cfg.MAX_WORDS])

        self._anchor_scales = cfg.ANCHOR_SCALES
        self._num_scales = len(self._anchor_scales)
        self._anchor_ratios = cfg.ANCHOR_RATIOS
        self._num_ratios = len(self._anchor_ratios)
        self._num_anchors = self._num_scales * self._num_ratios

        #FOR RegionPooling_HierarchicalRNN
        self.n_words = n_words
        self.batch_size = 256
        self.num_boxes = 50 # 50
        self.feats_dim = 4096 # 4096
        self.project_dim = 1024 # 1024
        self.S_max = 6 # 6
        self.N_max = 50 # 50
        self.word_embed_dim = 1024 # 1024

        self.sentRNN_lstm_dim = 512 # 512 hidden size
        self.sentRNN_FC_dim = 1024 # 1024 in fully connected layer
        self.wordRNN_lstm_dim = 512 # 512 hidden size
        self.bias_init_vector= None

        # word embedding, parameters of embedding
        # embedding shape: n_words x wordRNN_lstm_dim
        with tf.device('/cpu:0'):
            self.Wemb = tf.Variable(tf.random_uniform([self.n_words, self.word_embed_dim], -0.1, 0.1), name='Wemb')

        # regionPooling_W shape: 4096 x 1024
        # regionPooling_b shape: 1024
        self.regionPooling_W = tf.Variable(tf.random_uniform([self.feats_dim, self.project_dim], -0.1, 0.1), name='regionPooling_W')
        self.regionPooling_b = tf.Variable(tf.zeros([self.project_dim]), name='regionPooling_b')

        # sentence LSTM
        self.sent_LSTM = tf.nn.rnn_cell.BasicLSTMCell(self.sentRNN_lstm_dim, state_is_tuple=True)

        # logistic classifier
        self.logistic_Theta_W = tf.Variable(tf.random_uniform([self.sentRNN_lstm_dim, 2], -0.1, 0.1), name='logistic_Theta_W')
        self.logistic_Theta_b = tf.Variable(tf.zeros(2), name='logistic_Theta_b')

        # fc1_W: 512 x 1024, fc1_b: 1024
        # fc2_W: 1024 x 1024, fc2_b: 1024
        self.fc1_W = tf.Variable(tf.random_uniform([self.sentRNN_lstm_dim, self.sentRNN_FC_dim], -0.1, 0.1), name='fc1_W')
        self.fc1_b = tf.Variable(tf.zeros(self.sentRNN_FC_dim), name='fc1_b')
        self.fc2_W = tf.Variable(tf.random_uniform([self.sentRNN_FC_dim, 1024], -0.1, 0.1), name='fc2_W')
        self.fc2_b = tf.Variable(tf.zeros(1024), name='fc2_b')

        # word LSTM
        def wordLSTM():
            lstm = tf.nn.rnn_cell.BasicLSTMCell(self.wordRNN_lstm_dim, state_is_tuple=True)
            return lstm
        self.word_LSTM = tf.nn.rnn_cell.MultiRNNCell([wordLSTM() for _ in range(2)], state_is_tuple=True)

        self.embed_word_W = tf.Variable(tf.random_uniform([self.wordRNN_lstm_dim, self.n_words], -0.1,0.1), name='embed_word_W')

        tf.get_variable_scope().reuse_variables()

        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([self.n_words]), name='embed_word_b')

        if cfg.DEBUG_ALL:
            self._for_debug = {}
            self._tag = 'pre'
            self._mode = 'TRAIN'
            self._num_classes = 1

    def _add_act_summary(self, tensor):
        tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
        tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                          tf.nn.zero_fraction(tensor))

    def _add_score_summary(self, key, tensor):
        tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

    def _add_train_summary(self, var):
        tf.summary.histogram('TRAIN/' + var.op.name, var)

    def _reshape_layer(self, bottom, num_dim, name):
        input_shape = tf.shape(bottom)
        with tf.variable_scope(name) as scope:
            # change the channel to the caffe format
            to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
            # then force it to have channel 2
            reshaped = tf.reshape(to_caffe,
                                  tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))
            # then swap the channel back
            to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
            return to_tf

    def _softmax_layer(self, bottom, name):
        if name.startswith('rpn_cls_prob_reshape'):
            input_shape = tf.shape(bottom)
            bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
            reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
            return tf.reshape(reshaped_score, input_shape)
        return tf.nn.softmax(bottom, name=name)

    def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name) as scope:
            rois, rpn_scores = tf.py_func(proposal_top_layer,
                                          [rpn_cls_prob, rpn_bbox_pred, self._im_info,
                                           self._feat_stride, self._anchors, self._num_anchors],
                                          [tf.float32, tf.float32], name="proposal_top")
            rois.set_shape([cfg.TEST.RPN_TOP_N, 5])
            rpn_scores.set_shape([cfg.TEST.RPN_TOP_N, 1])

        return rois, rpn_scores

    def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name) as scope:
            rois, rpn_scores = tf.py_func(proposal_layer,
                                          [rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                                           self._feat_stride, self._anchors, self._num_anchors],
                                          [tf.float32, tf.float32], name="proposal")
            rois.set_shape([None, 5])
            rpn_scores.set_shape([None, 1])

            if cfg.DEBUG_ALL:
                self._for_debug['proposal_rois'] = rois
                self._for_debug['proposal_rpn_scores'] = rpn_scores

        return rois, rpn_scores

    # Only use it if you have roi_pooling op written in tf.image
    def _roi_pool_layer(self, bootom, rois, name):
        with tf.variable_scope(name) as scope:
            return tf.image.roi_pooling(bootom, rois,
                                        pooled_height=cfg.POOLING_SIZE,
                                        pooled_width=cfg.POOLING_SIZE,
                                        spatial_scale=1. / 16.)[0]

    def _crop_pool_layer(self, bottom, rois, name):
        with tf.variable_scope(name) as scope:
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            # Get the normalized coordinates of bounding boxes
            bottom_shape = tf.shape(bottom)
            height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
            width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
            # Won't be back-propagated to rois anyway, but to save time
            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
            pre_pool_size = cfg.POOLING_SIZE * 2
            crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids),
                                             [pre_pool_size, pre_pool_size],
                                             name="crops")

        # slim.max_pool2d has stride 2 in default
        return slim.max_pool2d(crops, [2, 2], padding='SAME')

    def _dropout_layer(self, bottom, name, ratio=0.5):
        return tf.nn.dropout(bottom, ratio, name=name)


    def _regions_layer(self, name, regions):
        with tf.variable_scope(name) as scope:
            regions_feature = tf.py_func(regions_project, [regions], [tf.float32], name=name)

            return regions_feature

    def _build_HierarchicalRNN_layer(self):
        # receive the feats in the current image
        # it's shape is 10 x 50 x 4096
        # tmp_feats: 500 x 4096
        feats = tf.placeholder(tf.float32, [self.batch_size, self.num_boxes, self.feats_dim])
        tmp_feats = tf.reshape(feats, [-1, self.feats_dim])

        # project_vec_all: 500 x 4096 * 4096 x 1024 --> 500 x 1024 ; project_vec: 10 x 1024
        project_vec_all = tf.matmul(tmp_feats, self.regionPooling_W) + self.regionPooling_b
        project_vec_all = tf.reshape(project_vec_all, [self.batch_size, 50, self.project_dim])
        project_vec = tf.reduce_max(project_vec_all, reduction_indices=1)

        # receive the [continue:0, stop:1] lists
        # example: [0, 0, 0, 0, 1, 1], it means this paragraph has five sentences
        num_distribution = tf.placeholder(tf.int32, [self.batch_size, self.S_max])

        # receive the ground truth words, which has been changed to idx use word2idx function
        captions = tf.placeholder(tf.int32, [self.batch_size, self.S_max, self.N_max+1])
        captions_masks = tf.placeholder(tf.float32, [self.batch_size, self.S_max, self.N_max+1])

        sent_state = self.sent_LSTM.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        probs = []
        loss = 0.0
        loss_sent = 0.0
        loss_word = 0.0
        lambda_sent = 5.0
        lambda_word = 1.0

        #----------------------------------------------------------------------------------------------
        # Hierarchical RNN: sentence RNN and words RNN
        # The word RNN has the max number, N_max = 50, the number in the papar is 50
        #----------------------------------------------------------------------------------------------
        for i in range(0, self.S_max):

            if i > 0:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope('sent_LSTM', reuse=tf.AUTO_REUSE):
                sent_output, sent_state = self.sent_LSTM(project_vec, sent_state)

            with tf.name_scope('fc1'):
                hidden1 = tf.nn.relu( tf.matmul(sent_output, self.fc1_W) + self.fc1_b )
            with tf.name_scope('fc2'):
                sent_topic_vec = tf.nn.relu( tf.matmul(hidden1, self.fc2_W) + self.fc2_b )

            # sent_state is a tuple, sent_state = (c, h)
            # 'c': shape=(1, 512) dtype=float32, 'h': shape=(1, 512) dtype=float32

            sentRNN_logistic_mu = tf.nn.xw_plus_b( sent_output, self.logistic_Theta_W, self.logistic_Theta_b )
            sentRNN_label = tf.stack([ 1 - num_distribution[:, i], num_distribution[:, i] ])
            sentRNN_label = tf.transpose(sentRNN_label)

            sentRNN_loss = tf.nn.softmax_cross_entropy_with_logits(labels=sentRNN_label, logits=sentRNN_logistic_mu)
            sentRNN_loss = tf.reduce_sum(sentRNN_loss)/self.batch_size
            loss += sentRNN_loss * lambda_sent
            loss_sent += sentRNN_loss

            # the begining input of word_LSTM is topic vector, and DON'T compute the loss
            topic = tf.nn.rnn_cell.LSTMStateTuple(sent_topic_vec[:, 0:512], sent_topic_vec[:, 512:])
            word_state = (topic, topic)
            # tf.reset_default_graph()
            for j in range(0, self.N_max):
                if j > 0:
                    tf.get_variable_scope().reuse_variables()

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
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels)
                cross_entropy = cross_entropy * captions_masks[:, i, j]
                loss_wordRNN = tf.reduce_sum(cross_entropy) / self.batch_size
                loss += loss_wordRNN * lambda_word
                loss_word += loss_wordRNN


        return feats, num_distribution, captions, captions_masks, loss, loss_sent, loss_word

    def _HierarchicalRNN_generate_layer(self):
        # feats: 1 x 50 x 4096
        feats = tf.placeholder(tf.float32, [1, self.num_boxes, self.feats_dim])
        # tmp_feats: 50 x 4096
        tmp_feats = tf.reshape(feats, [-1, self.feats_dim])

        # project_vec_all: 50 x 4096 * 4096 x 1024 + 1024 --> 50 x 1024
        project_vec_all = tf.matmul(tmp_feats, self.regionPooling_W) + self.regionPooling_b
        project_vec_all = tf.reshape(project_vec_all, [1, 50, self.project_dim])
        project_vec = tf.reduce_max(project_vec_all, reduction_indices=1)

        # initialize the sent_LSTM state
        sent_state = self.sent_LSTM.zero_state(batch_size=1, dtype=tf.float32)
        # save the generated paragraph to list, here I named generated_sents
        generated_paragraph = []

        # pred
        pred_re = []

        # T_stop: run the sentence RNN forward until the stopping probability p_i (STOP) exceeds a threshold T_stop
        T_stop = tf.constant(0.5)

        # Start build the generation model
        print 'Start build the generation model: '

        # sentence RNN
        for i in range(0, self.S_max):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            # sent_state:
            # LSTMStateTuple(c=<tf.Tensor 'sent_LSTM/BasicLSTMCell/add_2:0' shape=(1, 512) dtype=float32>,
            #                h=<tf.Tensor 'sent_LSTM/BasicLSTMCell/mul_2:0' shape=(1, 512) dtype=float32>)
            with tf.variable_scope('sent_LSTM', reuse=tf.AUTO_REUSE):
                sent_output, sent_state = self.sent_LSTM(project_vec, sent_state)

            # self.fc1_W: 512 x 1024, self.fc1_b: 1024
            # hidden1: 1 x 1024
            # sent_topic_vec: 1 x 1024
            with tf.name_scope('fc1'):
                hidden1 = tf.nn.relu( tf.matmul(sent_output, self.fc1_W) + self.fc1_b )
            with tf.name_scope('fc2'):
                sent_topic_vec = tf.nn.relu( tf.matmul(hidden1, self.fc2_W) + self.fc2_b )

            sentRNN_logistic_mu = tf.nn.xw_plus_b(sent_output, self.logistic_Theta_W, self.logistic_Theta_b)
            pred = tf.nn.softmax(sentRNN_logistic_mu)
            pred_re.append(pred)

            # save the generated sentence to list, named generated_sent
            generated_sent = []

            # initialize the word LSTM state
            topic = tf.nn.rnn_cell.LSTMStateTuple(sent_topic_vec[:, 0:512], sent_topic_vec[:, 512:])
            word_state = (topic, topic)
            # word RNN, unrolled to N_max time steps
            for j in range(0, self.N_max):
                if j > 0:
                    tf.get_variable_scope().reuse_variables()

                if j == 0:
                    with tf.device('/cpu:0'):
                        # get word embedding of BOS (index = 0)
                        current_embed = tf.nn.embedding_lookup(self.Wemb, tf.zeros([1], dtype=tf.int64))

                with tf.variable_scope('word_LSTM', reuse=tf.AUTO_REUSE):
                    word_output, word_state = self.word_LSTM(current_embed, word_state)

                # word_state:
                # (
                #     LSTMStateTuple(c=<tf.Tensor 'word_LSTM_152/MultiRNNCell/Cell0/BasicLSTMCell/add_2:0' shape=(1, 512) dtype=float32>,
                #                    h=<tf.Tensor 'word_LSTM_152/MultiRNNCell/Cell0/BasicLSTMCell/mul_2:0' shape=(1, 512) dtype=float32>),
                #     LSTMStateTuple(c=<tf.Tensor 'word_LSTM_152/MultiRNNCell/Cell1/BasicLSTMCell/add_2:0' shape=(1, 512) dtype=float32>,
                #                    h=<tf.Tensor 'word_LSTM_152/MultiRNNCell/Cell1/BasicLSTMCell/mul_2:0' shape=(1, 512) dtype=float32>)
                # )
                logit_words = tf.nn.xw_plus_b(word_output, self.embed_word_W, self.embed_word_b)
                max_prob_index = tf.argmax(logit_words, 1)[0]
                generated_sent.append(max_prob_index)

                with tf.device('/cpu:0'):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
                    current_embed = tf.expand_dims(current_embed, 0)

            generated_paragraph.append(generated_sent)

        return feats, generated_paragraph, pred_re, generated_sent

    def _regionPooling_HierarchicalRNN_layer(self, fc7, input_sentences, initializer, is_training):
        """
        1. Pooling the visual features into a single dense feature
        2. Then, build sentence LSTM, word LSTM
        """
        if self._mode == 'TRAIN':
            tf_feats, tf_num_distribution, tf_captions_matrix, tf_captions_masks, tf_loss, tf_loss_sent, tf_loss_word= _build_HierarchicalRNN_layer()

            with tf.variable_scope('LOSS_' + self._tag) as scope:

                self._losses['caption_loss']=tf_loss_word
                self._losses['sentence_loss']=tf_loss_sent
                self._losses['total_loss']= tf_loss
                self._event_summaries.update(self._losses)

                if cfg.DEBUG_ALL:
                    self._for_debug['caption_loss']=tf_loss_word
                    self._for_debug['sentence_loss']=tf_loss_sent
                    self._for_debug['total_loss']=tf_loss

        elif self._mode == 'TEST':
            tf_feats, tf_generated_paragraph, tf_pred_re, tf_sent_topic_vectors= _HierarchicalRNN_generate_layer()

        self._predictions['cap_probs']=tf_pred_re
        self._predictions['sent_topic_vectors']=tf_sent_topic_vectors
        self._predictions['predict_caption']=tf_generated_paragraph


    def _anchor_component(self):
        """
        bulid anchors for images
        """
        with tf.variable_scope('ANCHOR_' + self._tag) as scope:
            # just to get the shape right
            height=tf.to_int32(tf.ceil(self._im_info[0] / np.float32(self._feat_stride[0])))
            width=tf.to_int32(tf.ceil(self._im_info[1] / np.float32(self._feat_stride[0])))
            anchors, anchor_length=tf.py_func(generate_anchors_pre,
                                                [height, width,
                                                 self._feat_stride, self._anchor_scales, self._anchor_ratios],
                                                [tf.float32, tf.int32], name="generate_anchors")
            anchors.set_shape([None, 4])
            anchor_length.set_shape([])
            self._anchors=anchors
            self._anchor_length=anchor_length

        if cfg.DEBUG_ALL:
            self._for_debug['anchors']=anchors


    def _region_proposal(self, net_conv, is_training, initializer):
        rpn=slim.conv2d(net_conv, cfg.RPN_CHANNELS, [3, 3],
                          trainable=is_training and cfg.IM2P.FINETUNE,
                          weights_initializer=initializer,
                          scope="rpn_conv/3x3")
        self._act_summaries.append(rpn)
        rpn_cls_score=slim.conv2d(rpn, self._num_anchors * 2, [1, 1],
            trainable=is_training and cfg.IM2P.FINETUNE,
                                    weights_initializer=initializer,
                                    padding='VALID', activation_fn=None, scope='rpn_cls_score')
        # change it so that the score has 2 as its channel size
        rpn_cls_score_reshape=self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
        rpn_cls_prob_reshape=self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
        # rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis=1, name="rpn_cls_pred")
        rpn_cls_prob=self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
        rpn_bbox_pred=slim.conv2d(rpn, self._num_anchors * 4, [1, 1],
            trainable=is_training and cfg.IM2P.FINETUNE,
                                    weights_initializer=initializer,
                                    padding='VALID', activation_fn=None, scope='rpn_bbox_pred')

        if cfg.TEST.MODE == 'nms':
            rois, _=self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        elif cfg.TEST.MODE == 'top':
            rois, _=self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        else:
            raise NotImplementedError

        self._predictions["rpn_cls_score"]=rpn_cls_score
        self._predictions["rpn_cls_score_reshape"]=rpn_cls_score_reshape
        self._predictions["rpn_cls_prob"]=rpn_cls_prob
        # self._predictions["rpn_cls_pred"] = rpn_cls_pred
        self._predictions["rpn_bbox_pred"]=rpn_bbox_pred
        self._predictions["rois"]=rois



        if cfg.DEBUG_ALL:
            self._for_debug['rpn']=rpn
            self._for_debug['rpn_cls_score']=rpn_cls_score
            self._for_debug['rpn_cls_prob']=rpn_cls_prob
            self._for_debug['rpn_cls_prob_reshape']=rpn_cls_prob_reshape
            self._for_debug['rpn_cls_score_reshape']=rpn_cls_score_reshape
            self._for_debug['rpn_bbox_pred']=rpn_bbox_pred
        return rois


    def _image_to_head(self, is_training, reuse=None):
        raise NotImplementedError

    def _head_to_tail(self, pool5, is_training, reuse=None):
        raise NotImplementedError

    def _build_network(self, is_training=True):
        # select initializers
        if cfg.TRAIN.WEIGHT_INITIALIZER == 'truncated':
            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        elif cfg.TRAIN.WEIGHT_INITIALIZER == 'normal':
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)
        else:
            initializer=tf.contrib.layers.xavier_initializer()

        net_conv=self._image_to_head(is_training)

        with tf.variable_scope(self._scope + '/Extraction'):
            # build the anchors for the image
            self._anchor_component()
            # region proposal network
            rois=self._region_proposal(net_conv, is_training, initializer)
            # region of interest pooling
            if cfg.POOLING_MODE == 'crop':
                pool5=self._crop_pool_layer(net_conv, rois, "pool5")
            else:
                raise NotImplementedError

            if self._mode == 'TRAIN':

                input_sentences=self._regions_layer('regions_layer')

            elif self._mode == 'TEST':
                input_feed=tf.placeholder(dtype=tf.int32,
                                            shape=[None],
                                            name='input_feed')
                input_sentences=tf.expand_dims(input_feed, 1)
            else:
                raise NotImplementedError

        fc7=self._head_to_tail(pool5, is_training)
        #regionPooling_HierarchicalRNN_layer
        with tf.variable_scope(self._scope + '/Prediction'):
            self._regionPooling_HierarchicalRNN_layer(fc7, input_sentences, initializer, is_training)

        self._score_summaries.update(self._predictions)

        if cfg.DEBUG_ALL:
            self._for_debug['pool5']=pool5

        return rois


    def create_architecture(self, mode, num_classes=1, tag=None,
                            ):
        self._tag=tag

        self._num_classes=num_classes
        self._mode=mode

        training=mode == 'TRAIN'
        testing=mode == 'TEST'

        assert tag != None

        # handle most of the regularizers here
        weights_regularizer=tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
        if cfg.TRAIN.BIAS_DECAY:
            biases_regularizer=weights_regularizer
        else:
            biases_regularizer=tf.no_regularizer

        # list as many types of layers as possible, even if they are not used now
        with arg_scope([slim.conv2d, slim.conv2d_in_plane,
                        slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                       weights_regularizer=weights_regularizer,
                       biases_regularizer=biases_regularizer,
                       biases_initializer=tf.constant_initializer(0.0)):
            rois=self._build_network(training)

        layers_to_output={'rois': rois}

        for var in tf.trainable_variables():
            self._train_summaries.append(var)

        if testing:
            # stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_classes))
            stds=np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS)
            # means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self._num_classes))
            means=np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
            # self._predictions["bbox_pred"] *= stds
            # self._predictions["bbox_pred"] += means
        else:

            layers_to_output.update(self._losses)

            val_summaries=[]
            with tf.device("/cpu:0"):
                for key, var in self._event_summaries.items():
                    val_summaries.append(tf.summary.scalar(key, var))
                for key, var in self._score_summaries.items():
                    self._add_score_summary(key, var)
                for var in self._act_summaries:
                    self._add_act_summary(var)
                for var in self._train_summaries:
                    self._add_train_summary(var)

            self._summary_op=tf.summary.merge_all()
            self._summary_op_val=tf.summary.merge(val_summaries)

        layers_to_output.update(self._predictions)

        return layers_to_output

    def get_variables_to_restore(self, variables, var_keep_dic):
        raise NotImplementedError

    def fix_variables(self, sess, pretrained_model):
        raise NotImplementedError

    # Extract the head feature maps, for example for vgg16 it is conv5_3
    def extract_head(self, sess, image):
        feed_dict={self._image: image}
        features=sess.run(self._layers["head"], feed_dict=feed_dict)
        return features


    def feed_image(self, sess, image, im_info):
        feed_dict={self._image: image,
                     self._im_info: im_info}
        fetch_list=[
            '%s/Prediction/lstm/cap_init_state:0' % self._scope,
            self._sent_prob]

        fetch=sess.run(fetch_list, feed_dict=feed_dict)

        return fetch

    def inference_step(self, sess, input_feed, cap_state_feed):
        feed_dict={'%s/Extraction/input_feed:0' % self._scope: input_feed,
                     '%s/Prediction/lstm/cap_state_feed:0' % self._scope: cap_state_feed}
        fetch_list=['%s/Prediction/lstm/cap_probs:0' % self._scope,
                   '%s/Prediction/lstm/cap_state:0' % self._scope]

        fetch=sess.run(fetches=fetch_list, feed_dict=feed_dict)

        return fetch

    def get_summary(self, sess, blobs):
        feed_dict=self._feed_dict(blobs)

        summary=sess.run(self._summary_op_val, feed_dict=feed_dict)

        return summary

    def train_step(self, sess, blobs, train_op):
        feed_dict=self._feed_dict(blobs)
        sentence_loss, caption_loss, loss, \
            _=sess.run([
                               self._losses['sentence_loss'],
                               self._losses['caption_loss'],
                               self._losses['total_loss'],
                               train_op],
                              feed_dict=feed_dict)

        return sentence_loss, caption_loss, loss

    def train_step_with_summary(self, sess, blobs, train_op):
        feed_dict=self._feed_dict(blobs)
        sentence_loss, caption_loss, loss, \
            summary, _=sess.run([
                               self._losses['sentence_loss'],
                               self._losses['caption_loss'],
                               self._losses['total_loss'],
                               self._summary_op,
                               train_op],
                              feed_dict=feed_dict)

        return sentence_loss, caption_loss, loss, summary


    def _feed_dict(self, blobs):
        feed_dict={self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes'],
                     self._gt_ptokens: blobs['gt_ptokens']}


        return feed_dict
