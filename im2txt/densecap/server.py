from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join as pjoin
import sys
sys.path.append('../')

import six
import glob
import argparse
import json
import numpy as np
import tensorflow as tf

from densecap.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
from faster_rcnn.datasets.factory import get_imdb
import faster_rcnn.datasets.imdb

from densecap.train import get_training_roidb, train_net
from densecap.test import predict

from densecap.network.vgg16 import vgg16
from densecap.network.resnet_v1 import resnetv1
import pprint



CKPT= '../ckpt'
CFG = 'scripts/dense_cap_config.yml'
NET = 'res50'
VOCAB_FILE= '../ckpt/vocabulary.txt'
SET=  ['TEST.USE_BEAM_SEARCH', 'False', 'EMBED_DIM', '512', 'TEST.LN_FACTOR', '1.', 'TEST.RPN_NMS_THRESH', '0.7', 'TEST.NMS', '0.3']


class ModelWrapper(object):
    """Model wrapper for TensorFlow models in SavedModel format"""
    def __init__(self):

        if CFG is not None:
            cfg_from_file(CFG)
        if SET is not None:
            cfg_from_list(SET)

        if NET == 'vgg16':
            self.net = vgg16()
        elif NET == 'res50':
            self.net = resnetv1(num_layers=50)
        elif NET == 'res101':
            self.net = resnetv1(num_layers=101)
        elif NET == 'res152':
            self.net = resnetv1(num_layers=152)
        else:
            raise NotImplementedError

        self.net.create_architecture("TEST", num_classes=1, tag='pre')

        self.vocab = ['<PAD>', '<SOS>', '<EOS>']

        with open(VOCAB_FILE, 'r') as f:
            for line in f:
                self.vocab.append(line.strip())
        if CKPT:
            self.ckpt = tf.train.get_checkpoint_state(CKPT)
        else:
            raise ValueError

        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True

        saver = tf.train.Saver()

        self.sess = tf.Session(config=tfconfig)

        print('Restored from {}'.format(self.ckpt.model_checkpoint_path))

        saver.restore(self.sess, self.ckpt.model_checkpoint_path)


    def pred(self,image_data=False):

        if not image_data:
            im_paths = glob.glob('./data/demo/*.jpg')
            pre_results = {}

            for path in im_paths:
                pre_results = predict(self.sess, self.net, path, self.vocab, pre_results)
        else:
            pre_results = predict(self.sess, self.net, image_data, self.vocab, pre_results)
        return pre_results


if __name__ == '__main__':
    mode= ModelWrapper()
    print(mode.pred())
