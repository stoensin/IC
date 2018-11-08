from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Train a dense caption model"""

from os.path import join as pjoin
import sys
import six
import glob
import argparse
import json
import numpy as np
import tensorflow as tf

from .config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
from .datasets.factory import get_imdb
import .datasets.imdb
from .train import get_training_roidb, train_net
from .test import test_im
from .network.resnet_v1 import resnetv1
import pprint


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Dense Caption network')

    parser.add_argument('--ckpt', dest='ckpt',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    # TODO: add inception
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res50', type=str)
    parser.add_argument('--vocab', dest='vocabulary',
                        help='vocabulary file',
                        default=None, type=str)

    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print('------- called with args: --------')
    pprint.pprint(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # load network
    if args.net == 'res50':
        net = resnetv1(num_layers=50)
    elif args.net == 'res101':
        net = resnetv1(num_layers=101)
    elif args.net == 'res152':
        net = resnetv1(num_layers=152)
    else:
        raise NotImplementedError

    net.create_architecture("TEST", num_classes=1, tag='pre')
    vocab = ['<PAD>', '<SOS>', '<EOS>']
    with open(args.vocabulary, 'r') as f:
        for line in f:
            vocab.append(line.strip())

    # get the image paths
    im_paths = glob.glob('./data/demo/*.jpg')
    print(im_paths)

    # read checkpoint file
    if args.ckpt:
        ckpt = tf.train.get_checkpoint_state(args.ckpt)
    else:
        raise ValueError

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    saver = tf.train.Saver()
    with tf.Session(config=tfconfig) as sess:
        print('Restored from {}'.format(ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)

        # for n in tf.get_default_graph().as_graph_def().node:
        #     if 'input_feed' in n.name:
        #         print(n.name)
        # for html visualization
        pre_results = {}
        save_path = './vis/data'
        for path in im_paths:
            test_im(sess, net, path, vocab, pre_results)
