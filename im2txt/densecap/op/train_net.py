from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pprint
import sys
sys.path.append('../')

import tensorflow as tf
import numpy as np
import argparse
import six
from im2txt.densecap.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
from im2txt.densecap.network.resnet_v1 import resnetv1
from im2txt.densecap.network.vgg16 import vgg16
from im2txt.densecap.train import get_training_roidb, train_net
import im2txt.faster_rcnn.datasets.imdb
from im2txt.faster_rcnn.datasets.factory import get_imdb






def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Dense Caption network')
    parser.add_argument('--device', dest='device', help='device to use',
                        default='gpu', type=str)
    parser.add_argument('--device_id', dest='device_id', help='device id to use',
                        default=0, type=int)
    parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='weights',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='vg_1.2_train', type=str)
    parser.add_argument('--imdbval', dest='imdbval_name',
                        help='dataset to validation on',
                        default='vg_1.2_val', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')

    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default=None, type=str)
    parser.add_argument('--data_dir', dest='data_dir', type=str,
                        default='/content/output/visual_genome', help='dataset directory')
    parser.add_argument('--embed_dim', dest='embed_dim', type=int,
                        default=512, help='embed dimension of words')
    parser.add_argument('--context_fusion', dest='context_fusion',
                        action='store_true', help='train with context fusion.')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def combined_roidb(imdb_names):
    # for now: imdb_names='vg_1.2_train'
    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print('Loaded dataset `{:s}` for training'.format(imdb.name))
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        imdb = lib.datasets.imdb.imdb(imdb_names)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb


def main():
    args = parse_args()
    cfg.DATA_DIR = args.data_dir
    cfg.CONTEXT_FUSION = args.context_fusion

    print('------ called with args: -------')
    pprint.pprint(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    if cfg.INIT_BY_GLOVE and cfg.KEEP_AS_GLOVE_DIM:
        cfg.EMBED_DIM = cfg.GLOVE_DIM
    else:
        cfg.EMBED_DIM = args.embed_dim

    print("Using config:")
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        tf.set_random_seed(cfg.RNG_SEED)

    imdb, roidb = combined_roidb(args.imdb_name)

    output_dir = get_output_dir(imdb, args.tag)
    print("output will be saved to `{:s}`".format(output_dir))

    # tensorboard directory where the summaries are saved during training
    tb_dir = get_output_tb_dir(imdb, args.tag)
    print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

    # also add validation set, but with no flipping image
    orgflip = cfg.TRAIN.USE_FLIPPED
    cfg.TRAIN.USE_FLIPPED = False

    _, valroidb = combined_roidb(args.imdbval_name)

    cfg.TRAIN.USE_FLIPPED = orgflip

    # load network
    if args.net == 'vgg16':
        net = vgg16()
    elif args.net == 'res50':
        net = resnetv1(num_layers=50)
    elif args.net == 'res101':
        net = resnetv1(num_layers=101)
    elif args.net == 'res152':
        net = resnetv1(num_layers=152)
    else:
        raise NotImplementedError

    if args.weights and not args.weights.endswith('.ckpt'):
        try:
            ckpt = tf.train.get_checkpoint_state(args.weights)
            pretrained_model = ckpt.model_checkpoint_path
        except:
            raise ValueError("NO checkpoint found in {}".format(args.weights))
    else:
        pretrained_model = args.weights

    train_net(net, imdb, roidb, valroidb, output_dir, tb_dir,
              pretrained_model=pretrained_model,
              max_iters=args.max_iters)


if __name__ == '__main__':
    main()
