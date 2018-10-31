from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../')

from .config import cfg
import im2txt.faster_rcnn.roi_data_layer.roidb as rdl_roidb
from im2txt.faster_rcnn.roi_data_layer.layer import RoIDataLayer
from im2txt.faster_rcnn.utils.timer import Timer
from six.moves import cPickle as pickle
import numpy as np
import os
import cv2
import pdb

import glob
import time
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


class SolverWrapper(object):
    """A simple wrapper for training.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, sess, network, imdb, roidb, valroidb, output_dir, tb_dir,
                 pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.net = network
        self.imdb = imdb
        self.roidb = roidb
        self.valroidb = valroidb
        self.output_dir = output_dir
        self.tb_dir = tb_dir
        self.pretrained_model = pretrained_model

        self.tbvaldir = tb_dir + '_val'
        if not os.path.exists(self.tbvaldir):
            os.makedirs(self.tbvaldir)

        # TODO: disable BBOX NORMALIZE
        # if (cfg.TRAIN.HAS_RPN and cfg.TRAIN.BBOX_REG and
        #         cfg.TRAIN.BBOX_NORMALIZE_TARGETS):
            # RPN can only use precomputed normalization because there are no
            # fixed statistics to compute a prior
            # assert cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED

        # if cfg.TRAIN.BBOX_REG:
        #     print('Computing bounding-box regression targets...')
        #     self.bbox_means, self.bbox_stds = \
        #         rdl_roidb.add_bbox_regression_targets(roidb)
        #     print('done')

    def snapshot(self, sess, iters=0):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.net

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Store the model snapshot
        filename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{}'.format(iters) + '.ckpt'
        filename = os.path.join(self.output_dir, filename)
        self.saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

        # Also store some meta information, random state, etc.
        nfilename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{}'.format(iters) + '.pkl'
        nfilename = os.path.join(self.output_dir, nfilename)
        # current state of numpy random
        st0 = np.random.get_state()
        # current position in the database
        cur = self.data_layer._cur
        # current shuffled indexes of the database
        perm = self.data_layer._perm
        # current position in the validation database
        cur_val = self.data_layer_val._cur
        # current shuffled indexes of the validation database
        perm_val = self.data_layer_val._perm

        # Dump the meta info
        with open(nfilename, 'wb') as fid:
            pickle.dump(st0, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(cur, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(perm, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(cur_val, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(perm_val, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(iters, fid, pickle.HIGHEST_PROTOCOL)

        return filename, nfilename

    def from_snapshot(self, sess, sfile, nfile):
        print('Restoring model snapshots from {:s}'.format(sfile))
        self.saver.restore(sess, sfile)
        print('Restored.')
        # Needs to restore the other hyper-parameters/states for training, (TODO xinlei) I have
        # tried my best to find the random states so that it can be recovered exactly
        # However the Tensorflow state is currently not available
        with open(nfile, 'rb') as fid:
            st0 = pickle.load(fid)
            cur = pickle.load(fid)
            perm = pickle.load(fid)
            cur_val = pickle.load(fid)
            perm_val = pickle.load(fid)
            last_snapshot_iter = pickle.load(fid)

            np.random.set_state(st0)
            self.data_layer._cur = cur
            self.data_layer._perm = perm
            self.data_layer_val._cur = cur_val
            self.data_layer_val._perm = perm_val

        return last_snapshot_iter

    def get_variables_in_checkpoint_file(self, file_name):
        try:
            reader = pywrap_tensorflow.NewCheckpointReader(file_name)
            var_to_shape_map = reader.get_variable_to_shape_map()
            return var_to_shape_map
        except Exception as e:  # pylint: disable=broad-except
            print(str(e))
            if "corrupted compressed block contents" in str(e):
                print("It's likely that your checkpoint file has been compressed "
                      "with SNAPPY.")

    def construct_graph(self, sess):
        with sess.graph.as_default():
            # Set the random seed for tensorflow(done in the beginning)
            # tf.set_random_seed(cfg.RNG_SEED)
            # Build the main computation graph
            layers = self.net.create_architecture('TRAIN', num_classes=1, tag='default')
            # Define the loss
            loss = layers['total_loss']
            # Set learning rate and momentum
            lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
            print("learning rate {}".format(cfg.TRAIN.LEARNING_RATE))
            self.global_step = tf.Variable(0, trainable=False)
            if cfg.TRAIN.LR_DIY_DECAY:
                learning_rate = lr
            else:
                learning_rate = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE,
                                                           self.global_step,
                                                           cfg.TRAIN.EXP_DECAY_STEPS,
                                                           cfg.TRAIN.EXP_DECAY_RATE,
                                                           staircase=True)
            if cfg.TRAIN.OPTIMIZER == 'sgd_m':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate, cfg.TRAIN.MOMENTUM)
            elif cfg.TRAIN.OPTIMIZER == 'adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate)

                # must disable diy decay when using exponentially decay.
                assert cfg.TRAIN.LR_DIY_DECAY == False

            # Compute the gradients with regard to the loss
            gvs = self.optimizer.compute_gradients(loss)
            # gradient clipping
            capped_gvs = [(tf.clip_by_norm(grad, cfg.TRAIN.CLIP_NORM), var)
                          if grad is not None else (tf.zeros_like(var), var)
                          for grad, var in gvs]
            # Double the gradient of the bias if set
            if cfg.TRAIN.DOUBLE_BIAS:
                final_gvs = []
                with tf.variable_scope('Gradient_Mult') as scope:
                    for grad, var in capped_gvs:
                        scale = 1.
                        if cfg.TRAIN.DOUBLE_BIAS and '/biases:' in var.name:
                            scale *= 2.
                        if not np.allclose(scale, 1.0):
                            grad = tf.multiply(grad, scale)
                        final_gvs.append((grad, var))
                train_op = self.optimizer.apply_gradients(final_gvs,
                                                          global_step=self.global_step)
            else:
                train_op = self.optimizer.apply_gradients(capped_gvs,
                                                          global_step=self.global_step)

            # We will handle the snapshots ourselves
            self.saver = tf.train.Saver(max_to_keep=100000)
            # Write the train and validation information to tensorboard
            self.writer = tf.summary.FileWriter(self.tb_dir, sess.graph)
            self.valwriter = tf.summary.FileWriter(self.tbvaldir)

        return learning_rate, train_op

    def find_previous(self):
        sfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.ckpt.meta')
        sfiles = glob.glob(sfiles)
        sfiles.sort(key=os.path.getmtime)
        # Get the snapshot name in TensorFlow
        redfiles = []
        for stepsize in cfg.TRAIN.STEPSIZE:
            redfiles.append(os.path.join(self.output_dir,
                                         cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}.ckpt.meta'.format(stepsize + 1)))
        sfiles = [ss.replace('.meta', '') for ss in sfiles if ss not in redfiles]

        nfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.pkl')
        nfiles = glob.glob(nfiles)
        nfiles.sort(key=os.path.getmtime)
        redfiles = [redfile.replace('.ckpt.meta', '.pkl') for redfile in redfiles]
        nfiles = [nn for nn in nfiles if nn not in redfiles]

        lsf = len(sfiles)
        assert len(nfiles) == lsf

        return lsf, nfiles, sfiles

    def initialize(self, sess):
        # Initial file lists are empty
        np_paths = []
        ss_paths = []
        # Fresh train directly from ImageNet weights
        print('Loading initial model weights from {:s}'.format(self.pretrained_model))
        variables = tf.global_variables()
        # Initialize all variables first
        sess.run(tf.variables_initializer(variables, name='init'))
        var_keep_dic = self.get_variables_in_checkpoint_file(self.pretrained_model)
        # Get the variables to restore, ignoring the variables to fix
        variables_to_restore = self.net.get_variables_to_restore(variables, var_keep_dic)

        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, self.pretrained_model)
        print('Loaded.')
        # Need to fix the variables before loading, so that the RGB weights are changed to BGR
        # For VGG16 it also changes the convolutional weights fc6 and fc7 to
        # fully connected weights
        self.net.fix_variables(sess, self.pretrained_model)
        print('Fixed.')
        print("Ckpt path: {}".format(self.pretrained_model))
        # Added for continue traing when doing experiments.
        pkl_path = os.path.splitext(self.pretrained_model)[0] + '.pkl'
        if os.path.exists(pkl_path):
            print("Found pickle file, restore training process.")
            with open(pkl_path, 'rb') as fid:
                st0 = pickle.load(fid,encoding='bytes')
                cur = pickle.load(fid,encoding='bytes')
                perm = pickle.load(fid,encoding='bytes')
                cur_val = pickle.load(fid,encoding='bytes')
                perm_val = pickle.load(fid,encoding='bytes')
                last_snapshot_iter = pickle.load(fid,encoding='bytes')
            print("Last snapshot iters:{}".format(last_snapshot_iter))
        else:
            last_snapshot_iter = 0

        last_snapshot_iter = 0

        rate = cfg.TRAIN.LEARNING_RATE
        stepsizes = list(cfg.TRAIN.STEPSIZE)

        for stepsize in cfg.TRAIN.STEPSIZE:
            if last_snapshot_iter >= stepsize:
                rate *= cfg.TRAIN.GAMMA
                print("Decrease learning rate by gamma {}, changed to rate: {}, stepsize:{}, iters:{}".format(cfg.TRAIN.GAMMA, rate, stepsize, last_snapshot_iter))
            else:
                stepsizes.append(stepsize)

        return rate, last_snapshot_iter, stepsizes, np_paths, ss_paths

    def restore(self, sess, sfile, nfile):
        # Get the most recent snapshot and restore
        np_paths = [nfile]
        ss_paths = [sfile]
        # Restore model from snapshots
        last_snapshot_iter = self.from_snapshot(sess, sfile, nfile)
        # Set the learning rate
        rate = cfg.TRAIN.LEARNING_RATE
        stepsizes = []
        for stepsize in cfg.TRAIN.STEPSIZE:
            if last_snapshot_iter > stepsize:
                rate *= cfg.TRAIN.GAMMA
            else:
                stepsizes.append(stepsize)

        return rate, last_snapshot_iter, stepsizes, np_paths, ss_paths

    def remove_snapshot(self, np_paths, ss_paths):
        to_remove = len(np_paths) - cfg.TRAIN.SNAPSHOT_KEPT
        for c in range(to_remove):
            nfile = np_paths[0]
            os.remove(str(nfile))
            np_paths.remove(nfile)

        to_remove = len(ss_paths) - cfg.TRAIN.SNAPSHOT_KEPT
        for c in range(to_remove):
            sfile = ss_paths[0]
            # To make the code compatible to earlier versions of Tensorflow,
            # where the naming tradition for checkpoints are different
            if os.path.exists(str(sfile)):
                os.remove(str(sfile))
            else:
                os.remove(str(sfile + '.data-00000-of-00001'))
                os.remove(str(sfile + '.index'))
            sfile_meta = sfile + '.meta'
            os.remove(str(sfile_meta))
            ss_paths.remove(sfile)

    def train_model(self, sess, max_iters):
        # Build data layers for both training and validation set
        self.data_layer = RoIDataLayer(self.roidb)
        self.data_layer_val = RoIDataLayer(self.valroidb, random=True)

        # Construct the computation graph
        lr, train_op = self.construct_graph(sess)

        # Find previous snapshots if there is any to restore from
        lsf, nfiles, sfiles = self.find_previous()

        # Initialize the variables or restore them from the last snapshot
        if lsf == 0:
            rate, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.initialize(sess)
        else:
            rate, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.restore(sess, str(sfiles[-1]), str(nfiles[-1]))
        timer = Timer()
        iters = last_snapshot_iter + 1
        last_summary_time = time.time()
        # Make sure the lists are not empty
        stepsizes.append(max_iters)
        stepsizes.reverse()
        next_stepsize = stepsizes.pop()

        # In case the lr is restored from ckpt.
        # last_lr = sess.run(lr)
        # if rate != last_lr:
        #     sess.run(tf.assign(lr, rate))

        while iters < max_iters + 1:
            # Learning rate
            if cfg.TRAIN.LR_DIY_DECAY:
                # if iters == next_stepsize + 1:
                if iters % cfg.TRAIN.STEPSIZE[0] == 1 and iters > 1:
                    # Add snapshot here before reducing the learning rate
                    # self.snapshot(sess, iters)
                    rate *= cfg.TRAIN.GAMMA
                    sess.run(tf.assign(lr, rate))
                    print("Decrease learning rate to {}, with iters: {}, stepsize: {}, \
                        gamma: {}".format(rate, iters, cfg.TRAIN.STEPSIZE[0], cfg.TRAIN.GAMMA))
                    next_stepsize = stepsizes.pop()

            timer.tic()
            # Get training data, one batch at a time
            blobs = self.data_layer.forward()

            while blobs['gt_ptokens'].shape[0] == 0:
                print(" ######## Jump over a training example")
                blobs = self.data_layer.forward()

            now = time.time()
            if iters == 1 or now - last_summary_time > cfg.TRAIN.SUMMARY_INTERVAL:
                # Compute the graph with summary
                sentence_loss, caption_loss, total_loss, summary = \
                    self.net.train_step_with_summary(sess, blobs, train_op)
                self.writer.add_summary(summary, float(iters))
                # Also check the summary on the validation set
                # blobs_val = self.data_layer_val.forward()
                # summary_val = self.net.get_summary(sess, blobs_val)
                # self.valwriter.add_summary(summary_val, float(iters))
                last_summary_time = now
            else:
                # Compute the graph without summary
                sentence_loss, caption_loss, total_loss = \
                    self.net.train_step(sess, blobs, train_op)
            timer.toc()

            # Display training information
            if iters % (cfg.TRAIN.DISPLAY) == 0:
                if cfg.TRAIN.LR_DIY_DECAY:
                    learning_rate = lr
                else:
                    learning_rate = sess.run(lr)
                print('iters: %d / %d, total loss: %.6f\n >>> caption loss: %.6f\n >>> sentence_loss: %.6f\n '
                      '>>> lr: %f' %
                      (iters, max_iters, total_loss, caption_loss, sentence_loss,
                       float(learning_rate)))
                print('speed: {:.3f}s / iters'.format(timer.average_time))

            # Snapshotting
            if iters % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iters
                ss_path, np_path = self.snapshot(sess, iters)
                np_paths.append(np_path)
                ss_paths.append(ss_path)

                # Remove the old snapshots if there are too many
                if len(np_paths) > cfg.TRAIN.SNAPSHOT_KEPT:
                    self.remove_snapshot(np_paths, ss_paths)

            iters += 1

        if last_snapshot_iter != iters - 1:
            self.snapshot(sess, iters - 1)

        self.writer.close()
        self.valwriter.close()

    def vis_regions(self, im, regions, iter_n, save_path='debug'):
        """Visual debugging of detections by saving images with detected bboxes."""
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # mean_values = np.array([[[102.9801, 115.9465, 122.7717]]])
        im = im + cfg.PIXEL_MEANS  # offset to original values

        for i in xrange(len(regions)):
            bbox = regions[i, :4]
            region_id = regions[i, 4]
            if region_id == 0:
                continue
            caption = self.sentence(self._all_phrases[region_id])

            im_new = np.copy(im)
            cv2.rectangle(im_new, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            cv2.imwrite('%s/%d_%s.jpg' % (save_path, iter_n, caption), im_new)

    def sentence(self, vocab_indices):
        # consider <eos> tag with id 0 in vocabulary
        sentence = ' '.join([self._vocab[i] for i in vocab_indices])
        return sentence


def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print('Appending horizontally-flipped training examples...')
        imdb.append_flipped_images()
        print('done')

    print('Preparing training data...')
    rdl_roidb.prepare_roidb(imdb)
    print('done')

    return imdb.roidb


def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""

    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print('Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                       num, num_after))
    return filtered_roidb


def train_net(network, imdb, roidb, valroidb, output_dir, tb_dir,
              pretrained_model=None, max_iters=40000):
    """Train a Dense Caption network."""

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    with tf.Session(config=tfconfig) as sess:
        sw = SolverWrapper(sess, network, imdb, roidb, valroidb, output_dir, tb_dir,
                           pretrained_model=pretrained_model)

        print('Solving...')
        sw.train_model(sess, max_iters)
        print('done solving')
        # return model_paths
