from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy.sparse
import uuid
import json
import six
from tqdm import tqdm
from six.moves import xrange, cPickle
from os.path import join as pjoin

from .imdb import imdb
from im2txt.densecap.config import cfg


DEBUG = False
USE_CACHE = True
UNK_IDENTIFIER = '<unk>'


def pre_roidb(roidb):
    """Enrich the imdb's roidb by adding some derived quantities that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    """
    # need gt_overlaps as a dense array for argmax
    gt_overlaps = roidb['gt_overlaps'].toarray()
    # max overlap with gt over classes (columns)
    max_overlaps = gt_overlaps.max(axis=1)
    # gt class that had the max overlap
    max_classes = gt_overlaps.argmax(axis=1)
    roidb['max_classes'] = max_classes
    roidb['max_overlaps'] = max_overlaps
    # sanity checks
    # max overlap of 0 => class should be zero (background)
    zero_inds = np.where(max_overlaps == 0)[0]
    assert all(max_classes[zero_inds] == 0)
    # max overlap > 0 => class should not be zero (must be a fg class)
    # nonzero_inds = np.where(max_overlaps > 0)[0]
    # assert all(max_classes[nonzero_inds] != 0)
    return roidb


def is_valid_limitRam(entry):
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


def flip_image(roidb):
    """flip image and change the name for reading later"""

    boxes = roidb['boxes'].copy()
    oldx1 = boxes[:, 0].copy()
    oldx2 = boxes[:, 2].copy()
    boxes[:, 0] = roidb['width'] - oldx2 - 1
    boxes[:, 2] = roidb['width'] - oldx1 - 1
    assert (boxes[:, 2] >= boxes[:, 0]).all()
    entry = {'boxes': boxes,
             'gt_overlaps': roidb['gt_overlaps'],
             'gt_classes': roidb['gt_classes'],
             'flipped': True,
             'gt_ptokens': roidb['gt_ptokens'],
             'width': roidb['width'],
             'height': roidb['height'],
             'image': roidb['image'],
             'image_id': '%s_flip' % roidb['image_id']}

    return entry

class visual_genome(imdb):
    def __init__(self, image_set, version):
        imdb.__init__(self, 'vg_' + version + '_' + image_set)
        # image_set from ['train', 'val', 'test']
        self._image_set = image_set

        # self._data_path = '%s/%s' % (cfg.DATA_DIR, version)
        self._data_path = cfg.DATA_DIR
        cfg.CACHE_DIR = self._data_path

        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

        # return path of the json file
        self.region_imset_path = os.path.join(self._data_path,
                                              '%s_gt_paragraph.json' % image_set)
        with open(self.region_imset_path, 'r') as f:
            self._gt_regions = json.load(f)

        self._image_ext = '.jpg'
        print('data_path: %s' % self._data_path)

        self._classes = ('__background__', '__foreground__')

        self._image_index = self._load_image_set_index()

        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        vocab_path = os.path.join(self._data_path, 'vocabulary.txt')
        with open(vocab_path, 'r') as f:
            self._vocabulary_inverted = [line.strip() for line in f]

        self._vocabulary = dict([(w, i) for i, w in enumerate(self._vocabulary_inverted)])

        # test for overfitting a minibatch
        if cfg.ALL_TEST:
            if image_set == 'train':
                self._image_index = self._image_index[:cfg.ALL_TEST_NUM_TRAIN]
            elif image_set == 'val':
                self._image_index = self._image_index[:cfg.ALL_TEST_NUM_VAL]
            elif image_set == 'test':
                self._image_index = self._image_index[:cfg.ALL_TEST_NUM_TEST]
            else:
                raise ValueError('Please check the name of the image set.')

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = self._gt_regions[str(index)]['path']

        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self, ext='json'):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_index = [key for key in self._gt_regions]
        print("Number of examples: {}".format(len(image_index)))
        return image_index

    def get_gt_regions(self):

        raise NotImplementedError
        # return [v for k, v in six.iteritems(self._gt_regions)]

    def get_gt_regions_index(self, index):
        regions = self._gt_regions[index]

        return regions

    def get_vocabulary(self):
        return self._vocabulary_inverted

    def gt_roidb(self):
        gt_roidb = self.gt_roidbs()
        return gt_roidb

    def gt_roidbs(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = pjoin(self._data_path, self._image_set + '_gt_roidb.pkl')
        if os.path.exists(cache_file) and USE_CACHE:
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self._image_set, cache_file))
            return roidb

        # gt_roidb = [self._load_vg_annotation(index) for index in self._image_index]
        gt_roidb = []
        image_index = []
        exclude_index = []
        for i in tqdm(xrange(len(self._image_index)), desc="%s" % self._image_set):
            idx = self._image_index[i]
            dictionary = self._load_vg_annotation(idx)
            if is_valid_limitRam(pre_roidb(dictionary)):
                image_index.append(idx)
                gt_roidb.append(dictionary)
            else:
                exclude_index.append(idx)

           # check for flipping only during training
            if cfg.TRAIN.USE_FLIPPED and self._image_set == 'train':
                flip_dict = flip_image(dictionary)
                flip_id = flip_dict['image_id']
                if is_valid_limitRam(pre_roidb(flip_dict)):
                    image_index.append(flip_id)
                    gt_roidb.append(flip_dict)
                else:
                    exclude_index.append(flip_id)

        print("filter out {} images.".format(len(exclude_index)))
        print("remaining {} iamges for {} set".format(len(image_index),
                                                      self._image_set))
        self._image_index = image_index

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)

        print('wrote gt roidb to {}'.format(cache_file))
        return gt_roidb

    def rpn_roidb(self):
        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
            'layers data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _line_to_stream(self, sentence):
        stream = []
        for word in sentence:
            word = word.strip()
            if word in self._vocabulary:
                stream.append(self._vocabulary[word])
            else:  # unknown word; append UNK
                stream.append(self._vocabulary[UNK_IDENTIFIER])
        # increment the stream --
        # 0 will be the <pad> character
        # 1 will be the <SOS> character
        # 2 will be the <EOS> character

        stream = [s + 3 for s in stream]
        return stream

    def _load_vg_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        if not isinstance(index, six.string_types):
            index = str(index)
        data_json = self._gt_regions[index]

        regions = data_json['regions']

        num_regs = len(regions)
        boxes = np.zeros((num_regs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_regs), dtype=np.int32)
        overlaps = np.zeros((num_regs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_regs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, reg in enumerate(regions):
            # Make pixel indexes 0-based
            x1 = reg['x']
            y1 = reg['y']
            x2 = reg['x'] + reg['width']
            y2 = reg['y'] + reg['height']

            boxes[ix, :] = [x1, y1, x2, y2]
            # replace the class id with region id so that can retrieve the caption later
            gt_classes[ix] = reg['region_id']
            overlaps[ix, 1] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        gt_ptokens = []
        ptokens = self._gt_regions[index]['paragraph_tokens']
        for i, s in enumerate(ptokens):
            gt_ptokens.append(self._line_to_stream(s))
            if DEBUG:
                # CHECK consistency
                for wi, w in zip(gt_ptokens[i], s):
                    vocab_w = self._vocabulary_inverted[wi - 3]
                    print(vocab_w, w)
                    assert (vocab_w == UNK_IDENTIFIER or vocab_w == w)

        sparse_overlaps = scipy.sparse.csr_matrix(overlaps)
        dictionary = {'boxes': boxes,
                      'gt_classes': gt_classes,
                      'gt_overlaps': sparse_overlaps,
                      'flipped': False,
                      'gt_ptokens': gt_ptokens,
                      'seg_areas': seg_areas}
        dictionary.update({
            'image': data_json['path'],
            'width': data_json['width'],
            'height': data_json['height'],
            'image_id': data_json['id']
        })

        return dictionary


if __name__ == '__main__':

    from IPython import embed
    embed()
