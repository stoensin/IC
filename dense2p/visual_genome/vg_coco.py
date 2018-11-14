# -*- coding: utf-8 -*-
# File: coco.py

import numpy as np
import os
from termcolor import colored
from tabulate import tabulate
import tqdm
import json
import pickle

from tensorpack.utils import logger
from tensorpack.utils.timer import timed_operation
from tensorpack.utils.argtools import log_once

from region_detector.config import config as cfg


__all__ = ['COCODetection', 'COCOMeta']


class _COCOMeta(object):
    # handle the weird (but standard) split of train and val
    INSTANCE_TO_BASEDIR = {
        'valminusminival2014': 'val2014',
        'minival2014': 'val2014',
        'train': 'images'
    }

    def valid(self):
        return hasattr(self, 'cat_names')

    def create(self, cat_ids, cat_names):
        """
        cat_ids: list of ids
        cat_names: list of names
        """
        assert not self.valid()
        assert len(cat_ids) == cfg.DATA.NUM_CATEGORY and len(cat_names) == cfg.DATA.NUM_CATEGORY
        self.cat_names = cat_names
        self.class_names = ['BG'] + self.cat_names

        # background has class id of 0
        self.category_id_to_class_id = {
            v: i + 1 for i, v in enumerate(cat_ids)}
        self.class_id_to_category_id = {
            v: k for k, v in self.category_id_to_class_id.items()}
        cfg.DATA.CLASS_NAMES = self.class_names


COCOMeta = _COCOMeta()


class COCODetection(object):
    def __init__(self, basedir, name, tag='train'):
        self.name = name
        self._imgdir = os.path.realpath(os.path.join(
            basedir, COCOMeta.INSTANCE_TO_BASEDIR.get(name, name)))

        assert os.path.isdir(self._imgdir), self._imgdir

        annotation_file = 'visual_genome/{}_sets.json'.format(tag)
        img2paras = open('visual_genome/img2paragraph_modify_batch', 'rb')
        img2paras_data = pickle.load(img2paras)
        img2paras.close()
        self.paras = img2paras_data

        from pycocotools.coco import COCO
        self.coco = COCO(annotation_file)

        # initialize the meta
        cat_ids = self.coco.getCatIds()
        cat_names = [c['name'] for c in self.coco.loadCats(cat_ids)]
        if not COCOMeta.valid():
            COCOMeta.create(cat_ids, cat_names)
        else:
            assert COCOMeta.cat_names == cat_names

        logger.info("Instances loaded from {}.".format(annotation_file))

    def load(self, add_gt=True, add_mask=False):
        """
        Args:
            add_gt: whether to add ground truth bounding box annotations to the dicts
            add_mask: whether to also add ground truth mask

        Returns:
            a list of dict, each has keys including:
                'height', 'width', 'id', 'file_name',
                and (if add_gt is True) 'boxes', 'class', 'is_crowd', and optionally
                'segmentation'.
        """
        if add_mask:
            assert add_gt
        with timed_operation('Load Groundtruth Boxes for {}'.format(self.name)):
            img_ids = self.coco.getImgIds()
            img_ids.sort()
            # list of dict, each has keys: height,width,id,file_name
            imgs = self.coco.loadImgs(img_ids)

            for img in tqdm.tqdm(imgs):
                if add_gt:
                    self._add_detection_gt(img, add_mask)
            return imgs

    def _add_detection_gt(self, img, add_mask):
        """
        Add 'boxes', 'class', 'is_crowd' of this image to the dict, used by detection.
        If add_mask is True, also add 'segmentation' in coco poly format.
        """
        # ann_ids = self.coco.getAnnIds(imgIds=img['id'])
        # objs = self.coco.loadAnns(ann_ids)
        # objs = self.coco.imgToAnns[img['id']]  # equivalent but faster than the above two lines
        objs = img['regions']
        # clean-up boxes
        valid_objs = []
        width = img['width']
        height = img['height']
        region_len = len(objs)
        gt_classes = np.zeros((region_len), dtype=np.int32)
        for i, obj in enumerate(objs):
            if obj.get('ignore', 0) == 1:
                continue
            x1, y1, w, h = obj['bbox']
            # bbox is originally in float
            # x1/y1 means upper-left corner and w/h means true w/h. This can be verified by segmentation pixels.
            # But we do make an assumption here that (0.0, 0.0) is upper-left corner of the first pixel
            gt_classes[i] = region_len  # obj['region_id']
            x1 = np.clip(float(x1), 0, width)
            y1 = np.clip(float(y1), 0, height)
            w = np.clip(float(x1 + w), 0, width) - x1
            h = np.clip(float(y1 + h), 0, height) - y1
            # Require non-zero seg area and more than 1x1 box size
            if w > 0 and h > 0 and w * h >= 4:
                obj['bbox'] = [x1, y1, x1 + w, y1 + h]
                valid_objs.append(obj)

        # all geometrically-valid boxes are returned
        boxes = np.asarray([obj['bbox'] for obj in valid_objs], dtype='float32')  # (n, 4)
        cls = np.asarray([img['category_id']], dtype='int32')
        is_crowd = np.zeros((region_len), dtype=np.int8)  # np.asarray([obj['iscrowd'] for obj in valid_objs], dtype='int8')

        # add the keys
        img['boxes'] = boxes        # nx4
        img['class'] = gt_classes          # cls n, always >0
        img['is_crowd'] = is_crowd  # is_crowd n,
        img['sent_labels'] = self.paras[str(img['image_id'])]

    @staticmethod
    def load_many(basedir, names, add_gt=True, add_mask=False):
        """
        Load and merges several instance files together.

        Returns the same format as :meth:`COCODetection.load`.
        """
        if not isinstance(names, (list, tuple)):
            names = [names]
        ret = []
        for n in names:
            coco = COCODetection(basedir, n)
            ret.extend(coco.load(add_gt, add_mask=add_mask))
        return ret


if __name__ == '__main__':
    c = COCODetection(cfg.DATA.BASEDIR, 'train')
    gt_boxes = c.load(add_gt=True, add_mask=False)
    print("#Images:", len(gt_boxes))
