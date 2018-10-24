# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick, Sean Bell and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from densecap.config import cfg
from faster_rcnn.fast_rcnn.bbox_transform import bbox_transform
from faster_rcnn.utils.cython_bbox import bbox_overlaps
from faster_rcnn.rpn.rois_offset_layer import compute_rois_offset


def proposal_target_single_class_layer(rpn_rois, rpn_scores, gt_boxes, gt_phrases):
    """
  Assign object detection proposals to ground-truth targets. Produces proposal
  classification labels and bounding-box regression targets.
  """

    # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
    # (i.e., layers.proposal_layer.ProposalLayer), or any other source
    all_rois = rpn_rois
    all_scores = rpn_scores

    # Include ground-truth boxes in the set of candidate rois
    if cfg.TRAIN.USE_GT:
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
        )
        # not sure if it a wise appending, but anyway i am not using it
        all_scores = np.vstack((all_scores, zeros))

    num_images = 1
    rois_per_image = cfg.TRAIN.BATCH_SIZE // num_images
    fg_rois_per_image = int(cfg.TRAIN.FG_FRACTION * rois_per_image)

    # Sample rois with classification labels and bounding box regression
    # targets
    labels, rois, roi_scores, bbox_targets, bbox_inside_weights, phrases = _sample_rois(
        all_rois, all_scores, gt_boxes, gt_phrases, fg_rois_per_image,
        rois_per_image)

    rois = rois.reshape(-1, 5)
    roi_scores = roi_scores.reshape(-1)
    labels = labels.reshape(-1, 1)
    phrases = phrases.reshape(-1, cfg.MAX_WORDS)
    bbox_targets = bbox_targets.reshape(-1, 4)
    bbox_inside_weights = bbox_inside_weights.reshape(-1, 4)
    bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)
    clss = np.array(labels > 0).astype(np.int32)

    return rois, roi_scores, labels, bbox_targets, \
           bbox_inside_weights, bbox_outside_weights, clss, phrases


def _get_bbox_regression_labels(bbox_target_data):
    """Bounding-box regression targets (bbox_target_data) are stored in a
  compact form N x (class, tx, ty, tw, th)

  Returns:
      bbox_target (ndarray): N x 4 blob of regression targets
      bbox_inside_weights (ndarray): N x 4 blob of loss weights
  """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        bbox_targets[ind, :] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, :] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                   / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack(
        (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)


def _sample_rois(all_rois, all_scores, gt_boxes, gt_phrases, fg_rois_per_image, rois_per_image):
    """Generate a random sample of RoIs comprising foreground and background
  examples.
  """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]
    phrases = gt_phrases[gt_assignment]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]

    # Small modification to the original version where we ensure a fixed number of regions are sampled
    if cfg.SAMPLE_NUM_FIXED_REGIONS:
        if fg_inds.size > 0 and bg_inds.size > 0:
            fg_rois_per_image = min(fg_rois_per_image, fg_inds.size)
            fg_inds = npr.choice(fg_inds, size=int(fg_rois_per_image), replace=False)
            bg_rois_per_image = rois_per_image - fg_rois_per_image
            to_replace = bg_inds.size < bg_rois_per_image
            bg_inds = npr.choice(bg_inds, size=int(bg_rois_per_image), replace=to_replace)
        elif fg_inds.size > 0:
            to_replace = fg_inds.size < rois_per_image
            fg_inds = npr.choice(fg_inds, size=int(rois_per_image), replace=to_replace)
            fg_rois_per_image = rois_per_image
        elif bg_inds.size > 0:
            to_replace = bg_inds.size < rois_per_image
            bg_inds = npr.choice(bg_inds, size=int(rois_per_image), replace=to_replace)
            fg_rois_per_image = 0
        else:
            import pdb
            pdb.set_trace()
    else:
        # foreground RoIs
        fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
        # Sample foreground regions without replacement
        if fg_inds.size > 0:
            fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

        # Compute number of background RoIs to take from this image (guarding
        # against there being fewer than desired)
        bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
        bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
        # Sample background regions without replacement
        if bg_inds.size > 0:
            bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    phrases = phrases[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[int(fg_rois_per_image):] = 0
    phrases[int(fg_rois_per_image):, :] = 0
    rois = all_rois[keep_inds]
    roi_scores = all_scores[keep_inds]

    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    if cfg.DEBUG_ALL:
        target_boxes = compute_rois_offset(rois[:, 1:5], bbox_target_data[:, 1:5])
        match_boxes = gt_boxes[gt_assignment[keep_inds], :4]
        print('boxes consistency check')
        print(target_boxes[:2,:])
        print(match_boxes[:2,:])
        assert np.linalg.norm(target_boxes - match_boxes) < 0.01

    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data)

    return labels, rois, roi_scores, bbox_targets, bbox_inside_weights, phrases
