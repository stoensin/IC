from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import cPickle
from im2txt.densecap.config import cfg
from collections import Counter
import numpy as np
import six
from six.moves import xrange

# Pooling the visual features into a single dense feature
def paragraph_layer(feature_vec):
    assert feature_vec.ndim == 2

    num_sentences, n_w = feature_vec.shape

    assert n_w == cfg.MAX_WORDS

    if num_sentences > cfg.IM2P.S_MAX:
        num_sentences = cfg.IM2P.S_MAX
        feature_vec = feature_vec[: num_sentences]
    # else:
    #     repeats = [1] * (num_sentences - 1) + [cfg.IM2P.S_MAX - num_sentences + 1]
    #     feature_vec = np.repeat(feature_vec, repeats, axis=0)

    sentence_labels = np.array([0] * (num_sentences - 1) + [1], dtype=np.int32)
    #* (cfg.IM2P.S_MAX - num_sentences + 1), dtype=np.int32)
    target_sentences = np.zeros((num_sentences, cfg.TIME_STEPS), dtype=np.float32)
    input_sentences = np.zeros((num_sentences, cfg.TIME_STEPS - 1), dtype=np.float32)
    # add start token "1"
    target_sentences[:, 0] = 1
    input_sentences[:, 0] = 1
    for i in xrange(num_sentences):
        s = feature_vec[i]
        target_sentences[i, 1: -1] = s
        # "2" is end of sentence token
        target_sentences[i, np.sum(s > 0) + 1] = 2
        input_sentences[i, 1:] = s

    sentence_lengths = np.sum(target_sentences > 0, axis=1, dtype=np.int32)

    return np.array(num_sentences, dtype=np.int32), sentence_lengths, sentence_labels, input_sentences, target_sentences
