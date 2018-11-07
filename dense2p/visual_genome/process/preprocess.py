##########################################################
#### Preprocessing of visual genome dataset, including vocabularity generation,
#### removing invalid bboxes and phrases, tokenization, and result saving
#########################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os
import string
import json
import time
import numpy as np
from six.moves import xrange
from collections import Counter
from split.data_splits import data_splits
import os.path as osp
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Preprocessing visual genome')
parser.add_argument('--version', dest='version', type=float, default=1.2, help='the version of visual genome dataset.')
parser.add_argument('--path', dest='path', type=str, default='/home/ICR/im2txt/visual_genome/VG', help='directory saving the raw dataset')
parser.add_argument('--output_dir', dest='output_dir', type=str, default='/home/ICR/visual_genome/data', help='output directory of data files')
parser.add_argument('--max_words', dest='max_words', type=int, default=10, help='maximum length of words for training.')
args = parser.parse_args()

MAX_WORDS = args.max_words
VG_VERSION = args.version
VG_PATH = args.path
VG_IMAGE_ROOT = '%s/images' % VG_PATH
# read whole regions with a json file
VG_REGION_PATH = '%s/%s/region_descriptions.json' % (VG_PATH, VG_VERSION)

VG_PARA_PATH = '%s/%s/paragraphs_v1.json' % (VG_PATH, VG_VERSION)
VG_METADATA_PATH = '%s/%s/image_data.json' % (VG_PATH, VG_VERSION)
vocabulary_size = 10000
HAS_VOCAB = True
OUTPUT_DIR = args.output_dir
SPLITS_PATH = '/content/im2txt/visual_genome/process'+ '/%s_split.json'


# UNK_IDENTIFIER is the word used to identify unknown words
UNK_IDENTIFIER = '<unk>'


class VGDataProcessor:
    def __init__(self, split_name, image_data, regions_all=None, paras_all=None, vocab=None, split_ids=[], max_words=MAX_WORDS):
        self.max_words = max_words
        self.images = {}
        phrases_all = []
        num_invalid_bbox = 0
        num_bbox = 0
        num_empty_phrase = 0


        tic = time.time()
        for i in tqdm(xrange(len(image_data)), desc='%s' % split_name):
            image_info = image_data[i]
            # NOTE: for VG_1.2 and VG_1.0 the key in image_info about id is different.
            im_id = image_info['image_id']

            if im_id not in split_ids:
                continue


            item = regions_all[i]
            if item['id'] != im_id:
                print('region and image metadata inconsistent with regions id: %s, image id: %s' %
                      (item['id'], image_info['image_id']))
                exit()
            if im_id not in paras_all:
                print('region and image metadata inconsistent with paragraph id.')
                exit()
            # tokenize phrase
            num_bbox += len(item['regions'])
            regions_filt = []

            paragraph = paras_all[im_id]['paragraph']
            paragraph_tokens = paras_all[im_id]['paragraph_tokens']
            for obj in item['regions']:
                # remove invalid regions
                if obj['x'] < 0 or obj['y'] < 0 or \
                        obj['width'] <= 0 or obj['height'] <= 0 or \
                        obj['x'] + obj['width'] >= image_info['width'] or \
                        obj['y'] + obj['height'] >= image_info['height']:
                    num_invalid_bbox += 1
                    continue
                obj.pop('phrase')
                obj.pop('image_id')
                regions_filt.append(obj)

            phrases_all.extend(paragraph_tokens)
            im_path = '%s/%d.jpg' % (VG_IMAGE_ROOT, im_id)
            Dict = {'path': im_path, 'regions': regions_filt, 'id': im_id,
                    'height': image_info['height'], 'width': image_info['width'],
                    'paragraph': paragraph,
                    'paragraph_tokens': paragraph_tokens}

            self.images[item['id']] = Dict

        toc = time.time()
        print('processing %s set with time: %.2f seconds' % (split_name, toc - tic))
        print("there are %d invalid bboxes out of %d" % (num_invalid_bbox, num_bbox))
        print("there are %d empty phrases after triming" % num_empty_phrase)
        if vocab is None:
            self.init_vocabulary(phrases_all)
        else:
            self.vocabulary_inverted = vocab
        self.vocabulary = {}
        for index, word in enumerate(self.vocabulary_inverted):
            self.vocabulary[word] = index

    def init_vocabulary(self, phrases_all):
        word_freq = Counter(itertools.chain(*phrases_all))
        print("Found %d unique word tokens." % len(word_freq.items()))
        vocab_freq = word_freq.most_common(vocabulary_size - 1)
        self.vocabulary_inverted = [x[0] for x in vocab_freq]
        self.vocabulary_inverted.insert(0, UNK_IDENTIFIER)
        print("Using vocabulary size %d." % vocabulary_size)
        print("The least frequent word in our vocabulary is '%s' and appeared %d times." %
              (vocab_freq[-1][0], vocab_freq[-1][1]))

    def dump_vocabulary(self, vocab_filename):
        print('Dumping vocabulary to file: %s' % vocab_filename)
        with open(vocab_filename, 'wb') as vocab_file:
            for word in self.vocabulary_inverted:
                vocab_file.write('%s\n' % word)
        print('Done.')


VG_IMAGE_PATTERN = '%s/%%d.jpg' % VG_IMAGE_ROOT


def extract_paras(paras_json, max_words=MAX_WORDS):
    all_paras = {}
    num_filtered_sents = 0
    for img in paras_json:
        img_id = img['image_id']

        paragraph = img['paragraph']
        # sentences = []
        sentences_tokens = []
        for s in paragraph.split('.'):
            phrase = s.strip().encode('ascii', 'ignore').lower()
            if len(phrase) == 0:
                continue
            phrase_tokens = phrase.translate(None, string.punctuation).split()
            if len(phrase_tokens) > max_words:
                num_filtered_sents += 1
                continue
            # sentences.append(phrase)
            sentences_tokens.append(phrase_tokens)
        all_paras[img_id] = {"paragraph": paragraph,
                             "paragraph_tokens": sentences_tokens}
    print("Number of sentences filtered out: %d, with max_words: %d" % (num_filtered_sents, max_words))
    return all_paras


def process_dataset(split_name, paras_all, vocab=None):

    with open(SPLITS_PATH % split_name, 'r') as f:
        split_image_ids = json.load(f)
    print('split image number: %d for split name: %s' % (len(split_image_ids), split_name))

    print('start loading json files...')
    t1 = time.time()

    regions_all = json.load(open(VG_REGION_PATH))

    image_data = json.load(open(VG_METADATA_PATH))
    t2 = time.time()
    print('%f seconds for loading' % (t2 - t1))

    processor = VGDataProcessor(split_name, image_data, regions_all, paras_all,
                                split_ids=split_image_ids, vocab=vocab)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if vocab is None:
        vocab_out_path = '%s/vocabulary.txt' % OUTPUT_DIR
        processor.dump_vocabulary(vocab_out_path)

    # dump image region dict
    with open(OUTPUT_DIR + '/%s_gt_paragraph.json' % split_name, 'w') as f:
        json.dump(processor.images, f)

    return processor.vocabulary_inverted


def process_vg():
    vocab = None
    # use existing vocabulary
    if HAS_VOCAB:
        vocab_path = '%s/vocabulary.txt' % OUTPUT_DIR
        with open(vocab_path, 'r') as f:
            vocab = [line.strip() for line in f]

    paras_json = json.load(open(VG_PARA_PATH))
    paras_all = extract_paras(paras_json)

    datasets = ['train', 'val', 'test']
    for split_name in datasets:
        vocab = process_dataset(split_name, paras_all, vocab=vocab)


if __name__ == "__main__":
    process_vg()
