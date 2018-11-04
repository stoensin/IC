import math
import logging

import tensorflow as tf

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import vocabulary
from im2txt.inference_utils import caption_generator


logger = logging.getLogger()

CHECKPOINT_PATH = './assets/checkpoint/model2.ckpt-2000000'
VOCAB_FILE = './assets/word_counts.txt'


class ModelWrapper(object):
    """Model wrapper for TensorFlow models in SavedModel format"""

    def __init__(self):

        g = tf.Graph()
        with g.as_default():
            model = inference_wrapper.InferenceWrapper()
            restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                                       CHECKPOINT_PATH)
        g.finalize()
        self.model = model
        sess = tf.Session(graph=g)
        # Load the model from checkpoint.
        restore_fn(sess)
        self.sess = sess

    def predict(self, image_data):
        # Create the vocabulary.
        vocab = vocabulary.Vocabulary(VOCAB_FILE)

        # Prepare the caption generator. Here we are implicitly using the default
        # beam search parameters. See caption_generator.py for a description of the
        # available beam search parameters.
        generator = caption_generator.CaptionGenerator(self.model, vocab)

        captions = generator.beam_search(self.sess, image_data)

        results = []
        for i, caption in enumerate(captions):
            # Ignore begin and end words.
            sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
            sentence = " ".join(sentence)
            # print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
            results.append((i, sentence, math.exp(caption.logprob)))

        return results
