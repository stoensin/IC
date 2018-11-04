# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Image-to-text implementation based on http://arxiv.org/abs/1411.4555.
"Show and Tell: A Neural Image Caption Generator"
Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from im2txt.ops import image_embedding
from im2txt.ops import image_processing
from im2txt.ops import inputs as input_ops


class ShowAndTellModel(object):
    """Image-to-text implementation based on http://arxiv.org/abs/1411.4555.
    "Show and Tell: A Neural Image Caption Generator"
    Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
    """

    def __init__(self, config, mode, train_inception=False):
        """Basic setup.
        Args:
          config: Object containing configuration parameters.
          mode: "train", "eval" or "inference".
          train_inception: Whether the inception submodel variables are trainable.
        """
        assert mode in ["train", "eval", "inference"]
        self.config = config
        self.mode = mode
        self.train_inception = train_inception

        # Reader for the input data.
        self.reader = tf.TFRecordReader()

        # To match the "Show and Tell" paper we initialize all variables with a
        # random uniform initializer.
        self.initializer = tf.random_uniform_initializer(
            minval=-self.config.initializer_scale,
            maxval=self.config.initializer_scale)

        # A float32 Tensor with shape [batch_size, height, width, channels].
        self.images = None

        # An int32 Tensor with shape [batch_size, padded_length].
        self.input_seqs = None

        # An int32 Tensor with shape [batch_size, padded_length].
        self.target_seqs = None

        # An int32 0/1 Tensor with shape [batch_size, padded_length].
        self.input_mask = None

        # A float32 Tensor with shape [batch_size, embedding_size].
        self.image_embeddings = None

        # A float32 Tensor with shape [batch_size, padded_length, embedding_size].
        self.seq_embeddings = None

        # A float32 scalar Tensor; the total loss for the trainer to optimize.
        self.total_loss = None

        # A float32 Tensor with shape [batch_size * padded_length].
        self.target_cross_entropy_losses = None

        # A float32 Tensor with shape [batch_size * padded_length].
        self.target_cross_entropy_loss_weights = None

        # Collection of variables from the inception submodel.
        self.inception_variables = []

        # Function to restore the inception submodel from checkpoint.
        self.init_fn = None

        # Global step Tensor.
        self.global_step = None

        self.D = 1536  # imgae dimension [1]

        self.L = 64  # image dimension [0]

        self.H = 512

        self.M = 512

        self.V = 12000

        self.weight_initializer = self.initializer  # tf.contrib.layers.xavier_initializer()
        self.const_initializer = self.initializer  # tf.constant_initializer(0.0)

    def is_training(self):
        """Returns true if the model is built for training mode."""
        return self.mode == "train"

    def process_image(self, encoded_image, thread_id=0):
        """Decodes and processes an image string.
        Args:
          encoded_image: A scalar string Tensor; the encoded image.
          thread_id: Preprocessing thread id used to select the ordering of color
            distortions.
        Returns:
          A float32 Tensor of shape [height, width, 3]; the processed image.
        """
        return image_processing.process_image(encoded_image,
                                              is_training=self.is_training(),
                                              height=self.config.image_height,
                                              width=self.config.image_width,
                                              thread_id=thread_id,
                                              image_format=self.config.image_format)

    def build_inputs(self):
        """Input prefetching, preprocessing and batching.
        Outputs:
          self.images
          self.input_seqs
          self.target_seqs (training and eval only)
          self.input_mask (training and eval only)
        """
        if self.mode == "inference":
            # In inference mode, images and inputs are fed via placeholders.
            image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed")
            input_feed = tf.placeholder(dtype=tf.int64,
                                        shape=[None],  # batch_size
                                        name="input_feed")

            # Process image and insert batch dimensions.
            images = tf.expand_dims(self.process_image(image_feed), 0)
            input_seqs = tf.expand_dims(input_feed, 1)

            # No target sequences or input mask in inference mode.
            target_seqs = None
            input_mask = None
        else:
            # Prefetch serialized SequenceExample protos.
            input_queue = input_ops.prefetch_input_data(
                self.reader,
                self.config.input_file_pattern,
                is_training=self.is_training(),
                batch_size=self.config.batch_size,
                values_per_shard=self.config.values_per_input_shard,
                input_queue_capacity_factor=self.config.input_queue_capacity_factor,
                num_reader_threads=self.config.num_input_reader_threads)

            # Image processing and random distortion. Split across multiple threads
            # with each thread applying a slightly different distortion.
            assert self.config.num_preprocess_threads % 2 == 0
            images_and_captions = []
            for thread_id in range(self.config.num_preprocess_threads):
                serialized_sequence_example = input_queue.dequeue()
                encoded_image, caption = input_ops.parse_sequence_example(
                    serialized_sequence_example,
                    image_feature=self.config.image_feature_name,
                    caption_feature=self.config.caption_feature_name)
                image = self.process_image(encoded_image, thread_id=thread_id)
                images_and_captions.append([image, caption])

            # Batch inputs.
            queue_capacity = (2 * self.config.num_preprocess_threads *
                              self.config.batch_size)
            images, input_seqs, target_seqs, input_mask = (
                input_ops.batch_with_dynamic_pad(images_and_captions,
                                                 batch_size=self.config.batch_size,
                                                 queue_capacity=queue_capacity))

        self.images = images
        self.input_seqs = input_seqs
        self.target_seqs = target_seqs
        self.input_mask = input_mask

    def build_image_embeddings(self):
        """Builds the image model subgraph and generates image embeddings.
        Inputs:
          self.images
        Outputs:
          self.image_embeddings
        """
        inception_output = image_embedding.inception_v4(
            self.images,
            trainable=self.train_inception,
            is_training=self.is_training())
        self.inception_variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV4")

        # Map inception output into embedding space.
        if self.mode == "inference":
            with tf.variable_scope("image_embedding") as scope:
                image_embeddings = tf.reshape(tf.expand_dims(
                    inception_output, 1), shape=[1, 64, 1536])
        else:
            with tf.variable_scope("image_embedding") as scope:
               # image_embeddings = tf.contrib.layers.fully_connected(
                #  inputs=inception_output,
               # num_outputs=self.config.embedding_size,
             #   activation_fn=None,
                #  weights_initializer=self.initializer,
                #  biases_initializer=None,
                #  scope=scope)
                image_embeddings = tf.reshape(tf.expand_dims(inception_output, 1), shape=[
                                              self.config.batch_size, 64, 1536])
        # Save the embedding size in the graph.
        tf.constant(self.config.embedding_size, name="embedding_size")

        self.image_embeddings = image_embeddings

    def build_seq_embeddings(self):
        """Builds the input sequence embeddings.
        Inputs:
          self.input_seqs
        Outputs:
          self.seq_embeddings
        """
        with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
            embedding_map = tf.get_variable(
                name="map",
                shape=[self.config.vocab_size, self.config.embedding_size],
                initializer=self.initializer)
            seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.input_seqs)

        self.seq_embeddings = seq_embeddings

    def _initial_lstm(self, features):
        with tf.variable_scope('initial_lstm'):
            features_mean = tf.reduce_mean(features, 1)

            w_h = tf.get_variable('w_h1', [self.D, self.H], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h1', [self.H], initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

            w_c = tf.get_variable('w_c', [self.D, self.H], initializer=self.weight_initializer)
            b_c = tf.get_variable('b_c', [self.H], initializer=self.const_initializer)
            c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
            return c, h

    def _project_features(self, features):
        with tf.variable_scope('project_features'):
            w = tf.get_variable('w1', [self.D, self.D], initializer=self.weight_initializer)
            features_flat = tf.reshape(features, [-1, self.D])
            features_proj = tf.matmul(features_flat, w)
            features_proj = tf.reshape(features_proj, [-1, self.L, self.D])
            return features_proj

    def _attention_layer(self, features, features_proj, h, reuse=False):
        with tf.variable_scope('attention_layer', reuse=reuse):
            w = tf.get_variable('w2', [self.H, self.D], initializer=self.weight_initializer)
            b = tf.get_variable('b2', [self.D], initializer=self.const_initializer)
            w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)
            h_att = tf.nn.relu(
                features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b)    # (N, L, D)
            out_att = tf.reshape(tf.matmul(tf.reshape(
                h_att, [-1, self.D]), w_att), [-1, self.L])   # (N, L)
            alpha = tf.nn.softmax(out_att)
            context = tf.reduce_sum(features * tf.expand_dims(alpha, 2),
                                    1, name='context')  # (N, D)
            return context, alpha

    def _selector(self, context, h, reuse=False):
        with tf.variable_scope('selector', reuse=reuse):
            w = tf.get_variable('w3', [self.H, 1], initializer=self.weight_initializer)
            b = tf.get_variable('b3', [1], initializer=self.const_initializer)
            beta = tf.nn.sigmoid(tf.matmul(h, w)+b, 'beta')
            context = tf.multiply(beta, context, name='selected_context')
            return context, beta

    def _decode_lstm(self, x, h, context, reuse=False):
        with tf.variable_scope('logits', reuse=reuse) as logits_scope:
            w_h = tf.get_variable('w_h', [self.H, self.M], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.M], initializer=self.const_initializer)
            w_out = tf.get_variable('w_out', [self.M, self.V], initializer=self.weight_initializer)
            b_out = tf.get_variable('b_out', [self.V], initializer=self.const_initializer)

            if self.mode == "train":
                h = tf.nn.dropout(h, self.config.lstm_dropout_keep_prob)
            h_logits = tf.matmul(h, w_h) + b_h

            w_ctx2out = tf.get_variable(
                'w_ctx2out', [self.D, self.M], initializer=self.weight_initializer)
            h_logits += tf.matmul(context, w_ctx2out)

            h_logits += x
            h_logits = tf.nn.tanh(h_logits)

            if self.mode == "train":
                h_logits = tf.nn.dropout(h_logits, self.config.lstm_dropout_keep_prob)
            #out_logits = tf.matmul(h_logits, w_out) + b_out
            # return out_logits
            return tf.contrib.layers.fully_connected(inputs=h_logits, num_outputs=12000, activation_fn=None, weights_initializer=self.initializer, scope=logits_scope)

    def build_model(self):
        """Builds the model.

        Inputs:
          self.image_embeddings
          self.seq_embeddings
          self.target_seqs (training and eval only)
          self.input_mask (training and eval only)

        Outputs:
          self.total_loss (training and eval only)
          self.target_cross_entropy_losses (training and eval only)
          self.target_cross_entropy_loss_weights (training and eval only)
        """
        # This LSTM cell has biases and outputs tanh(new_c) * sigmoid(o), but the
        # modified LSTM in the "Show and Tell" paper has no biases and outputs
        # new_c * sigmoid(o).

        batch_size = self.config.batch_size
        features = self.image_embeddings
        feature_proj = self._project_features(features)
        targets = self.target_seqs
        mask = self.input_mask
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.config.num_lstm_units)
        c, h = self._initial_lstm(features)
        logits = []

        if self.mode == "inference":
            x = tf.reshape(self.seq_embeddings, shape=[1, 512])
            tf.concat(axis=1, values=(c, h), name="initial_state")
            state_feed = tf.placeholder(dtype=tf.float32, shape=[
                                        None, sum(lstm_cell.state_size)], name="state_feed")
            (c, h) = tf.split(state_feed, 2, 1)
            context, alpha = self._attention_layer(features, feature_proj, h, reuse=tf.AUTO_REUSE)
            context, beta = self._selector(context, h, reuse=tf.AUTO_REUSE)
            with tf.variable_scope('lstm', reuse=tf.AUTO_REUSE):
                _, (c, h) = lstm_cell(inputs=tf.concat([x, context], 1), state=[c, h])
            tf.concat(axis=1, values=(c, h), name="state")
            logit = self._decode_lstm(x, h, context, reuse=tf.AUTO_REUSE)
            tf.nn.softmax(tf.cast(logit, dtype=tf.float32), name="softmax")
        else:
            alpha_list = []
            loss = 0.0
            x = self.seq_embeddings
            losses = []
            for t in range(63):
                context, alpha = self._attention_layer(features, feature_proj, h, reuse=(t != 0))
                alpha_list.append(alpha)
                context, beta = self._selector(context, h, reuse=(t != 0))
                with tf.variable_scope('lstm', reuse=(t != 0)) as lstm_scope:
                    _, (c, h) = lstm_cell(inputs=tf.concat([x[:, t, :], context], 1), state=[c, h])
                logits = self._decode_lstm(x[:, t, :], h, context, reuse=(t != 0))
                los = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=targets[:, t], logits=logits)
                los = los * mask[:, t]
                loss += tf.reduce_sum(los)

            # alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))     # (N, T, L)
            # alphas_all = tf.reduce_sum(alphas, 1)      # (N, L)
            #alpha_reg = 0.5 * tf.reduce_sum((16./196 - alphas_all) ** 2)
            #loss += alpha_reg

            weights = tf.to_float(tf.reshape(mask, [-1]))

            batch_loss = tf.div(loss,  # tf.reduce_sum(tf.multiply(loss, weights)),
                                tf.reduce_sum(weights),
                                name="batch_loss")
            tf.losses.add_loss(batch_loss)
            total_loss = tf.losses.get_total_loss()
            tf.summary.scalar("losses/batch_loss", batch_loss)
            tf.summary.scalar("losses/total_loss", total_loss)
            # for var in tf.trainable_variables():
            # tf.summary.histogram("parameters/" + var.op.name, var)

            self.total_loss = total_loss
            self.target_cross_entropy_losses = loss  # Used in evaluation.
            self.target_cross_entropy_loss_weights = weights  # Used in evaluation.

    def setup_inception_initializer(self):
        """Sets up the function to restore inception variables from checkpoint."""
        if self.mode != "inference":
            # Restore inception variables only.
            saver = tf.train.Saver(self.inception_variables)

            def restore_fn(sess):
                tf.logging.info("Restoring Inception variables from checkpoint file %s",
                                self.config.inception_checkpoint_file)
                saver.restore(sess, self.config.inception_checkpoint_file)

            self.init_fn = restore_fn

    def setup_global_step(self):
        """Sets up the global step Tensor."""
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        self.global_step = global_step

    def build(self):
        """Creates all ops for training and evaluation."""
        self.build_inputs()
        self.build_image_embeddings()
        self.build_seq_embeddings()
        self.build_model()
        self.setup_inception_initializer()
        self.setup_global_step()
