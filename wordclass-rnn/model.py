from enum import Enum

import tensorflow as tf
from tensorflow.contrib import rnn

## Note:
##
## This model plus the given hyperparameters give an accuracy on the
## validation set of 87.21.

class Phase(Enum):
    Train = 0
    Validation = 1
    Predict = 2

def bidi_gru_layers(
        config,
        inputs,
        hidden_sizes,
        seq_lens=None,
        bidirectional=False,
        phase=Phase.Predict):
    fcells = [rnn.GRUCell(size) for size in hidden_sizes]
    if phase == Phase.Train:
        fcells = [rnn.DropoutWrapper(cell, output_keep_prob=config.rnn_output_dropout, state_keep_prob=config.rnn_state_dropout) for cell in fcells]

    bcells = [rnn.GRUCell(size) for size in hidden_sizes]
    if phase == Phase.Train:
        bcells = [rnn.DropoutWrapper(cell, output_keep_prob=config.rnn_output_dropout, state_keep_prob=config.rnn_state_dropout) for cell in bcells]

    return rnn.stack_bidirectional_dynamic_rnn(fcells, bcells, inputs, dtype=tf.float32, sequence_length=seq_lens)

class Model:
    def __init__(
            self,
            config,
            batch,
            lens_batch,
            label_batch,
            n_chars,
            phase=Phase.Predict):
        batch_size = batch.shape[1]
        input_size = batch.shape[2]
        label_size = label_batch.shape[2]

        # Integer-encoded word characters.
        self._x = tf.placeholder(tf.int32, shape=[batch_size, input_size])

        # Sequence (word) lengths
        self._lens = tf.placeholder(tf.int32, shape=[batch_size])

        if phase != Phase.Predict:
            # Gold-standard label distributions.
            self._y = tf.placeholder(
                tf.float32, shape=[batch_size, label_size])

        # Create an embedding matrix and look up each character.
        char_embeds = tf.get_variable("char_embeds", shape=[n_chars, config.char_embed_size])
        input_layer = tf.nn.embedding_lookup(char_embeds, self._x, None)

        # Apply dropout to the character embeddings during training.
        if phase == Phase.Train:
            input_layer = tf.nn.dropout(input_layer, config.input_dropout)

        # Apply one or more bidirectional GRU layers to the inputs.
        _, fstate, bstate = bidi_gru_layers(config, input_layer, config.hidden_sizes, seq_lens=self.lens, bidirectional = True, phase=phase)

        # Concatenate the forward and backward representations.
        hidden_state = tf.concat([fstate[-1], bstate[-1]], axis=1)
       
        # Compute logits for softmax.
        w = tf.get_variable("w", shape=[hidden_state.shape[1], label_size])
        b = tf.get_variable("b", shape=[label_size])
        logits = tf.matmul(hidden_state, w) + b

        if phase == Phase.Train or Phase.Validation:
            losses = tf.nn.softmax_cross_entropy_with_logits(
                labels=self._y, logits=logits)
            self._loss = loss = tf.reduce_sum(losses)

        if phase == Phase.Train:
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(config.start_lr, global_step,
                                                       decay_steps=config.decay_steps, decay_rate=config.decay_rate)

            self._train_op = tf.train.AdamOptimizer(learning_rate) \
                .minimize(losses, global_step=global_step)
            self._probs = probs = tf.nn.softmax(logits)

        if phase == Phase.Validation:
            # Highest probability labels of the gold data.
            hp_labels = tf.argmax(self.y, axis=1)

            # Predicted labels
            labels = tf.argmax(logits, axis=1)

            correct = tf.equal(hp_labels, labels)
            correct = tf.cast(correct, tf.float32)
            self._accuracy = tf.reduce_mean(correct)

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def lens(self):
        return self._lens

    @property
    def loss(self):
        return self._loss

    @property
    def probs(self):
        return self._probs

    @property
    def train_op(self):
        return self._train_op

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y
