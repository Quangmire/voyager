import tensorflow as tf

from voyager.losses import HierarchicalSequenceLoss, HierarchicalCrossEntropyWithLogitsLoss
from voyager.metrics import *


def get_model(model_name):
    if model_name == 'voyager':
        return Voyager
    elif model_name == 'offset_voyager':
        return OffsetVoyager
    elif model_name == 'offset_lstm':
        return OffsetLSTM


class Stateless:
    '''
    Stateless dropout implementation for reproducible dropout
    '''

    @staticmethod
    def dropout_input(x, rate, seed, training):
        # Testing -> no dropout
        if not training:
            return x
        # Training -> Stateless dropout
        keep_mask = tf.cast(tf.random.stateless_uniform(tf.shape(x), seed=seed) >= rate, dtype=x.dtype)
        return x * keep_mask


class Voyager(tf.keras.Model):

    def __init__(self, config, pc_vocab_size, page_vocab_size):
        super(Voyager, self).__init__()
        # Needed for stateless dropout seed
        self.steps_per_epoch = config.steps_per_epoch
        self.step = 0
        self.epoch = 1

        self.config = config
        # Model params
        self.offset_size = (1 << config.offset_bits)
        self.pc_embed_size = config.pc_embed_size
        self.page_embed_size = config.page_embed_size
        self.offset_embed_size = config.page_embed_size * config.num_experts
        self.lstm_size = config.lstm_size
        self.num_layers = config.lstm_layers
        self.sequence_length = config.sequence_length
        self.pc_vocab_size = pc_vocab_size
        self.page_vocab_size = page_vocab_size
        self.batch_size = config.batch_size
        self.sequence_loss = config.sequence_loss
        self.dropout = config.lstm_dropout

        self.init()

    def init(self):
        self.init_embed()
        self.init_mha()
        self.init_lstm()
        self.init_linear()

    def init_embed(self):
        # Embedding Layers
        self.pc_embedding = tf.keras.layers.Embedding(self.pc_vocab_size, self.pc_embed_size, embeddings_regularizer='l1')
        self.page_embedding = tf.keras.layers.Embedding(self.page_vocab_size, self.page_embed_size, embeddings_regularizer='l1')
        self.offset_embedding = tf.keras.layers.Embedding(self.offset_size, self.offset_embed_size, embeddings_regularizer='l1')

    def init_mha(self):
        # Page-Aware Offset Embedding
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=1,
            key_dim=self.page_embed_size,
            attention_axes=(2, 3),
            kernel_regularizer='l1',
        )

    def init_linear(self):
        # Linear layers
        self.page_linear = tf.keras.layers.Dense(self.page_vocab_size, input_shape=(self.lstm_size,), activation=None, kernel_regularizer='l1')
        self.offset_linear = tf.keras.layers.Dense(self.offset_size, input_shape=(self.lstm_size,), activation=None, kernel_regularizer='l1')

    def init_lstm(self):
        # LSTM Layers
        if self.sequence_loss:
            self.coarse_layers = tf.keras.Sequential([
                tf.keras.layers.LSTM(
                    self.lstm_size,
                    return_sequences=True,
                    kernel_regularizer='l1',
                    dropout=self.config.lstm_dropout,
                ) for i in range(self.num_layers)
            ])
            self.fine_layers = tf.keras.Sequential([
                tf.keras.layers.LSTM(
                    self.lstm_size,
                    return_sequences=True,
                    kernel_regularizer='l1',
                    dropout=self.config.lstm_dropout,
                ) for i in range(self.num_layers)
            ])
        else:
            self.coarse_layers = tf.keras.Sequential([
                tf.keras.layers.LSTM(
                    self.lstm_size,
                    return_sequences=(i != self.num_layers - 1),
                    kernel_regularizer='l1',
                    dropout=self.config.lstm_dropout,
                ) for i in range(self.num_layers)
            ])
            self.fine_layers = tf.keras.Sequential([
                tf.keras.layers.LSTM(
                    self.lstm_size,
                    return_sequences=(i != self.num_layers - 1),
                    kernel_regularizer='l1',
                    dropout=self.config.lstm_dropout,
                ) for i in range(self.num_layers)
            ])

    def train_step(self, data):
        ret = super(Voyager, self).train_step(data)

        # Need to keep track of step + epoch for dropout random seed
        self.step += 1
        if self.step >= self.steps_per_epoch:
            self.step = 0
            self.epoch += 1

        # Still need to return original logs
        return ret

    def address_embed(self, pages, offsets, training=False):
        page_embed = self.page_embedding(pages)
        offset_embed = self.offset_embedding(offsets)

        # Compute page-aware offset embedding
        tmp_page_embed = tf.reshape(page_embed, shape=(-1, self.sequence_length, 1, self.page_embed_size))
        offset_embed = tf.reshape(offset_embed, shape=(-1, self.sequence_length, self.offset_embed_size // self.page_embed_size, self.page_embed_size))
        offset_embed = tf.reshape(self.mha(tmp_page_embed, offset_embed, training=training), shape=(-1, self.sequence_length, self.page_embed_size))

        return page_embed, offset_embed

    def lstm_output(self, lstm_inputs, training=False):
        # Run through LSTM
        # Manual embedding dropout to have reproducible dropout randomness
        #lstm_inputs = Stateless.dropout_input(lstm_inputs, self.dropout, seed=(self.epoch, self.step), training=training)
        coarse_out = self.coarse_layers(lstm_inputs, training=training)
        fine_out = self.fine_layers(lstm_inputs, training=training)

        return coarse_out, fine_out

    def linear(self, lstm_output):
        coarse_out, fine_out = lstm_output
        # Pass through linear layers
        coarse_logits = self.page_linear(coarse_out)
        fine_logits = self.offset_linear(fine_out)

        # Full sequence has a time dimension
        if self.sequence_loss:
            return tf.concat([coarse_logits, fine_logits], 2)
        else:
            return tf.concat([coarse_logits, fine_logits], 1)

    def call(self, inputs, training=False):
        pcs = inputs[:, :self.sequence_length]
        pages = inputs[:, self.sequence_length:2 * self.sequence_length]
        offsets = inputs[:, 2 * self.sequence_length:3 * self.sequence_length]

        # Compute embeddings
        pc_embed = self.pc_embedding(pcs)
        page_embed, offset_embed = self.address_embed(pages, offsets, training=training)

        if self.config.pc_localized and self.config.global_stream:
            pc_localized_pcs = inputs[:, 3 * self.sequence_length:4 * self.sequence_length]
            pc_localized_pages = inputs[:, 4 * self.sequence_length:5 * self.sequence_length]
            pc_localized_offsets = inputs[:, 5 * self.sequence_length:6 * self.sequence_length]

            # Compute embeddings
            pc_localized_pc_embed = self.pc_embedding(pc_localized_pcs)
            pc_localized_page_embed, pc_localized_offset_embed = self.address_embed(pc_localized_pages, pc_localized_offsets, training=training)

            lstm_inputs = tf.concat([pc_embed, page_embed, offset_embed,
                pc_localized_pc_embed, pc_localized_page_embed, pc_localized_offset_embed], 2)
        else:
            lstm_inputs = tf.concat([pc_embed, page_embed, offset_embed], 2)

        lstm_output = self.lstm_output(lstm_inputs, training)

        return self.linear(lstm_output)

    def load(self, model_path):
        '''
        Load weights from checkpoint. expect_partial() necessary to silence
        some extraneous TensorFlow output
        '''
        self.load_weights(model_path).expect_partial()

    @classmethod
    def compile_model(cls, config, num_unique_pcs, num_unique_pages):
        '''
        Create and compile the model
        '''
        model = cls(config, num_unique_pcs, num_unique_pages)
        num_offsets = (1 << config.offset_bits)

        if config.sequence_loss:
            loss = HierarchicalSequenceLoss(multi_label=config.multi_label, num_offsets=num_offsets)
        else:
            loss = HierarchicalCrossEntropyWithLogitsLoss(multi_label=config.multi_label, num_offsets=num_offsets)

        metrics = [
            PageHierarchicalAccuracy(sequence_loss=config.sequence_loss, multi_label=config.multi_label, num_offsets=num_offsets),
            OffsetHierarchicalAccuracy(sequence_loss=config.sequence_loss, multi_label=config.multi_label, num_offsets=num_offsets),
            OverallHierarchicalAccuracy(sequence_loss=config.sequence_loss, multi_label=config.multi_label, num_offsets=num_offsets),
        ]

        # Only add prediction accuracy for multi-label since the above 3 metrics
        # are the same as the prediction accuracy for global stream
        if config.multi_label:
            metrics.extend([
                PagePredictionHierarchicalAccuracy(sequence_loss=config.sequence_loss, num_offsets=num_offsets),
                OffsetPredictionHierarchicalAccuracy(sequence_loss=config.sequence_loss, num_offsets=num_offsets),
                OverallPredictionHierarchicalAccuracy(sequence_loss=config.sequence_loss, num_offsets=num_offsets),
            ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(config.learning_rate),
            loss=loss,
            metrics=metrics,
        )

        return model


class OffsetVoyager(Voyager):

    def init_linear(self):
        # Linear layers
        self.page_linear = tf.keras.layers.Dense(self.page_vocab_size, input_shape=(self.lstm_size,), activation=None, kernel_regularizer='l1')
        self.offset_linear = tf.keras.layers.Dense(self.offset_size, input_shape=(self.lstm_size + self.page_embed_size,), activation=None, kernel_regularizer='l1')

    def linear(self, lstm_output):
        coarse_out, fine_out = lstm_output
        # Pass through linear layers
        coarse_logits = self.page_linear(coarse_out)
        if self.sequence_loss:
            fine_logits = self.offset_linear(tf.concat([fine_out, self.page_embedding(tf.argmax(coarse_logits, axis=-1))], axis=2))
        else:
            fine_logits = self.offset_linear(tf.concat([fine_out, self.page_embedding(tf.argmax(coarse_logits, axis=-1))], axis=1))

        # Full sequence has a time dimension
        if self.sequence_loss:
            return tf.concat([coarse_logits, fine_logits], 2)
        else:
            return tf.concat([coarse_logits, fine_logits], 1)


class OffsetLSTM(Voyager):

    def init_lstm(self):
        # LSTM Layers
        if self.sequence_loss:
            self.coarse_layers = tf.keras.Sequential([
                tf.keras.layers.LSTM(
                    self.lstm_size,
                    return_sequences=True,
                    kernel_regularizer='l1',
                ) for i in range(self.num_layers)
            ])
        else:
            self.coarse_layers = tf.keras.Sequential([
                tf.keras.layers.LSTM(
                    self.lstm_size,
                    return_sequences=(i != self.num_layers - 1),
                    kernel_regularizer='l1',
                ) for i in range(self.num_layers)
            ])

    def init_linear(self):
        # Linear layers
        self.page_linear = tf.keras.layers.Dense(self.page_vocab_size, input_shape=(self.lstm_size,), activation=None, kernel_regularizer='l1')
        self.offset_linear = tf.keras.layers.Dense(self.offset_size, input_shape=(self.lstm_size + self.page_embed_size,), activation=None, kernel_regularizer='l1')

    def lstm_output(self, lstm_inputs, training=False):
        # Run through LSTM
        # Manual embedding dropout to have reproducible dropout randomness
        lstm_inputs = Stateless.dropout_input(lstm_inputs, self.dropout, seed=(self.epoch, self.step), training=training)
        coarse_out = self.coarse_layers(lstm_inputs, training=training)

        return coarse_out

    def linear(self, lstm_output):
        coarse_out = lstm_output
        # Pass through linear layers
        coarse_logits = self.page_linear(coarse_out)
        if self.sequence_loss:
            fine_logits = self.offset_linear(tf.concat([coarse_out, self.page_embedding(tf.argmax(coarse_logits, axis=-1))], axis=2))
        else:
            fine_logits = self.offset_linear(tf.concat([coarse_out, self.page_embedding(tf.argmax(coarse_logits, axis=-1))], axis=1))

        # Full sequence has a time dimension
        if self.sequence_loss:
            return tf.concat([coarse_logits, fine_logits], 2)
        else:
            return tf.concat([coarse_logits, fine_logits], 1)
