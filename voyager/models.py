import tensorflow as tf

from voyager.losses import HierarchicalSequenceLoss, HierarchicalCrossEntropyWithLogitsLoss
from voyager.metrics import *


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


class HierarchicalLSTM(tf.keras.Model):

    def __init__(self, config, pc_vocab_size, page_vocab_size):
        super(HierarchicalLSTM, self).__init__()
        # Needed for stateless dropout seed
        self.steps_per_epoch = config.steps_per_epoch
        self.step = 0
        self.epoch = 1

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

        # Embedding Layers
        self.pc_embedding = tf.keras.layers.Embedding(self.pc_vocab_size, self.pc_embed_size, embeddings_regularizer='l1')
        self.page_embedding = tf.keras.layers.Embedding(self.page_vocab_size, self.page_embed_size, embeddings_regularizer='l1')
        self.offset_embedding = tf.keras.layers.Embedding(self.offset_size, self.offset_embed_size, embeddings_regularizer='l1')

        # Page-Aware Offset Embedding
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=1,
            key_dim=self.page_embed_size,
            attention_axes=(2, 3),
            kernel_regularizer='l1',
        )

        # LSTM Layers
        if self.sequence_loss:
            self.coarse_layers = tf.keras.Sequential([
                tf.keras.layers.LSTM(
                    self.lstm_size,
                    return_sequences=True,
                    kernel_regularizer='l1',
                ) for i in range(self.num_layers)
            ])
            self.fine_layers = tf.keras.Sequential([
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
            self.fine_layers = tf.keras.Sequential([
                tf.keras.layers.LSTM(
                    self.lstm_size,
                    return_sequences=(i != self.num_layers - 1),
                    kernel_regularizer='l1',
                ) for i in range(self.num_layers)
            ])

        # Linear layers
        self.page_linear = tf.keras.layers.Dense(self.page_vocab_size, input_shape=(self.lstm_size,), activation=None, kernel_regularizer='l1')
        self.offset_linear = tf.keras.layers.Dense(self.offset_size, input_shape=(self.lstm_size,), activation=None, kernel_regularizer='l1')

    def train_step(self, data):
        ret = super(HierarchicalLSTM, self).train_step(data)

        # Need to keep track of step + epoch for dropout random seed
        self.step += 1
        if self.step >= self.steps_per_epoch:
            self.step = 0
            self.epoch += 1

        # Still need to return original logs
        return ret

    def call(self, inputs, training=False):
        # Compute embeddings
        pc_embed = self.pc_embedding(inputs[:, :self.sequence_length])
        page_embed = self.page_embedding(inputs[:, self.sequence_length:2 * self.sequence_length])
        offset_embed = self.offset_embedding(inputs[:, 2 * self.sequence_length:])

        # Compute page-aware offset embedding
        tmp_page_embed = tf.reshape(page_embed, shape=(-1, self.sequence_length, 1, self.page_embed_size))
        offset_embed = tf.reshape(offset_embed, shape=(-1, self.sequence_length, self.offset_embed_size // self.page_embed_size, self.page_embed_size))
        offset_embed = tf.reshape(self.mha(tmp_page_embed, offset_embed), shape=(-1, self.sequence_length, self.page_embed_size))

        # Run through LSTM
        lstm_inputs = tf.concat([pc_embed, page_embed, offset_embed], 2)
        # Manual embedding dropout to have reproducible dropout randomness
        lstm_inputs = Stateless.dropout_input(lstm_inputs, self.dropout, seed=(self.epoch, self.step), training=training)
        coarse_out = self.coarse_layers(lstm_inputs, training=training)
        fine_out = self.fine_layers(lstm_inputs, training=training)

        # Pass through linear layers
        coarse_logits = self.page_linear(coarse_out)
        fine_logits = self.offset_linear(fine_out)

        # Full sequence has a time dimension
        if self.sequence_loss:
            return tf.concat([coarse_logits, fine_logits], 2)
        else:
            return tf.concat([coarse_logits, fine_logits], 1)

    def load(self, model_path):
        '''
        Load weights from checkpoint. expect_partial() necessary to silence
        some extraneous TensorFlow output
        '''
        self.load_weights(model_path).expect_partial()

    @staticmethod
    def compile_model(config, num_unique_pcs, num_unique_pages):
        '''
        Create and compile the model
        '''
        model = HierarchicalLSTM(config, num_unique_pcs, num_unique_pages)
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
            optimizer='adam',
            loss=loss,
            metrics=metrics,
        )

        return model
