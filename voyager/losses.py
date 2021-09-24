import tensorflow as tf
import tensorflow_addons as tfa

from voyager.math import multi_one_hot

class HierarchicalSequenceLoss(tf.keras.losses.Loss):
    '''
    Seq2Seq sequence loss to train Voyager to make every timestep produce
    consistent output. Namely, every step has to predict the next instead of
    just looking at the final timestep
    '''
    def __init__(self, multi_label):
        super(HierarchicalSequenceLoss, self).__init__()
        self.multi_label = multi_label

    def call(self, y_true, y_pred):
        # y_true is passed in as a tuple
        y_page_labels = y_true[0]
        y_offset_labels = y_true[1]

        # Can't use sequence loss on seq2seq unfortunately, but the following is equivalent
        if self.multi_label:
            # Create one_hot representations for labels for cross_entropy
            y_page = multi_one_hot(y_page_labels, tf.shape(y_pred)[-1] - 64)
            y_offset = multi_one_hot(y_offset_labels, 64)

            page_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_page, y_pred[:, :, :-64]))
            offset_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_offset, y_pred[:, :, -64:]))
        else:
            page_loss = tfa.seq2seq.sequence_loss(
                y_pred[:, :, :-64],
                y_page_labels,
                tf.ones((tf.shape(y_pred)[:2])),
                average_across_timesteps=True, average_across_batch=True,
                sum_over_timesteps=False, sum_over_batch=False,
            )

            offset_loss = tfa.seq2seq.sequence_loss(
                y_pred[:, :, -64:],
                y_offset_labels,
                tf.ones((tf.shape(y_pred)[:2])),
                average_across_timesteps=True, average_across_batch=True,
                sum_over_timesteps=False, sum_over_batch=False,
            )

        return page_loss + offset_loss

    def get_config(self):
        return {'multi_label': self.multi_label}

    @classmethod
    def from_config(cls, config):
        return HierarchicalSequenceLoss(config['multi_label'])

class HierarchicalCrossEntropyWithLogitsLoss(tf.keras.losses.Loss):
    '''
    Hierarchical CrossEntropy loss that Voyager was trained with in original paper
    '''
    def __init__(self, multi_label):
        super(HierarchicalCrossEntropyWithLogitsLoss, self).__init__()
        self.multi_label = multi_label
        if not self.multi_label:
            self.cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def call(self, y_true, y_pred):
        if self.multi_label:
            # Extra time access with only one timestep
            y_page_labels = tf.squeeze(y_true[0], axis=1)
            y_offset_labels = tf.squeeze(y_true[1], axis=1)

            # Create one-hot representation for cross_entropy
            y_page = multi_one_hot(y_page_labels, tf.shape(y_pred)[-1] - 64)
            y_offset = multi_one_hot(y_offset_labels, 64)

            page_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_page, y_pred[:, :-64]))
            offset_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_offset, y_pred[:, -64:]))
        else:
            page_loss = self.cross_entropy(y_true[0], y_pred[:, :-64])
            offset_loss = self.cross_entropy(y_true[1], y_pred[:, -64:])

        return page_loss + offset_loss

    def get_config(self):
        return {'multi_label': self.multi_label}

    @classmethod
    def from_config(cls, config):
        return HierarchicalCrossEntropyWithLogitsLoss(config['multi_label'])
