import tensorflow as tf
import tensorflow_addons as tfa

class HierarchicalSequenceLoss(tf.keras.losses.Loss):
    '''
    Seq2Seq sequence loss to train Voyager to make every timestep produce
    consistent output. Namely, every step has to predict the next instead of
    just looking at the final timestep
    '''

    def call(self, y_true, y_pred):
        page_loss = tfa.seq2seq.sequence_loss(
            y_pred[:, :, :-64],
            y_true[:, 0],
            tf.ones((tf.shape(y_pred)[:2])),
            average_across_timesteps=False, average_across_batch=False,
            sum_over_timesteps=True, sum_over_batch=True,
        )

        offset_loss = tfa.seq2seq.sequence_loss(
            y_pred[:, :, -64:],
            y_true[:, 1],
            tf.ones((tf.shape(y_pred)[:2])),
            average_across_timesteps=False, average_across_batch=False,
            sum_over_timesteps=True, sum_over_batch=True,
        )
        return page_loss + offset_loss


class HierarchicalCrossEntropyWithLogitsLoss(tf.keras.losses.Loss):
    '''
    Hierarchical CrossEntropy loss that Voyager was trained with in original paper
    '''

    def __init__(self):
        super(HierarchicalCrossEntropyWithLogitsLoss, self).__init__()
        self.cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def call(self, y_true, y_pred):
        page_loss = self.cross_entropy(y_true[:, 0], y_pred[:, :-64])
        offset_loss = self.cross_entropy(y_true[:, 1], y_pred[:, -64:])
        return page_loss + offset_loss
