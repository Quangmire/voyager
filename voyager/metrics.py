import tensorflow as tf

from voyager.math import multi_one_hot, reduce_sum_det

# Custom accuracy metric base class for other accuracy metrics in this file that
# implements some basic functionality that is extensively reused
class CustomAccuracy(tf.keras.metrics.Metric):

    def __init__(self, sequence_loss=False, multi_label=False, num_offsets=64, **kwargs):
        super(CustomAccuracy, self).__init__(**kwargs)
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')
        self.sequence_loss = sequence_loss
        self.multi_label = multi_label
        self.num_offsets = num_offsets

    def result(self):
        return self.correct / self.total * 100

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)

    def get_config(self):
        '''
        Return information needed to save state
        '''
        return {'total': self.total.value().numpy(),
                'correct': self.correct.value().numpy(),
                'sequence_loss': self.sequence_loss,
                'multi_label': self.multi_label,
        }

    def load_config(self, config):
        '''
        Restore state
        '''
        self.correct.assign(config['correct'])
        self.total.assign(config['total'])
        self.sequence_loss = config['sequence_loss']
        self.multi_label = config['multi_label']

    @classmethod
    def from_config(cls, config):
        '''
        Restore state
        '''
        self = cls()
        self.correct.assign(config['correct'])
        self.total.assign(config['total'])
        self.sequence_loss = config['sequence_loss']
        self.multi_label = config['multi_label']
        return self

class OverallPredictionHierarchicalAccuracy(CustomAccuracy):
    '''
    Accuracy of the full predicted address
    '''

    def __init__(self, **kwargs):
        super(OverallPredictionHierarchicalAccuracy, self).__init__(name='pred_acc', **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Compute page / offset correctness of final (only for non-sequential) timestep
        y_page_labels = y_true[0][:, -1]
        y_offset_labels = y_true[1][:, -1]

        # Grab final timestep from prediction
        if self.sequence_loss:
            y_pred = y_pred[:, -1]

        # LHS (batch size x # labels per batch) tensor of labels
        # RHS is (batch size x 1) tensor of predicted values
        # Result is (batch size x # labels per batch) boolean array where
        #   Result[i, j] = LHS[i, j] == RHS[i, 0]
        page_correct = y_page_labels == tf.expand_dims(tf.argmax(y_pred[:, :-self.num_offsets], axis=-1, output_type=tf.int32), axis=-1)
        offset_correct = y_offset_labels == tf.expand_dims(tf.argmax(y_pred[:, -self.num_offsets:], axis=-1, output_type=tf.int32), axis=-1)

        # Only correct overall if both page / offset are correct
        values = tf.logical_and(page_correct, offset_correct)
        # We reduce along rows since we only need at least 1 label to be correct
        values = tf.cast(tf.reduce_any(values, axis=-1), self.dtype)

        # Multiply sample weights if provided
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)

        # Update internal stats
        self.correct.assign_add(reduce_sum_det(values))
        # Increase by batch size
        self.total.assign_add(tf.cast(tf.shape(y_pred)[0], tf.float32))

class PagePredictionHierarchicalAccuracy(CustomAccuracy):
    '''
    Accuracy of the predicted pages
    '''

    def __init__(self, **kwargs):
        super(PagePredictionHierarchicalAccuracy, self).__init__(name='page_pred_acc', **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Compute page correctness of final timestep
        y_page_labels = y_true[0][:, -1]

        # Grab final timestep from prediction
        if self.sequence_loss:
            y_pred = y_pred[:, -1]

        # LHS (batch size x # labels per batch) tensor of labels
        # RHS is (batch size x 1) tensor of predicted values
        # Result is (batch size x # labels per batch) boolean array where
        #   Result[i, j] = LHS[i, j] == RHS[i, 0]
        page_correct = y_page_labels == tf.expand_dims(tf.argmax(y_pred[:, :-self.num_offsets], axis=-1, output_type=tf.int32), axis=-1)

        values = page_correct
        # We reduce along rows since we only need at least 1 label to be correct
        values = tf.cast(tf.reduce_any(values, axis=-1), self.dtype)

        # Multiply sample weights if provided
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)

        # Update internal stats
        self.correct.assign_add(reduce_sum_det(values))
        # Increase by batch size
        self.total.assign_add(tf.cast(tf.shape(y_pred)[0], tf.float32))

class OffsetPredictionHierarchicalAccuracy(CustomAccuracy):
    '''
    Accuracy of the predicted offsets
    '''

    def __init__(self, **kwargs):
        super(OffsetPredictionHierarchicalAccuracy, self).__init__(name='offset_pred_acc', **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Compute offset correctness of final timestep
        y_offset_labels = y_true[1][:, -1]

        # Grab final timestep from prediction
        if self.sequence_loss:
            y_pred = y_pred[:, -1]

        # LHS (batch size x # labels per batch) tensor of labels
        # RHS is (batch size x 1) tensor of predicted values
        # Result is (batch size x # labels per batch) boolean array where
        #   Result[i, j] = LHS[i, j] == RHS[i, 0]
        offset_correct = y_offset_labels == tf.expand_dims(tf.argmax(y_pred[:, -self.num_offsets:], axis=-1, output_type=tf.int32), axis=-1)

        values = offset_correct
        # We reduce along rows since we only need at least 1 label to be correct
        values = tf.cast(tf.reduce_any(values, axis=-1), self.dtype)

        # Multiply sample weights if provided
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)

        # Update internal stats
        self.correct.assign_add(reduce_sum_det(values))
        # Increase by batch size
        self.total.assign_add(tf.cast(tf.shape(y_pred)[0], tf.float32))

class OverallHierarchicalAccuracy(CustomAccuracy):
    '''
    Accuracy of the full predicted address
    '''

    def __init__(self, **kwargs):
        super(OverallHierarchicalAccuracy, self).__init__(name='acc', **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Compute page / offset correctness of final timestep
        y_page_labels = y_true[0][:, -1]
        y_offset_labels = y_true[1][:, -1]

        # Grab final timestep from prediction
        if self.sequence_loss:
            y_pred = y_pred[:, -1]

        if self.multi_label:
            # Convert to multi one-hot tensor
            y_page = multi_one_hot(y_page_labels, tf.shape(y_pred)[-1] - self.num_offsets)
            y_offset = multi_one_hot(y_offset_labels, self.num_offsets)

            # LHS[i, j] is True if y_page[i, j] is 1
            # RHS[i, j] is True if y_page[i, j] is non-negative. This was chosen due to
            #           the fact that it is trained with sigmoid final layer
            # We'll denote the logical and as AND
            # The gather creates an array the size of the labels where
            #     Result[i, j] = AND[Label[i,j]]
            page_correct = tf.gather((y_page > 0.5) & (y_pred[:, :-self.num_offsets] >= 0), y_page_labels, batch_dims=1)
            offset_correct = tf.gather((y_offset > 0.5) & (y_pred[:, -self.num_offsets:] >= 0), y_offset_labels, batch_dims=1)
        else:
            # Compare labels against argmax
            page_correct = y_page_labels == tf.argmax(y_pred[:, :-self.num_offsets], -1, output_type=tf.int32)
            offset_correct = y_offset_labels == tf.argmax(y_pred[:, -self.num_offsets:], -1, output_type=tf.int32)

        # Only correct overall if both page / offset are correct
        values = tf.logical_and(page_correct, offset_correct)
        values = tf.cast(values, self.dtype)

        # Multiply sample weights if provided
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)

        # Update internal stats
        self.correct.assign_add(reduce_sum_det(values))
        if self.multi_label:
            # Only looking at accuracy of valid addresses. -1 was a filler
            self.total.assign_add(reduce_sum_det(tf.cast(y_page_labels != -1, tf.float32)))
        else:
            # Increase by batch size
            self.total.assign_add(tf.cast(tf.shape(y_pred)[0], self.dtype))

class PageHierarchicalAccuracy(CustomAccuracy):
    '''
    Accuracy of the predicted pages
    '''

    def __init__(self, **kwargs):
        super(PageHierarchicalAccuracy, self).__init__(name='page_acc', **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Compute page correctness of final timestep
        y_page_labels = y_true[0][:, -1]

        # Grab final timestep from prediction
        if self.sequence_loss:
            y_pred = y_pred[:, -1]

        if self.multi_label:
            # Convert to multi one-hot tensor
            y_page = multi_one_hot(y_page_labels, tf.shape(y_pred)[-1] - self.num_offsets)

            # LHS[i, j] is True if y_page[i, j] is 1
            # RHS[i, j] is True if y_page[i, j] is non-negative. This was chosen due to
            #           the fact that it is trained with sigmoid final layer
            # We'll denote the logical and as AND
            # The gather creates an array the size of the labels where
            #     Result[i, j] = AND[Label[i,j]]
            page_correct = (y_page > 0.5) & (y_pred[:, :-self.num_offsets] >= 0)
        else:
            # Compare labels against argmax
            page_correct = y_page_labels == tf.argmax(y_pred[:, :-self.num_offsets], -1, output_type=tf.int32)

        values = page_correct
        values = tf.cast(values, self.dtype)

        # Multiply sample weights if provided
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)

        # Update internal stats
        self.correct.assign_add(reduce_sum_det(values))
        if self.multi_label:
            # Only looking at accuracy of valid addresses. -1 was a filler
            self.total.assign_add(reduce_sum_det(tf.cast(y_page_labels != -1, tf.float32)))
        else:
            self.total.assign_add(tf.cast(tf.shape(y_pred)[0], self.dtype))

class OffsetHierarchicalAccuracy(CustomAccuracy):
    '''
    Accuracy of the predicted offsets
    '''

    def __init__(self, **kwargs):
        super(OffsetHierarchicalAccuracy, self).__init__(name='offset_acc', **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Compute offset correctness of final timestep
        y_offset_labels = y_true[1][:, -1]

        # Grab final timestep from prediction
        if self.sequence_loss:
            y_pred = y_pred[:, -1]

        if self.multi_label:
            # Convert to multi one-hot tensor
            y_offset = multi_one_hot(y_offset_labels, self.num_offsets)

            # LHS[i, j] is True if y_page[i, j] is 1
            # RHS[i, j] is True if y_page[i, j] is non-negative. This was chosen due to
            #           the fact that it is trained with sigmoid final layer
            # We'll denote the logical and as AND
            # The gather creates an array the size of the labels where
            #     Result[i, j] = AND[Label[i,j]]
            offset_correct = (y_offset > 0.5) & (y_pred[:, -self.num_offsets:] >= 0)
        else:
            # Compare labels against argmax
            offset_correct = y_offset_labels == tf.argmax(y_pred[:, -self.num_offsets:], -1, output_type=tf.int32)

        values = offset_correct
        values = tf.cast(values, self.dtype)

        # Multiply sample weights if provided
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)

        # Update internal stats
        self.correct.assign_add(reduce_sum_det(values))
        if self.multi_label:
            # Only looking at accuracy of valid addresses. -1 was a filler
            self.total.assign_add(reduce_sum_det(tf.cast(y_offset_labels != -1, tf.float32)))
        else:
            # Increase by batch size
            self.total.assign_add(tf.cast(tf.shape(y_pred)[0], self.dtype))
