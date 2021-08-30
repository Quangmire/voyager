import tensorflow as tf

class OverallHierarchicalAccuracy(tf.keras.metrics.Metric):
    '''
    Accuracy of the full predicted address
    '''

    def __init__(self, sequence_loss=False, name='acc', **kwargs):
        super(OverallHierarchicalAccuracy, self).__init__(name=name, **kwargs)
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')
        self.sequence_loss = sequence_loss

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Compute page / offset correctness of final timestep
        if self.sequence_loss:
            page_correct = y_true[:, 0, -1] == tf.argmax(y_pred[:, -1, :-64], -1)
            offset_correct = y_true[:, 1, -1] == tf.argmax(y_pred[:, -1, -64:], -1)
        else:
            page_correct = y_true[:, 0] == tf.argmax(y_pred[:, :-64], -1)
            offset_correct = y_true[:, 1] == tf.argmax(y_pred[:, -64:], -1)

        # Only correct overall if both page / offset are correct
        values = tf.logical_and(page_correct, offset_correct)
        values = tf.cast(values, self.dtype)

        # Multiply sample weights if provided
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)

        # Update internal stats
        self.correct.assign_add(tf.reduce_sum(values))
        self.total.assign_add(tf.cast(tf.shape(y_true)[0], self.dtype))

    def result(self):
        return self.correct / self.total * 100

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)

class PageHierarchicalAccuracy(tf.keras.metrics.Metric):
    '''
    Accuracy of the predicted pages
    '''

    def __init__(self, sequence_loss=False, name='page_acc', **kwargs):
        super(PageHierarchicalAccuracy, self).__init__(name=name, **kwargs)
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')
        self.sequence_loss = sequence_loss

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Compute page correctness of final timestep
        if self.sequence_loss:
            page_correct = y_true[:, 0, -1] == tf.argmax(y_pred[:, -1, :-64], -1)
        else:
            page_correct = y_true[:, 0] == tf.argmax(y_pred[:, :-64], -1)

        values = page_correct
        values = tf.cast(values, self.dtype)

        # Multiply sample weights if provided
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)

        # Update internal stats
        self.correct.assign_add(tf.reduce_sum(values))
        self.total.assign_add(tf.cast(tf.shape(y_true)[0], self.dtype))

    def result(self):
        return self.correct / self.total * 100

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)

class OffsetHierarchicalAccuracy(tf.keras.metrics.Metric):
    '''
    Accuracy of the predicted offsets
    '''

    def __init__(self, sequence_loss=False, name='offset_acc', **kwargs):
        super(OffsetHierarchicalAccuracy, self).__init__(name=name, **kwargs)
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')
        self.sequence_loss = sequence_loss

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Compute offset correctness of final timestep
        if self.sequence_loss:
            offset_correct = y_true[:, 1, -1] == tf.argmax(y_pred[:, -1, -64:], -1)
        else:
            offset_correct = y_true[:, 1] == tf.argmax(y_pred[:, -64:], -1)

        values = offset_correct
        values = tf.cast(values, self.dtype)

        # Multiply sample weights if provided
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)

        # Update internal stats
        self.correct.assign_add(tf.reduce_sum(values))
        self.total.assign_add(tf.cast(tf.shape(y_true)[0], self.dtype))

    def result(self):
        return self.correct / self.total * 100

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)

