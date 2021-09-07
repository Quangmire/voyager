import tensorflow as tf

# Modified from https://github.com/keras-team/keras/issues/2850
class NBatchLogger(tf.keras.callbacks.Callback):
    '''
    Logger prints metrics every N batches
    '''

    BATCH_TYPES = ('train', 'test')

    def __init__(self, display):
        self.display = display
        self.epoch = 1

        # State for both train and test
        self.step = {batch_type: 0 for batch_type in NBatchLogger.BATCH_TYPES}
        self.metric_cache = {batch_type: {} for batch_type in NBatchLogger.BATCH_TYPES}

    def on_train_batch_end(self, batch, logs):
        self._on_batch_end(batch, logs, 'train')

    def on_test_batch_end(self, batch, logs):
        self._on_batch_end(batch, logs, 'test')

    def _on_batch_end(self, batch, logs, batch_type):
        self.step[batch_type] += 1

        # Save aggregate metrics
        for k in logs:
            self.metric_cache[batch_type][k] = logs[k]

        # Display mean aggregate metric values
        if self.step[batch_type] % self.display == 0:
            self.print(batch_type)

    def on_epoch_end(self, epoch, logs):
        # Reset counters and update epoch counter
        for batch_type in NBatchLogger.BATCH_TYPES:
            self.step[batch_type] = 0
        self.epoch += 1

    def print(self, batch_type):
        metrics_log = ''

        for (k, v) in self.metric_cache[batch_type].items():
            if abs(v) > 1e-3:
                metrics_log += ' - %s: %.4f' % (k, v)
            else:
                metrics_log += ' - %s: %.4e' % (k, v)

        print('[{}] Epoch {} Step {} {}'.format(batch_type, self.epoch, self.step[batch_type],  metrics_log.strip()))
