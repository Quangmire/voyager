import tensorflow as tf

# Modified from https://github.com/keras-team/keras/issues/2850
class NBatchLogger(tf.keras.callbacks.Callback):
    '''
    Logger prints metrics every N batches
    '''

    def __init__(self, display):
        self.step = 0
        self.display = display
        self.metric_cache = {}
        self.on_test_batch_end = self.on_batch_end

    def on_batch_end(self, batch, logs):
        self.step += 1

        # Save aggregate metrics
        for k in logs:
            self.metric_cache[k] = logs[k]

        # Display mean aggregate metric values
        if self.step % self.display == 0:
            self.print()

    def print(self):
        metrics_log = ''

        for (k, v) in self.metric_cache.items():
            if abs(v) > 1e-3:
                metrics_log += ' - %s: %.4f' % (k, v)
            else:
                metrics_log += ' - %s: %.4e' % (k, v)

        print('step: {}/{} ... {}'.format(self.step, self.params['steps'], metrics_log))
