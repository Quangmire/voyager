import tensorflow as tf

# From https://github.com/keras-team/keras/issues/2850
class NBatchLogger(tf.keras.callbacks.Callback):
    '''
    Logger prints metrics every N batches
    '''

    def __init__(self, display):
        self.step = 0
        self.display = display
        self.metric_cache = {}

    def on_batch_end(self, batch, logs):
        self.step += 1

        # Save aggregate metrics by addition
        for k in logs:
            self.metric_cache[k] = self.metric_cache.get(k, 0) + logs[k]

        # Display mean aggregate metric values
        if self.step % self.display == 0:
            metrics_log = ''

            for (k, v) in self.metric_cache.items():
                val = v / self.display
                if abs(val) > 1e-3:
                    metrics_log += ' - %s: %.4f' % (k, val)
                else:
                    metrics_log += ' - %s: %.4e' % (k, val)

            print('step: {}/{} ... {}'.format(self.step, self.params['steps'], metrics_log))
            self.metric_cache.clear()
