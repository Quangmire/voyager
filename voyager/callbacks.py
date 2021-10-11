import json
import os

import tensorflow as tf

# Modified from https://github.com/keras-team/keras/issues/2850
class NBatchLogger(tf.keras.callbacks.Callback):
    '''
    Logger prints metrics every N batches
    '''

    BATCH_TYPES = ('train', 'test')

    def __init__(self, display, start_epoch=0, start_step=0, online=False, start_phase=0):
        self.display = display
        self.epoch = start_epoch
        self.phase = start_phase
        self.online = online

        # State for both train and test
        self.step = {batch_type: 0 for batch_type in NBatchLogger.BATCH_TYPES}
        self.step['train'] = start_step
        self.metric_cache = {batch_type: {} for batch_type in NBatchLogger.BATCH_TYPES}

    def set_online(self, online, start_phase):
        self.phase = start_phase
        self.online = online

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
            self.print(batch_type)
            self.step[batch_type] = 0
        self.epoch += 1

    def print(self, batch_type):
        metrics_log = ''

        for (k, v) in self.metric_cache[batch_type].items():
            if abs(v) > 1e-3:
                metrics_log += ' - %s: %.4f' % (k, v)
            else:
                metrics_log += ' - %s: %.4e' % (k, v)

        if self.online:
            print('[{}] Phase {} Epoch {} Step {} {}'.format(batch_type, self.phase, self.epoch, self.step[batch_type],  metrics_log.strip()))
        else:
            print('[{}] Epoch {} Step {} {}'.format(batch_type, self.epoch, self.step[batch_type],  metrics_log.strip()))

    def advance_phase(self):
        self.phase += 1
        self.step = {batch_type: 0 for batch_type in NBatchLogger.BATCH_TYPES}


class ReduceLROnPlateauWithConfig(tf.keras.callbacks.ReduceLROnPlateau):

    def get_config(self):
        '''
        Return the data required to save state
        '''
        return {
            'factor': self.factor,
            'min_lr': self.min_lr,
            'min_delta': self.min_delta,
            'patience': self.patience,
            'verbose': self.verbose,
            'cooldown': self.cooldown,
            'cooldown_counter': self.cooldown_counter,
            'wait': self.wait,
            'best': self.best,
            'mode': self.mode,
        }

    def load_config(self, config):
        '''
        Restore state

        monitor_op could not be saved since it was a lambda
        '''
        for k, v in config.items():
            setattr(self, k, v)
        if (self.mode == 'min' or (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)

class ResumeCheckpoint(tf.keras.callbacks.Callback):
    '''
    Backs up model wrapper
    '''

    def __init__(self, model_wrapper, checkpoint_every, model_path, step):
        self.model_wrapper = model_wrapper
        self.checkpoint_every = checkpoint_every
        self.model_path = model_path
        self.step = step

    # Save every N steps
    def on_train_batch_end(self, batch, logs):
        self.step += 1
        if self.step % self.checkpoint_every == 0:
            self.model_wrapper.create_checkpoint(self.model_path)
