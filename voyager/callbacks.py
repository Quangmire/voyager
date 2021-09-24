import json
import os

import numpy as np
import tensorflow as tf
import keras.backend as K

# Modified from https://github.com/keras-team/keras/issues/2850
class NBatchLogger(tf.keras.callbacks.Callback):
    '''
    Logger prints metrics every N batches
    '''

    BATCH_TYPES = ('train', 'test')

    def __init__(self, display, start_epoch=1, start_step=0):
        self.display = display
        self.epoch = start_epoch

        # State for both train and test
        self.step = {batch_type: 0 for batch_type in NBatchLogger.BATCH_TYPES}
        self.step['train'] = start_step
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

        print('[{}] Epoch {} Step {} {}'.format(batch_type, self.epoch, self.step[batch_type],  metrics_log.strip()))


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


# Utility functions for backup and restore
def cast_numerical(v):
    if isinstance(v, (np.int8, np.int16, np.int32, np.int64, np.int)):
        return int(v)
    elif isinstance(v, (np.float16, np.float32, np.float64, np.float)):
        return float(v)
    return v


def config_to_python(config):
    return {k: cast_numerical(v) for k, v in config.items()}


def optim_weights_to_python(weights):
    ret = []
    for weight in weights:
        if not isinstance(weight, np.ndarray):
            ret.append(cast_numerical(weight))
        else:
            ret.append(weight.tolist())
    return ret


def python_to_optim_weights(weights):
    ret = []
    for weight in weights:
        if not isinstance(weight, list):
            if isinstance(weight, int):
                ret.append(np.int64(weight))
            else:
                ret.append(np.float32(weight))
        else:
            ret.append(np.array(weight))
    return ret

class ResumeCheckpoint(tf.keras.callbacks.Callback):
    '''
    Backs up and restores model + metric state + optimizer state
    Inspired by:
    https://gitlab.idiap.ch/bob/bob.learn.tensorflow/-/blob/7a498fd907de139fb89bdfecb092a8070a546f79/bob/learn/tensorflow/callbacks.py#L10
    '''

    def __init__(self, model, model_path, backups, checkpoint_every, epoch=1, step=0, resume=False):
        self.model = model
        self.model_path = os.path.join(model_path + '_resume', 'model')
        self.backup_path = os.path.join(model_path + '_resume', 'backup.json')

        self.checkpoint_every = checkpoint_every
        self.epoch = epoch
        self.step = step
        self.resume = resume
        self.backups = backups
        self.backups['optim'] = self.model.optimizer
        for metric in self.model.metrics:
            self.backups[metric.name] = metric

    def backup(self):
        print('\nCreating checkpoint...', end='')

        # Backup model
        self.model.save_weights(self.model_path)

        # Backup callbacks, metrics, optimizer
        backup_data = {}
        for name, item in self.backups.items():
            backup_data[name] = config_to_python(item.get_config())

        # Optimizer weights aren't saved in get_config() unfortunately
        symbolic_weights = getattr(self.model.optimizer, 'weights')
        if symbolic_weights:
            weight_values = K.batch_get_value(symbolic_weights)
            backup_data['optim']['weights'] = optim_weights_to_python(weight_values)

        # Dump to json
        with open(self.backup_path, 'w') as f:
            json.dump(backup_data, f, indent=4)

        print('Done')

    def restore(self):
        print('\nRestoring checkpoint...', end='')

        # Reload model state
        self.model.load(self.model_path)
        self.model.step = self.step
        self.model.epoch = self.epoch

        # Restore callbacks, metrics, optimizer state
        if os.path.exists(self.backup_path):
            with open(self.backup_path) as f:
                backup_data = json.load(f)
            for name, config in backup_data.items():
                if name == 'optim':
                    # Pop out weights and manually set
                    weights = config.pop('weights')
                    for k, v in config.items():
                        setattr(self.model.optimizer, k, v)

                    # Need to do a single forward pass so that model is initialized
                    self.model(tf.zeros((1, self.model.sequence_length * 3,)))

                    # Need to do a single backwards pass so that optimizer is initialized
                    zero_grads = [tf.zeros_like(w) for w in self.model.trainable_weights]
                    self.model.optimizer.apply_gradients(zip(zero_grads, self.model.trainable_weights))

                    # Now we can finally sest the optimizer weights
                    self.model.optimizer.set_weights(python_to_optim_weights(weights))

                elif name in self.backups:
                    self.backups[name].load_config(config)
        print('Done')

    # Save every N steps
    def on_train_batch_end(self, batch, logs):
        self.step += 1
        if self.step % self.checkpoint_every == 0:
            self.backup()

    # Resume on epoch begin and not train begin due to state resetting in model.fit(..)
    def on_epoch_begin(self, epoch, logs):
        self.step = 0
        if self.resume:
            self.restore()
            # Resume only once
            self.resume = False

