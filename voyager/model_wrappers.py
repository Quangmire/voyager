import json
import os
import shutil
import time

import keras.backend as K
import numpy as np
import tensorflow as tf
import attrdict

from voyager.callbacks import NBatchLogger, ReduceLROnPlateauWithConfig, ResumeCheckpoint
from voyager.data_loader import read_benchmark_trace
from voyager.losses import HierarchicalSequenceLoss, HierarchicalCrossEntropyWithLogitsLoss
from voyager.models import get_model
from voyager.utils import load_config, create_prefetch_file, timefunction


class ModelWrapper:

    VERBOSITY_QUIET   = 0
    VERBOSITY_PROGBAR = 1
    VERBOSITY_EPOCH   = 2

    def __init__(self, config, benchmark, model_name, verbosity=1):
        self.config = config
        self.benchmark = benchmark
        self.verbosity = verbosity
        self._callbacks = []
        self.callbacks = None
        self.step = 0
        self.epoch = 0
        self.phase = 0
        self.backups = {}
        self.batch_logger = None
        self.lr_decay = None
        self.tensorboard_path = None
        self.num_offsets = (1 << config.offset_bits)
        self.model_path = None
        self.monitor = None
        self.recreate_chkpt = False
        self.chkpt = None

        # Create a model
        print('DEBUG : Creating a model with...')
        print('    pc vocab size   :', benchmark.num_pcs())
        print('    page vocab size :', benchmark.num_pages())
        self.model = get_model(model_name).compile_model(config, benchmark.num_pcs(), benchmark.num_pages())
        self._compile_metrics()
        self.backups['optim'] = self.model.optimizer

    def _compile_metrics(self):
        # The following is necessary to "compile" the metrics.
        if self.config.sequence_loss:
            y_true = tf.zeros((2, 1, 16, 1), dtype=tf.int32)
            y_pred = tf.zeros((1, 16, self.benchmark.num_pages() + self.num_offsets))
        else:
            y_true = tf.zeros((2, 1, 1, 1), dtype=tf.int32)
            y_pred = tf.zeros((1, self.benchmark.num_pages() + self.num_offsets))
        self.model.compiled_metrics.update_state(y_true, y_pred)

        # Set up list of things to backup
        for metric in self.model.metrics:
            self.backups[metric.name] = metric

        # This is necessary if we also wanted to compile the loss
        #self.model.compiled_loss(y_true, y_pred)

    def _init_callbacks(self, callbacks):
        if self.tensorboard_path:
            # Create a new one everytime
            # Need to do this because the write objects get destroyed at the end
            # of every train phase, which we have multiple of for online training
            tensorboard = tf.keras.callbacks.TensorBoard(
                log_dir=self.tensorboard_path,
                histogram_freq=1
            )
            # Do this to not affect the callbacks list which .append(..) would
            callbacks = callbacks + [tensorboard]
        if self.config.learning_rate_decay > 1:
            self.lr_decay = ReduceLROnPlateauWithConfig(
                monitor='val_acc',
                factor=1 / self.config.learning_rate_decay,
                patience=5,
                mode='max',
                verbose=1,
                min_lr=self.config.min_learning_rate,
                min_delta=0.005,
            )
            callbacks = callbacks + [self.lr_decay]
            self.backups['lr_decay'] = self.lr_decay
        if self.model_path is not None:
            if self.recreate_chkpt or self.chkpt is None:
                self.chkpt = tf.keras.callbacks.ModelCheckpoint(
                        filepath=self.model_path,
                        save_weights_only=True,
                        monitor='val_acc' if self.monitor is None else self.monitor,
                        mode='max',
                        save_best_only=True,
                        verbose=1,
                )
                callbacks = callbacks + [self.chkpt]

        # Create callback list
        self.callbacks = tf.keras.callbacks.CallbackList(
            callbacks=self._callbacks + callbacks,
            add_progbar=self.verbosity == ModelWrapper.VERBOSITY_PROGBAR,
            verbose=self.verbosity,
            epochs=self.config.num_epochs,
            steps=self.config.steps_per_epoch,
            model=self.model,
        )

    def setup_callbacks(self, args):
        # Set-up batch logger callback.
        if args.print_every is not None:
            self.batch_logger = NBatchLogger(args.print_every, start_epoch=self.epoch, start_step=self.step)
            self._callbacks.append(self.batch_logger)

        # Set-up model checkpoint callback.
        if args.model_path:
            self.model_path = args.model_path
        else:
            print('Notice: Not checkpointing the model. To do so, please provide a path to --model-path.')

        # Set-up Tensorboard callback.
        if args.tb_dir:
            self.tensorboard_path = args.tb_dir
        else:
            print('Notice: Not logging to Tensorboard. To do so, please provide a directory to --tb-dir.')

        # Set-up ResumeCheckpoint callback.
        # Set-up learning rate decay callback.
        if self.config.learning_rate_decay <= 1:
            print('Notice: Not decaying learning rate. To do so, please provide learning_rate_decay > 1.0 in the config file.')

        # Create and add in ResumeCheckpoint
        if args.checkpoint_every is not None:
            self._callbacks.append(ResumeCheckpoint(
                    self,
                    args.checkpoint_every,
                    args.model_path,
                    self.step,
                )
            )

    def load(self, model_path):
        if not self.model.built:
            self.model(tf.zeros((1, self.model.sequence_length * 3,)))
        self.model.load(model_path)

    @tf.function
    def train_step(self, x, y):
        # Single train step
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss_value = self.model.loss(y, logits)
            # regularization_loss = tf.add_n(self.model.losses)
            # loss_value += 0.001 * regularization_loss

        # Update gradient
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        # Update logs of loss and metrics
        logs = {'loss': loss_value}
        for metric in self.model.metrics:
            metric.update_state(y, logits)
            logs[metric.name] = metric.result()
        return logs

    def train(self, train_ds=None, valid_ds=None, callbacks=None):
        # Create default datasets if there are None
        if train_ds is None:
            train_ds, valid_ds, test_ds = self.benchmark.split(self.epoch, self.step)

        # Create callbacks list anew
        self._init_callbacks(callbacks if callbacks is not None else [])

        # Reset metrics and start callbacks
        self.reset_metrics()
        self.callbacks.on_train_begin()
        self.callbacks.on_epoch_begin(self.epoch)

        epoch_ended = False
        logs = {}

        # Main training loop
        for _, _, x, y_page, y_offset in train_ds:
            epoch_ended = False
            self.step += 1

            # Needed for resume, without this, there is an off-by-one error
            if self.step <= self.config.steps_per_epoch:
                # Do one train step
                self.callbacks.on_train_batch_begin(self.step)
                logs = self.train_step(x, (y_page, y_offset))
                self.callbacks.on_train_batch_end(self.step, logs)

            # Advance an epoch
            if self.step >= self.config.steps_per_epoch:
                self.step = 0
                self.epoch += 1
                # Evaluate on validation dataset if it was passed in
                if valid_ds is not None:
                    val_logs = self.evaluate([valid_ds], training=True)
                    logs.update(val_logs)
                self.callbacks.on_epoch_end(self.epoch - 1, logs)
                epoch_ended = True
                if self.epoch >= self.config.num_epochs:
                    break
                self.callbacks.on_epoch_begin(self.epoch)
                self.reset_metrics()

        # Make sure epochs are ended properly when we run out of data prematurely
        if not epoch_ended:
            self.epoch += 1
            self.callbacks.on_epoch_end(self.epoch)
        self.callbacks.on_train_end(logs)

    def train_online(self, prefetch_file=None, callbacks=None):
        # Create datasets
        train_datasets, eval_datasets = self.benchmark.split(self.epoch % self.config.num_epochs_online, self.step, online=True, start_phase=self.phase)
        # Change # of epochs to # of online epochs
        orig_num_epochs = self.config.num_epochs
        self.config.num_epochs = (self.phase + 1) * self.config.num_epochs_online
        self.monitor = 'acc'
        self.recreate_chkpt = True

        # Enables online training printing
        if self.batch_logger:
            self.batch_logger.set_online(True, self.phase)

        # Online training phase
        for train_ds, eval_ds in zip(train_datasets, eval_datasets):
            # Train while showing eval performance
            self.train(train_ds, eval_ds)
            # Reload the best model
            self.load(self.model_path)
            # Generate on eval dataset
            inst_ids, addresses, _ = self.generate([eval_ds])
            # TODO: Resume functionality doesn't work when the file writing fails
            create_prefetch_file(prefetch_file, inst_ids, addresses, append=True)
            # Reset for reproducible dropout
            self.step = 0
            self.model.step = 0
            self.phase += 1
            # Manually advance state
            if self.batch_logger:
                self.batch_logger.advance_phase()

            # New LR Decay every phase
            if self.lr_decay:
                self.lr_decay._reset()

            # Need to advance number of epochs so that we have num_epochs_online
            # for every phase so that the tensorboard callback works
            self.config.num_epochs += self.config.num_epochs_online

        # Restore original # of epochs
        self.config.num_epochs = orig_num_epochs

    @tf.function
    def evaluate_step(self, x, y):
        # One evaluate step
        logits = self.model(x, training=False)
        loss_value = self.model.loss(y, logits)

        # Save logs of loss and metrics
        logs = {'val_loss': loss_value}
        for metric in self.model.metrics:
            metric.update_state(y, logits)
            logs['val_' + metric.name] = metric.result()
        return logs

    def evaluate(self, datasets=None, callbacks=None, training=False):
        # Setup default datasets if there are None
        if datasets is None:
            train_ds, valid_ds, test_ds = self.benchmark.split(self.epoch, self.step)
            datasets = [test_ds]

        # Create callbacks list anew
        if not training:
            self._init_callbacks(callbacks if callbacks is not None else [])

        # Validation over dataset
        self.reset_metrics()
        self.callbacks.on_test_begin()

        # Validation loop
        for ds in datasets:
            for step, (_, _, x, y_page, y_offset) in enumerate(ds):
                self.callbacks.on_test_batch_begin(step)
                logs = self.evaluate_step(x, (y_page, y_offset))
                self.callbacks.on_test_batch_end(step, logs)
        self.callbacks.on_test_end(logs)

        return logs

    @tf.function
    def generate_step(self, x, y):
        # Generate step
        logits = self.model(x, training=False)
        loss_value = self.model.loss(y, logits)

        # Save logs, but they may not end up getting used here
        logs = {'val_loss': loss_value}
        for metric in self.model.metrics:
            metric.update_state(y, logits)
            logs['val_' + metric.name] = metric.result()
        return logits, logs

    def generate(self, datasets=None, prefetch_file=None, callbacks=None):
        # Create default datasets if there are none
        if datasets is None:
            train_ds, valid_ds, test_ds = self.benchmark.split(self.epoch, self.step)
            datasets = [test_ds]

        # Create callbacks list anew
        self._init_callbacks(callbacks if callbacks is not None else [])

        addresses = []
        inst_ids = []
        self.reset_metrics()
        self.callbacks.on_test_begin()

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        for ds in datasets:
            correct = 0
            total = 0
            for step, (idx, batch_inst_ids, x, y_page, y_offset) in enumerate(ds):
                self.callbacks.on_test_batch_begin(step)
                logits, logs = self.generate_step(x, (y_page, y_offset))

                # Grab the final (only) timestep logits
                if self.config.sequence_loss:
                    page_logits = logits[:, -1, :-self.num_offsets]
                    offset_logits = logits[:, -1, -self.num_offsets:]
                else:
                    page_logits = logits[:, :-self.num_offsets]
                    offset_logits = logits[:, -self.num_offsets:]

                # Argmax for prediction
                pred_pages = tf.argmax(page_logits, -1).numpy().tolist()
                pred_offsets = tf.argmax(offset_logits, -1).numpy().tolist()

                page_logits = page_logits.numpy()
                offset_logits = offset_logits.numpy()
                # Unmap addresses
                for i, (idxi, xi, inst_id, pred_page, pred_offset, yp, yo) in enumerate(zip(idx.numpy().tolist(), x.numpy().tolist(), batch_inst_ids.numpy().tolist(), pred_pages, pred_offsets, y_page.numpy().tolist(), y_offset.numpy().tolist())):
                    '''
                    # TODO: Possibly threshold here
                    if sigmoid(page_logits[i, pred_page]) < 0.99999 or sigmoid(offset_logits[i, pred_offset]) < 0.99999:
                        continue
                    '''
                    total += 1
                    # OOV
                    if pred_page == 0:
                        continue
                    if pred_page == yp[-1] and pred_offset == yo[-1]:
                        correct += 1
                    addresses.append(self.benchmark.unmap(idxi, xi, pred_page, pred_offset, self.config.sequence_length))
                    inst_ids.append(inst_id)

                self.callbacks.on_test_batch_end(step, logs)
            print(correct / total * 100, total)

        self.callbacks.on_test_end(logs)

        # Create a prefetch file if path is given
        if prefetch_file is not None:
            create_prefetch_file(prefetch_file, inst_ids, addresses)
        else:
            # Return if no prefetch file
            return inst_ids, addresses, logs

    def reset_metrics(self):
        # Reset all the metrics with one convenient call
        for metric in self.model.metrics:
            metric.reset_state()

    def get_datasets(self, train, valid, test):
        datasets = []
        train_ds, valid_ds, test_ds = self.benchmark.split(self.epoch, self.step)
        if train:
            datasets.append(train_ds)
        if valid:
            datasets.append(valid_ds)
        if test:
            datasets.append(test_ds)
        return datasets


    # Need the newline if run using the progress bar instead of NBatchLogger
    @timefunction('\nCreating checkpoint')
    def create_checkpoint(self, model_path):
        # Paths to main resume and backup resume
        checkpoint_path = os.path.join(model_path, 'resume')
        backup_path = os.path.join(model_path, 'resume_backup')

        # If checkpoint already exists, copy to backup path
        if os.path.exists(os.path.join(checkpoint_path, 'done')):
            shutil.copytree(checkpoint_path, backup_path)
            # Remove the done file from the current checkpoint path
            os.remove(os.path.join(checkpoint_path, 'done'))

        # Backup model
        self.model.save_weights(os.path.join(checkpoint_path, 'model'))

        # Backup callbacks, metrics, optimizer
        backup_data = {
            'epoch': self.epoch,
            'step': self.step,
            'phase': self.phase,
        }

        for name, item in self.backups.items():
            backup_data[name] = config_to_python(item.get_config())

        # Optimizer weights aren't saved in get_config() unfortunately
        np.save(os.path.join(checkpoint_path, 'optim_weights.npy'), self.model.optimizer.get_weights(), allow_pickle=True)

        # Dump to json
        with open(os.path.join(checkpoint_path, 'data.json'), 'w') as f:
            json.dump(backup_data, f, indent=4)

        # Create empty done file to signify that we're done
        with open(os.path.join(checkpoint_path, 'done'), 'w') as _:
            pass

        # Safe to remove backup now
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)

    @timefunction('Restoring checkpoint')
    def restore_checkpoint(self, model_path):
        # Paths to main resume and backup resume
        checkpoint_path = os.path.join(model_path, 'resume')
        backup_path = os.path.join(model_path, 'resume_backup')

        # Check main path and then backup
        if os.path.exists(os.path.join(checkpoint_path, 'done')):
            load_path = checkpoint_path
        elif os.path.exists(os.path.join(backup_path, 'done')):
            load_path = backup_path
        else:
            print('No valid checkpoints', end='')
            return

        # Restore callbacks, metrics, optimizer state
        with open(os.path.join(load_path, 'data.json')) as f:
            backup_data = json.load(f)

        for name, config in backup_data.items():
            if name == 'epoch':
                self.epoch = config
                self.model.epoch = config
            elif name == 'step':
                self.step = config
                self.model.step = config
            elif name == 'phase':
                self.phase = config
            elif name == 'optim':
                weights = np.load(os.path.join(load_path, 'optim_weights.npy'), allow_pickle=True)

                for k, v in config.items():
                    setattr(self.model.optimizer, k, v)

                # Need to do a single forward pass so that model is initialized
                self.model(tf.zeros((1, self.model.sequence_length * 3,)))

                # Need to do a single backwards pass so that optimizer is initialized
                zero_grads = [tf.zeros_like(w) for w in self.model.trainable_weights]
                self.model.optimizer.apply_gradients(zip(zero_grads, self.model.trainable_weights))

                # Now we can finally set the optimizer weights
                self.model.optimizer.set_weights(python_to_optim_weights(weights))

            elif name in self.backups:
                self.backups[name].load_config(config)

        # Reload model state
        self.model.load(os.path.join(load_path, 'model'))

    @staticmethod
    def setup_from_args(args):
        print(args)

        # Parse config file
        config = load_config(args.config, args.debug)
        print(config)

        # Load and process benchmark
        benchmark = read_benchmark_trace(args.benchmark, config)

        # Create and compile the model
        model_wrapper = ModelWrapper(config, benchmark, args.model_name, verbosity=1 if args.print_every is None else 2)

        if args.auto_resume:
            model_wrapper.restore_checkpoint(args.model_path)

        model_wrapper.setup_callbacks(args)

        return model_wrapper
    
    @staticmethod
    def setup_from_ray_config(config):
        # Model config is already loaded.
        # Config is generated using raytune.utils.load_tuning_config
        args = config.args
        
        # Load and process benchmark
        benchmark = read_benchmark_trace(args.benchmark, config)
        
        # Create and compile the model
        model_wrapper = ModelWrapper(config, benchmark, args.model_name, verbosity=1 if args.print_every is None else 2)
        
        if args.auto_resume:
            model_wrapper.restore_checkpoint(args.model_path)
            
        model_wrapper.setup_callbacks(args)
        
        return model_wrapper
        


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
