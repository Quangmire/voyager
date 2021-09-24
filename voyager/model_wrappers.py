import time
import tensorflow as tf

from voyager.callbacks import NBatchLogger, ReduceLROnPlateauWithConfig, ResumeCheckpoint
from voyager.data_loader import read_benchmark_trace
from voyager.losses import HierarchicalSequenceLoss, HierarchicalCrossEntropyWithLogitsLoss
from voyager.models import HierarchicalLSTM
from voyager.utils import load_config, create_prefetch_file

class ModelWrapper:

    VERBOSITY_QUIET   = 0
    VERBOSITY_PROGBAR = 1
    VERBOSITY_EPOCH   = 2

    def __init__(self, config, benchmark, verbosity=1, callbacks=None):
        self.config = config
        self.benchmark = benchmark
        self.verbosity = verbosity
        self._callbacks = []
        self.callbacks = None
        self.args = None

        # Create a model
        self.model = HierarchicalLSTM.compile_model(config, benchmark.num_pcs(), benchmark.num_pages())
        self._compile_metrics()

    def _compile_metrics(self):
        # The following is necessary to "compile" the metrics.
        if self.config.sequence_loss:
            y_true = tf.zeros((2, 1, 16, 1), dtype=tf.int32)
            y_pred = tf.zeros((1, 16, self.benchmark.num_pages() + 64))
        else:
            y_true = tf.zeros((2, 1, 1, 1), dtype=tf.int32)
            y_pred = tf.zeros((1, self.benchmark.num_pages() + 64))
        self.model.compiled_metrics.update_state(y_true, y_pred)

        # This is necessary if we also wanted to compile the loss
        #self.model.compiled_loss(y_true, y_pred)

    def _init_callbacks(self, callbacks):
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
            self._callbacks.append(
                NBatchLogger(args.print_every, start_epoch=args.start_epoch, start_step=args.start_step)
            )

        # Set-up model checkpoint callback.
        if args.model_path:
            self._callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=args.model_path,
                    save_weights_only=True,
                    monitor='val_acc',
                    mode='max',
                    save_best_only=True,
                    verbose=1,
            ))
        else:
            print('Notice: Not checkpointing the model. To do so, please provide a path to --model-path.')

        # Set-up Tensorboard callback.
        if args.tb_dir:
            self._callbacks.append(
                tf.keras.callbacks.TensorBoard(
                    log_dir=args.tb_dir,
                    histogram_freq=1
            ))
        else:
            print('Notice: Not logging to Tensorboard. To do so, please provide a directory to --tb-dir.')

        # Set-up ResumeCheckpoint callback.
        # Dictionary of things to backup
        backup = {}

        # Set-up learning rate decay callback.
        if self.config.learning_rate_decay > 1:
            lr_decay_callback = ReduceLROnPlateauWithConfig(
                monitor='val_acc',
                factor=1 / self.config.learning_rate_decay,
                patience=5,
                mode='max',
                verbose=1,
                min_lr=self.config.min_learning_rate,
                min_delta=0.005,
            )
            self._callbacks.append(lr_decay_callback)
            backup['lr_decay'] = lr_decay_callback
        else:
            print('Notice: Not decaying learning rate. To do so, please provide learning_rate_decay > 1.0 in the config file.')

        # Create and add in ResumeCheckpoint
        if args.checkpoint_every is not None:
            self._callbacks.append(
                # TODO: Save epoch and step into the checkpoint
                ResumeCheckpoint(
                    self.model,
                    args.model_path,
                    backup,
                    args.checkpoint_every,
                    epoch=args.start_epoch,
                    step=args.start_step,
                    resume=(args.start_step != 0 or args.start_epoch != 1)
                )
            )

    def load(self, model_path):
        self.model.load(model_path)

    @tf.function
    def train_step(self, x, y):
        # Single train step
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss_value = self.model.loss(y, logits)

        # Update gradient
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        # Update logs of loss and metrics
        logs = {'loss': loss_value}
        for metric in self.model.metrics:
            metric.update_state(y, logits)
            logs[metric.name] = metric.result()
        return logs

    def train(self, train_ds=None, valid_ds=None, start_epoch=0, start_step=0, callbacks=None):
        # Create default datasets if there are None
        if train_ds is None:
            train_ds, valid_ds, test_ds = self.benchmark.split(self.config, self.args.start_epoch, self.args.start_step)

        # Create callbacks list anew
        self._init_callbacks(callbacks if callbacks is not None else [])
        epoch = start_epoch
        step = start_step

        # Reset metrics and start callbacks
        self.reset_metrics()
        self.callbacks.on_train_begin()
        self.callbacks.on_epoch_begin(epoch)

        # Main training loop
        for _, x, y_page, y_out in train_ds:
            step += 1

            # Advance an epoch
            if step >= self.config.steps_per_epoch:
                # Evaluate on validation dataset if it was passed in
                if valid_ds is not None:
                    val_logs = self.evaluate([valid_ds])
                    logs.update(val_logs)
                self.callbacks.on_epoch_end(epoch, logs)
                step = 0
                epoch += 1
                self.callbacks.on_epoch_begin(epoch)
                self.reset_metrics()

            # Do one train step
            self.callbacks.on_train_batch_begin(step)
            logs = self.train_step(x, (y_page, y_out))
            self.callbacks.on_train_batch_end(step, logs)

        self.callbacks.on_epoch_end(epoch)
        self.callbacks.on_train_end(logs)

    @tf.function
    def evaluate_step(self, x, y):
        # One evaluate step
        logits = self.model(x, training=False)
        loss_value = self.model.loss(y, logits)

        # Save logs of loss and metrics
        logs = {'loss': loss_value}
        for metric in self.model.metrics:
            metric.update_state(y, logits)
            logs['val_' + metric.name] = metric.result()
        return logs

    def evaluate(self, datasets=None, callbacks=None):
        # Setup default datasets if there are None
        if datasets is None:
            train_ds, valid_ds, test_ds = self.benchmark.split(self.config, self.args.start_epoch, self.args.start_step)
            datasets = [test_ds]

        # Setup callbacks if there are None
        if not self.callbacks:
            self._init_callbacks(callbacks if callbacks is not None else [])

        # Validation over dataset
        self.reset_metrics()
        self.callbacks.on_test_begin()

        # Validation loop
        for ds in datasets:
            for step, (_, x, y_page, y_out) in enumerate(ds):
                self.callbacks.on_test_batch_begin(step)
                logs = self.evaluate_step(x, (y_page, y_out))
                self.callbacks.on_test_batch_end(step, logs)
        self.callbacks.on_test_end(logs)

        return logs

    @tf.function
    def generate_step(self, x, y):
        # Generate step
        logits = self.model(x, training=False)
        loss_value = self.model.loss(y, logits)

        # Save logs, but they may not end up getting used here
        logs = {'loss': loss_value}
        for metric in self.model.metrics:
            metric.update_state(y, logits)
            logs['val_' + metric.name] = metric.result()
        return logits, logs

    def generate(self, datasets=None, prefetch_file=None, callbacks=None):
        # Create default datasets if there are none
        if datasets is None:
            train_ds, valid_ds, test_ds = self.benchmark.split(self.config, self.args.start_epoch, self.args.start_step)
            datasets = [test_ds]

        # Setup callbacks if there are None
        if not self.callbacks:
            self._init_callbacks(callbacks if callbacks is not None else [])

        addresses = []
        inst_ids = []
        self.reset_metrics()
        self.callbacks.on_test_begin()
        for ds in datasets:
            for step, (batch_inst_ids, x, y_page, y_out) in enumerate(ds):
                self.callbacks.on_test_batch_begin(step)
                logits, logs = self.generate_step(x, (y_page, y_out))

                # Grab the final (only) timestep logits
                if self.config.sequence_loss:
                    page_logits = logits[:, -1, :-64]
                    offset_logits = logits[:, -1, -64:]
                else:
                    page_logits = logits[:, :-64]
                    offset_logits = logits[:, -64:]

                # Argmax for prediction
                # TODO: Possibly threshold here
                pred_pages = tf.argmax(page_logits, -1).numpy().tolist()
                pred_offsets = tf.argmax(offset_logits, -1).numpy().tolist()

                # Unmap addresses
                for xi, inst_id, pred_page, pred_offset in zip(x.numpy().tolist(), batch_inst_ids.numpy().reshape(-1).tolist(), pred_pages, pred_offsets):
                    # OOV
                    if pred_page == 0:
                        continue
                    addresses.append(self.benchmark.unmap(xi, pred_page, pred_offset, self.config.sequence_length))
                    inst_ids.append(inst_id)

                self.callbacks.on_test_batch_end(step, logs)

        self.callbacks.on_test_end(logs)

        # Create a prefetch file if path is given
        if prefetch_file is not None:
            create_prefetch_file(prefetch_file, inst_ids, addresses)

        # Return if no prefetch file
        return inst_ids, addresses, logs

    def reset_metrics(self):
        # Reset all the metrics with one convenient call
        for metric in self.model.metrics:
            metric.reset_state()

    def get_datasets(self, train, valid, test):
        datasets = []
        train_ds, valid_ds, test_ds = self.benchmark.split(self.config, self.args.start_epoch, self.args.start_step)
        if train:
            datasets.append(train_ds)
        if valid:
            datasets.append(valid_ds)
        if test:
            datasets.append(test_ds)
        return datasets

    @staticmethod
    def setup_from_args(args):
        print(args)

        # Parse config file
        config = load_config(args.config, args.debug)
        print(config)

        # Load and process benchmark
        benchmark = read_benchmark_trace(args.benchmark, config.multi_label)

        # Create and compile the model
        model_wrapper = ModelWrapper(config, benchmark, verbosity=1 if args.print_every is None else 2)
        model_wrapper.args = args
        model_wrapper.setup_callbacks(args)

        return model_wrapper
