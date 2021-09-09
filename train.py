import os

# Reduce extraneous TensorFlow output. Needs to occur before tensorflow import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

from voyager.callbacks import NBatchLogger, ReduceLROnPlateauWithConfig, ResumeCheckpoint
from voyager.data_loader import read_benchmark_trace
from voyager.models import HierarchicalLSTM
from voyager.utils import get_parser, load_config, pick_gpu_lowest_memory


# For reproducibility
tf.random.set_seed(0)
np.random.seed(0)


# Select the lowest utilization GPU if not preset
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    gpu = pick_gpu_lowest_memory()
    print(os.uname(), gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Parse config file
    config = load_config(args.config, args.debug)
    print(config)

    # Load and process benchmark
    benchmark = read_benchmark_trace(args.benchmark)
    train_ds, valid_ds, test_ds = benchmark.split(config, args.start_epoch, args.start_step)

    # Create and compile the model
    model, metrics = HierarchicalLSTM.compile_model(config, benchmark.num_pcs(), benchmark.num_pages())

    # Set-up callbacks for training
    callbacks = []

    # Set-up batch logger callback.
    if args.print_every is not None:
        callbacks.append(NBatchLogger(args.print_every, start_epoch=args.start_epoch, start_step=args.start_step))

    # Set-up model checkpoint callback.
    if args.model_path:
        callbacks.append(
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
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=args.tb_dir,
                histogram_freq=1
        ))
    else:
        print('Notice: Not logging to Tensorboard. To do so, please provide a directory to --tb-dir.')

    # Set-up ResumeCheckpoint callback.
    # Dictionary of things to backup
    backup = {
        'optim': model.optimizer,
    }

    for metric in metrics:
        backup[metric.name] = metric

    # Set-up learning rate decay callback.
    if config.learning_rate_decay > 1:
        lr_decay_callback = ReduceLROnPlateauWithConfig(
            monitor='val_acc',
            factor=1 / config.learning_rate_decay,
            patience=5,
            mode='max',
            verbose=1,
            min_lr=config.min_learning_rate,
            min_delta=0.005,
        )
        callbacks.append(lr_decay_callback)
        backup['lr_decay'] = lr_decay_callback
    else:
        print('Notice: Not decaying learning rate. To do so, please provide learning_rate_decay > 1.0 in the config file.')

    # Create and add in ResumeCheckpoint
    callbacks.append(
        ResumeCheckpoint(args.model_path, backup, args.checkpoint_every, epoch=args.start_epoch, step=args.start_step, resume=(args.start_step != 0 or args.start_epoch != 1))
    )

    # If resuming partway through an epoch, finish it before starting the rest
    # Note that here and main training loop below, initial_epoch is zero-indexed hence the "- 1"
    if args.start_step != 0:
        print('Finishing resume epoch')
        model.fit(
            train_ds,
            epochs=1,
            steps_per_epoch=config.steps_per_epoch - args.start_step,
            validation_data=valid_ds,
            verbose='auto' if args.print_every is None else 2,
            callbacks=callbacks,
            initial_epoch=args.start_epoch - 1,
        )
        args.start_epoch += 1

    model.fit(
        train_ds,
        epochs=config.num_epochs,
        steps_per_epoch=config.steps_per_epoch,
        validation_data=valid_ds,
        verbose='auto' if args.print_every is None else 2,
        callbacks=callbacks,
        initial_epoch=args.start_epoch - 1,
    )


if __name__ == '__main__':
    main()
