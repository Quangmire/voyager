import os

import numpy as np
import tensorflow as tf

from voyager.data_loader import read_benchmark_trace
from voyager.logging import NBatchLogger
from voyager.models import HierarchicalLSTM
from voyager.utils import get_parser, load_config, pick_gpu_lowest_memory


# For reproducibility
tf.random.set_seed(0)
np.random.seed(0)

# Reduce extraneous TensorFlow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
    train_ds, valid_ds, test_ds = benchmark.split(config)

    # Set TensorBoard log path
    tb_path = args.tb_path

    # Create and compile the model
    model = HierarchicalLSTM.compile_model(config, benchmark.num_pcs(), benchmark.num_pages())

    # Set-up callbacks for training
    callbacks = []

    # Set-up batch logger callback.
    if args.print_every is not None:
        callbacks.append(
            NBatchLogger(args.print_every)
        )

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
    if args.tb_path:
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=tb_path,
                histogram_freq=1
        ))
    else:
        print('Notice: Not logging to Tensorboard. To do so, please provide a path to --tb-path.')

    # Set-up learning rate callbacks (plus anything else).
    callbacks.extend([
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_acc',
            factor=1 / config.learning_rate_decay,
            patience=5,
            mode='max',
            verbose=1,
            min_lr=config.min_learning_rate,
            min_delta=0.005,
        ),
        
    ])

    model.fit(
        train_ds,
        epochs=config.num_epochs,
        steps_per_epoch=config.steps_per_epoch,
        validation_data=valid_ds,
        verbose='auto' if args.print_every is None else 2,
        callbacks=callbacks,
    )


if __name__ == '__main__':
    main()
