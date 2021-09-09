import os

# Reduce extraneous TensorFlow output. Needs to occur before tensorflow import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

from voyager.data_loader import read_benchmark_trace
from voyager.logging import NBatchLogger
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

    assert args.model_path, 'No model path provided. Please provide a path to --model-path.'

    # Parse config file
    config = load_config(args.config, args.debug)
    print(config)

    # Load and process benchmark
    benchmark = read_benchmark_trace(args.benchmark)
    train_ds, valid_ds, test_ds = benchmark.split(config)

    # Create and compile the model
    model, _ = HierarchicalLSTM.compile_model(config, benchmark.num_pcs(), benchmark.num_pages())
    model.load(args.model_path)

    # Set-up callbacks for testing
    callbacks = []

    # Set-up batch logger callback.
    if args.print_every is not None:
        callbacks.append(NBatchLogger(args.print_every, args.start_epoch, args.start_step))

    # Set up Tensorboard callback.
    if args.tb_dir:
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=args.tb_dir,
                histogram_freq=1
        ))
    else:
        print('Notice: Not logging to Tensorboard. To do so, please provide a directory to --tb-dir.')

    model.evaluate(
        test_ds,
        verbose=1 if args.print_every is None else 0,
        callbacks=callbacks,
    )

    if callbacks != [] and args.print_every is not None:
        callbacks[0].print()

if __name__ == '__main__':
    main()
