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


def setup_callbacks(args, config, model, metrics):
    """Setup callbacks for prefetch trace generation.
    (NOTE: Not currently used for generation.)
    """
    callbacks = []

    if args.print_every is not None: # Set-up batch logger callback.
        callbacks.append(NBatchLogger(
            args.print_every, 
            start_epoch=args.start_epoch, 
            start_step=args.start_step
        ))

    if args.model_path: # Set-up model checkpoint callback.
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=args.model_path,
            save_weights_only=True,
            monitor='val_acc',
            mode='max',
            save_best_only=True,
            verbose=1,
        ))
    else:
        print('Notice: Not checkpointing the model. To do so, please provide a path to --model-path.')

    if args.tb_dir: # Set-up Tensorboard callback.
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=args.tb_dir,
                histogram_freq=1
        ))
    else:
        print('Notice: Not logging to Tensorboard. To do so, please provide a directory to --tb-dir.')


def main():
    parser = get_parser()
    parser.add_argument('--prefetch-file', required=True)
    args = parser.parse_args()

    # Parse config file
    print('Loading config...')
    config = load_config(args.config, args.debug)
    print(config)

    # Load and process benchmark
    print('Reading benchmark trace...')
    benchmark = read_benchmark_trace(args.benchmark)
    print('Processing benchmark...')
    train_ds, valid_ds, test_ds = benchmark.split(config, args.start_epoch, args.start_step)

    # Create and compile the model
    print('Compiling model..')
    model, metrics = HierarchicalLSTM.compile_model(config, benchmark.num_pcs(), benchmark.num_pages())

    # Set-up callbacks for generation
    # print('Setting up callbacks...')
    # callbacks = setup_callbacks(args, config, model, metrics)

    print('Loading model...')
    model.load(args.model_path)

    print('Generating prefetch trace...')
    ds_len = len(test_ds)
    with open(args.prefetch_file, 'w') as f:
        for i, (x, _) in enumerate(test_ds):
            logits = model(x, training=False)

            # Print an update every (args.print_every) samples.
            if args.print_every is not None:
                if i % args.print_every == 0:
                    print(f'({i/ds_len*100:4.1f} %) Sample {i} / {ds_len}')

            # Get prediction from logits
            if config.sequence_loss:
                pages = tf.argmax(logits[:, -1, :-64], -1).numpy()
                offsets = tf.argmax(logits[:, -1, -64:], -1).numpy()
            else:
                pages = tf.argmax(logits[:, :-64], -1).numpy().tolist()
                offsets = tf.argmax(logits[:, -64:], -1).numpy().tolist()

            for inst_id, page, offset in zip(benchmark.test_inst_ids[i * config.batch_size:], pages, offsets):
                # Skip OOV
                if page == 0:
                    continue
                addr = (benchmark.reverse_page_mapping[page] << 6) + offset
                print(inst_id, hex(addr), file=f)

if __name__ == '__main__':
    main()
