import numpy as np
import tensorflow as tf
from ray import tune

from voyager.utils import get_parser, load_config
from voyager.data_loader import read_benchmark_trace
from voyager.logging import NBatchLogger
from voyager.models import HierarchicalLSTM

# For reproducibility
tf.random.set_seed(0)
np.random.seed(0)


def _setup_callbacks(config, print_every=None, model_path=None):
    """Set up callbacks for training/validation."""
    callbacks = []
    if print_every is not None: # Set-up batch logger callback.
        callbacks.append(NBatchLogger(print_every))
    if model_path is not None: # Set-up model checkpoint callback.
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                save_weights_only=True,
                monitor='val_acc',
                mode='max',
                save_best_only=True,
                verbose=1,
        ))
    else:
        print('Notice: Not checkpointing the model. To do so, please provide a path to --model-path.')

    callbacks.extend([ # Set-up learning rate callbacks (plus anything else).
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
    return callbacks


def train_voyager(config, print_every=None, model_path=None):
    """Train/validate an instance of Voyager."""
    # Load and process benchmark
    benchmark = read_benchmark_trace(args.benchmark)
    train_ds, valid_ds, _ = benchmark.split(config)

    # Create and compile the model
    model = HierarchicalLSTM.compile_model(config, benchmark.num_pcs(), benchmark.num_pages())
    callbacks = _setup_callbacks(config, print_every, model_path)

    model.fit(
        train_ds,
        epochs=config.num_epochs,
        steps_per_epoch=config.steps_per_epoch,
        validation_data=valid_ds,
        verbose='auto' if print_every is None else 2,
        callbacks=callbacks,
    )


def main():
    parser = get_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    print('Base config:', config)

    # Define hyperparameter search space by building on
    # the base config.
    config.pc_embed_size = tune.choice([16, 32, 64, 128, 256])
    config.page_embed_size = tune.choice([32, 64, 128, 256])
    config.learning_rate = tune.loguniform(1e-5, 1e-2)
    config.batch_size = tune.lograndint(16, 1024)
    config.lstm_size = tune.lograndint(32, 512)


if __name__ == '__main__':
    main()