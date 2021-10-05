import os

# Reduce extraneous TensorFlow output. Needs to occur before tensorflow import
# NOTE: You may want to unset this if you want to see GPU-related error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

from voyager.model_wrappers import ModelWrapper
from voyager.utils import get_parser, pick_gpu_lowest_memory


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
    parser.add_argument('--prefetch-file', required=True, help='Path to generated prefetch file')
    parser.add_argument('--train', action='store_true', default=False, help='Generate for train dataset too')
    parser.add_argument('--valid', action='store_true', default=False, help='Generate for valid dataset too')
    parser.add_argument('--no-test', action='store_true', default=False, help='Do not generate for the test dataset')
    args = parser.parse_args()

    assert args.model_path, 'No model path provided. Please provide a path to --model-path.'

    # Create model wrapper
    model_wrapper = ModelWrapper.setup_from_args(args)
    model_wrapper.load(args.model_path)

    # Start generating prefetches using the model
    model_wrapper.generate(
        datasets=model_wrapper.get_datasets(
            train=args.train,
            valid=args.valid,
            test=not args.no_test,
        ),
        prefetch_file=args.prefetch_file,
    )

if __name__ == '__main__':
    main()
