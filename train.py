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
    args = parser.parse_args()

    # Create model wrapper
    model_wrapper = ModelWrapper.setup_from_args(args)

    # Start training the model
    model_wrapper.train()

if __name__ == '__main__':
    main()
