import os
import yaml

# Reduce extraneous TensorFlow output. Needs to occur before tensorflow import
# NOTE: You may want to unset this if you want to see GPU-related error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import ray
ray.init()

from ray import tune
import numpy as np
import tensorflow as tf
 
from raytune.utils import get_tuning_parser, load_tuning_config, train_voyager

# For reproducibility
tf.random.set_seed(0)
np.random.seed(0)

def main():
    args = get_tuning_parser().parse_args()
    
    #print('Ray cluster  :')
    #print('    Address:', args.address)
    print('Benchmark    :')
    print('    Path   :', args.benchmark)
    print('Model config :')
    print('    Path   :', args.config)
    
    tuning_config = load_tuning_config(args)
    print('Tuning config:')
    print('    Workers:', args.num_workers)
    print('    Path   :', args.tuning_config)
    print('    Data   :', tuning_config)
    print('Dry run?     :', args.dry_run)
    
    # https://docs.ray.io/en/latest/tune/user-guide.html
    #ray.init()
    analysis = tune.run(
        train_voyager, # DEBUG
        config = tuning_config,
        resources_per_trial={'gpu': 1},
    )
    
if __name__ == '__main__':
    main()