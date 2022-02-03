import os
import sys
import yaml

# Reduce extraneous TensorFlow output. Needs to occur before tensorflow import
# NOTE: You may want to unset this if you want to see GPU-related error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import ray
from ray import tune

# Ray
ray.init(address='auto')


from ray.tune.suggest.skopt import SkOptSearch
from ray.tune.schedulers import MedianStoppingRule
import numpy as np
import tensorflow as tf


from raytune.utils import get_tuning_parser, load_tuning_config

# For reproducibility
tf.random.set_seed(0)
np.random.seed(0)


def train_voyager(config):
    # os.chdir('/home/ray/voyager')
    # os.environ['PYTHONPATH'] = '/home/ray/voyager'
    # print('======= WORKING DIRECTORY: ========', os.getcwd())
    # print('======= PYTHONPATH:        ========', os.environ['PYTHONPATH'])
    # print('STUFF INSIDE:', os.listdir(os.getcwd()))
    
    sys.path.append('/home/ray/voyager')
    from voyager.model_wrappers import ModelWrapper
    
    
    """Train/validate an instance of Voyager."""
    print('Benchmark:')
    print('    Path  :', config.args.benchmark)
    print('Model    :')
    print('    Name  :', config.args.model_name)
    print('    Config:', config)
    
    
    if config.args.dry_run:
        return

    model_wrapper = ModelWrapper.setup_from_ray_config(config)
    model_wrapper.train()


def main():
    args = get_tuning_parser().parse_args()
    
    #print('Ray cluster  :')
    #print('    Address:', args.address)
    print('Dry run?       :', args.dry_run)
    print('Benchmark     :')
    print('    Path      :', args.benchmark)
    print('Model config  :')
    print('    Path      :', args.config)
   
    
    tuning_config, initial_config = load_tuning_config(args)
    print('Tuning config  :')
    #print('    Workers   :', args.num_workers)
    print('    Name         :', args.sweep_name)
    print('    Path         :', args.tuning_config)
    print('    Max epochs   :', tuning_config.num_epochs)
    print('    Base start?  :', args.base_start)
    print('    Grace period :', args.grace_period, 'hours')
    print('    Data         :', tuning_config)
    
    
    # https://docs.ray.io/en/latest/tune/user-guide.html
    #ray.init()
    search = SkOptSearch(
        metric='mean_loss',
        mode='min',
        points_to_evaluate=[initial_config] if args.base_start else None # Use base config as intiial start.
    )
    
    sched = MedianStoppingRule(
        metric='mean_loss',
        mode='min',
        grace_period=args.grace_period * 60 * 60
    )
    
    analysis = tune.run(
        train_voyager, # DEBUG
        config = tuning_config,
        search_alg = search,
        scheduler = sched,
        name = args.sweep_name,
        resources_per_trial={'gpu': 1},
    )
    
if __name__ == '__main__':
    main()