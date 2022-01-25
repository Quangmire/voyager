import os
import yaml
import attrdict
import ray
from ray import tune
import numpy as np
import tensorflow as tf

from voyager.utils import get_parser, load_config
from voyager.model_wrappers import ModelWrapper
from voyager.models import Voyager

# For reproducibility
tf.random.set_seed(0)
np.random.seed(0)

def train_voyager(config):
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
    
    
    
def get_tuning_parser():
    """Get parser for Ray Tune sweep
    that builds on the default arguments for single-model training.
    """
    parser = get_parser() # Base arguments
    parser.add_argument(
        '--tuning-config', 
        default='./configs/ray/tune_ray.yaml',
        help='Path to tuning configuration'
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Set up the sweep and print tuning parameters, but do not actually do tuning.'
    )
    
    return parser


# Reference: https://docs.ray.io/en/latest/tune/api_docs/search_space.html
TYPE_DICT = {
    'choice': tune.choice, 
    'loguniform': tune.loguniform, 
    'lograndint': tune.lograndint,
    'qlograndint': tune.qlograndint,
    'grid_search': tune.grid_search
}

def load_tuning_config(args):
    """Parse .yaml file for the tuning
    config, converting it to a Ray Tune
    compatible dictionary.
    
    Any tuned variable will overwrite its option
    in the model configuration.
    """
    with open(args.config, 'r') as f:
        config = attrdict.AttrDict(yaml.safe_load(f))
    
    with open(args.tuning_config, 'r') as f:
        tuning = attrdict.AttrDict(yaml.safe_load(f))

    # Replace variables in config with Ray Tune tuning counterparts
    for k, v in tuning.items():
        vclass = TYPE_DICT[v["type"]]
        del v['type']
        config[k] = vclass(**v)
        
    config.args = args
        
    return config
