import argparse
import os
import yaml
import attrdict
import ray
from ray import tune
import numpy as np
import tensorflow as tf

from voyager.utils import load_config
from voyager.model_wrappers import ModelWrapper
from voyager.models import Voyager

# For reproducibility
tf.random.set_seed(0)
np.random.seed(0)
    

def get_eval_parser():
    """Get parser for Ray Tune evaluation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'ray_results_dir',
        help='Local path to the ray results directory',
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file for the .csv results.',
        default='./results.csv'
    )
    return parser
        
    
def get_tuning_parser():
    """Get parser for Ray Tune sweep
    that builds on the default arguments for single-model training.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-b', '--benchmark', 
        help='Path to the benchmark trace', 
        required=True
    )
    parser.add_argument(
        '-c', '--config', 
        default='./configs/base.yaml', 
        help='Path to configuration file for the model. Default: ./configs/base.yaml'
    )
    parser.add_argument(
        '-e', '--epochs', 
        default=None,
        type=int,
        help='Maximum number of epochs to train. Default: num_epochs in <config>.'
    )
    parser.add_argument(
        '-g', '--grace-period', 
        default=4,
        type=int,
        help='Minimum time (in hours) a trial can run before the median stopping rule can prune it. Default: 4'
    )
    parser.add_argument(
        '-m', '--model-name', 
        default='voyager',
        help='Name of the model (NOT TESTED). Default: voyager'
    )
    parser.add_argument(
        '-n', '--sweep-name',
        default=None,
        help='Name of sweep (e.g. name it after the trace to tune).'
    )
    parser.add_argument(
        '-p', '--checkpoint-every', 
        type=int,
        default=None,
        help='Save a model checkpoint every <checkpoint_every> steps. Default: No checkpointing'
    )
    parser.add_argument(
        '-r', '--auto-resume', 
        action='store_true', 
        default=False, 
        help='Automatically resume if checkpoint detected.'
    )
    parser.add_argument(
        '-s', '--num-samples', 
        default=1,
        type=int,
        help='Number of simultaneous samples to queue for the tuning sweep. Default: 1'
    )
    parser.add_argument(
        '-t', '--tuning-config', 
        default='./configs/ray/tune_ray.yaml',
        help='Path to tuning configuration. Default: ./configs/ray/tune_ray.yaml'
    )
    parser.add_argument(
        '--print-every', 
        default=None,
        type=int,
        help='Print statistics every <print-every> batches. Make sure to set when outputting to a file. Default: Progress bar'
    )
    parser.add_argument(
        '--base-start', 
        action='store_true',
        default=False,
        help='Initialize the Bayesian optimization using the config from <config>.'
    )
    
    
    return parser


# Reference: https://docs.ray.io/en/latest/tune/api_docs/search_space.html
TYPE_DICT = {
    'choice': tune.choice, 
    'loguniform': tune.loguniform, 
    'lograndint': tune.lograndint,
    'randint': tune.randint,
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
        
    # Build intial config from the base config
    initial = attrdict.AttrDict({})
    for k, v in tuning.items():
        initial[k] = config[k]

    # Replace variables in config with Ray Tune tuning counterparts
    for k, v in tuning.items():
        vclass = TYPE_DICT[v["type"]]
        del v['type']
        config[k] = vclass(**v)
        
    # Replace epochs with args.epochs if provided
    if args.epochs:
        config.num_epochs = args.epochs
        
    # Add args and upload_dest
    upload_dest = f'gs://voyager-tune/checkpoints/{os.path.basename(args.tuning_config).replace(".yaml", "")}'
    config.args = args
    config.upload_dest = upload_dest
    
    return config, initial, tuning
