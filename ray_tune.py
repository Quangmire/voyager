#!/bin/python

"""Perform a Ray Tune sweep.
"""

import os
import sys
import yaml
import attrdict
import skopt

# Reduce extraneous TensorFlow output. Needs to occur before tensorflow import
# NOTE: You may want to unset this if you want to see GPU-related error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import ray
from ray import tune

# Ray
ray.init(address='auto')
from ray.tune.suggest.skopt import SkOptSearch
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import MedianStoppingRule
import numpy as np
import tensorflow as tf

from raytune.utils import get_tuning_parser, load_tuning_config
from voyager.models import Voyager
from voyager.data_loader import read_benchmark_trace

# For reproducibility
tf.random.set_seed(0)
np.random.seed(0)


"""
Trainable class
"""
class VoyagerTrainable(tune.Trainable):
    def _print_trial_parameters(self):
        print(f'''[setup] ======== TRIAL PARAMETERS ========
Benchmark:
     Path       : {self.config.args.benchmark}
Model:
     Name       : {self.config.args.model_name}
     Path       : {self.model_path}
Trial:
     Name       : {self.trial_name}
     Auto resume: {self.config.args.auto_resume}
     Config     :
{pretty_dict_string(self.config, indent=8)}
        ''')
        
    
    def setup(self, config, upload_dest=None, sweep_name=None, use_local_benchmark=False):  
        """Setup the training configuration.
        """
        # Imports / configs
        sys.path.append('/home/ray/voyager')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # (Try to) reduce extraneous TensorFlow output.
        from voyager.data_loader import read_benchmark_trace
        from voyager.model_wrappers import ModelWrapper
        
        # Clear GPU memory, for compatibility with resuming.
        # (On resuming, Tune tries to reload the model itself,
        # but our Trainable already handles this).
        #print('[setup] Clearing GPU memory...')
        #cuda.get_current_device().reset()
        
        # Initialize configuration
        self.config = attrdict.AttrDict(config) # re-cast as attrdict for compatibility with resuming
         
        # Iniitalize model path
        if upload_dest is not None:
            self.model_path = upload_dest + f'/{sweep_name}/{self.trial_name}_model/'
        else:
            self.model_path = config.upload_dest +  f'/{sweep_name}/{self.trial_name}_model/' # May raise an error on resuming.
            
        # Print trial parameters
        self._print_trial_parameters()
        
        # Initialize benchmark
        if use_local_benchmark:
            benchmark_path = get_local_benchmark_copy(self.config.args.benchmark)
        else:
            benchmark_path = self.config.args.benchmark

        print(f'[setup] Loading benchmark from {benchmark_path}...')
        self.benchmark = read_benchmark_trace(benchmark_path, self.config)
        
        # Initialize model
        print(f'[setup] Setting up model...')
        self.model_wrapper = ModelWrapper.setup_from_ray_config(
            self.config, 
            benchmark = self.benchmark,
            model_path = self.model_path,
        )  
        
        
    def step(self):
        """Train the model for one epoch, or the remainder of an epoch
        if resuming.
        """
        res = self.model_wrapper.train_one_epoch(model_path = self.model_path)
        res.update(should_checkpoint=True) # Do not remove - otherwise Tune will checkpoint the full Trainable and create memory leaks.
        return res
    
    
    def save_checkpoint(self, checkpoint_dir):
        """Override save_checkpoint and load_checkpoint, as the model wrapper
        handles checkpointing for us with more frequent checkpoints.
        """
        print('[save_checkpoint] [DEBUG] Called save_checkpoint with checkpoint_dir', checkpoint_dir)
        return checkpoint_dir
    
    
    def load_checkpoint(self, checkpoint_dir):
        print('[load_checkpoint] [DEBUG] Called load_checkpoint with checkpoint_dir', checkpoint_dir)
        return
        

    
"""
Helper functions for Trainable
"""
def pretty_dict_string(dic, indent=0):
    return '\n'.join([f'{" "*indent}{k}={v}' for k, v in dic.items()])
        
    
def get_local_benchmark_copy(benchmark_path):
    # If benchmark_path on local machine:
    # Just load it from where it is.
    if not benchmark_path.startswith('gs://'):
        return benchmark_path 
    
    # If benchmark_path on gcp:
    # Create a local copy, then return the path to the local copy
    local_copy = f'/tmp/benchmarks/{os.path.basename(benchmark_path)}'
    if not os.path.exists(local_copy):
        print(f'[setup] Copying benchmark from GCP ({benchmark_path}) to {local_copy}...')
        os.makedirs('/tmp/benchmarks/', exist_ok=True)
        os.system(f'gsutil cp {benchmark_path} {local_copy}') # TODO FIXME: Fails on worker nodes.
    return local_copy

def name_trial(trial):
    global sweep_config
    return '_'.join([f'{k}={trial.config[k]}' for k in sweep_config.keys()] + [trial.trial_id])


"""
Main function
"""
global sweep_config
def main():
    args = get_tuning_parser().parse_args()
    global sweep_config
    tuning_config, initial_config, sweep_config = load_tuning_config(args)
    
    print(f'''
======== SWEEP PARAMETERS ========
Benchmark:
     Path        : {args.benchmark}
Model config:
     Path        : {args.config}
Sweep:
     Name        : {args.sweep_name}
     Checkpoints : {tuning_config.upload_dest}
     Auto resume : {args.auto_resume}
Tuning:
     Path        : {args.tuning_config}
     Max epochs  : {tuning_config.num_epochs}
     Base start  : {args.base_start}
     Grace period: {args.grace_period} hours
     Data        :
{pretty_dict_string(tuning_config, indent=8)}
    ''')

    # scikit-optimize BO search
    # - https://docs.ray.io/en/latest/tune/user-guide.html   
    # - https://scikit-optimize.github.io/stable/modules/generated/skopt.Optimizer.html
    # search = SkOptSearch(
    #    metric='val_acc',
    #    mode='max',
    #    points_to_evaluate=[initial_config] if args.base_start else None # Use base config as intial start.
    # )
    
    # HyperOpt Search
    # - https://docs.ray.io/en/master/tune/api_docs/suggestion.html#hyperopt-tune-suggest-hyperopt-hyperoptsearch
    # - http://hyperopt.github.io/hyperopt/
    search = HyperOptSearch(
        metric='val_acc',
        mode='max',
        points_to_evaluate=[initial_config] if args.base_start else None # Use base config as intial start.
    )
    
    # Median stopping rule for early termination
    #sched = MedianStoppingRule(
    #   metric='val_acc',
    #   mode='max',
    #   grace_period=args.grace_period * 60 * 60
    #)
    
    # https://docs.ray.io/en/latest/tune/user-guide.html#checkpointing-and-synchronization
    # Synchronize checkpoints on GCP cloud storage
    sync_config = tune.SyncConfig(
        upload_dir=tuning_config.upload_dest
    ) 
                 
    # Run tuning sweep
    analysis = tune.run(
        tune.with_parameters(
           VoyagerTrainable,
           upload_dest = tuning_config.upload_dest,
           sweep_name = args.sweep_name,
           use_local_benchmark = False
        ),
        num_samples=args.num_samples,
        config=tuning_config,
        search_alg=search,
        #scheduler=sched,
        sync_config=sync_config,
        name=args.sweep_name,
        trial_name_creator=name_trial,
        resources_per_trial={'gpu': 1, 'cpu': 2},
        stop={'training_iteration': tuning_config.num_epochs},
        max_failures=1000, # Check configurations for GPU OOM errors, or else your sweep won't finish.
        resume='AUTO' if args.auto_resume else False,
        #fail_fast='raise',
    )
    
if __name__ == '__main__':
    main()
