import os
import sys
import yaml
import attrdict

# Reduce extraneous TensorFlow output. Needs to occur before tensorflow import
# NOTE: You may want to unset this if you want to see GPU-related error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import ray
from ray import tune

# Ray
ray.init(address='auto')

from numba import cuda
from ray.tune.suggest.skopt import SkOptSearch
from ray.tune.schedulers import MedianStoppingRule
import numpy as np
import tensorflow as tf


from raytune.utils import get_tuning_parser, load_tuning_config
from voyager.models import Voyager
from voyager.data_loader import read_benchmark_trace

# For reproducibility
tf.random.set_seed(0)
np.random.seed(0)


class VoyagerTrainable(tune.Trainable):
    

    def _print_trial_parameters(self):
        print('======== TRIAL PARAMETERS ========')
        print('Benchmark:')
        print('    Path       :', self.config.args.benchmark)
        print('Model:')
        print('    Name       :', self.config.args.model_name)
        print('    Path       :', self.model_path)
        print('Trial:')
        print('    Name       :', self.trial_name)
        print('    Auto resume:', self.config.args.auto_resume)
        print('    Config     :')
        pretty_print_dict(self.config, indent=8)
    
    
    def setup(self, config, upload_dest = None, sweep_name = None):#, benchmark_obj=None):   
        sys.path.append('/home/ray/voyager')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # (Try to) reduce extraneous TensorFlow output.
        from voyager.data_loader import read_benchmark_trace
        from voyager.model_wrappers import ModelWrapper
        
        # For compatibility with resuming,
        # clear GPU memory (Tune tries to reload the model on its own,
        # but our Trainable already handles this.
        print('Clearing GPU memory...')
        cuda.get_current_device().reset()
        
        self.config = attrdict.AttrDict(config) # For compatibility with resuming
        self.model_path = upload_dest + f'/{sweep_name}/{self.trial_name}_model/'
        self._print_trial_parameters()
            
        self.benchmark = read_benchmark_trace(self.config.args.benchmark, self.config)
        self.model_wrapper = ModelWrapper.setup_from_ray_config(
            self.config, 
            benchmark = self.benchmark,
            model_path = self.model_path,
        )  
        
        
    def step(self):
        # TODOS:
        # - Fix checkpoint resuming when the last checkpoint ended the epoch
        #   (does not advance to next epoch, and repeats validation).
        return self.model_wrapper.train_one_epoch(model_path = self.model_path)
    
    
    def cleanup(self):
        print('CLEANUP')
        del self.model_wrapper
        
        
    # def step(self):
    #     """Train Voyager for one epoch.
    #     """
    #     result = self.model_wrapper.train_one_epoch()
    #     return result
    

# def train_voyager(config):
#     sys.path.append('/home/ray/voyager')
#     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # (Try to) reduce extraneous TensorFlow output.
#     from voyager.model_wrappers import ModelWrapper

#     config = attrdict.AttrDict(config) # For compatibility with resuming

#     # TODO - Check if we can use the trial name. If not, find a more robust name 
#     # based on the trial parameters.
    
#     trial_name = 'DUMMY_NAME'

#     model_path = config.upload_dest + f'/{config.args.sweep_name}/{trial_name}_model/'

#     print('======== TRIAL PARAMETERS ========')
#     print('Benchmark:')
#     print('    Path       :', config.args.benchmark)
#     print('Model:')
#     print('    Name       :', config.args.model_name)
#     print('    Path       :', model_path)
#     print('Trial:')
#     print('    Name       :', trial_name)
#     print('    Auto resume:', config.args.auto_resume)
#     print('    Config     :')
#     pretty_print_dict(config, indent=8)

#     model_wrapper = ModelWrapper.setup_from_ray_config(config, model_path = model_path)   
#     start_epoch = model_wrapper.epoch
    
#     for epoch in range(start_epoch, config.num_epochs):
#         yield model_wrapper.train_one_epoch() # Sends metrics to Tune
    

    
def pretty_print_dict(dic, indent=0):
    for k, v in dic.items():
        print(f'{" "*indent}{k}={v}')
    
    

def main():
    args = get_tuning_parser().parse_args()
    
    tuning_config, initial_config = load_tuning_config(args)
    
    print('======== SWEEP PARAMETERS ========')
    print('Benchmark:')
    print('    Path        :', args.benchmark)
    print('Model config:')
    print('    Path        :', args.config)
    print('Sweep:')
    print('    Name        :', args.sweep_name)
    print('    Checkpoints :', tuning_config.upload_dest)
    print('    Auto resume?:', args.auto_resume)
    print('Tuning:')
    print('    Path        :', args.tuning_config)
    print('    Max epochs  :', tuning_config.num_epochs)
    print('    Base start? :', args.base_start)
    print('    Grace period:', args.grace_period, 'hours')
    print('    Data        :')
    pretty_print_dict(tuning_config, indent=8)
    
    
    # https://docs.ray.io/en/latest/tune/user-guide.html   
    # Bayesian Optimization search using scikit-optimize
    search = SkOptSearch(
       metric='val_acc',
       mode='max',
       points_to_evaluate=[initial_config] if args.base_start else None # Use base config as intial start.
    )
    
    # Median stopping rule for early termination
    sched = MedianStoppingRule(
       metric='val_acc',
       mode='max',
       grace_period=args.grace_period * 60 * 60
    )
    
    # https://docs.ray.io/en/latest/tune/user-guide.html#checkpointing-and-synchronization
    # Synchronize checkpoints on GCP cloud storage
    sync_config = tune.SyncConfig(
        upload_dir=tuning_config.upload_dest
    ) 
    
    
    #benchmark = read_benchmark_trace(args.benchmark)
    #ref = ray.put(benchmark)
                         
    # Run tuning sweep
    analysis = tune.run(
        tune.with_parameters(
           VoyagerTrainable,
           upload_dest = tuning_config.upload_dest,
           sweep_name = args.sweep_name,
           #benchmark_obj = benchmark
        ),
        #VoyagerTrainable,
        num_samples=args.num_samples,
        config=tuning_config,
        search_alg=search,
        scheduler=sched,
        sync_config=sync_config,
        name=args.sweep_name,
        resources_per_trial={'gpu': 1, 'cpu': 2},
        #trial_executor=executor,
        resume='AUTO' if args.auto_resume else False,
        fail_fast='raise',
    )
    
if __name__ == '__main__':
    main()
