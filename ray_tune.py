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


from ray.tune.suggest.skopt import SkOptSearch
from ray.tune.schedulers import MedianStoppingRule
import numpy as np
import tensorflow as tf


from raytune.utils import get_tuning_parser, load_tuning_config

# For reproducibility
tf.random.set_seed(0)
np.random.seed(0)

# TODO : Try custom tune.Trainable class to fix checkpoints resetting.

# def train_voyager(config):
#     """Train/validate an instance of Voyager."""
#     sys.path.append('/home/ray/voyager')
#     from voyager.model_wrappers import ModelWrapper
    
#     config = attrdict.AttrDict(config) # For compatibility with resuming
#     print('Benchmark:')
#     print('    Path  :', config.args.benchmark)
#     print('Model    :')
#     print('    Name  :', config.args.model_name)
#     print('    Config:', config)
    
#     if config.args.dry_run:
#         return

#     model_wrapper = ModelWrapper.setup_from_ray_config(config)
#     model_wrapper.train()

class VoyagerTrainable(tune.Trainable):
    def setup(self, config):
        sys.path.append('/home/ray/voyager')
        from voyager.model_wrappers import ModelWrapper
        
        config = attrdict.AttrDict(config) # For compatibility with resuming
        
        # TODO - Check if we can use the trial name. If not, find a more robust name 
        # based on the trial parameters.
        model_path = config.upload_dest + f'/{config.args.sweep_name}/{self.trial_name}_model'

        print('Benchmark:')
        print('    Path  :', config.args.benchmark)
        print('Model    :')
        print('    Name  :', config.args.model_name)
        print('    Config:', config)
        print('Trial name:', self.trial_name)
        print('Model path:', model_path)
        
        if config.args.dry_run:
            return
        
        self.model_wrapper = ModelWrapper.setup_from_ray_config(
            config, 
            model_path = model_path
        )
        
        
        
    def step(self):
        """Train Voyager for one epoch.
        """
        logs = self.model_wrapper.train_one_epoch()
        
        print('[step] LOGS:', logs)
        return logs
        
    def save_checkpoint(self, checkpoint_dir):
        print('[VoyagerTrainable.save_checkpoint] Entering save checkpoint (no effect)')
        return # Model wrapper handles checkpointing on a more granular basis for us.
    
    def load_checkpoint(self, checkpoint_dir):
        print('[VoyagerTrainable.load_checkpoint] Loading from checkpoint...')
        self.model_wrapper.restore_checkpoint(self.model_wrapper.model_path)

    

def main():
    args = get_tuning_parser().parse_args()
    
    #print('Ray cluster  :')
    #print('    Address:', args.address)
    print('Dry run?      :', args.dry_run)
    print('Auto resume?  :', args.auto_resume)
    print('Benchmark     :')
    print('    Path      :', args.benchmark)
    print('Model config  :')
    print('    Path      :', args.config)
   
    
    tuning_config, initial_config = load_tuning_config(args)
    upload_dest = f'gs://voyager-tune/checkpoints/{os.path.basename(args.tuning_config).replace(".yaml", "")}'
    tuning_config['upload_dest'] = upload_dest
    
    print('Checkpoints   :', upload_dest)
    print('Tuning config :')
    #print('    Workers   :', args.num_workers)
    print('    Name         :', args.sweep_name)
    print('    Path         :', args.tuning_config)
    print('    Max epochs   :', tuning_config.num_epochs)
    print('    Base start?  :', args.base_start)
    print('    Grace period :', args.grace_period, 'hours')
    print('    Data         :', tuning_config)
    
    
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
        upload_dir=upload_dest
    ) 
                         
    # Run tuning sweep
    analysis = tune.run(
        #train_voyager,
        VoyagerTrainable,
        config=tuning_config,
        search_alg=search,
        scheduler=sched,
        sync_config=sync_config,
        name=args.sweep_name,
        resources_per_trial={'gpu': 1, 'cpu': 2},
        checkpoint_score_attr='val_acc',
        keep_checkpoints_num=5,
        resume='AUTO' if args.auto_resume else False,
        fail_fast='raise'
    )
    
if __name__ == '__main__':
    main()
