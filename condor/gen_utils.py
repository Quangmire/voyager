import os
import argparse
import attrdict
import itertools
import yaml
import copy

CONDOR_GPU = """
+Group="GRAD"
+Project="ARCHITECTURE"
+ProjectDescription="Voyager hyperparameter tuning"

universe=vanilla
getenv=true
Rank=Memory >= {memory}
notification=Error
notify_user=cmolder@cs.utexas.edu
error={err_file}
output={out_file}
initial_dir={init_dir}
executable={exe}

requirements=Cuda8 && TARGET.GPUSlot
request_GPUs=1
+GPUJob=true

queue
"""

CONDOR_CPU = """
+Group="GRAD"
+Project="ARCHITECTURE"
+ProjectDescription="Voyager hyperparameter tuning"

universe=vanilla
getenv=true
Rank=Memory >= {memory}
notification=Error
notify_user=cmolder@cs.utexas.edu
error={err_file}
output={out_file}
initial_dir={init_dir}
executable={exe}

queue
"""

def get_parser():
    """Return base parser for scripts.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Tuning configuration file (examples: configs/tuning)')
    return parser
    

def generate(gpu=False, memory=0, **params):
    """Generate Condor launch scripts.
    Sourced from: github.com/Quangmire/condor/condor_common.py
    """
    base = CONDOR_GPU if gpu else CONDOR_CPU
    params = {
        'memory': memory,
        **params
    }
    return base.format(**params)


def load_tuning_config(path):
    """Load base tuning configuration."""
    with open(path, 'r') as f:
        config = attrdict.AttrDict(yaml.safe_load(f))
    return config


def variation_string(variation):
    """Generate a string representing the variation."""
    pm_str = ''
    for k, v in variation.items():
        pm_str += f'{k}-{v}_'
    return pm_str.rstrip('_')


def permute_variations(config):
    """Generate all variations (and their names) of the tuning config."""

    # Get permute keys
    permuted_vars = {}
    for k, v in config.config.items():
        if isinstance(v, list):
            permuted_vars[k] = v

    # If we have no permutations, we are not tuning any variables.
    if len(permuted_vars) == 0:
        # Return dictionary version of config.
        variation = {}
        for k, v in config.config.items():
            variation[k] = v
        return [variation], ['default']

    # Generate permutations of the subset.
    keys, _ = zip(*permuted_vars.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*permuted_vars.values())]

    # Inject each permutation into its own config attrdict.
    base_config = config.config
    variations = []
    variation_names = []
    for pm in permutations:
        variation = {}
        for k, v in base_config.items():
            variation[k] = v
        for k, v in pm.items():
            variation[k] = v
        variations.append(variation)
        variation_names.append(variation_string(pm))

    return variations, variation_names


def permute_trace_paths(config, trace_type='load'):
    """Generate all trace paths of the tuning config.
    
    If load traces, use the load base.
    If champsim traces, use the champsim base."""
    base_dir = config.meta.trace_dirs[trace_type]
    trace_paths = [os.path.join(base_dir, tr) for tr in config.traces]
    return trace_paths
