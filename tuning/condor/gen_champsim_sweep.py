"""After completing a hyperparameter sweep (gen_tuning_sweep.py) 
and ChampSim prefetch trace generation sweep (gen_trace_sweep.py), 
this code takes the series of saved models and their traces, and 
creates a series of Condor scripts to evaluate them on the traces 
on ChampSim.

A file that lists each launch condor config line-by-line
is saved to {BASE_DIR}/condor_configs_champsim.txt. You
like Quangmire/condor/condor_submit_batch.py to launch them.

Please don't use the eldar (GPU) machines to do this, as
ChampSim reads the traces directly instead of invoking an
instance of the model.

Based on: github.com/Quangmire/condor/condor_pc.py

TODO Work in progress (awaiting script)
"""


import os
import itertools
import copy
import yaml

from condor_common import generate # Used for template for condor submit scripts

CHAMPSIM_PATH = '/u/cmolder/GitHub/ChampSim'
BASE_DIR = '/scratch/cluster/cmolder/voyager_hypertuning/experts_lrdecay/'

TRACE_DIR = '/scratch/cluster/qduong/ML-DPC/data/load_traces/'
TRACES = [
    'spec17/605.mcf-s0.txt.xz',
]

VARIATIONS = {
    #'learning_rate': [0.01, 0.001, 0.0001, 0.00001], # best mcf-s0: 0.001 (run 1)
    #'batch_size': [32, 64, 128, 256, 512],           # best mcf-s0: 512   (run 1)
    #'pc_embed_size': [16, 32, 64, 128, 256],          # (pc=128, page=512, bsz=512) runs out of memory on GTX 1080
    #'page_embed_size': [32, 64, 128, 256]
    'page_embed_size': [64, 256],
    'num_experts': [10, 25, 50, 75, 100],
    'learning_rate_decay': [1, 2] # 1 disables LR decay
}


# Template for bash script
# TODO Implement correctly
SH_TEMPLATE = '''#!/bin/bash
source /u/cmolder/miniconda3/etc/profile.d/conda.sh
conda activate tensorflow
python3 -u {script_file} run {prefetch_trace_file}
'''


def permutation_string(permutation):
    """Generate a string representing the permutation."""
    pm_str = ''
    for k, v in permutation.items():
        pm_str += f'{k}-{v}_'
    return pm_str.rstrip('_')


def permute_variations(variations):
    """Generate all permutations of the variations."""
    keys, values = zip(*variations.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return permutations
    

def main():
    # Generate all permutations of the variations.
    permutations = permute_variations(VARIATIONS)
    print(f'Generating {len(permutations)} configurations.')

    # Track condor files generated so they can be batch launched later
    # (paths saved line-by-line into the same file)
    condor_files = []

    # For each trace, generate each permuted configuration and its script.
    for tr in TRACES:
        for pm in permutations:
            tr_name, pm_name = tr.split('.')[1], permutation_string(pm)

            # Setup initial output directories/files per experiment
            log_file_base = os.path.join(BASE_DIR, 'champsim', 'logs', tr_name, pm_name)
            config_file = os.path.join(BASE_DIR, 'champsim', 'configs', f'{pm_name}.yaml')
            condor_file = os.path.join(BASE_DIR, 'champsim', 'condor', tr_name, f'{pm_name}.condor')
            script_file = os.path.join(BASE_DIR, 'champsim', 'scripts', tr_name, f'{pm_name}.sh')
            
            print(f'\nFiles for {tr_name}, {pm_name}:')
            print(f'    output log  : {log_file_base}.OUT')
            print(f'    error log   : {log_file_base}.ERR')
            print(f'    model       : {log_file_base}.model')
            print(f'    config      : {config_file}')
            print(f'    condor      : {condor_file}')
            print(f'    script      : {script_file}')

            # Create directories
            os.makedirs(os.path.join(BASE_DIR, 'logs', tr_name), exist_ok=True)
            os.makedirs(os.path.join(BASE_DIR, 'configs'), exist_ok=True)
            os.makedirs(os.path.join(BASE_DIR, 'condor', tr_name), exist_ok=True)
            os.makedirs(os.path.join(BASE_DIR, 'scripts', tr_name), exist_ok=True)

            # Build condor file
            condor = generate(
                gpu=False,
                err_file=log_file_base + '.ERR',
                out_file=log_file_base + '.OUT',
                init_dir=CHAMPSIM_PATH,
                exe=script_file
            )
            with open(condor_file, 'w') as f:
                print(condor, file=f)

            # TODO implement and calculate trace file path.
            prefetch_trace_file = None

            # Build script file
            with open(script_file, 'w') as f:
                print(SH_TEMPLATE.format(
                    script_file=os.path.join(CHAMPSIM_PATH, 'ml_prefetch_sim.py'),
                    prefetch_trace_file=prefetch_trace_file,
                ), file=f)
            
            # Make script executable
            os.chmod(script_file, 0o777)

            # Add condor file to the list
            condor_files.append(condor_file)

    print(f'\nCondor file list : {os.path.join(BASE_DIR, "condor_configs_champsim.txt")}')
    with open(
        os.path.join(BASE_DIR, 'condor_configs.txt'), 'w'
    ) as f:
        for cf in condor_files:
            print(cf, file=f)


if __name__ == '__main__':
    main()
