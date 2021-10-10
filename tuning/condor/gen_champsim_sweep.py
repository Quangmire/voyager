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
"""

import os

# generate - Used for template for condor submit scripts
from gen_utils import generate, get_parser, load_tuning_config, \
                      permute_variations, permute_trace_paths

# Template for bash script
SH_TEMPLATE = '''#!/bin/bash
source /u/cmolder/miniconda3/etc/profile.d/conda.sh
conda activate tensorflow
python3 -u {script_file} run {prefetch_trace_file}
'''


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Open tuning config and get it as a dictionary
    config = load_tuning_config(args.config)

    # Generate all permutations of the variations.
    variations, variation_names = permute_variations(config)
    print(f'Generating {len(variations)} configurations.')

    # Generate all trace paths.
    trace_paths = permute_trace_paths(config, trace_type='champsim')
    print('Generating configurations for these traces:', trace_paths)

    # Track condor files generated so they can be batch launched later 
    # (paths saved line-by-line into the same file)
    condor_files = []
    base_dir = config.meta.sweep_dir

    # For each trace, generate each permuted configuration and its script.
    for tr_path in trace_paths:
        tr = tr_path.split('/')[-1].split('.')[1]
        for var, var_name in zip(variations, variation_names):
            # Setup initial output directories/files per experiment
            log_file_base = os.path.join(base_dir, 'logs', tr, 'champsim', var_name)
            config_file = os.path.join(base_dir, 'configs', f'{var_name}.yaml')
            condor_file = os.path.join(base_dir, 'condor', tr, 'champsim', f'{var_name}.condor')
            script_file = os.path.join(base_dir, 'scripts', tr, 'champsim', f'{var_name}.sh')
            prefetch_file = os.path.join(base_dir, 'prefetch_traces', tr, f'{var_name}.txt')
            
            print(f'\nFiles for {tr}, {var_name}:')
            print(f'    output log  : {log_file_base}.OUT')
            print(f'    error log   : {log_file_base}.ERR')
            print(f'    model       : {log_file_base}.model')
            print(f'    config      : {config_file}')
            print(f'    condor      : {condor_file}')
            print(f'    script      : {script_file}')
            print(f'    prefetches  : {prefetch_file}')

            # Create directories
            os.makedirs(os.path.join(base_dir, 'logs', tr, 'champsim'), exist_ok=True)
            os.makedirs(os.path.join(base_dir, 'configs'), exist_ok=True)
            os.makedirs(os.path.join(base_dir, 'condor', tr, 'champsim'), exist_ok=True)
            os.makedirs(os.path.join(base_dir, 'scripts', tr,  'champsim'), exist_ok=True)
            os.makedirs(os.path.join(base_dir, 'prefetch_traces', tr), exist_ok=True)

            # Build condor file
            condor = generate(
                gpu=False,
                err_file=log_file_base + '.ERR',
                out_file=log_file_base + '.OUT',
                init_dir=config.meta.software_dirs.champsim,
                exe=script_file
            )
            with open(condor_file, 'w') as f:
                print(condor, file=f)

            # Build script file
            with open(script_file, 'w') as f:
                print(SH_TEMPLATE.format(
                    script_file=os.path.join(config.meta.software_dirs.champsim, 'ml_prefetch_sim.py'),
                    prefetch_trace_file=prefetch_file,
                ), file=f)
            
            # Make script executable
            os.chmod(script_file, 0o777)

            # Add condor file to the list
            condor_files.append(condor_file)

    print(f'\nCondor file list : {os.path.join(base_dir, "condor_configs_champsim.txt")}')
    with open(
        os.path.join(base_dir, 'condor_configs.txt'), 'w'
    ) as f:
        for cf in condor_files:
            print(cf, file=f)


if __name__ == '__main__':
    main()
