"""Generate a hyperparameter sweep for Voyager.
Using the parameters from a base configuration, vary them
as desired, saving the new configurations to {BASE_DIR}/configs.

Then, generate condor launch scripts for each configuration +
benchmark.

A file that lists each launch condor config line-by-line is
saved to {BASE_DIR}/condor_configs_train.txt. You can run a script
like Quangmire/condor/condor_submit_batch.py to launch them.

Based on: github.com/Quangmire/condor/condor_pc.py
"""

import os
import yaml

# generate - Used for template for condor submit scripts
from gen_utils import generate, get_parser, load_tuning_config, \
                      permute_variations, permute_trace_paths

# Template for bash script
SH_TEMPLATE = '''#!/bin/bash
source /u/cmolder/miniconda3/etc/profile.d/conda.sh
conda activate tensorflow
python3 -u {script_file} --benchmark {benchmark} \\
    --config {config_file} --tb-dir {tensorboard_dir} \\
    --model-path {model_path} --print-every {print_every} \\
    --checkpoint-every {checkpoint_period}
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
    trace_paths = permute_trace_paths(config, trace_type='load')
    print('Generating configurations for these traces:', trace_paths)

    # Track condor files generated so they can be batch launched later 
    # (paths saved line-by-line into the same file)
    condor_files = []
    base_dir = config.meta.sweep_dir

    # For each trace, generate each permuted configuration and its script.
    for tr_path in trace_paths:
        tr = tr_path.split('/')[-1].rstrip('.xz').rstrip('.gz').rstrip('.bz')
        for var, var_name in zip(variations, variation_names):
            # Setup initial output directories/files per experiment
            tensorboard_dir = os.path.join(base_dir, 'tensorboard', tr, 'train', var_name + '/')
            log_file_base = os.path.join(base_dir, 'logs', tr, 'train', var_name)
            config_file = os.path.join(base_dir, 'configs', f'{var_name}.yaml')
            condor_file = os.path.join(base_dir, 'condor', tr, 'train', f'{var_name}.condor')
            script_file = os.path.join(base_dir, 'scripts', tr, 'train', f'{var_name}.sh')
            model_file = os.path.join(base_dir, 'models', tr, f'{var_name}.model')
            
            print(f'\nFiles for {tr}, {var_name}:')
            print(f'    output log  : {log_file_base}.OUT')
            print(f'    error log   : {log_file_base}.ERR')
            print(f'    model       : {model_file}')
            print(f'    tensorboard : {tensorboard_dir}')
            print(f'    config      : {config_file}')
            print(f'    condor      : {condor_file}')
            print(f'    script      : {script_file}')

            # Create directories
            os.makedirs(tensorboard_dir, exist_ok=True)
            os.makedirs(os.path.join(base_dir, 'logs', tr, 'train'), exist_ok=True)
            os.makedirs(os.path.join(base_dir, 'configs'), exist_ok=True)
            os.makedirs(os.path.join(base_dir, 'condor', tr, 'train'), exist_ok=True)
            os.makedirs(os.path.join(base_dir, 'scripts', tr, 'train'), exist_ok=True)
            os.makedirs(os.path.join(base_dir, 'models', tr), exist_ok=True)

            # Save variation config file
            with open(config_file, 'w') as f:
                yaml.dump(var, f)

            # Build condor file
            condor = generate(
                gpu=config.meta.use_gpu,
                err_file=log_file_base + '.ERR',
                out_file=log_file_base + '.OUT',
                init_dir=config.meta.software_dirs.voyager,
                exe=script_file
            )
            with open(condor_file, 'w') as f:
                print(condor, file=f)

            # Build script file
            with open(script_file, 'w') as f:
                print(SH_TEMPLATE.format(
                    script_file=os.path.join(config.meta.software_dirs.voyager, 'train.py'),
                    benchmark=tr_path,
                    config_file=config_file,
                    tensorboard_dir=tensorboard_dir,
                    model_path=model_file,
                    print_every=config.meta.print_every,
                    checkpoint_period=config.meta.checkpoint_every,
                ), file=f)
            os.chmod(script_file, 0o777) # Make script executable

            condor_files.append(condor_file) # Add condor file to the list

    print(f'\nCondor file list : {os.path.join(base_dir, "condor_configs_train.txt")}')
    with open(os.path.join(base_dir, 'condor_configs_train.txt'), 'w') as f:
        for cf in condor_files:
            print(cf, file=f)


if __name__ == '__main__':
    main()
