"""After completing a hyperparameter tuning sweep,
(gen_train_sweep.py), this code takes the series 
of saved models, and creates a series of Condor 
scripts to generate a ChampSim prefetch trace for 
each model.

A file that lists each launch condor config line-by-line
is saved to {BASE_DIR}/condor_configs_generate.txt. You
like Quangmire/condor/condor_submit_batch.py to launch them.

Based on: github.com/Quangmire/condor/condor_pc.py
"""

import os
import glob
import argparse

from gen_utils import generate, get_parser, load_tuning_config, \
                      permute_variations, permute_trace_paths

# Template for bash script
SCRIPT = '/u/cmolder/GitHub/voyager-analysis/corr/corr_loadbranch.py'
PYPATH = '/u/cmolder/GitHub/voyager-analysis/'

SH_TEMPLATE = '''#!/bin/bash
source /u/cmolder/miniconda3/etc/profile.d/conda.sh
conda activate tensorflow
cd {pypath}
export PYTHONPATH=.
python3 -u {script_file} {loadbranch_trace} \\
    -d {depth} -l {max_hist_len} -b {max_branch_len}
'''

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('loadbranch_trace_dir')
    parser.add_argument('output_dir')
    parser.add_argument('-d', '--depth', type=int, default=1)
    parser.add_argument('-l', '--max-hist-len', type=int, default=4)
    parser.add_argument('-b', '--max-branch-len', type=int, default=4)
    parser.add_argument('-m', '--memory', type=int, default=32768) # Request memory in MB
    args = parser.parse_args()
    
    print('Arguments:')
    print('    Load-branch trace dir:', args.loadbranch_trace_dir)
    print('    Output dir      :', args.output_dir)
    print('    Depth           :', args.depth)
    print('    Max history len :', args.max_hist_len)
    print('    Max branch len  :', args.max_branch_len)
    print('    Allocate memory :', args.memory, 'MB')
    return args


def main():
    args = get_arguments()
    trace_paths = glob.glob(os.path.join(args.loadbranch_trace_dir, '*.*'))
    trace_names = [t.split('/')[-1].split('.')[0] for t in trace_paths]
    print('Candidate traces:', trace_names)
    
    condor_files = []
    condor_list_file = os.path.join(args.output_dir, 'condor_configs_loadbranch_corr.txt')
    
    for tr, tr_path in zip(trace_names, trace_paths):
        log_file_base = os.path.join(args.output_dir, 'logs', f'{tr}_{args.max_hist_len}h_{args.max_branch_len}b')
        condor_file = os.path.join(args.output_dir, 'condor', f'{tr}_{args.max_hist_len}h_{args.max_branch_len}b.condor')
        script_file = os.path.join(args.output_dir, 'scripts', f'{tr}_{args.max_hist_len}h_{args.max_branch_len}b.sh')
                              
        print(f'\nFiles for {tr}:')
        print(f'    load-branch trace : {tr_path}')
        print(f'    output log     : {log_file_base}.OUT')
        print(f'    error log      : {log_file_base}.ERR')
        print(f'    condor         : {condor_file}')
        print(f'    script         : {script_file}')
        print(f'    condor list    : {condor_list_file}')
        
        # Create directories
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'condor'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'scripts'), exist_ok=True)
        
        # Build condor file
        condor = generate(
            gpu=False,
            err_file=log_file_base + '.ERR',
            out_file=log_file_base + '.OUT',
            init_dir=PYPATH,
            memory=args.memory,
            exe=script_file
        )
        with open(condor_file, 'w') as f:
            print(condor, file=f)

        # Build script file
        with open(script_file, 'w') as f:
            print(SH_TEMPLATE.format(
                script_file=SCRIPT,
                pypath=PYPATH,
                loadbranch_trace=tr_path,
                depth=args.depth,
                max_hist_len=args.max_hist_len,
                max_branch_len=args.max_branch_len
            ), file=f)
        os.chmod(script_file, 0o777) # Make script executable

        condor_files.append(condor_file) # Add condor file path to the list

    # Build condor list file
    with open(condor_list_file, 'w') as f:
        for cf in condor_files:
            print(cf, file=f)


if __name__ == '__main__':
    main()
