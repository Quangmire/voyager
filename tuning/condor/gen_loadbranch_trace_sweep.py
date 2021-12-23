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
SCRIPT = '/u/cmolder/GitHub/voyager-analysis/corr/loadbranch_trace.py'
PYPATH = '/u/cmolder/GitHub/voyager-analysis/'

SH_TEMPLATE = '''#!/bin/bash
source /u/cmolder/miniconda3/etc/profile.d/conda.sh
conda activate tensorflow
cd {pypath}
export PYTHONPATH=.
python3 -u {script_file} {champsim_trace} {load_trace} \\
    {n_branches} -o {output_trace}
'''

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('champsim_trace_dir')
    parser.add_argument('load_trace_dir')
    parser.add_argument('n_branches', type=int)
    parser.add_argument('output_trace_dir')
    parser.add_argument('-i', '--max-inst', default=None, type=int)
    parser.add_argument('-e', '--ext', default='txt', type=str)
    args = parser.parse_args()
    
    print('Arguments:')
    print('    ChampSim trace dir :', args.champsim_trace_dir)
    print('    Load trace dir     :', args.load_trace_dir)
    print('    Num branches       :', args.n_branches)
    print('    Output trace dir   :', args.output_trace_dir)
    print('    Saving as          :', f'.{args.ext}')
    return args


def main():
    args = get_arguments()
    
    csim_traces = glob.glob(os.path.join(args.champsim_trace_dir, '*.*'))
    load_traces = glob.glob(os.path.join(args.load_trace_dir, '*.*'))
    
    csim_trace_names = set(ct.split('/')[-1].split('.')[0] for ct in csim_traces)
    load_trace_names = set(lt.split('/')[-1].split('.')[0] for lt in load_traces)
    
    traces = csim_trace_names.intersection(load_trace_names)
    print('Candidate traces:', traces)
    
    condor_files = []
    condor_list_file = os.path.join(args.output_trace_dir, 'condor', 'condor_configs_loadbranch.txt')
    
    for tr in traces:
        csim_tr = glob.glob(f'{args.champsim_trace_dir}/{tr}*')[0]
        load_tr = glob.glob(f'{args.load_trace_dir}/{tr}*')[0]
        out_tr = os.path.join(args.output_trace_dir, f'{tr}_{args.n_branches}.{args.ext}')
        
        log_file_base = os.path.join(args.output_trace_dir, 'condor', 'logs', f'{tr}_{args.n_branches}')
        condor_file = os.path.join(args.output_trace_dir, 'condor', 'condor', f'{tr}_{args.n_branches}.condor')
        script_file = os.path.join(args.output_trace_dir, 'condor', 'scripts', f'{tr}_{args.n_branches}.sh')
                              
        print(f'\nFiles for {tr}:')
        print(f'    ChampSim trace : {csim_tr}')
        print(f'    load trace     : {load_tr}')
        print(f'    output trace   : {out_tr}')
        print(f'    output log     : {log_file_base}.OUT')
        print(f'    error log      : {log_file_base}.ERR')
        print(f'    condor         : {condor_file}')
        print(f'    script         : {script_file}')
        print(f'    condor list    : {condor_list_file}')
        
        # Create directories
        os.makedirs(args.output_trace_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_trace_dir, 'condor'), exist_ok=True)
        os.makedirs(os.path.join(args.output_trace_dir, 'condor', 'logs'), exist_ok=True)
        os.makedirs(os.path.join(args.output_trace_dir, 'condor', 'condor'), exist_ok=True)
        os.makedirs(os.path.join(args.output_trace_dir, 'condor', 'scripts'), exist_ok=True)
        
        # Build condor file
        condor = generate(
            gpu=False,
            err_file=log_file_base + '.ERR',
            out_file=log_file_base + '.OUT',
            init_dir=PYPATH,
            exe=script_file
        )
        with open(condor_file, 'w') as f:
            print(condor, file=f)

        # Build script file
        with open(script_file, 'w') as f:
            print(SH_TEMPLATE.format(
                script_file=SCRIPT,
                pypath=PYPATH,
                champsim_trace=csim_tr,
                load_trace=load_tr,
                n_branches=args.n_branches,
                output_trace=out_tr
            ), file=f)
        os.chmod(script_file, 0o777) # Make script executable

        condor_files.append(condor_file) # Add condor file path to the list

    # Build condor list file
    with open(condor_list_file, 'w') as f:
        for cf in condor_files:
            print(cf, file=f)


if __name__ == '__main__':
    main()
