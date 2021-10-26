"""This code takes a series of traces, and creates a series of
Condor scripts to evaluate them on the BASELINE prefetchers (e.g. ISB, BO)
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
import argparse
import glob

# generate - Used for template for condor submit scripts
from gen_utils import generate

# Template for bash script
SH_TEMPLATE = '''#!/bin/bash
source /u/cmolder/miniconda3/etc/profile.d/conda.sh
conda activate tensorflow
cd {champsim_dir}
python3 -u {script_file} run {champsim_trace_file} \\
        --results {results_dir} \\
        --num-instructions {num_instructions} \\
        --num-prefetch-warmup-instructions {num_warmup_instructions} \\
        --stat-printing-period {stat_printing_period}
'''
def get_baseline_parser():
    """Arugement parser for *BASELINE* ChampSim configs. Instead
    of using a config file, you supply a variable length of ChampSim 
    (NOT LOAD) trace files."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'champsim_trace_dir', 
        help='Directory to ChampSim (not load) traces, inside can be as many as you want.', 
    )
    parser.add_argument(
        '--base-dir',
        help='Directory to place results.',
        default='/scratch/cluster/cmolder/base_prefetchers/'
    )
    parser.add_argument(
        '--champsim-dir',
        help='Directory to ChampSim.',
        default='/u/cmolder/GitHub/ChampSim/'
    )
    parser.add_argument(
        '--warmup-pct',
        help='Percentage of the trace to warmup instructions in percent (default 90)',
        default=90
    )
    parser.add_argument(
        '--n-heartbeat',
        help='Heartbeat interval, in millions of instructions in millions (default 1m)',
        default=1
    )
    return parser


def main():
    parser = get_baseline_parser()
    args = parser.parse_args()
    print(args)

    trace_path_dir = args.champsim_trace_dir
    trace_paths = glob.glob(os.path.join(trace_path_dir, '*/*.xz'))
    print('Generating configurations for these traces:', trace_paths)

    # Track condor files generated so they can be batch launched later 
    # (paths saved line-by-line into the same file)
    condor_files = []
    base_dir = args.base_dir

    # For each trace, generate each permuted configuration and its script.
    for tr_path in trace_paths:
        # Splice .txt from file name
        tr_path = tr_path.replace('.txt', '.trace')
        tr = tr_path.split('/')[-1].rstrip('.xz')
        print(tr)

        # Setup initial output directories/files per experiment
        log_file_base = os.path.join(base_dir, 'logs', tr)
        condor_file = os.path.join(base_dir, 'condor', f'{tr}.condor')
        script_file = os.path.join(base_dir, 'scripts', f'{tr}.sh')
        results_dir = os.path.join(base_dir, 'champsim_results')
        
        print(f'\nFiles for {tr}:')
        print(f'    output log  : {log_file_base}.OUT')
        print(f'    error log   : {log_file_base}.ERR')
        print(f'    condor      : {condor_file}')
        print(f'    script      : {script_file}')
        print(f'    results dir : {results_dir}')

        # Create directories
        os.makedirs(os.path.join(base_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'condor'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'scripts'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, results_dir), exist_ok=True)

        # Build condor file
        condor = generate(
            gpu=False,
            err_file=log_file_base + '.ERR',
            out_file=log_file_base + '.OUT',
            init_dir=args.champsim_dir,
            exe=script_file
        )
        with open(condor_file, 'w') as f:
            print(condor, file=f)

        # Determine number of warmup instructions, and total instructions
        # Spec 06/17: 500m total
        # gap       : 300m totat,
        # warmup    : if online, 0
        #           : if offline, total * (train_split + valid_split)
        #             - prefetch trace covers last testing x% (if offline).
        num_inst   = 500 if 'spec' in tr_path else 300 # In millions
        num_warmup = int(round(num_inst * (args.warmup_pct / 100))) # First (train+valid)% go to warmup.
        stat_interval = args.n_heartbeat

        print(f'ChampSim simulation parameters for {tr}:')
        print(f'    champsim path  : {args.champsim_dir}')
        print(f'    results dir    : {results_dir}')
        print(f'    # instructions : {num_inst} million')
        print(f'    # warmup insts : {num_warmup} million')
        print(f'    stat intervals : {stat_interval} million')

        # Build script file
        with open(script_file, 'w') as f:
            print(SH_TEMPLATE.format(
                champsim_dir=args.champsim_dir,
                script_file='ml_prefetch_sim.py',
                champsim_trace_file=tr_path,
                results_dir=results_dir,
                num_instructions=num_inst,
                num_warmup_instructions=num_warmup,
                stat_printing_period=stat_interval
            ), file=f)
        
        # Make script executable
        os.chmod(script_file, 0o777)

        # Add condor file to the list
        condor_files.append(condor_file)

    print(f'\nCondor file list : {os.path.join(base_dir, "condor_configs_champsim.txt")}')
    with open(
       os.path.join(base_dir, 'condor_configs_champsim.txt'), 'w'
    ) as f:
       for cf in condor_files:
           print(cf, file=f)


if __name__ == '__main__':
    main()
