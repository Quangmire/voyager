#!/bin/python

"""Evaluate the performance of a Ray Tune sweep.
Reference: https://docs.ray.io/en/latest/tune/api_docs/analysis.html
"""
import os
import glob

import pandas as pd
import ray
from ray.tune import ExperimentAnalysis
from raytune.utils import get_eval_parser

columns = [
    'run', #'training_iteration',
    'val_loss', 'val_acc', 'val_page_acc', 'val_offset_acc', 'done',
    'offset_bits' # TODO: Keep all config/ for the full tuning sweep.
]

def main():
    args = get_eval_parser().parse_args()
    
    print(f'''
======== EVAL PARAMETERS ========
Results:
     Path        : {args.ray_results_dir}
Output:
     Path        : {args.output}
    ''')
    
    df = pd.DataFrame(columns=columns)
    
    
    for run in glob.glob(os.path.join(args.ray_results_dir, '*/')):
        analysis = ExperimentAnalysis(run)
        run_name = os.path.basename(run.rstrip('/'))
        run_df = analysis.dataframe()
        run_df.columns = run_df.columns.str.replace('config/', '')
        print(run_df.columns) # DEBUG
        
        run_df['run'] = run_name
        run_df = run_df[columns]
        run_df = run_df.dropna(axis=0)
        
        df = df.append(run_df)
        
    print(df)
    df.to_csv(args.output)
        
        
    
    
if __name__ == '__main__':
    main()
