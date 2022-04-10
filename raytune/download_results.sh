#!/bin/bash

# Script to download results from a tuning sweep on GCP. Change <gcp_dir>
# and <out_dir> to reflect where the results are / where you want to
# save them.
#
# Author: Carson Molder

gcp_dir="gs://voyager-tune/checkpoints/offset_bits"
out_dir="/home/ray/ray_results_from_gcp"

mkdir -p "$out_dir"
readarray -t runs < <(gsutil -m ls "$gcp_dir")

# Copy experiment state/results (without models)
for run in "${runs[@]}"
do
    run_name=`basename "$run"`
    out_run_dir="$out_dir/$run_name"
    
    mkdir -p "$out_run_dir"  
    gsutil -m cp -r "$gcp_dir/$run_name/*.json" "$out_run_dir"
    gsutil -m cp -r "$gcp_dir/$run_name/*.pkl" "$out_run_dir"
    gsutil -m cp -r "$gcp_dir/$run_name/*_args*/" "$out_run_dir"
done
