#!/bin/bash

#killall screen;

declare -A traces

traces[astar_313B]=200
traces[pr]=200
traces[cc]=200
traces[mcf_46B]=100

#cd /home/ray/voyager
for trace in "${!traces[@]}"; do
    cd "/home/ray/voyager"
    launchcmd="python ray_tune.py -b gs://voyager-tune/zhan_traces/load/${trace}.txt -c /home/ray/voyager/configs/base_fast.yaml -p 50 -t /home/ray/voyager/configs/ray/offset_bits.yaml -e ${traces[$trace]} --base-start -g 1 -n offset_bits_fast_${trace} --print-every 10 -s 4";
    
    echo;
    echo "=== LAUNCHING SWEEP FOR $trace (on screen $trace) ===";
    echo $launchcmd;
    screen -AmdS $trace;
    screen -S $trace -X exec $launchcmd
done

echo;
echo "SWEEP LAUNCH COMPLETE.";