#!/bin/bash

#killall screen;

declare -A traces

traces[astar_313B]=20 #200
traces[pr]=20 #200
traces[cc]=20 #200
traces[mcf_46B]=10 #100

#cd /home/ray/voyager
for trace in "${!traces[@]}"; do
    cd "/home/ray/voyager"
    launchcmd="python ray_tune.py -b gs://voyager-tune/zhan_traces/load/${trace}.txt -c /home/ray/voyager/configs/base_fast.yaml -p 100 -t /home/ray/voyager/configs/ray/offset_bits.yaml -e ${traces[$trace]} --base-start -g 2 -n ${trace}_fast --print-every 25 -r -s 4";
    
    echo;
    echo "=== LAUNCHING SWEEP FOR $trace (on screen $trace) ===";
    echo $launchcmd;
    screen -AmdS $trace;
    screen -S $trace -X exec $launchcmd
done

echo;
echo "SWEEP LAUNCH COMPLETE.";