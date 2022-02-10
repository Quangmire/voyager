#!/bin/bash

#killall screen;

declare -A traces

traces[omnetpp_340B]=100
traces[soplex_66B]=100
traces[sphinx3_2520B]=80
traces[xalancbmk_99B]=70
traces[bfs]=50

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