#!/bin/bash

#killall screen;

declare -A traces

traces[omnetpp_340B]=10 #100
traces[soplex_66B]=10 #100
traces[sphinx3_2520B]=8 #80
traces[xalancbmk_99B]=7 #70
traces[bfs]=5 #50

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