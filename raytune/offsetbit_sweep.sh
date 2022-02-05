#!/bin/bash

#killall screen;

declare -A traces

traces[astar_313B]=200
traces[bfs]=50
traces[cc]=200
traces[mcf_46B]=100
traces[omnetpp_340B]=100
traces[pr]=200
traces[soplex_66B]=100
traces[sphinx3_2520B]=80
traces[xalancbmk_99B]=70

#cd /home/ray/voyager
for trace in "${!traces[@]}"; do
    cd "/home/ray/voyager"
    launchcmd="python ray_tune.py -b gs://voyager-tune/zhan_traces/load/${trace}.txt -c /home/ray/voyager/configs/base.yaml -p -t /home/ray/voyager/configs/ray/offset_bits.yaml -e ${traces[$trace]} --base-start -g 6 -n ${trace}";
    
    echo;
    echo "=== LAUNCHING SWEEP FOR $trace (on screen $trace) ===";
    echo $launchcmd;
    screen -AmdS $trace;
    screen -S $trace -X exec $launchcmd
done

echo;
echo "SWEEP LAUNCH COMPLETE.";