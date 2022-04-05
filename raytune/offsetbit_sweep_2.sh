#!/bin/bash

#killall screen;

declare -A traces

#traces[astar_313B]=30
traces[bfs]=10
#traces[cc]=30
#traces[mcf_46B]=15
#traces[omnetpp_340B]=15
#traces[pr]=30
#traces[soplex_66B]=15
traces[sphinx3_2520B]=10
traces[xalancbmk_99B]=10

#cd /home/ray/voyager
for trace in "${!traces[@]}"; do
    cd "/home/ray/voyager"
    launchcmd="python ray_tune.py -b gs://voyager-tune/zhan_traces/load/${trace}.txt -c /home/ray/voyager/configs/base_fast.yaml -p 1000 -t /home/ray/voyager/configs/ray/offset_bits.yaml -e ${traces[$trace]} --base-start -n ${trace} --print-every 250 -s 12 -r";
    
    echo;
    echo "=== LAUNCHING SWEEP FOR $trace (on screen $trace) ===";
    echo $launchcmd;
    screen -AmdS $trace;
    screen -S $trace -X exec $launchcmd
done

echo;
echo "SWEEP LAUNCH COMPLETE.";