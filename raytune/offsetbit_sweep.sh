#!/bin/bash

#killall screen;

declare -A traces

traces[astar_313b]=200
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
    launchcmd="python ray_tune.py -b /home/ray/zhan_traces/load/${trace}.txt -p /home/ray/models/${trace} -c /home/ray/voyager/configs/base.yaml -r --tb-dir /home/ray/tensorboard/${trace} --checkpoint-every 10000 --print-every 100 --tuning-config /home/ray/voyager/configs/ray/offset_bits.yaml --epochs ${traces[$trace]} --base-start --grace-period 6 --sweep-name ${trace}";
    
    echo;
    echo "=== LAUNCHING SWEEP FOR $trace (on screen $trace) ===";
    echo $launchcmd;
    screen -AmdS $trace;
    screen -S $trace -X exec $launchcmd
done

echo;
echo "SWEEP LAUNCH COMPLETE.";