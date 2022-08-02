#!/bin/bash 

trap break INT
for noise in 1.5 5 10 20
do
    for run in `seq 30 $max`
    do
    echo "noise: $noise, run: $run"
    python -m GenKI.train --train_out sum_x_$noise -x $noise
    echo -e "completed\n"
    done
done
trap - INT


# trap break INT
# for run in `seq 30 $max`
# do
# echo "run: $run"
# python -m GenKI.train --train_out train_sum --do_test --generank_out genelist --r2_out r2_score
# echo -e "completed\n"
# done
# trap - INT
