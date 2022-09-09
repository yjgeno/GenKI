#!/bin/bash 

# trap break INT
# for noise in 1 2 3 4 5
# do
#     for run in `seq 30 $max`
#     do
#     echo "noise: $noise, run: $run"
#     python -m GenKI.train --dir data --train_out train_X_$noise -X $noise
#     echo -e "completed\n"
#     done
# done


# for noise in 1 2 3 4 5
# do
#     for run in `seq 30 $max`
#     do
#     echo "noise: $noise, run: $run"
#     python -m GenKI.train --dir data --train_out train_E_$noise -E $noise
#     echo -e "completed\n"
#     done
# done
# trap - INT


for noise in 0.1 0.3 0.5 0.7 0.9
do
    for run in `seq 10 $max`
    do
    echo "dropout: $noise, run: $run"
    python -m GenKI.train --dir data --train_out train_XO_$noise -XO $noise
    echo -e "completed\n"
    done
done
trap - INT


# trap break INT
# for run in `seq 30 $max`
# do
# echo "run: $run"
# python -m GenKI.train --train_out train_log --do_test --generank_out genelist --r2_out r2_score
# echo -e "completed\n"
# done
# trap - INT


# trap break INT
# for cutoff in 30 55 70 95
# do
#     trap break INT
#     for run in `seq 30 $max`
#     do
#     echo "run: $run"
#     python -m GenKI.train --train_out train_log --dir data_$cutoff
#     echo -e "completed\n"
#     done
# done
# trap - INT
