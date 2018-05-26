#!/usr/bin/env bash
for trial in {0..9}
do
    CUDA_VISIBLE_DEVICES=0 python3 main.py --KEY=APY --DIR=APY_Alexnet_Graph_$trial --TA=1 --SELATT=1 --OPT=2 --maxSteps=10 --HEADER=1 --numClass=32 --numAtt=64 --lr=1e-6 --SEED=$trial > Log$trial.txt
    CUDA_VISIBLE_DEVICES=0 python3 main.py --KEY=APY --DIR=APY_Alexnet_Graph_$trial --TC=1 --SELATT=1 --OPT=3 --maxSteps=50000 --numClass=32 --numAtt=64 --lr=1e-4 --SEED=$trial >> Log$trial.txt
    CUDA_VISIBLE_DEVICES=0 python3 main.py --KEY=APY --DIR=APY_Alexnet_Graph_$trial --SELATT=1 --OPT=9 --numClass=32 --numAtt=64 --SEED=$trial >> Log$trial.txt

    start=20
    for number in {0..18}
    do
        CUDA_VISIBLE_DEVICES=0 python3 main.py --KEY=APY --DIR=APY_Alexnet_Graph_$trial --TA=0 --SELATT=1 --OPT=2 --maxSteps=$start --numClass=32 --numAtt=64 --lr=1e-6 --SEED=$trial >> Log$trial.txt
        CUDA_VISIBLE_DEVICES=0 python3 main.py --KEY=APY --DIR=APY_Alexnet_Graph_$trial --SELATT=1 --OPT=9 --numClass=32 --numAtt=64 --SEED=$trial >> Log$trial.txt
        start=$(($start+10))
    done
done

