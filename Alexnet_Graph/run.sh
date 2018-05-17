CUDA_VISIBLE_DEVICES=0 python3 main.py --KEY=APY --DIR=APY_Alexnet_Graph --TA=1 --SELATT=1 --OPT=2 --maxSteps=10 --numClass=32 --numAtt=64 --lr=1e-6 > Log.txt
CUDA_VISIBLE_DEVICES=0 python3 main.py --KEY=APY --DIR=APY_Alexnet_Graph --SELATT=1 --OPT=9 --numClass=32 --numAtt=64 >> Log.txt

start=20
for number in {0..18}
do
    CUDA_VISIBLE_DEVICES=0 python3 main.py --KEY=APY --DIR=APY_Alexnet_Graph --TA=0 --SELATT=1 --OPT=2 --maxSteps=$start --numClass=32 --numAtt=64 --lr=1e-6 >> Log.txt
    CUDA_VISIBLE_DEVICES=0 python3 main.py --KEY=APY --DIR=APY_Alexnet_Graph --SELATT=1 --OPT=9 --numClass=32 --numAtt=64 >> Log.txt
    start=$(($start+10))
done