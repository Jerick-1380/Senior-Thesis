#!/bin/bash
#SBATCH --job-name=main
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=1
#SBATCH --time=23:00:00
#SBATCH --output=logs/main.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=junkais@andrew.cmu.edu
export TMPDIR=/data/user_data/junkais/tmp
export MKL_SERVICE_FORCE_INTEL=1

mkdir -p $TMPDIR

#python test_run.py
#export NCCL_P2P_DISABLE=1
#mkdir "output"


python small_sim.py  --folder "output1000" --epsilon 1 --num_conversations 5 --args_length 3
#python small_sim.py  --folder "output2" --epsilon 1 --num_conversations 150 --args_length 5
#python small_sim.py  --folder "output3" --epsilon 1 --num_conversations 150 --args_length 8

#python small_sim.py  --folder "output4" --epsilon 1 --num_conversations 150 --args_length 8 --temperature 1.2
#python small_sim.py  --folder "output5" --epsilon 0 --num_conversations 150 --args_length 8
#python small_sim.py  --folder "output6" --epsilon 0.05 --num_conversations 150 --args_length 8
#python small_sim.py  --folder "output7" --epsilon 0.1 --num_conversations 150 --args_length 8

#python small_sim.py  --folder "output14" --epsilon 0 --num_conversations 300 --args_length 3
#python small_sim.py  --folder "output15" --epsilon 0.05 --num_conversations 300 --args_length 3
#python small_sim.py  --folder "output16" --epsilon 0.1 --num_conversations 300 --args_length 3
#python small_sim.py  --folder "output17" --epsilon 0.2 --num_conversations 300 --args_length 3
#python small_sim.py  --folder "output18" --epsilon 0.5 --num_conversations 300 --args_length 3
#python small_sim.py  --folder "output19" --epsilon 1 --num_conversations 300 --args_length 3

#python small_sim.py  --folder "output21" --epsilon 0.1 --num_conversations 150 --args_length 8
#python small_sim.py  --folder "output23" --epsilon 0.2 --num_conversations 150 --args_length 8
#python small_sim.py  --folder "output24" --epsilon 0.3 --num_conversations 150 --args_length 8
#python small_sim.py  --folder "output22" --epsilon 1 --num_conversations 150 --args_length 8


#python small_sim.py  --folder "output88" --epsilon 1 --num_conversations 500 --args_length 8 --topic temp
#python small_sim.py  --folder "output89" --epsilon 1 --num_conversations 500 --args_length 8 --topic temp
#python small_sim.py  --folder "output90" --epsilon 1 --num_conversations 500 --args_length 8 --topic temp

#python small_sim.py  --folder "output91" --epsilon 1 --num_conversations 500 --args_length 8 --topic god
#python small_sim.py  --folder "output92" --epsilon 1 --num_conversations 500 --args_length 8 --topic god
#python small_sim.py  --folder "output93" --epsilon 1 --num_conversations 500 --args_length 8 --topic god

#python small_sim.py  --folder "output94" --epsilon 1 --num_conversations 500 --args_length 8 --topic water
#python small_sim.py  --folder "output95" --epsilon 1 --num_conversations 500 --args_length 8 --topic water
#python small_sim.py  --folder "output96" --epsilon 1 --num_conversations 500 --args_length 8 --topic water

#python small_sim.py  --folder "output97" --epsilon 1 --num_conversations 500 --args_length 8 --topic american
#python small_sim.py  --folder "output98" --epsilon 1 --num_conversations 500 --args_length 8 --topic american
#python small_sim.py  --folder "output99" --epsilon 1 --num_conversations 500 --args_length 8 --topic american

#python small_sim.py  --folder "output100" --epsilon 1 --num_conversations 500 --args_length 8 --topic monarchy
#python small_sim.py  --folder "output101" --epsilon 1 --num_conversations 500 --args_length 8 --topic monarchy
#python small_sim.py  --folder "output102" --epsilon 1 --num_conversations 500 --args_length 8 --topic monarchy

#python small_sim.py  --folder "output100" --epsilon 0 --num_conversations 500 --args_length 8 --topic temp
#python small_sim.py  --folder "output101" --epsilon 0 --num_conversations 500 --args_length 8 --topic temp
#python small_sim.py  --folder "output102" --epsilon 0 --num_conversations 500 --args_length 8 --topic temp

#python small_sim.py  --folder "output103" --epsilon 1 --agents 100 --num_conversations 2000 --args_length 8 --topic temp

#python small_sim.py  --folder "output104" --epsilon 0.05 --num_conversations 500 --args_length 8 --topic temp
#python small_sim.py  --folder "output105" --epsilon 0.1 --num_conversations 500 --args_length 8 --topic temp
#python small_sim.py  --folder "output106" --epsilon 0.2 --num_conversations 500 --args_length 8 --topic temp



#python small_sim.py  --folder "output89" --epsilon 1 --num_conversations 200 --args_length 8 --topic american
#python small_sim.py  --folder "output90" --epsilon 1 --num_conversations 200 --args_length 8 --topic water
#python small_sim.py  --folder "output91" --epsilon 1 --num_conversations 200 --args_length 8 --topic monarchy




#rm -rf "output"
#[ -d "frames" ] && rm -rf "frames"
#[ -d "edgedistribution" ] && rm -rf "edgedistribution"