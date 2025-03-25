#!/bin/bash
#SBATCH --job-name=main3
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=1
#SBATCH --time=1-23:00:00
#SBATCH --output=logs/main3.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=junkais@andrew.cmu.edu
export TMPDIR=/data/user_data/junkais/tmp
export MKL_SERVICE_FORCE_INTEL=1

mkdir -p $TMPDIR3

#python test_run.py
#export NCCL_P2P_DISABLE=1
#mkdir "output"



#python one_agent.py --folder "output127" --args_length 8 --num_conversations 10000 --topic water

#pro, then con

#3000 is all pro water

#This was half half
#python one_agent.py --folder "output3001" --args_length 4 --init_args 4 --num_conversations 1000 --topic water
#python one_agent.py --folder "output3002" --args_length 8 --init_args 8 --num_conversations 1000 --topic water
#python one_agent.py --folder "output3003" --args_length 12 --init_args 12 --num_conversations 1000 --topic water



#python one_agent.py --folder "output3011" --args_length 4 --init_args 4 --num_conversations 1000 --topic god
#python one_agent.py --folder "output3012" --args_length 8 --init_args 8 --num_conversations 1000 --topic god
#python one_agent.py --folder "output3013" --args_length 12 --init_args 12 --num_conversations 1000 --topic god

#This is all pro
#python one_agent.py --folder "output3004" --args_length 4 --init_args 4 --num_conversations 1000 --topic water
#python one_agent.py --folder "output3005" --args_length 8 --init_args 8 --num_conversations 1000 --topic water
#python one_agent.py --folder "output3006" --args_length 12 --init_args 12 --num_conversations 1000 --topic water



#python one_agent.py --folder "output3014" --args_length 4 --init_args 4 --num_conversations 1000 --topic god
#python one_agent.py --folder "output3015" --args_length 8 --init_args 8 --num_conversations 1000 --topic god
#python one_agent.py --folder "output3016" --args_length 12 --init_args 12 --num_conversations 1000 --topic god

#This is all con
#python one_agent.py --folder "output3007" --args_length 4 --init_args 4 --num_conversations 1000 --topic water
#python one_agent.py --folder "output3008" --args_length 8 --init_args 8 --num_conversations 1000 --topic water
#python one_agent.py --folder "output3009" --args_length 12 --init_args 12 --num_conversations 1000 --topic water



#python one_agent.py --folder "output3017" --args_length 4 --init_args 4 --num_conversations 1000 --topic god
#python one_agent.py --folder "output3018" --args_length 8 --init_args 8 --num_conversations 1000 --topic god
#python one_agent.py --folder "output3019" --args_length 12 --init_args 12 --num_conversations 1000 --topic god

#python small_sim.py  --folder "output5000" --epsilon 1 --num_conversations 500 --args_length 4 --init_args 4 --topic god --initial_condition extreme
#python small_sim.py  --folder "output5001" --epsilon 1 --num_conversations 500 --args_length 4 --init_args 4 --topic god --initial_condition extreme
#python small_sim.py  --folder "output5002" --epsilon 1 --num_conversations 500 --args_length 4 --init_args 4 --topic god --initial_condition moderate
#python small_sim.py  --folder "output5003" --epsilon 1 --num_conversations 500 --args_length 4 --init_args 4 --topic god --initial_condition moderate

#python small_sim.py  --folder "output4007" --epsilon 1 --num_conversations 500 --args_length 4 --init_args 4 --topic water --initial_condition moderate
#python small_sim.py  --folder "output4008" --epsilon 1 --num_conversations 500 --args_length 4 --init_args 4 --topic water --initial_condition moderate
#python small_sim.py  --folder "output4009" --epsilon 1 --num_conversations 500 --args_length 4 --init_args 4 --topic water --initial_condition moderate


python small_sim.py  --folder "output5015" --epsilon 1 --num_conversations 500 --args_length 4 --init_args 4 --topic god --initial_condition moderate
python small_sim.py  --folder "output5016" --epsilon 1 --num_conversations 500 --args_length 4 --init_args 4 --topic god --initial_condition moderate
python small_sim.py  --folder "output5017" --epsilon 1 --num_conversations 500 --args_length 4 --init_args 4 --topic god --initial_condition moderate
python small_sim.py  --folder "output5018" --epsilon 1 --num_conversations 500 --args_length 4 --init_args 4 --topic god --initial_condition moderate
python small_sim.py  --folder "output5019" --epsilon 1 --num_conversations 500 --args_length 4 --init_args 4 --topic god --initial_condition moderate

#python image_compiler.py