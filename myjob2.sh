#!/bin/bash
#SBATCH --job-name=main2
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=1
#SBATCH --time=1-23:00:00
#SBATCH --output=logs/main2.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=junkais@andrew.cmu.edu
export TMPDIR=/data/user_data/junkais/tmp
export MKL_SERVICE_FORCE_INTEL=1

mkdir -p $TMPDIR2

#python test_run.py
#export NCCL_P2P_DISABLE=1
#mkdir "output"



#python one_agent.py --folder "output127" --args_length 8 --num_conversations 10000 --topic water

#pro, then con

#2007-2009,2016-2018 are half half
#python one_agent.py --folder "output_temp" --args_length 4 --init_args 4 --num_conversations 100 --topic god
#python one_agent.py --folder "output2011" --args_length 8 --init_args 8 --num_conversations 10000 --topic water
#python one_agent.py --folder "output2012" --args_length 12 --init_args 12 --num_conversations 10000 --topic water

#python one_agent.py --folder "output2019" --args_length 4 --init_args 4 --num_conversations 10000 --topic god
#python one_agent.py --folder "output2020" --args_length 8 --init_args 8 --num_conversations 10000 --topic god
#python one_agent.py --folder "output2021" --args_length 12 --init_args 12 --num_conversations 10000 --topic god


#python one_agent.py --folder "output207" --args_length 8 --init_args 8 --num_conversations 10000 --topic water
#python one_agent.py --folder "output208" --args_length 12 --init_args 12 --num_conversations 10000 --topic water

#python one_agent.py --folder "output209" --args_length 4 --init_args 4 --num_conversations 10000 --topic god
#python one_agent.py --folder "output210" --args_length 8 --init_args 8 --num_conversations 10000 --topic god
#python one_agent.py --folder "output211" --args_length 12 --init_args 12 --num_conversations 10000 --topic god
#python image_compiler.py 

#python small_sim.py  --folder "output4000" --epsilon 1 --num_conversations 500 --args_length 4 --init_args 4 --topic water --initial_condition extreme
#python small_sim.py  --folder "output4001" --epsilon 1 --num_conversations 500 --args_length 4 --init_args 4 --topic water --initial_condition extreme
#python small_sim.py  --folder "output4002" --epsilon 1 --num_conversations 500 --args_length 4 --init_args 4 --topic water --initial_condition moderate
#python small_sim.py  --folder "output4003" --epsilon 1 --num_conversations 500 --args_length 4 --init_args 4 --topic water --initial_condition moderate

python small_sim.py  --folder "output5025" --epsilon 1 --num_conversations 500 --args_length 2 --init_args 0 --topic god --initial_condition none
python small_sim.py  --folder "output5026" --epsilon 1 --num_conversations 500 --args_length 2 --init_args 0 --topic god --initial_condition none
python small_sim.py  --folder "output5027" --epsilon 1 --num_conversations 500 --args_length 2 --init_args 0 --topic god --initial_condition none
python small_sim.py  --folder "output5028" --epsilon 1 --num_conversations 500 --args_length 2 --init_args 0 --topic god --initial_condition none
python small_sim.py  --folder "output5029" --epsilon 1 --num_conversations 500 --args_length 2 --init_args 0 --topic god --initial_condition none



#python small_sim.py  --folder "output4005" --epsilon 1 --num_conversations 500 --args_length 4 --init_args 4 --topic water --initial_condition extreme
#python small_sim.py  --folder "output4006" --epsilon 1 --num_conversations 500 --args_length 4 --init_args 4 --topic water --initial_condition extreme
