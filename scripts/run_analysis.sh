#!/bin/bash
#SBATCH --job-name=analysis
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=1
#SBATCH --time=23:59:00
#SBATCH --output=logs/embedding.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=junkais@andrew.cmu.edu
export TMPDIR=/data/user_data/junkais/tmp
export MKL_SERVICE_FORCE_INTEL=1

mkdir -p $TMPDIR

# Aggregate results from parallel analysis splits
python /home/junkais/test/src/analysis/pipeline_analysis/aggregate_analysis_results.py \
    --input-pattern "analysis_split_*.json" \
    --results-dir "/home/junkais/test/scripts"

#python compare_simulation_stats.py --folder1 output7010 --folder2 output7009 --output temperature_filtered.txt
#python aggregate_and_plot_stats.py --folder output7009
#python aggregate_and_plot_stats.py --folder output7010
#python summarize_simulation_stats.py --folders output7009 output7010

#python argument_analysis.py --folder output7009 --output_name argument_survival_results_temperature_moderate
#python argument_analysis.py --folder output7010 --output_name argument_survival_results_temperature_none


#python compare_simulation_stats.py --folder1 output7002 --folder2 output7001 --output god_none_extreme.txt
#python embedding_creator.py output7007
#python embedding_plotter.py /data/user_data/junkais/all_simulation_vectors_free.json images /data/user_data/junkais/opinion_scores.json lk99_human_filtered
#python embedding_creator.py output7008
#python embedding_plotter.py /data/user_data/junkais/all_simulation_vectors_free.json images /data/user_data/junkais/opinion_scores.json lk99_llm_filtered

#python temp.py
#python argument_analysis.py --folder output7001 --output_name argument_survival_results_god_extreme
#python argument_analysis.py --folder output7002 --output_name argument_survival_results_god_none



#python temp.py
#python embedding_creator.py output7000
#python embedding_plotter.py /data/user_data/junkais/all_simulation_vectors_free.json images /data/user_data/junkais/opinion_scores.json god_moderate
#python embedding_creator.py output7001
#python embedding_plotter.py /data/user_data/junkais/all_simulation_vectors_free.json images /data/user_data/junkais/opinion_scores.json god_extreme
#python embedding_creator.py output7002
#python embedding_plotter.py /data/user_data/junkais/all_simulation_vectors_free.json images /data/user_data/junkais/opinion_scores.json god_none
#rm all_simulation_vectors_free.json
#rm opinion_scores.json

#python aggregate_and_plot_stats.py --folder output6001
#python aggregate_and_plot_stats.py --folder output6002
#python one_agent.py --folder "output127" --args_length 8 --num_conversations 10000 --topic water

#pro, then con

#3000 is all pro water

#This was half half
#python one_agent.py --folder "output3001" --args_length 4 --init_args 4 --num_conversations 1000 --topic water
#python one_agent.py --folder "output3002" --args_length 8 --init_args 8 --num_conversations 1000 --topic water
#python one_agent.py --folder "output3003" --args_length 12 --init_args 12 --num_conversations 1000 --topic water

#python aggregate.py --folder output5045
#python aggregate.py --folder output5046
#python aggregate.py --folder output5047
#python aggregate.py --folder output5048
#python aggregate.py --folder output5049
#python aggregate.py --folder output5050
#python stats2.py --folders output5034 output5035 output5036 output5037 output5038 output5039 output5040 output5041 output5042 output5043 output5044

#python plotter.py --folder output5045 --output god_bound_low_var.png
#python plotter.py --folder output5046 --output god_bound_high_var.png

#python plotter.py --folder output5047 --output water_bound_low_var.png
#python plotter.py --folder output5048 --output water_bound_high_var.png

#python plotter.py --folder output5049 --output drugs_bound_low_var.png
#python plotter.py --folder output5050 --output drugs_bound_high_var.png
#python plotter.py
#python plotter.py --folder output5035 --output god_extreme_var.png
#python plotter.py --folder output5042 --output god_none_var.png

#python plotter.py --folder output5042 --output god_none_mean.png
#python plotter.py --folder output5034 --output god_moderate_mean.png
#python plotter.py --folder output5035 --output god_extreme_mean.png

#python plotter.py --folder output5043 --output water_none_mean.png
#python plotter.py --folder output5036 --output water_moderate_mean.png
#python plotter.py --folder output5037 --output water_extreme_mean.png

#python plotter.py --folder output5044 --output drugs_none_mean.png
#python plotter.py --folder output5038 --output drugs_moderate_mean.png
#python plotter.py --folder output5039 --output drugs_extreme_mean.png

#python plotter.py --folder output5040 --output random1_mean.png
#python plotter.py --folder output5041 --output random2_mean.png
#python plotter.py --folder output5039 --output water_extreme_var.png

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


#python small_sim.py  --folder "output5031" --epsilon 1 --num_conversations 100 --args_length 4 --init_args 4 --topic god --initial_condition moderate --num_pairs 5
#python small_sim.py  --folder "output5016" --epsilon 1 --num_conversations 500 --args_length 4 --init_args 4 --topic god --initial_condition moderate
#python small_sim.py  --folder "output5017" --epsilon 1 --num_conversations 500 --args_length 4 --init_args 4 --topic god --initial_condition moderate
#python small_sim.py  --folder "output5018" --epsilon 1 --num_conversations 500 --args_length 4 --init_args 4 --topic god --initial_condition moderate
#python small_sim.py  --folder "output5019" --epsilon 1 --num_conversations 500 --args_length 4 --init_args 4 --topic god --initial_condition moderate
