import json
import asyncio
from datetime import datetime
from typing import List, Dict, Optional
import statistics
from tqdm import tqdm
import time
import argparse
import os
import sys
import math
from random import sample

# Add the project root to Python path to find helpers
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from helpers.model import Llama, GPT4o
from helpers.graph import Grapher, Writer  
from helpers.bots import Agent
from helpers.data import ALL_NAMES, PERSONAS
from helpers.conversation import ConversationCreator
from helpers.advanced_prompts import (
    PredictionPrompts, 
    ArgumentPrompts,
    StrengthCalculationHelpers
)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Parallel Multi-method Prediction Analysis with Conversational Agents')
    
    # Predictor Method Selection
    predictor_group = parser.add_argument_group('Prediction Methods', 'Choose which prediction approaches to run')
    predictor_group.add_argument('--baseline', action='store_true', help='Use community baseline predictions from historical data')
    predictor_group.add_argument('--basic', action='store_true', help='Use basic LLM predictions without additional context')
    predictor_group.add_argument('--argument-based', action='store_true', help='Generate arguments first, then make informed predictions')
    predictor_group.add_argument('--adaptive-conv', action='store_true', help='Use adaptive conversations between agents to develop perspectives')
    predictor_group.add_argument('--extended-conv', action='store_true', help='Use extended adaptive conversations (more overall rounds for deeper perspective development)')
    
    # Parallel Processing Configuration
    parallel_group = parser.add_argument_group('Parallel Processing', 'Configuration for parallel dataset processing')
    parallel_group.add_argument('--split-id', type=int, default=0, help='Which dataset split to process (0-based index)')
    parallel_group.add_argument('--total-splits', type=int, default=8, help='Total number of dataset splits')
    parallel_group.add_argument('--output-prefix', type=str, default='analysis', help='Prefix for output files')
    
    # Model Configuration
    model_group = parser.add_argument_group('Model Configuration', 'LLM model settings')
    model_group.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature for model generation (0.0-1.0, default: 0.7)')
    model_group.add_argument('--model-url', type=str, default="http://babel-5-31:9000/v1", help='Base URL for the LLM API endpoint')
    
    # Method-Specific Parameters
    params_group = parser.add_argument_group('Method Parameters', 'Configure specific prediction method parameters')
    params_group.add_argument('--args-per-trial', type=int, default=4, help='Number of arguments to generate per prediction trial (default: 4)')
    params_group.add_argument('--overall-rounds', type=int, default=4, help='Number of overall conversation rounds between agent groups (default: 4)')
    params_group.add_argument('--pairwise-exchanges', type=int, default=6, help='Number of back-and-forth exchanges between each pair of agents (default: 6)')
    params_group.add_argument('--extended-rounds', type=int, default=4, help='Number of overall rounds for extended conversations (default: 8)')
    
    return parser.parse_args()

class BatchConversationalAgent:
    """Optimized wrapper around Agent class for batch prediction experiments."""
    
    def __init__(self, agent_id: int, llama_model, topic: str = "general"):
        self.agent_id = agent_id
        self.model = llama_model
        self.topic = topic
        self.arguments = []
        
        # Create a minimal Agent instance for advanced capabilities
        self.agent = Agent(
            name=f"agent_{agent_id}",
            persona="You are a thoughtful conversationalist.",
            model=llama_model,
            topic=topic,
            claims={"pro": [], "con": [], "connector": ""},
            init_args=[],
            memory_length=5,
            args_length=10,
            remove_irrelevant=False
        )
    
    async def batch_start_conversations(self, topics: List[str], partner_ids: List[int]) -> List[str]:
        """Start multiple conversations in batch."""
        tasks = []
        for topic, partner_id in zip(topics, partner_ids):
            tasks.append(self.agent.start_conversation(topic))
        return await asyncio.gather(*tasks)
    
    async def batch_continue_conversations(self, topics: List[str], histories: List[str], partner_ids: List[int]) -> List[str]:
        """Continue multiple conversations in batch."""
        tasks = []
        for topic, history, partner_id in zip(topics, histories, partner_ids):
            tasks.append(self.agent.continue_conversation(topic, history))
        return await asyncio.gather(*tasks)
    
    async def batch_predict_with_arguments(self, questions: List[str]) -> List[float]:
        """Make multiple predictions using the agent's collected arguments in batch."""
        if not self.arguments:
            return [0.5] * len(questions)
        
        # Update the agent's arguments with our collected ones
        self.agent.args = self.arguments.copy()
        
        # Use the enhanced prediction method in batch
        tasks = [self.agent.predict_with_arguments(question) for question in questions]
        return await asyncio.gather(*tasks)

class ParallelPredictionAnalyzer:
    def __init__(self, llama_model, args, run_baseline=True, run_basic=True, run_argument=True, run_conversational=True, run_extended=True):
        self.model = llama_model
        self.args = args
        self.run_baseline = run_baseline
        self.run_basic = run_basic
        self.run_argument = run_argument
        self.run_conversational = run_conversational
        self.run_extended = run_extended
        
        # Validate that at least one predictor is enabled
        if not any([run_baseline, run_basic, run_argument, run_conversational, run_extended]):
            raise ValueError("At least one predictor type must be enabled")
    
    def load_and_split_questions(self, filepath: str, split_id: int, total_splits: int) -> List[Dict]:
        """Load questions from JSON file, filter by date criteria, and return only the assigned split."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        filtered_questions = []
        for item in data:
            # Parse dates
            try:
                date_begin = datetime.strptime(item['date_begin'], '%Y-%m-%d')
                date_resolve = datetime.strptime(item['date_resolve_at'], '%Y-%m-%d')
                
                # Filter: began before 2024, resolved in 2024
                if (date_begin.year < 2024 and 
                    date_resolve.year == 2024 and 
                    item['is_resolved'] and
                    item['question_type'] == 'binary'):
                    filtered_questions.append(item)
            except (ValueError, KeyError):
                continue
        
        # Split the dataset
        total_questions = len(filtered_questions)
        questions_per_split = math.ceil(total_questions / total_splits)
        start_idx = split_id * questions_per_split
        end_idx = min((split_id + 1) * questions_per_split, total_questions)
        
        split_questions = filtered_questions[start_idx:end_idx]
        
        return split_questions
    
    def extract_baseline_prediction(self, question_data: Dict) -> Optional[float]:
        """Extract the last community prediction before 2024."""
        try:
            # Parse the community_predictions string
            community_predictions_str = question_data.get('community_predictions', '[]')
            if not community_predictions_str or community_predictions_str == '[]':
                return None
            
            # Parse the JSON string
            predictions = json.loads(community_predictions_str)
            
            if not predictions:
                return None
            
            # Find the last prediction before 2024
            last_prediction_before_2024 = None
            cutoff_date = datetime(2024, 1, 1)
            
            for prediction in predictions:
                if len(prediction) >= 2:
                    date_str, prob = prediction[0], prediction[1]
                    try:
                        pred_date = datetime.strptime(date_str, '%Y-%m-%d')
                        if pred_date < cutoff_date:
                            last_prediction_before_2024 = prob
                    except (ValueError, TypeError):
                        continue
            
            return last_prediction_before_2024
            
        except (json.JSONDecodeError, KeyError, TypeError, IndexError) as e:
            return None
    
    async def batch_calculate_basic_strengths(self, questions: List[str]) -> List[float]:
        """Calculate basic strengths for multiple questions in batch."""
        # Use improved basic prediction prompts
        prompts = [PredictionPrompts.basic_prediction_prompt(question) for question in questions]
        
        # Get probabilities for all prompts in parallel
        tasks = [self.model.get_probabilities(prompt, "") for prompt in prompts]
        
        try:
            prob_results = await asyncio.gather(*tasks)
            strengths = []
            
            for probs_dict in prob_results:
                yes_prob, no_prob = StrengthCalculationHelpers.extract_yes_no_probabilities(probs_dict)
                strength = StrengthCalculationHelpers.calculate_strength_from_probabilities(yes_prob, no_prob)
                strengths.append(strength)
            
            return strengths
            
        except Exception as e:
            return [0.5] * len(questions)
    
    async def batch_generate_arguments(self, questions: List[str]) -> List[List[str]]:
        """Generate arguments for multiple questions in batch."""
        all_argument_tasks = []
        
        # Create tasks for all questions and all arguments per question
        for question in questions:
            argument_prompt = ArgumentPrompts.generate_argument_prompt(question)
            question_tasks = [self.model.generate(argument_prompt, [], []) for _ in range(self.args.args_per_trial)]
            all_argument_tasks.extend(question_tasks)
        
        # Execute all argument generation tasks in parallel
        all_arguments = await asyncio.gather(*all_argument_tasks)
        
        # Group arguments back by question
        arguments_per_question = []
        args_per_trial = self.args.args_per_trial
        
        for i in range(0, len(all_arguments), args_per_trial):
            question_arguments = all_arguments[i:i+args_per_trial]
            
            # Clean arguments
            clean_arguments = []
            for arg in question_arguments:
                cleaned = arg.strip()
                if "Argument:" in cleaned:
                    cleaned = cleaned.split("Argument:")[-1].strip()
                if cleaned and cleaned != "0":
                    clean_arguments.append(cleaned)
            
            arguments_per_question.append(clean_arguments)
        
        return arguments_per_question
    
    async def batch_calculate_argument_based_strengths(self, questions: List[str]) -> List[float]:
        """Calculate argument-based strengths for multiple questions, using batching within each trial."""
        question_strengths = []
        
        # Process all questions in parallel for each trial
        for trial_num in range(20):  # 20 trials as in original
            
            # Generate arguments for all questions in this trial
            arguments_per_question = await self.batch_generate_arguments(questions)
            
            # Create informed prompts for all questions
            informed_prompts = []
            valid_question_indices = []
            
            for i, (question, arguments) in enumerate(zip(questions, arguments_per_question)):
                if arguments:
                    informed_prompt = PredictionPrompts.predict_with_arguments_prompt(question, arguments)
                    informed_prompts.append(informed_prompt)
                    valid_question_indices.append(i)
            
            # Get probabilities for all valid prompts in parallel
            if informed_prompts:
                try:
                    prob_tasks = [self.model.get_probabilities(prompt, "") for prompt in informed_prompts]
                    prob_results = await asyncio.gather(*prob_tasks)
                    
                    # Calculate strengths
                    trial_strengths = [None] * len(questions)
                    for idx, probs_dict in zip(valid_question_indices, prob_results):
                        yes_prob, no_prob = StrengthCalculationHelpers.extract_yes_no_probabilities(probs_dict)
                        strength = StrengthCalculationHelpers.calculate_strength_from_probabilities(yes_prob, no_prob)
                        if trial_strengths[idx] is None:
                            trial_strengths[idx] = []
                        trial_strengths[idx] = strength
                    
                    # Initialize question_strengths on first trial
                    if trial_num == 0:
                        question_strengths = [[] for _ in range(len(questions))]
                    
                    # Add trial results to question strengths
                    for i, strength in enumerate(trial_strengths):
                        if strength is not None:
                            question_strengths[i].append(strength)
                            
                except Exception as e:
                    continue
        
        # Average strengths across trials for each question
        final_strengths = []
        for strengths in question_strengths:
            if strengths:
                final_strengths.append(statistics.mean(strengths))
            else:
                final_strengths.append(0.5)
        
        return final_strengths
    
    async def batch_calculate_conversational_strengths(self, questions: List[str]) -> List[float]:
        """Calculate conversational strengths using batched agent interactions."""
        
        # Initialize 20 agents (will be reused across questions for efficiency)
        agents = [BatchConversationalAgent(i, self.model, topic="general") for i in range(20)]
        question_strengths = []
        
        # Process questions in smaller batches to manage memory
        batch_size = 5  # Process 5 questions at a time
        for batch_start in range(0, len(questions), batch_size):
            batch_end = min(batch_start + batch_size, len(questions))
            batch_questions = questions[batch_start:batch_end]
            
            batch_strengths = []
            for question in batch_questions:
                # Reset agents for each question
                for agent in agents:
                    agent.arguments = []
                    agent.topic = question
                
                # Run conversation rounds
                for round_num in range(self.args.overall_rounds):
                    await self.batch_run_conversation_round(agents, question, self.args.pairwise_exchanges)
                    await asyncio.sleep(0.05)  # Small delay to prevent API overwhelming
                
                # Get predictions from all agents for this question
                predictions = await agents[0].batch_predict_with_arguments([question] * len(agents))
                
                # Calculate average prediction
                valid_predictions = [p for p in predictions if p is not None]
                if valid_predictions:
                    batch_strengths.append(statistics.mean(valid_predictions))
                else:
                    batch_strengths.append(0.5)
            
            question_strengths.extend(batch_strengths)
        
        return question_strengths
    
    async def batch_run_conversation_round(self, agents: List[BatchConversationalAgent], topic: str, 
                                         conversation_length: int = 6) -> None:
        """Run one round of conversations with all agents paired up, using batching."""
        import random
        shuffled_agents = agents.copy()
        random.shuffle(shuffled_agents)
        pairs = [(shuffled_agents[i], shuffled_agents[i+1]) for i in range(0, len(shuffled_agents), 2)]
        
        # Run all paired conversations in parallel
        conversation_tasks = []
        for agent1, agent2 in pairs:
            task = self.batch_run_paired_conversation(agent1, agent2, topic, conversation_length)
            conversation_tasks.append(task)
        
        await asyncio.gather(*conversation_tasks)
    
    async def batch_run_paired_conversation(self, agent1: BatchConversationalAgent, agent2: BatchConversationalAgent, 
                                          topic: str, conversation_length: int) -> None:
        """Run a conversation between two agents using batching where possible."""
        conversation_history = []
        
        # Agent1 starts the conversation
        response1 = await agent1.agent.start_conversation(topic, agent2.agent_id)
        conversation_history.append(f"Agent {agent1.agent_id}: {response1}")
        
        # Continue conversation for specified length
        current_speaker = agent2
        other_speaker = agent1
        
        for turn in range(conversation_length - 1):
            history_text = "\n".join(conversation_history)
            response = await current_speaker.agent.continue_conversation(topic, history_text, other_speaker.agent_id)
            conversation_history.append(f"Agent {current_speaker.agent_id}: {response}")
            
            current_speaker, other_speaker = other_speaker, current_speaker
        
        # Extract perspectives from both agents in parallel
        history_text = "\n".join(conversation_history)
        
        # Store the conversation in agents' history temporarily for perspective extraction
        lines = history_text.split('\n')
        for agent in [agent1, agent2]:
            agent.agent.user_history = []
            agent.agent.model_history = []
            for i, line in enumerate(lines):
                if line.strip():
                    if i % 2 == 0:  # User lines
                        agent.agent.user_history.append(line.strip())
                    else:  # Model lines  
                        agent.agent.model_history.append(line.strip())
        
        # Use the agent's improved add_perspective method
        perspective1_task = agent1.agent.add_perspective()
        perspective2_task = agent2.agent.add_perspective()
        
        await asyncio.gather(perspective1_task, perspective2_task)
        
        # Get the new perspectives if they were added
        perspective1 = agent1.agent.args[-1] if agent1.agent.args else ""
        perspective2 = agent2.agent.args[-1] if agent2.agent.args else ""
        
        if perspective1:
            agent1.arguments.append(perspective1)
        if perspective2:
            agent2.arguments.append(perspective2)
    
    def calculate_brier_score(self, predicted_prob: float, actual_outcome: int) -> float:
        """Calculate Brier score for a single prediction."""
        return (predicted_prob - actual_outcome) ** 2
    
    async def analyze_split_questions(self, questions: List[Dict]) -> Dict:
        """Analyze a split of questions with maximum parallelization."""
        results = []
        brier_scores_baseline = []
        brier_scores_basic = []
        brier_scores_argument = []
        brier_scores_conversational = []
        brier_scores_extended = []
        
        start_time = time.time()
        
        enabled_predictors = []
        if self.run_baseline:
            enabled_predictors.append("Baseline")
        if self.run_basic:
            enabled_predictors.append("Basic")
        if self.run_argument:
            enabled_predictors.append("Argument-based")
        if self.run_conversational:
            enabled_predictors.append("Conversational")
        if self.run_extended:
            enabled_predictors.append("Extended-Conv")
        
        
        # Extract question texts for batch processing
        question_texts = [q['question'] for q in questions]
        
        # Initialize result dictionaries
        for i, question_data in enumerate(questions):
            result = {
                'question': question_data['question'],
                'resolution': question_data['resolution'],
                'background': question_data.get('background', ''),
                'date_begin': question_data['date_begin'],
                'date_resolve_at': question_data['date_resolve_at']
            }
            
            # Handle baseline predictions (not batchable)
            if self.run_baseline:
                baseline_prob = self.extract_baseline_prediction(question_data)
                if baseline_prob is not None:
                    result['baseline_strength'] = baseline_prob
                    brier_score = self.calculate_brier_score(baseline_prob, question_data['resolution'])
                    result['baseline_brier'] = brier_score
                    brier_scores_baseline.append(brier_score)
                else:
                    result['baseline_strength'] = None
                    result['baseline_brier'] = None
            
            results.append(result)
        
        # Batch process different predictor types
        predictor_results = {}
        
        if self.run_basic:
            basic_strengths = await self.batch_calculate_basic_strengths(question_texts)
            predictor_results['basic'] = basic_strengths
        
        if self.run_argument:
            argument_strengths = await self.batch_calculate_argument_based_strengths(question_texts)
            predictor_results['argument'] = argument_strengths
        
        if self.run_conversational:
            conversational_strengths = await self.batch_calculate_conversational_strengths(question_texts)
            predictor_results['conversational'] = conversational_strengths
        
        if self.run_extended:
            # For extended, reuse conversational logic but with more rounds
            old_rounds = self.args.overall_rounds
            self.args.overall_rounds += self.args.extended_rounds
            extended_strengths = await self.batch_calculate_conversational_strengths(question_texts)
            self.args.overall_rounds = old_rounds  # Reset
            predictor_results['extended'] = extended_strengths
        
        # Add batch results to individual question results
        for i, result in enumerate(results):
            question_data = questions[i]
            
            for predictor_name, strengths in predictor_results.items():
                if i < len(strengths):
                    strength = strengths[i]
                    result[f'{predictor_name}_strength'] = strength
                    brier_score = self.calculate_brier_score(strength, question_data['resolution'])
                    result[f'{predictor_name}_brier'] = brier_score
                    
                    # Add to appropriate brier score list
                    if predictor_name == 'basic':
                        brier_scores_basic.append(brier_score)
                    elif predictor_name == 'argument':
                        brier_scores_argument.append(brier_score)
                    elif predictor_name == 'conversational':
                        brier_scores_conversational.append(brier_score)
                    elif predictor_name == 'extended':
                        brier_scores_extended.append(brier_score)
        
        # Calculate overall metrics
        mean_brier_baseline = statistics.mean(brier_scores_baseline) if brier_scores_baseline else None
        mean_brier_basic = statistics.mean(brier_scores_basic) if brier_scores_basic else None
        mean_brier_argument = statistics.mean(brier_scores_argument) if brier_scores_argument else None
        mean_brier_conversational = statistics.mean(brier_scores_conversational) if brier_scores_conversational else None
        mean_brier_extended = statistics.mean(brier_scores_extended) if brier_scores_extended else None
        
        total_time = time.time() - start_time
        
        return {
            'split_id': self.args.split_id,
            'total_questions': len(questions),
            'mean_brier_baseline': mean_brier_baseline,
            'mean_brier_basic': mean_brier_basic,
            'mean_brier_argument': mean_brier_argument,
            'mean_brier_conversational': mean_brier_conversational,
            'mean_brier_extended': mean_brier_extended,
            'enabled_predictors': {
                'baseline': self.run_baseline,
                'basic': self.run_basic,
                'argument': self.run_argument,
                'conversational': self.run_conversational,
                'extended': self.run_extended
            },
            'configuration': {
                'args_per_trial': self.args.args_per_trial,
                'overall_rounds': self.args.overall_rounds,
                'pairwise_exchanges': self.args.pairwise_exchanges,
                'extended_rounds': self.args.extended_rounds,
                'model_temperature': self.args.temperature,
                'model_url': self.args.model_url
            },
            'results': results,
            'total_time_minutes': total_time / 60,
            'baseline_questions_available': len(brier_scores_baseline) if brier_scores_baseline else 0
        }
    
    def print_split_summary(self, analysis_results: Dict):
        """Print summary of split analysis results."""
        print("\n" + "="*80)
        print(f"📊 SPLIT {analysis_results['split_id']} ANALYSIS SUMMARY")
        print("="*80)
        print(f"Questions in this split: {analysis_results['total_questions']}")
        print(f"Split analysis time: {analysis_results['total_time_minutes']:.2f} minutes")
        print(f"Average time per question: {analysis_results['total_time_minutes']/analysis_results['total_questions']:.2f} minutes")
        
        # Print configuration
        config = analysis_results.get('configuration', {})
        print(f"\n🔧 Configuration:")
        print(f"   Model URL: {config.get('model_url', 'N/A')}")
        print(f"   Arguments per trial: {config.get('args_per_trial', 'N/A')}")
        print(f"   Overall rounds: {config.get('overall_rounds', 'N/A')}")
        print(f"   Temperature: {config.get('model_temperature', 'N/A')}")
        
        # Display results for enabled predictors only
        enabled = analysis_results.get('enabled_predictors', {})
        scores = {}
        
        if enabled.get('baseline', False) and analysis_results['mean_brier_baseline'] is not None:
            score = analysis_results['mean_brier_baseline']
            print(f"\n📈 Mean Brier Score (Baseline): {score:.4f}")
            scores['Baseline'] = score
            
        if enabled.get('basic', False) and analysis_results['mean_brier_basic'] is not None:
            score = analysis_results['mean_brier_basic']
            print(f"📈 Mean Brier Score (Basic): {score:.4f}")
            scores['Basic'] = score
        
        if enabled.get('argument', False) and analysis_results['mean_brier_argument'] is not None:
            score = analysis_results['mean_brier_argument']
            print(f"📈 Mean Brier Score (Argument-based): {score:.4f}")
            scores['Argument-based'] = score
        
        if enabled.get('conversational', False) and analysis_results['mean_brier_conversational'] is not None:
            score = analysis_results['mean_brier_conversational']
            print(f"📈 Mean Brier Score (Conversational): {score:.4f}")
            scores['Conversational'] = score
        
        if enabled.get('extended', False) and analysis_results['mean_brier_extended'] is not None:
            score = analysis_results['mean_brier_extended']
            print(f"📈 Mean Brier Score (Extended-Conv): {score:.4f}")
            scores['Extended-Conv'] = score
        
        if scores:
            best_method = min(scores.keys(), key=lambda k: scores[k])
            print(f"\n🏆 Best performing method in this split: {best_method} (Brier: {scores[best_method]:.4f})")

async def main():
    args = parse_arguments()
    

    import os
    llama_model = Llama(
        base_url=args.model_url,
        api_key=os.getenv("VLLM_API_KEY", "token-abc123"), 
        model="meta-llama/Llama-3.1-8B-Instruct",
        max_tokens=100,
        temperature=args.temperature
    )
    
    # Create analyzer with predictor flags
    try:
        analyzer = ParallelPredictionAnalyzer(
            llama_model, 
            args,
            run_baseline=args.baseline,
            run_basic=args.basic,
            run_argument=args.argument_based, 
            run_conversational=args.adaptive_conv,
            run_extended=args.extended_conv
        )
    except ValueError as e:
        return
    
    # Load and split questions
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '../../config/topics/combined.json')
    config_path = os.path.normpath(config_path)
    questions = analyzer.load_and_split_questions(config_path, args.split_id, args.total_splits)
    
    if not questions:
        return
    
    # Analyze split questions
    results = await analyzer.analyze_split_questions(questions)
    
    # Create output filename with split info
    predictor_suffix = ""
    if args.baseline:
        predictor_suffix += "_baseline"
    if args.basic:
        predictor_suffix += "_basic"
    if args.argument_based:
        predictor_suffix += "_arg"
    if args.adaptive_conv:
        predictor_suffix += "_adaptive"
    if args.extended_conv:
        predictor_suffix += "_extended"
    
    config_suffix = f"_args{args.args_per_trial}_rounds{args.overall_rounds}_exchanges{args.pairwise_exchanges}_extrounds{args.extended_rounds}_temp{args.temperature}"
    output_filename = f'{args.output_prefix}{predictor_suffix}{config_suffix}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())