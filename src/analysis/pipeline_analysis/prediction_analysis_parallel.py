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
import matplotlib.pyplot as plt
import numpy as np

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
    params_group.add_argument('--extended-rounds', type=int, default=4, help='Number of overall rounds for extended conversations (default: 4)')
    params_group.add_argument('--track-brier-rounds', action='store_true', help='Track and plot Brier scores after each conversation round for extended-conv method')
    
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
    
    async def start_conversation(self, topic: str, partner_id: int) -> str:
        """Start a conversation about the given topic."""
        return await self.agent.start_conversation(topic)
    
    async def continue_conversation(self, topic: str, conversation_history: str, partner_id: int) -> str:
        """Continue an ongoing conversation."""
        return await self.agent.continue_conversation(topic, conversation_history)
    
    async def extract_perspective(self, topic: str, conversation_history: str) -> str:
        """Extract a new perspective from the conversation - matches working version."""
        # Store the conversation in agent's history temporarily
        lines = conversation_history.split('\n')
        
        # Clear and rebuild history (matches working version exactly)
        self.agent.user_history = []
        self.agent.model_history = []
        
        for i, line in enumerate(lines):
            if line.strip():
                if i % 2 == 0:  # User lines
                    self.agent.user_history.append(line.strip())
                else:  # Model lines  
                    self.agent.model_history.append(line.strip())
        
        # Use the agent's improved add_perspective method
        old_args_count = len(self.agent.args)
        await self.agent.add_perspective()
        
        # Get the new perspective if one was added
        if len(self.agent.args) > old_args_count:
            new_perspective = self.agent.args[-1]
            self.arguments.append(new_perspective)
            return new_perspective
        
        return ""
    
    async def predict_with_arguments(self, question: str) -> float:
        """Make a prediction using the agent's collected arguments."""
        print(f"Agent {self.agent_id} has {len(self.arguments)} arguments: {self.arguments[:2] if self.arguments else 'None'}")
        if not self.arguments:
            return 0.5
        
        # Update the agent's arguments with our collected ones
        self.agent.args = self.arguments.copy()
        
        # Use the enhanced prediction method
        result = await self.agent.predict_with_arguments(question)
        print(f"Agent {self.agent_id} prediction: {result}")  # Debug
        return result
    
    def reset_for_new_question(self, new_topic: str):
        """Properly reset agent for a new question."""
        self.topic = new_topic
        self.arguments = []
        
        # Reset the underlying Agent's state
        self.agent.topic = new_topic
        self.agent.args = []  # Clear previous arguments
        self.agent.user_history = []  # Clear conversation history
        self.agent.model_history = []  # Clear model history
        
        # Reset any other stateful components
        self.agent.claims = {"pro": [], "con": [], "connector": ""}

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
        """Load questions from JSON file and return only the assigned split."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        filtered_questions = []
        for item in data:
            # Only filter for resolved binary questions
            if (item.get('status') == 'resolved' and 
                item.get('question_type') == 'BINARY'):
                # Convert field names to match expected format
                converted_item = {
                    'question': item['question'],
                    'resolution': item.get('resolution', 0),  # Already 0 or 1
                    'background': item.get('description', ''),
                    'date_begin': item['begin_date'][:10] if 'begin_date' in item else '',  # Extract date part
                    'date_resolve_at': item['resolve_date'][:10] if 'resolve_date' in item else '',  # Extract date part
                    'is_resolved': True,
                    'question_type': 'binary',
                    'community_predictions': item.get('community_predictions', [])
                }
                filtered_questions.append(converted_item)
        
        # Split the dataset
        total_questions = len(filtered_questions)
        questions_per_split = math.ceil(total_questions / total_splits)
        start_idx = split_id * questions_per_split
        end_idx = min((split_id + 1) * questions_per_split, total_questions)
        
        split_questions = filtered_questions[start_idx:end_idx]
        
        return split_questions
    
    def extract_baseline_prediction(self, question_data: Dict) -> Optional[float]:
        """Extract the latest community prediction before 2024."""
        try:
            from datetime import datetime
            
            # Get community predictions - could be a list or string
            community_predictions = question_data.get('community_predictions', [])
            
            # If it's a string, try to parse it as JSON
            if isinstance(community_predictions, str):
                if not community_predictions or community_predictions == '[]':
                    return None
                try:
                    predictions = json.loads(community_predictions)
                except json.JSONDecodeError:
                    return None
            else:
                predictions = community_predictions
            
            if not predictions:
                return None
            
            # Filter predictions before 2024 and sort by date
            valid_predictions = []
            
            for prediction in predictions:
                if len(prediction) >= 2:
                    date_str, prob = prediction[0], prediction[1]
                    try:
                        # Parse date and check if before 2024
                        pred_date = datetime.strptime(date_str, "%Y-%m-%d")
                        if pred_date.year < 2024:
                            valid_predictions.append((pred_date, prob))
                    except (ValueError, TypeError):
                        continue
            
            if not valid_predictions:
                return None
            
            # Sort by date and return the latest probability before 2024
            valid_predictions.sort(key=lambda x: x[0])
            return valid_predictions[-1][1]
            
        except (KeyError, TypeError, IndexError) as e:
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
    
    async def batch_calculate_conversational_strengths(self, questions: List[str], question_data: List[Dict] = None, track_rounds: bool = False) -> Dict:
        """Calculate conversational strengths using batched agent interactions."""
        
        # Initialize 20 agents (will be reused across questions for efficiency)
        agents = [BatchConversationalAgent(i, self.model, topic="general") for i in range(20)]
        question_strengths = []
        
        # Initialize round tracking if needed
        round_data = None
        if track_rounds and question_data:
            round_data = {
                'brier_scores_by_round': [],
                'predictions_by_round': [],
                'mean_brier_by_round': []
            }
        
        # Process questions in smaller batches to manage memory
        batch_size = 5  # Process 5 questions at a time
        for batch_start in range(0, len(questions), batch_size):
            batch_end = min(batch_start + batch_size, len(questions))
            batch_questions = questions[batch_start:batch_end]
            batch_question_data = question_data[batch_start:batch_end] if question_data else None
            
            batch_strengths = []
            for q_idx, question in enumerate(batch_questions):
                # Reset agents for each question
                for agent in agents:
                    agent.reset_for_new_question(question)
                    agent.topic = question
                
                question_round_predictions = []
                question_round_brier_scores = []
                
                # Run conversation rounds
                for round_num in range(self.args.overall_rounds):
                    await self.batch_run_conversation_round(agents, question, self.args.pairwise_exchanges)
                    await asyncio.sleep(0.05)  # Small delay to prevent API overwhelming
                    
                    # Get predictions from all agents for this question after this round
                    prediction_tasks = [agent.agent.predict_with_arguments(question) for agent in agents]
                    predictions = await asyncio.gather(*prediction_tasks)
                    
                    # Calculate average prediction
                    valid_predictions = [p for p in predictions if p is not None]
                    avg_prediction = statistics.mean(valid_predictions) if valid_predictions else 0.5
                    
                    # Track round data if needed
                    if track_rounds and batch_question_data:
                        question_round_predictions.append(avg_prediction)
                        brier_score = self.calculate_brier_score(avg_prediction, batch_question_data[q_idx]['resolution'])
                        question_round_brier_scores.append(brier_score)
                
                # Store final prediction
                final_prediction = question_round_predictions[-1] if question_round_predictions else 0.5
                batch_strengths.append(final_prediction)
                
                # Store round data if tracking
                if track_rounds and question_round_predictions:
                    global_q_idx = batch_start + q_idx
                    
                    # Initialize round data structures if this is the first question
                    if global_q_idx == 0:
                        round_data['brier_scores_by_round'] = [[] for _ in range(len(question_round_brier_scores))]
                        round_data['predictions_by_round'] = [[] for _ in range(len(question_round_predictions))]
                    
                    # Add this question's data to each round
                    for round_idx in range(len(question_round_predictions)):
                        round_data['predictions_by_round'][round_idx].append(question_round_predictions[round_idx])
                        round_data['brier_scores_by_round'][round_idx].append(question_round_brier_scores[round_idx])
            
            question_strengths.extend(batch_strengths)
        
        # Calculate mean Brier scores by round if tracking
        if track_rounds and round_data:
            for round_scores in round_data['brier_scores_by_round']:
                mean_score = statistics.mean(round_scores) if round_scores else None
                round_data['mean_brier_by_round'].append(mean_score)
        
        # Return results
        if track_rounds and round_data:
            return {
                'final_predictions': question_strengths,
                'round_data': round_data
            }
        else:
            return {'final_predictions': question_strengths}
    
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
        response1 = await agent1.start_conversation(topic, agent2.agent_id)
        conversation_history.append(f"Agent {agent1.agent_id}: {response1}")
        
        # Continue conversation for specified length
        current_speaker = agent2
        other_speaker = agent1
        
        for turn in range(conversation_length - 1):  # -1 because we already had the first turn
            history_text = "\n".join(conversation_history)
            response = await current_speaker.continue_conversation(topic, history_text, other_speaker.agent_id)
            conversation_history.append(f"Agent {current_speaker.agent_id}: {response}")
            
            # Switch speakers
            current_speaker, other_speaker = other_speaker, current_speaker
        
        # Extract perspectives from both agents
        history_text = "\n".join(conversation_history)
        
        perspective1_task = agent1.extract_perspective(topic, history_text)
        perspective2_task = agent2.extract_perspective(topic, history_text)
        
        perspective1, perspective2 = await asyncio.gather(perspective1_task, perspective2_task)
    
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
            conv_results = await self.batch_calculate_conversational_strengths(question_texts)
            predictor_results['conversational'] = conv_results['final_predictions']
        
        if self.run_extended:
            # For extended, reuse conversational logic but with more rounds
            old_rounds = self.args.overall_rounds
            self.args.overall_rounds += self.args.extended_rounds
            
            # Check if we need to track Brier scores by round
            track_rounds = hasattr(self.args, 'track_brier_rounds') and self.args.track_brier_rounds
            extended_results = await self.batch_calculate_conversational_strengths(question_texts, questions if track_rounds else None, track_rounds)
            
            self.args.overall_rounds = old_rounds  # Reset
            predictor_results['extended'] = extended_results['final_predictions']
            
            # Store round data if tracking
            if track_rounds and 'round_data' in extended_results:
                self._extended_round_data = extended_results['round_data']
        
        # Store round data for aggregation if tracking
        round_data = None
        if hasattr(self, '_extended_round_data'):
            # Don't store individual predictions/brier scores to avoid huge files
            # Just store the mean brier scores by round for aggregation
            round_data = {
                'mean_brier_by_round': self._extended_round_data['mean_brier_by_round'],
                'num_questions': len(questions),
                'num_rounds': len(self._extended_round_data['mean_brier_by_round'])
            }
        
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
        
        result_dict = {
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
        
        # Add round data if available
        if round_data is not None:
            result_dict['round_data'] = round_data
            
        return result_dict
    
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
    config_path = os.path.join(script_dir, '../../config/topics/new_combined.json')
    config_path = os.path.normpath(config_path)
    questions = analyzer.load_and_split_questions(config_path, args.split_id, args.total_splits)
    
    if not questions:
        return
    
    # Analyze split questions
    results = await analyzer.analyze_split_questions(questions)
    
    # Print split summary
    analyzer.print_split_summary(results)
    
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
    
    # Note: Brier score plots will be generated after aggregating all splits

def create_brier_plot(round_data: Dict, filename: str):
    """Create and save a plot of Brier scores over rounds."""
    if 'mean_brier_by_round' not in round_data:
        return
    
    rounds = list(range(1, len(round_data['mean_brier_by_round']) + 1))
    brier_scores = round_data['mean_brier_by_round']
    
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, brier_scores, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Conversation Round')
    plt.ylabel('Mean Brier Score')
    plt.title('Brier Score Evolution During Extended Conversations')
    plt.grid(True, alpha=0.3)
    
    # Add improvement annotation
    if len(brier_scores) > 1:
        improvement = brier_scores[0] - brier_scores[-1]
        plt.text(0.02, 0.98, f'Total Improvement: {improvement:.4f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    asyncio.run(main())