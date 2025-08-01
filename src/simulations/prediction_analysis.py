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

# Add the project root to Python path to find helpers
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.core.model import Llama, GPT4o
from src.core.graph import Grapher, Writer  
from src.core.bots import Agent
from src.core.data import ALL_NAMES, PERSONAS
from src.core.conversation import ConversationCreator
from src.core.advanced_prompts import (
    PredictionPrompts, 
    ArgumentPrompts,
    StrengthCalculationHelpers
)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Multi-method Prediction Analysis with Conversational Agents')
    
    # Predictor Method Selection
    predictor_group = parser.add_argument_group('Prediction Methods', 'Choose which prediction approaches to run')
    predictor_group.add_argument('--baseline', action='store_true', help='Use community baseline predictions from historical data')
    predictor_group.add_argument('--basic', action='store_true', help='Use basic LLM predictions without additional context')
    predictor_group.add_argument('--argument-based', action='store_true', help='Generate arguments first, then make informed predictions')
    predictor_group.add_argument('--adaptive-conv', action='store_true', help='Use adaptive conversations between agents to develop perspectives')
    predictor_group.add_argument('--extended-conv', action='store_true', help='Use extended adaptive conversations (more overall rounds for deeper perspective development)')
    
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

class ConversationalAgent:
    """Wrapper around enhanced Agent class for prediction experiments."""
    
    def __init__(self, agent_id: int, llama_model, topic: str = "general"):
        self.agent_id = agent_id
        self.model = llama_model
        self.topic = topic
        self.arguments = []  # Store arguments from conversations
        
        # Create a minimal Agent instance for advanced capabilities
        self.agent = Agent(
            name=f"agent_{agent_id}",
            persona="You are a thoughtful conversationalist.",
            model=llama_model,
            topic=topic,
            claims={"pro": [], "con": [], "connector": ""},  # Minimal claims
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
        """Extract a new perspective from the conversation."""
        # Store the conversation in agent's history temporarily
        lines = conversation_history.split('\n')
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
        if not self.arguments:
            return 0.5
        
        # Update the agent's arguments with our collected ones
        self.agent.args = self.arguments.copy()
        
        # Use the enhanced prediction method
        return await self.agent.predict_with_arguments(question)



class PredictionAnalyzer:
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
    
    def load_questions(self, filepath: str) -> List[Dict]:
        """Load questions from JSON file and filter by date criteria."""
        print("Loading and filtering questions...")
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        filtered_questions = []
        for item in tqdm(data, desc="Filtering questions", unit="questions", disable=True):
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
        
        print(f"✓ Loaded {len(filtered_questions)} questions matching criteria")
        return filtered_questions
    
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
    
    
    async def calculate_question_strength(self, question: str) -> float:
        """Calculate the strength for a single question (only need one call since it's deterministic)."""
        strength = await self._single_trial_strength(question)
        
        if strength is None:
            return 0.5
        
        return strength
    
    async def run_conversation_round(self, agents: List[ConversationalAgent], topic: str, 
                                   conversation_length: int = 6) -> None:
        """Run one round of conversations with all agents paired up."""
        # Shuffle agents and pair them up
        import random
        shuffled_agents = agents.copy()
        random.shuffle(shuffled_agents)
        pairs = [(shuffled_agents[i], shuffled_agents[i+1]) for i in range(0, len(shuffled_agents), 2)]
        
        # Run all conversations in parallel
        conversation_tasks = []
        for agent1, agent2 in pairs:
            task = self.run_paired_conversation(agent1, agent2, topic, conversation_length)
            conversation_tasks.append(task)
        
        await asyncio.gather(*conversation_tasks)
    
    async def run_paired_conversation(self, agent1: ConversationalAgent, agent2: ConversationalAgent, 
                                    topic: str, conversation_length: int) -> None:
        """Run a conversation between two agents."""
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
        
        # Add perspectives to agents' arguments
        if perspective1:
            agent1.arguments.append(perspective1)
        if perspective2:
            agent2.arguments.append(perspective2)
    
    async def calculate_conversational_strength(self, question: str, progress_bar=None) -> float:
        """Calculate strength using conversational agents approach."""
        
        # Initialize 20 agents with the question as topic
        agents = [ConversationalAgent(i, self.model, topic=question) for i in range(20)]
        
        if progress_bar:
            progress_bar.set_postfix_str("Initializing 20 agents")
        
        # Run overall rounds of conversations
        for round_num in range(self.args.overall_rounds):
            if progress_bar:
                progress_bar.set_postfix_str(f"Conversation round {round_num+1}/{self.args.overall_rounds}")
            
            await self.run_conversation_round(agents, question, self.args.pairwise_exchanges)
            
            # Small delay to prevent overwhelming the API
            await asyncio.sleep(0.1)
        
        if progress_bar:
            progress_bar.set_postfix_str("Getting agent predictions")
        
        # Get predictions from all agents in parallel
        prediction_tasks = [agent.predict_with_arguments(question) for agent in agents]
        predictions = await asyncio.gather(*prediction_tasks)
        
        # Calculate average prediction
        valid_predictions = [p for p in predictions if p is not None]
        if not valid_predictions:
            return 0.5
        
        return statistics.mean(valid_predictions)
    
    async def calculate_extended_conv_strength(self, question: str, progress_bar=None) -> float:
        """Calculate strength using extended adaptive conversations (overall rounds + extended rounds)."""
        
        # Initialize 20 agents with the question as topic
        agents = [ConversationalAgent(i, self.model, topic=question) for i in range(20)]
        
        if progress_bar:
            progress_bar.set_postfix_str("Initializing 20 agents for extended conversations")
        
        # Run overall rounds of conversations first (same as adaptive-conv)
        total_rounds = self.args.overall_rounds + self.args.extended_rounds
        for round_num in range(total_rounds):
            if round_num < self.args.overall_rounds:
                phase = "adaptive"
                progress_text = f"Adaptive round {round_num+1}/{self.args.overall_rounds}"
            else:
                phase = "extended"
                extended_round = round_num - self.args.overall_rounds + 1
                progress_text = f"Extended round {extended_round}/{self.args.extended_rounds}"
            
            if progress_bar:
                progress_bar.set_postfix_str(progress_text)
            
            await self.run_conversation_round(agents, question, self.args.pairwise_exchanges)
            
            # Small delay to prevent overwhelming the API
            await asyncio.sleep(0.1)
        
        if progress_bar:
            progress_bar.set_postfix_str("Getting extended conversation agent predictions")
        
        # Get predictions from all agents in parallel
        prediction_tasks = [agent.predict_with_arguments(question) for agent in agents]
        predictions = await asyncio.gather(*prediction_tasks)
        
        # Calculate average prediction
        valid_predictions = [p for p in predictions if p is not None]
        if not valid_predictions:
            return 0.5
        
        return statistics.mean(valid_predictions)
    
    async def generate_arguments(self, question: str) -> List[str]:
        """Generate NUM_ARGUMENTS arguments for a question using improved prompting."""
        argument_prompt = ArgumentPrompts.generate_argument_prompt(question)
        
        # Generate arguments in parallel
        tasks = [self.model.generate(argument_prompt, [], []) for _ in range(self.args.args_per_trial)]
        arguments = await asyncio.gather(*tasks)
        
        # Clean arguments
        clean_arguments = []
        for arg in arguments:
            cleaned = arg.strip()
            if "Argument:" in cleaned:
                cleaned = cleaned.split("Argument:")[-1].strip()
            if cleaned and cleaned != "0":
                clean_arguments.append(cleaned)
        
        return clean_arguments
    
    async def calculate_argument_based_strength(self, question: str, progress_bar=None) -> float:
        """Calculate strength using argument-based prompting, averaged over 20 trials."""
        strengths = []
        
        # Create nested progress bar for the 20 trials
        trial_desc = f"Argument trials"
        
        # Run 20 trials
        for trial_num in range(20):
            if progress_bar:
                progress_bar.set_postfix(trial=f"{trial_num+1}/20")
            
            # Generate NUM_ARGUMENTS arguments
            arguments = await self.generate_arguments(question)
            
            if not arguments:
                continue
                
            # Create argument-informed prompt using improved helpers
            informed_prompt = PredictionPrompts.predict_with_arguments_prompt(question, arguments)
            
            # Get probabilities and calculate strength
            try:
                probs_dict = await self.model.get_probabilities(informed_prompt, "")
                yes_prob, no_prob = StrengthCalculationHelpers.extract_yes_no_probabilities(probs_dict)
                strength = StrengthCalculationHelpers.calculate_strength_from_probabilities(yes_prob, no_prob)
                strengths.append(strength)
                    
            except Exception as e:
                if progress_bar:
                    progress_bar.write(f"Error in argument-based trial: {e}")
                continue
        
        if not strengths:
            return 0.5
            
        return statistics.mean(strengths)
    
    async def _single_trial_strength(self, question: str) -> float:
        """Calculate strength for a single trial of a question using improved prompts."""
        try:
            # Use the improved basic prediction prompt
            prompt = PredictionPrompts.basic_prediction_prompt(question)
            
            # Get probabilities and calculate strength using helper functions
            probs_dict = await self.model.get_probabilities(prompt, "")
            yes_prob, no_prob = StrengthCalculationHelpers.extract_yes_no_probabilities(probs_dict)
            return StrengthCalculationHelpers.calculate_strength_from_probabilities(yes_prob, no_prob)
            
        except Exception as e:
            print(f"Error in trial: {e}")
            return None
    
    def calculate_brier_score(self, predicted_prob: float, actual_outcome: int) -> float:
        """Calculate Brier score for a single prediction."""
        return (predicted_prob - actual_outcome) ** 2
    
    async def analyze_all_questions(self, questions: List[Dict]) -> Dict:
        """Analyze all questions and calculate overall metrics."""
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
        
        print(f"\n🚀 Starting analysis of {len(questions)} questions...")
        print(f"📊 Running predictors: {', '.join(enabled_predictors)}")
        print(f"🔧 Configuration: {self.args.args_per_trial} arguments, {self.args.overall_rounds} overall rounds, {self.args.pairwise_exchanges} pairwise exchanges, {self.args.temperature} temperature")
        
        # Create main progress bar
        with tqdm(total=len(questions), desc="Processing questions", unit="questions", disable=True) as pbar:
            
            for i, question_data in enumerate(questions):
                question_preview = question_data['question'][:80] + "..." if len(question_data['question']) > 80 else question_data['question']
                pbar.set_description(f"Q{i+1}/{len(questions)}: {question_preview}")
                
                # Initialize result dictionary
                result = {
                    'question': question_data['question'],
                    'resolution': question_data['resolution'],
                    'background': question_data.get('background', ''),
                    'date_begin': question_data['date_begin'],
                    'date_resolve_at': question_data['date_resolve_at']
                }
                
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
                
                # Calculate strengths for enabled predictors
                tasks = []
                
                if self.run_basic:
                    tasks.append(('basic', self.calculate_question_strength(question_data['question'])))
                
                if self.run_argument:
                    tasks.append(('argument', self.calculate_argument_based_strength(question_data['question'], pbar)))
                
                if self.run_conversational:
                    tasks.append(('conversational', self.calculate_conversational_strength(question_data['question'], pbar)))
                
                if self.run_extended:
                    tasks.append(('extended', self.calculate_extended_conv_strength(question_data['question'], pbar)))
                
                # Run all enabled predictors
                if tasks:
                    task_names, task_coroutines = zip(*tasks)
                    strengths = await asyncio.gather(*task_coroutines)
                    
                    # Process results
                    for name, strength in zip(task_names, strengths):
                        result[f'{name}_strength'] = strength
                        brier_score = self.calculate_brier_score(strength, question_data['resolution'])
                        result[f'{name}_brier'] = brier_score
                        
                        if name == 'basic':
                            brier_scores_basic.append(brier_score)
                        elif name == 'argument':
                            brier_scores_argument.append(brier_score)
                        elif name == 'conversational':
                            brier_scores_conversational.append(brier_score)
                        elif name == 'extended':
                            brier_scores_extended.append(brier_score)
                
                results.append(result)
                
                # Update progress bar with current stats
                current_stats = {}
                if brier_scores_baseline:
                    current_stats['Base_Brier'] = f'{statistics.mean(brier_scores_baseline):.4f}'
                if brier_scores_basic:
                    current_stats['Basic_Brier'] = f'{statistics.mean(brier_scores_basic):.4f}'
                if brier_scores_argument:
                    current_stats['Arg_Brier'] = f'{statistics.mean(brier_scores_argument):.4f}'
                if brier_scores_conversational:
                    current_stats['Conv_Brier'] = f'{statistics.mean(brier_scores_conversational):.4f}'
                if brier_scores_extended:
                    current_stats['Ext_Brier'] = f'{statistics.mean(brier_scores_extended):.4f}'
                
                elapsed_time = time.time() - start_time
                avg_time_per_question = elapsed_time / (i + 1)
                remaining_questions = len(questions) - (i + 1)
                eta_seconds = avg_time_per_question * remaining_questions
                current_stats['ETA'] = f'{eta_seconds/60:.1f}m'
                
                pbar.set_postfix(current_stats)
                pbar.update(1)
        
        # Calculate overall metrics
        mean_brier_baseline = statistics.mean(brier_scores_baseline) if brier_scores_baseline else None
        mean_brier_basic = statistics.mean(brier_scores_basic) if brier_scores_basic else None
        mean_brier_argument = statistics.mean(brier_scores_argument) if brier_scores_argument else None
        mean_brier_conversational = statistics.mean(brier_scores_conversational) if brier_scores_conversational else None
        mean_brier_extended = statistics.mean(brier_scores_extended) if brier_scores_extended else None
        
        total_time = time.time() - start_time
        print(f"\n✅ Analysis completed in {total_time/60:.2f} minutes")
        
        return {
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
                'model_temperature': self.args.temperature
            },
            'results': results,
            'total_time_minutes': total_time / 60,
            'baseline_questions_available': len(brier_scores_baseline) if brier_scores_baseline else 0
        }
    
    def print_summary(self, analysis_results: Dict):
        """Print summary of analysis results."""
        print("\n" + "="*80)
        print("📊 ANALYSIS SUMMARY")
        print("="*80)
        print(f"Total questions analyzed: {analysis_results['total_questions']}")
        print(f"Total analysis time: {analysis_results['total_time_minutes']:.2f} minutes")
        print(f"Average time per question: {analysis_results['total_time_minutes']/analysis_results['total_questions']:.2f} minutes")
        
        # Print configuration
        config = analysis_results.get('configuration', {})
        print(f"\n🔧 Configuration:")
        print(f"   Arguments per trial: {config.get('args_per_trial', 'N/A')}")
        print(f"   Overall rounds: {config.get('overall_rounds', 'N/A')}")
        print(f"   Pairwise exchanges: {config.get('pairwise_exchanges', 'N/A')}")
        print(f"   Extended rounds: {config.get('extended_rounds', 'N/A')}")
        print(f"   Model temperature: {config.get('model_temperature', 'N/A')}")
        
        # Get enabled predictors info
        enabled = analysis_results.get('enabled_predictors', {})
        
        # Display results for enabled predictors only
        if enabled.get('baseline', False) and analysis_results['mean_brier_baseline'] is not None:
            print(f"\n📈 Mean Brier Score (Community Baseline): {analysis_results['mean_brier_baseline']:.4f}")
            
        if enabled.get('basic', False) and analysis_results['mean_brier_basic'] is not None:
            print(f"\n📈 Mean Brier Score (Basic): {analysis_results['mean_brier_basic']:.4f}")
        
        if enabled.get('argument', False) and analysis_results['mean_brier_argument'] is not None:
            print(f"📈 Mean Brier Score (Argument-based): {analysis_results['mean_brier_argument']:.4f}")
        
        if enabled.get('conversational', False) and analysis_results['mean_brier_conversational'] is not None:
            print(f"📈 Mean Brier Score (Conversational): {analysis_results['mean_brier_conversational']:.4f}")
        
        if enabled.get('extended', False) and analysis_results['mean_brier_extended'] is not None:
            print(f"📈 Mean Brier Score (Extended-Conv): {analysis_results['mean_brier_extended']:.4f}")
        
        # Calculate improvements only for enabled predictors
        improvements = []
        scores = {}
        
        if enabled.get('baseline', False) and analysis_results['mean_brier_baseline'] is not None:
            scores['Community Baseline'] = analysis_results['mean_brier_baseline']
        
        if enabled.get('basic', False) and analysis_results['mean_brier_basic'] is not None:
            scores['Basic'] = analysis_results['mean_brier_basic']
        
        if enabled.get('argument', False) and analysis_results['mean_brier_argument'] is not None:
            scores['Argument-based'] = analysis_results['mean_brier_argument']
        
        if enabled.get('conversational', False) and analysis_results['mean_brier_conversational'] is not None:
            scores['Conversational'] = analysis_results['mean_brier_conversational']
        
        if enabled.get('extended', False) and analysis_results['mean_brier_extended'] is not None:
            scores['Extended-Conv'] = analysis_results['mean_brier_extended']
        
        best_method = min(scores.keys(), key=lambda k: scores[k])
        print(f"\n🏆 Best performing method: {best_method} (Brier: {scores[best_method]:.4f})")
    

async def main():
    args = parse_arguments()
    
    print("🤖 Prediction Analysis Script Starting...")
    print("="*50)
    
    print("🔧 Predictor Configuration:")
    print(f"   Baseline: {'✅ Enabled' if args.baseline else '❌ Disabled'}")
    print(f"   Basic: {'✅ Enabled' if args.basic else '❌ Disabled'}")
    print(f"   Argument-based: {'✅ Enabled' if args.argument_based else '❌ Disabled'}")
    print(f"   Adaptive-Conv: {'✅ Enabled' if args.adaptive_conv else '❌ Disabled'}")
    print(f"   Extended-Conv: {'✅ Enabled' if args.extended_conv else '❌ Disabled'}")
    
    print(f"\n⚙️  Model Configuration:")
    print(f"   Model URL: {args.model_url}")
    print(f"   Temperature: {args.temperature}")
    print(f"   Arguments per trial: {args.args_per_trial}")
    print(f"   Overall rounds: {args.overall_rounds}")
    print(f"   Pairwise exchanges: {args.pairwise_exchanges}")
    print(f"   Extended rounds: {args.extended_rounds}")

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
        analyzer = PredictionAnalyzer(
            llama_model, 
            args,
            run_baseline=args.baseline,
            run_basic=args.basic,
            run_argument=args.argument_based, 
            run_conversational=args.adaptive_conv,
            run_extended=args.extended_conv
        )
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("Please enable at least one predictor type.")
        return
    
    # Load and filter questions
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '../../config/topics/combined.json')
    config_path = os.path.normpath(config_path)
    questions = analyzer.load_questions(config_path)
    
    if not questions:
        print("❌ No questions found matching the criteria (began before 2024, resolved in 2024)")
        return
    
    print(f"📋 Found {len(questions)} questions matching criteria")
    
    
    # Ask for confirmation if it's a large dataset
    if len(questions) > 10:
        predictor_list = []
        if args.baseline:
            predictor_list.append("Baseline")
        if args.basic:
            predictor_list.append("Basic")
        if args.argument_based:
            predictor_list.append("Argument-based")
        if args.adaptive_conv:
            predictor_list.append("Adaptive-Conv")
        if args.extended_conv:
            predictor_list.append("Extended-Conv")
        
        print(f"\nThis will analyze {len(questions)} questions using: {', '.join(predictor_list)}")
        print(f"Configuration: {args.args_per_trial} args, {args.overall_rounds} overall rounds, {args.pairwise_exchanges} exchanges, {args.extended_rounds} extended rounds, temp={args.temperature}")
    
    # Analyze all questions
    results = await analyzer.analyze_all_questions(questions)
    
    # Print summary
    analyzer.print_summary(results)
    
    # Create output filename with enabled predictors and configuration
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
    output_filename = f'prediction_analysis_results{predictor_suffix}{config_suffix}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Detailed results saved to '{output_filename}'")

if __name__ == "__main__":
    asyncio.run(main())