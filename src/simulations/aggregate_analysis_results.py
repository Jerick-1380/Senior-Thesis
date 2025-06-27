#!/usr/bin/env python3
"""
Aggregate results from parallel prediction analysis splits into a single combined result.
"""
import json
import glob
import argparse
import statistics
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description='Aggregate parallel prediction analysis results')
    parser.add_argument('--input-pattern', type=str, required=True, 
                       help='Glob pattern for input JSON files (e.g., "analysis_split_*.json")')
    parser.add_argument('--output-file', type=str, default=None,
                       help='Output filename (auto-generated if not provided)')
    parser.add_argument('--results-dir', type=str, default='.',
                       help='Directory containing result files')
    return parser.parse_args()

def load_split_results(results_dir: str, pattern: str):
    """Load all split result files matching the pattern."""
    search_pattern = os.path.join(results_dir, pattern)
    result_files = glob.glob(search_pattern)
    
    if not result_files:
        raise ValueError(f"No files found matching pattern: {search_pattern}")
    
    print(f"Found {len(result_files)} result files to aggregate:")
    
    split_results = []
    for file_path in sorted(result_files):
        print(f"  - {os.path.basename(file_path)}")
        with open(file_path, 'r') as f:
            data = json.load(f)
            split_results.append(data)
    
    return split_results

def aggregate_results(split_results):
    """Aggregate results from multiple splits into a single combined result."""
    
    # Combine all individual question results
    all_results = []
    round_data_list = []
    
    for split_data in split_results:
        all_results.extend(split_data['results'])
        # Collect round data if available
        if 'round_data' in split_data:
            round_data_list.append(split_data['round_data'])
    
    # Calculate combined Brier scores
    def collect_brier_scores(score_key):
        scores = []
        for result in all_results:
            if result.get(score_key) is not None:
                scores.append(result[score_key])
        return scores
    
    brier_baseline = collect_brier_scores('baseline_brier')
    brier_basic = collect_brier_scores('basic_brier')
    brier_argument = collect_brier_scores('argument_brier')
    brier_conversational = collect_brier_scores('conversational_brier')
    brier_extended = collect_brier_scores('extended_brier')
    
    # Calculate combined means
    mean_brier_baseline = statistics.mean(brier_baseline) if brier_baseline else None
    mean_brier_basic = statistics.mean(brier_basic) if brier_basic else None
    mean_brier_argument = statistics.mean(brier_argument) if brier_argument else None
    mean_brier_conversational = statistics.mean(brier_conversational) if brier_conversational else None
    mean_brier_extended = statistics.mean(brier_extended) if brier_extended else None
    
    # Calculate total time across all splits
    total_time_minutes = sum(split_data['total_time_minutes'] for split_data in split_results)
    
    # Aggregate round data if available
    aggregated_round_data = None
    if round_data_list:
        aggregated_round_data = aggregate_round_data(round_data_list)
    
    # Use configuration from first split (should be identical across splits)
    first_split = split_results[0]
    
    # Aggregate per-split summaries
    split_summaries = []
    for split_data in split_results:
        summary = {
            'split_id': split_data['split_id'],
            'questions_processed': split_data['total_questions'],
            'time_minutes': split_data['total_time_minutes'],
            'mean_brier_baseline': split_data.get('mean_brier_baseline'),
            'mean_brier_basic': split_data.get('mean_brier_basic'),
            'mean_brier_argument': split_data.get('mean_brier_argument'),
            'mean_brier_conversational': split_data.get('mean_brier_conversational'),
            'mean_brier_extended': split_data.get('mean_brier_extended')
        }
        split_summaries.append(summary)
    
    # Create aggregated result
    aggregated_result = {
        'aggregation_timestamp': datetime.now().isoformat(),
        'num_splits_aggregated': len(split_results),
        'total_questions': len(all_results),
        'mean_brier_baseline': mean_brier_baseline,
        'mean_brier_basic': mean_brier_basic,
        'mean_brier_argument': mean_brier_argument,
        'mean_brier_conversational': mean_brier_conversational,
        'mean_brier_extended': mean_brier_extended,
        'enabled_predictors': first_split['enabled_predictors'],
        'configuration': first_split['configuration'],
        'results': all_results,
        'total_time_minutes': total_time_minutes,
        'baseline_questions_available': len(brier_baseline),
        'split_summaries': split_summaries,
        'brier_score_counts': {
            'baseline': len(brier_baseline),
            'basic': len(brier_basic),
            'argument': len(brier_argument),
            'conversational': len(brier_conversational),
            'extended': len(brier_extended)
        },
        'round_data': aggregated_round_data
    }
    
    return aggregated_result

def aggregate_round_data(round_data_list):
    """Aggregate round-by-round data from multiple splits."""
    if not round_data_list:
        return None
    
    # Calculate weighted average of mean Brier scores by round
    total_questions = sum(data['num_questions'] for data in round_data_list)
    num_rounds = round_data_list[0]['num_rounds']  # Should be same across splits
    
    # Weighted average by number of questions in each split
    aggregated_mean_brier = []
    for round_idx in range(num_rounds):
        weighted_sum = 0
        for data in round_data_list:
            if round_idx < len(data['mean_brier_by_round']) and data['mean_brier_by_round'][round_idx] is not None:
                weighted_sum += data['mean_brier_by_round'][round_idx] * data['num_questions']
        
        weighted_avg = weighted_sum / total_questions if total_questions > 0 else None
        aggregated_mean_brier.append(weighted_avg)
    
    return {
        'mean_brier_by_round': aggregated_mean_brier,
        'total_questions': total_questions,
        'num_rounds': num_rounds,
        'num_splits': len(round_data_list)
    }

def create_brier_plot(round_data, filename: str):
    """Create and save a plot of Brier scores over rounds."""
    if not round_data or 'mean_brier_by_round' not in round_data:
        return
    
    rounds = list(range(1, len(round_data['mean_brier_by_round']) + 1))
    brier_scores = round_data['mean_brier_by_round']
    
    # Filter out None values
    valid_data = [(r, s) for r, s in zip(rounds, brier_scores) if s is not None]
    if not valid_data:
        return
    
    rounds, brier_scores = zip(*valid_data)
    
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, brier_scores, 'b-', linewidth=2, marker='o', markersize=6)
    plt.xlabel('Conversation Round')
    plt.ylabel('Mean Brier Score')
    plt.title(f'Brier Score Evolution During Extended Conversations\n({round_data["total_questions"]} questions across {round_data["num_splits"]} splits)')
    plt.grid(True, alpha=0.3)
    
    # Add improvement annotation
    if len(brier_scores) > 1:
        improvement = brier_scores[0] - brier_scores[-1]
        improvement_pct = (improvement / brier_scores[0]) * 100
        plt.text(0.02, 0.98, f'Total Improvement: {improvement:.4f} ({improvement_pct:.1f}%)', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add final score annotation
    plt.text(0.98, 0.02, f'Final Brier Score: {brier_scores[-1]:.4f}', 
            transform=plt.gca().transAxes, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

def print_aggregated_summary(aggregated_result):
    """Print summary of aggregated results."""
    print("\n" + "="*80)
    print("📊 AGGREGATED ANALYSIS SUMMARY")
    print("="*80)
    print(f"Total splits processed: {aggregated_result['num_splits_aggregated']}")
    print(f"Total questions analyzed: {aggregated_result['total_questions']}")
    print(f"Total analysis time: {aggregated_result['total_time_minutes']:.2f} minutes")
    print(f"Average time per question: {aggregated_result['total_time_minutes']/aggregated_result['total_questions']:.2f} minutes")
    
    # Print configuration
    config = aggregated_result.get('configuration', {})
    print(f"\n🔧 Configuration:")
    print(f"   Arguments per trial: {config.get('args_per_trial', 'N/A')}")
    print(f"   Overall rounds: {config.get('overall_rounds', 'N/A')}")
    print(f"   Pairwise exchanges: {config.get('pairwise_exchanges', 'N/A')}")
    print(f"   Extended rounds: {config.get('extended_rounds', 'N/A')}")
    print(f"   Model temperature: {config.get('model_temperature', 'N/A')}")
    
    # Get enabled predictors info
    enabled = aggregated_result.get('enabled_predictors', {})
    
    # Display results for enabled predictors only
    scores = {}
    
    if enabled.get('baseline', False) and aggregated_result['mean_brier_baseline'] is not None:
        score = aggregated_result['mean_brier_baseline']
        print(f"\n📈 Mean Brier Score (Community Baseline): {score:.4f} ({aggregated_result['brier_score_counts']['baseline']} questions)")
        scores['Community Baseline'] = score
        
    if enabled.get('basic', False) and aggregated_result['mean_brier_basic'] is not None:
        score = aggregated_result['mean_brier_basic']
        print(f"📈 Mean Brier Score (Basic): {score:.4f} ({aggregated_result['brier_score_counts']['basic']} questions)")
        scores['Basic'] = score
    
    if enabled.get('argument', False) and aggregated_result['mean_brier_argument'] is not None:
        score = aggregated_result['mean_brier_argument']
        print(f"📈 Mean Brier Score (Argument-based): {score:.4f} ({aggregated_result['brier_score_counts']['argument']} questions)")
        scores['Argument-based'] = score
    
    if enabled.get('conversational', False) and aggregated_result['mean_brier_conversational'] is not None:
        score = aggregated_result['mean_brier_conversational']
        print(f"📈 Mean Brier Score (Conversational): {score:.4f} ({aggregated_result['brier_score_counts']['conversational']} questions)")
        scores['Conversational'] = score
    
    if enabled.get('extended', False) and aggregated_result['mean_brier_extended'] is not None:
        score = aggregated_result['mean_brier_extended']
        print(f"📈 Mean Brier Score (Extended-Conv): {score:.4f} ({aggregated_result['brier_score_counts']['extended']} questions)")
        scores['Extended-Conv'] = score
    
    if scores:
        best_method = min(scores.keys(), key=lambda k: scores[k])
        print(f"\n🏆 Best performing method overall: {best_method} (Brier: {scores[best_method]:.4f})")
    
    # Print per-split summary
    print(f"\n📋 Per-Split Summary:")
    print("   Split | Questions | Time(min) | Baseline | Basic    | Argument | Conv     | Extended")
    print("   ------|-----------|-----------|----------|----------|----------|----------|----------")
    
    for split_summary in aggregated_result['split_summaries']:
        split_id = split_summary['split_id']
        questions = split_summary['questions_processed']
        time_min = split_summary['time_minutes']
        
        def format_score(score):
            return f"{score:.4f}" if score is not None else "  N/A  "
        
        baseline_str = format_score(split_summary['mean_brier_baseline'])
        basic_str = format_score(split_summary['mean_brier_basic'])
        argument_str = format_score(split_summary['mean_brier_argument'])
        conv_str = format_score(split_summary['mean_brier_conversational'])
        extended_str = format_score(split_summary['mean_brier_extended'])
        
        print(f"   {split_id:5d} | {questions:9d} | {time_min:9.1f} | {baseline_str} | {basic_str} | {argument_str} | {conv_str} | {extended_str}")

def save_detailed_summary_to_txt(aggregated_result, output_dir: str):
    """Save detailed summary and analysis to text files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create summary text file
    summary_filename = os.path.join(output_dir, f"analysis_summary_{timestamp}.txt")
    
    with open(summary_filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write("AGGREGATED PREDICTION ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total splits processed: {aggregated_result['num_splits_aggregated']}\n")
        f.write(f"Total questions analyzed: {aggregated_result['total_questions']}\n")
        f.write(f"Total analysis time: {aggregated_result['total_time_minutes']:.2f} minutes\n")
        f.write(f"Average time per question: {aggregated_result['total_time_minutes']/aggregated_result['total_questions']:.2f} minutes\n")
        
        # Configuration
        config = aggregated_result.get('configuration', {})
        f.write(f"\nCONFIGURATION:\n")
        f.write(f"   Arguments per trial: {config.get('args_per_trial', 'N/A')}\n")
        f.write(f"   Overall rounds: {config.get('overall_rounds', 'N/A')}\n")
        f.write(f"   Pairwise exchanges: {config.get('pairwise_exchanges', 'N/A')}\n")
        f.write(f"   Extended rounds: {config.get('extended_rounds', 'N/A')}\n")
        f.write(f"   Model temperature: {config.get('model_temperature', 'N/A')}\n")
        f.write(f"   Model URL: {config.get('model_url', 'N/A')}\n")
        
        # Results
        enabled = aggregated_result.get('enabled_predictors', {})
        scores = {}
        
        f.write(f"\nPERFORMANCE RESULTS (Mean Brier Scores):\n")
        f.write("   Lower scores indicate better prediction accuracy\n\n")
        
        if enabled.get('baseline', False) and aggregated_result['mean_brier_baseline'] is not None:
            score = aggregated_result['mean_brier_baseline']
            f.write(f"   Community Baseline: {score:.4f} ({aggregated_result['brier_score_counts']['baseline']} questions)\n")
            scores['Community Baseline'] = score
            
        if enabled.get('basic', False) and aggregated_result['mean_brier_basic'] is not None:
            score = aggregated_result['mean_brier_basic']
            f.write(f"   Basic LLM: {score:.4f} ({aggregated_result['brier_score_counts']['basic']} questions)\n")
            scores['Basic'] = score
        
        if enabled.get('argument', False) and aggregated_result['mean_brier_argument'] is not None:
            score = aggregated_result['mean_brier_argument']
            f.write(f"   Argument-based: {score:.4f} ({aggregated_result['brier_score_counts']['argument']} questions)\n")
            scores['Argument-based'] = score
        
        if enabled.get('conversational', False) and aggregated_result['mean_brier_conversational'] is not None:
            score = aggregated_result['mean_brier_conversational']
            f.write(f"   Conversational: {score:.4f} ({aggregated_result['brier_score_counts']['conversational']} questions)\n")
            scores['Conversational'] = score
        
        if enabled.get('extended', False) and aggregated_result['mean_brier_extended'] is not None:
            score = aggregated_result['mean_brier_extended']
            f.write(f"   Extended-Conv: {score:.4f} ({aggregated_result['brier_score_counts']['extended']} questions)\n")
            scores['Extended-Conv'] = score
        
        if scores:
            best_method = min(scores.keys(), key=lambda k: scores[k])
            f.write(f"\nBEST PERFORMING METHOD: {best_method} (Brier: {scores[best_method]:.4f})\n")
        
        # Per-split breakdown
        f.write(f"\nPER-SPLIT BREAKDOWN:\n")
        f.write("Split | Questions | Time(min) | Baseline | Basic    | Argument | Conv     | Extended\n")
        f.write("------|-----------|-----------|----------|----------|----------|----------|----------\n")
        
        for split_summary in aggregated_result['split_summaries']:
            split_id = split_summary['split_id']
            questions = split_summary['questions_processed']
            time_min = split_summary['time_minutes']
            
            def format_score(score):
                return f"{score:.4f}" if score is not None else "  N/A  "
            
            baseline_str = format_score(split_summary['mean_brier_baseline'])
            basic_str = format_score(split_summary['mean_brier_basic'])
            argument_str = format_score(split_summary['mean_brier_argument'])
            conv_str = format_score(split_summary['mean_brier_conversational'])
            extended_str = format_score(split_summary['mean_brier_extended'])
            
            f.write(f"{split_id:5d} | {questions:9d} | {time_min:9.1f} | {baseline_str} | {basic_str} | {argument_str} | {conv_str} | {extended_str}\n")
    
    # Create detailed results CSV
    csv_filename = os.path.join(output_dir, f"detailed_results_{timestamp}.csv")
    
    with open(csv_filename, 'w') as f:
        # Write header
        headers = ['question', 'resolution', 'date_begin', 'date_resolve_at']
        enabled = aggregated_result.get('enabled_predictors', {})
        
        if enabled.get('baseline', False):
            headers.extend(['baseline_strength', 'baseline_brier'])
        if enabled.get('basic', False):
            headers.extend(['basic_strength', 'basic_brier'])
        if enabled.get('argument', False):
            headers.extend(['argument_strength', 'argument_brier'])
        if enabled.get('conversational', False):
            headers.extend(['conversational_strength', 'conversational_brier'])
        if enabled.get('extended', False):
            headers.extend(['extended_strength', 'extended_brier'])
        
        f.write(','.join(headers) + '\n')
        
        # Write data rows
        for result in aggregated_result['results']:
            row = [
                f'"{result.get("question", "").replace('"', '""')}"',
                str(result.get('resolution', '')),
                result.get('date_begin', ''),
                result.get('date_resolve_at', '')
            ]
            
            if enabled.get('baseline', False):
                row.extend([
                    str(result.get('baseline_strength', '')),
                    str(result.get('baseline_brier', ''))
                ])
            if enabled.get('basic', False):
                row.extend([
                    str(result.get('basic_strength', '')),
                    str(result.get('basic_brier', ''))
                ])
            if enabled.get('argument', False):
                row.extend([
                    str(result.get('argument_strength', '')),
                    str(result.get('argument_brier', ''))
                ])
            if enabled.get('conversational', False):
                row.extend([
                    str(result.get('conversational_strength', '')),
                    str(result.get('conversational_brier', ''))
                ])
            if enabled.get('extended', False):
                row.extend([
                    str(result.get('extended_strength', '')),
                    str(result.get('extended_brier', ''))
                ])
            
            f.write(','.join(row) + '\n')
    
    return summary_filename, csv_filename

def main():
    args = parse_arguments()
    
    print("🔄 Aggregating parallel prediction analysis results...")
    print(f"📁 Results directory: {args.results_dir}")
    print(f"🔍 Pattern: {args.input_pattern}")
    
    # Load split results
    split_results = load_split_results(args.results_dir, args.input_pattern)
    
    # Aggregate results
    print(f"\n🔗 Aggregating {len(split_results)} splits...")
    aggregated_result = aggregate_results(split_results)
    
    # Print summary
    print_aggregated_summary(aggregated_result)
    
    # Save detailed text summary and CSV files
    summary_txt, results_csv = save_detailed_summary_to_txt(aggregated_result, args.results_dir)
    
    # Generate Brier score plot if round data is available
    plot_filename = None
    if aggregated_result.get('round_data') is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"brier_evolution_plot_{timestamp}.png"
        plot_path = os.path.join(args.results_dir, plot_filename)
        
        try:
            created_plot = create_brier_plot(aggregated_result['round_data'], plot_path)
            if created_plot:
                print(f"\n📊 Brier score evolution plot generated!")
        except Exception as e:
            print(f"\n⚠️  Warning: Could not generate plot: {e}")
            plot_filename = None
    
    # Save aggregated results
    if args.output_file is None:
        # Auto-generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_file = f"aggregated_prediction_analysis_{timestamp}.json"
    
    output_path = os.path.join(args.results_dir, args.output_file)
    with open(output_path, 'w') as f:
        json.dump(aggregated_result, f, indent=2)
    
    print(f"\n💾 Files saved:")
    print(f"   📄 Summary report: {os.path.basename(summary_txt)}")
    print(f"   📊 Detailed CSV: {os.path.basename(results_csv)}")
    print(f"   🗂️  JSON data: {os.path.basename(output_path)}")
    if plot_filename:
        print(f"   📈 Brier plot: {plot_filename}")
    print(f"\n📊 Total questions processed: {aggregated_result['total_questions']}")
    print(f"⏱️  Total time: {aggregated_result['total_time_minutes']:.1f} minutes")

if __name__ == "__main__":
    main()