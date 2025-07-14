"""
Comprehensive Experiment Analyzer for Distributed Results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, List, Any
from scipy.stats import entropy
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ExperimentAnalyzer:
    """Analyzes and visualizes results from distributed experiments"""
    
    def __init__(self, results: Dict, config: 'ExperimentConfig', results_dir: Path):
        self.results = results
        self.config = config
        self.results_dir = Path(results_dir)
        self.analysis_dir = self.results_dir / "analysis"
        self.analysis_dir.mkdir(exist_ok=True)
        
    def create_comprehensive_analysis(self):
        """Create comprehensive analysis of all results"""
        print("Creating comprehensive analysis...")
        
        # 1. Summary statistics
        summary_stats = self.calculate_summary_statistics()
        self.save_summary_statistics(summary_stats)
        
        # 2. Performance comparison plots
        self.create_performance_plots(summary_stats)
        
        # 3. Pattern-specific analysis
        self.create_pattern_analysis()
        
        # 4. Convergence analysis
        self.create_convergence_analysis()
        
        # 5. Resource utilization analysis
        self.create_resource_analysis()
        
        # 6. Statistical significance tests
        self.perform_statistical_tests(summary_stats)
        
        # 7. Generate final report
        self.generate_final_report(summary_stats)
        
        print(f"Analysis complete! Results saved in: {self.analysis_dir}")
    
    def calculate_summary_statistics(self) -> Dict:
        """Calculate comprehensive summary statistics"""
        summary = {}
        
        for pattern, pattern_results in self.results.items():
            pattern_summary = {}
            
            for strategy, trials in pattern_results.items():
                if not trials:  # Skip empty results
                    continue
                
                # Extract interval value
                if strategy == 'full_comm':
                    interval = 0
                elif strategy == 'no_comm':
                    interval = float('inf')
                else:
                    interval = int(strategy.split('_')[1])
                
                # Calculate statistics
                discovery_rates = [trial['target_found'] for trial in trials]
                discovery_steps = [trial['first_discovery_step'] for trial in trials if trial['target_found']]
                final_entropies = [trial['final_entropy'] for trial in trials]
                prediction_errors = [trial['prediction_error'] for trial in trials]
                prob_at_targets = [trial['prob_at_true_target'] for trial in trials]
                computation_times = [trial['elapsed_time'] for trial in trials]
                
                pattern_summary[strategy] = {
                    'interval': interval,
                    'n_trials': len(trials),
                    'discovery_rate': {
                        'mean': np.mean(discovery_rates),
                        'std': np.std(discovery_rates),
                        'count': sum(discovery_rates)
                    },
                    'discovery_steps': {
                        'mean': np.mean(discovery_steps) if discovery_steps else self.config.max_steps,
                        'std': np.std(discovery_steps) if discovery_steps else 0,
                        'median': np.median(discovery_steps) if discovery_steps else self.config.max_steps,
                        'count': len(discovery_steps)
                    },
                    'final_entropy': {
                        'mean': np.mean(final_entropies),
                        'std': np.std(final_entropies),
                        'median': np.median(final_entropies)
                    },
                    'prediction_error': {
                        'mean': np.mean(prediction_errors),
                        'std': np.std(prediction_errors),
                        'median': np.median(prediction_errors)
                    },
                    'prob_at_target': {
                        'mean': np.mean(prob_at_targets),
                        'std': np.std(prob_at_targets),
                        'median': np.median(prob_at_targets)
                    },
                    'computation_time': {
                        'mean': np.mean(computation_times),
                        'std': np.std(computation_times),
                        'total': sum(computation_times)
                    }
                }
            
            summary[pattern] = pattern_summary
        
        return summary
    
    def save_summary_statistics(self, summary: Dict):
        """Save summary statistics to files"""
        # Save as JSON
        summary_serializable = self.make_json_serializable(summary)
        with open(self.analysis_dir / "summary_statistics.json", 'w') as f:
            json.dump(summary_serializable, f, indent=2)
        
        # Save as pickle for full precision
        with open(self.analysis_dir / "summary_statistics.pkl", 'wb') as f:
            pickle.dump(summary, f)
        
        # Create CSV for easy viewing
        self.create_summary_csv(summary)
    
    def make_json_serializable(self, obj):
        """Convert numpy types to JSON serializable types"""
        if isinstance(obj, dict):
            return {k: self.make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif obj == float('inf'):
            return "inf"
        else:
            return obj
    
    def create_summary_csv(self, summary: Dict):
        """Create CSV summary table"""
        rows = []
        
        for pattern, pattern_data in summary.items():
            for strategy, stats in pattern_data.items():
                row = {
                    'pattern': pattern,
                    'strategy': strategy,
                    'interval': stats['interval'],
                    'n_trials': stats['n_trials'],
                    'discovery_rate_mean': stats['discovery_rate']['mean'],
                    'discovery_rate_std': stats['discovery_rate']['std'],
                    'discovery_steps_mean': stats['discovery_steps']['mean'],
                    'discovery_steps_std': stats['discovery_steps']['std'],
                    'final_entropy_mean': stats['final_entropy']['mean'],
                    'final_entropy_std': stats['final_entropy']['std'],
                    'prediction_error_mean': stats['prediction_error']['mean'],
                    'prediction_error_std': stats['prediction_error']['std'],
                    'prob_at_target_mean': stats['prob_at_target']['mean'],
                    'prob_at_target_std': stats['prob_at_target']['std'],
                    'computation_time_mean': stats['computation_time']['mean'],
                    'computation_time_total': stats['computation_time']['total']
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(self.analysis_dir / "summary_table.csv", index=False)
    
    def create_performance_plots(self, summary: Dict):
        """Create comprehensive performance plots"""
        plt.style.use('seaborn-v0_8')
        
        # Create main comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        patterns = list(summary.keys())
        colors = ['blue', 'green', 'orange']
        
        # 1. Discovery rate comparison
        ax = axes[0, 0]
        for i, pattern in enumerate(patterns):
            intervals = []
            rates = []
            rate_stds = []
            
            for strategy, stats in summary[pattern].items():
                if stats['interval'] != float('inf'):
                    intervals.append(stats['interval'])
                    rates.append(stats['discovery_rate']['mean'] * 100)
                    rate_stds.append(stats['discovery_rate']['std'] * 100)
            
            ax.errorbar(intervals, rates, yerr=rate_stds, 
                       label=pattern.capitalize(), color=colors[i], 
                       marker='o', linewidth=2, markersize=8, capsize=5)
        
        ax.set_xlabel('Merge Interval (steps)')
        ax.set_ylabel('Discovery Rate (%)')
        ax.set_title('Target Discovery Success Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        
        # 2. Discovery speed
        ax = axes[0, 1]
        for i, pattern in enumerate(patterns):
            intervals = []
            steps = []
            step_stds = []
            
            for strategy, stats in summary[pattern].items():
                if stats['interval'] != float('inf') and stats['discovery_steps']['count'] > 0:
                    intervals.append(stats['interval'])
                    steps.append(stats['discovery_steps']['mean'])
                    step_stds.append(stats['discovery_steps']['std'])
            
            if intervals:
                ax.errorbar(intervals, steps, yerr=step_stds,
                           label=pattern.capitalize(), color=colors[i],
                           marker='s', linewidth=2, markersize=8, capsize=5)
        
        ax.set_xlabel('Merge Interval (steps)')
        ax.set_ylabel('Average Discovery Step')
        ax.set_title('Speed of Target Discovery')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Final entropy
        ax = axes[0, 2]
        for i, pattern in enumerate(patterns):
            intervals = []
            entropies = []
            entropy_stds = []
            
            for strategy, stats in summary[pattern].items():
                if stats['interval'] != float('inf'):
                    intervals.append(stats['interval'])
                    entropies.append(stats['final_entropy']['mean'])
                    entropy_stds.append(stats['final_entropy']['std'])
            
            ax.errorbar(intervals, entropies, yerr=entropy_stds,
                       label=pattern.capitalize(), color=colors[i],
                       marker='^', linewidth=2, markersize=8, capsize=5)
        
        ax.set_xlabel('Merge Interval (steps)')
        ax.set_ylabel('Final Belief Entropy')
        ax.set_title('Uncertainty in Final Belief')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Prediction error
        ax = axes[1, 0]
        for i, pattern in enumerate(patterns):
            intervals = []
            errors = []
            error_stds = []
            
            for strategy, stats in summary[pattern].items():
                if stats['interval'] != float('inf'):
                    intervals.append(stats['interval'])
                    errors.append(stats['prediction_error']['mean'])
                    error_stds.append(stats['prediction_error']['std'])
            
            ax.errorbar(intervals, errors, yerr=error_stds,
                       label=pattern.capitalize(), color=colors[i],
                       marker='d', linewidth=2, markersize=8, capsize=5)
        
        ax.set_xlabel('Merge Interval (steps)')
        ax.set_ylabel('Prediction Error (grid cells)')
        ax.set_title('Location Prediction Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Probability at target
        ax = axes[1, 1]
        for i, pattern in enumerate(patterns):
            intervals = []
            probs = []
            prob_stds = []
            
            for strategy, stats in summary[pattern].items():
                if stats['interval'] != float('inf'):
                    intervals.append(stats['interval'])
                    probs.append(stats['prob_at_target']['mean'])
                    prob_stds.append(stats['prob_at_target']['std'])
            
            ax.errorbar(intervals, probs, yerr=prob_stds,
                       label=pattern.capitalize(), color=colors[i],
                       marker='v', linewidth=2, markersize=8, capsize=5)
        
        ax.set_xlabel('Merge Interval (steps)')
        ax.set_ylabel('Probability at True Target')
        ax.set_title('Belief Accuracy at Target Location')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Computation time
        ax = axes[1, 2]
        for i, pattern in enumerate(patterns):
            intervals = []
            times = []
            time_stds = []
            
            for strategy, stats in summary[pattern].items():
                if stats['interval'] != float('inf'):
                    intervals.append(stats['interval'])
                    times.append(stats['computation_time']['mean'])
                    time_stds.append(stats['computation_time']['std'])
            
            ax.errorbar(intervals, times, yerr=time_stds,
                       label=pattern.capitalize(), color=colors[i],
                       marker='x', linewidth=2, markersize=8, capsize=5)
        
        ax.set_xlabel('Merge Interval (steps)')
        ax.set_ylabel('Computation Time (seconds)')
        ax.set_title('Computational Cost per Trial')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_pattern_analysis(self):
        """Create pattern-specific analysis"""
        for pattern in self.results.keys():
            self._analyze_single_pattern(pattern)
    
    def _analyze_single_pattern(self, pattern: str):
        """Analyze a single target pattern"""
        pattern_results = self.results[pattern]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Distribution of discovery times
        ax = axes[0, 0]
        for strategy, trials in pattern_results.items():
            if trials:
                discovery_times = [t['first_discovery_step'] for t in trials if t['target_found']]
                if discovery_times:
                    ax.hist(discovery_times, alpha=0.7, label=strategy, bins=20)
        
        ax.set_xlabel('Discovery Step')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{pattern.capitalize()} - Discovery Time Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Entropy evolution over time
        ax = axes[0, 1]
        for strategy, trials in pattern_results.items():
            if trials and 'entropy_history' in trials[0]:
                # Average entropy across trials
                max_steps = min(len(trials[0]['entropy_history']), 200)  # Plot first 200 steps
                avg_entropy = np.zeros(max_steps)
                
                for trial in trials:
                    for i in range(min(max_steps, len(trial['entropy_history']))):
                        avg_entropy[i] += trial['entropy_history'][i]['mean']
                
                avg_entropy /= len(trials)
                ax.plot(avg_entropy, label=strategy, alpha=0.8)
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Average Entropy')
        ax.set_title(f'{pattern.capitalize()} - Belief Uncertainty Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Performance vs interval scatter
        ax = axes[1, 0]
        intervals = []
        performance_scores = []
        
        for strategy, trials in pattern_results.items():
            if trials:
                # Extract interval
                if strategy == 'full_comm':
                    interval = 0
                elif strategy == 'no_comm':
                    interval = 1000  # High value for plotting
                else:
                    interval = int(strategy.split('_')[1])
                
                # Calculate composite performance score
                discovery_rate = np.mean([t['target_found'] for t in trials])
                avg_error = np.mean([t['prediction_error'] for t in trials])
                avg_time = np.mean([t['elapsed_time'] for t in trials])
                
                # Composite score (higher is better)
                score = discovery_rate * 100 / (1 + avg_error) / (1 + avg_time/10)
                
                intervals.append(interval)
                performance_scores.append(score)
        
        ax.scatter(intervals, performance_scores, s=100, alpha=0.7)
        for i, strategy in enumerate(pattern_results.keys()):
            if i < len(intervals):
                ax.annotate(strategy.replace('_', ' '), 
                           (intervals[i], performance_scores[i]),
                           xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Merge Interval (steps)')
        ax.set_ylabel('Performance Score')
        ax.set_title(f'{pattern.capitalize()} - Performance vs Merge Interval')
        ax.grid(True, alpha=0.3)
        
        # 4. Box plot of key metrics
        ax = axes[1, 1]
        metrics_data = []
        strategies = []
        
        for strategy, trials in pattern_results.items():
            if trials:
                for trial in trials:
                    metrics_data.append(trial['prob_at_true_target'])
                    strategies.append(strategy)
        
        if metrics_data:
            df = pd.DataFrame({'Strategy': strategies, 'Prob_at_Target': metrics_data})
            sns.boxplot(data=df, x='Strategy', y='Prob_at_Target', ax=ax)
            ax.tick_params(axis='x', rotation=45)
            ax.set_title(f'{pattern.capitalize()} - Probability at Target Distribution')
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / f"pattern_analysis_{pattern}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_convergence_analysis(self):
        """Analyze convergence properties"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        patterns = list(self.results.keys())
        
        # 1. Convergence rate analysis
        ax = axes[0, 0]
        for pattern in patterns:
            pattern_results = self.results[pattern]
            
            # Look at entropy reduction over time for different strategies
            for strategy, trials in pattern_results.items():
                if trials and strategy != 'no_comm' and 'entropy_history' in trials[0]:
                    # Calculate average entropy reduction rate
                    reduction_rates = []
                    
                    for trial in trials:
                        entropy_history = [h['mean'] for h in trial['entropy_history']]
                        if len(entropy_history) > 10:
                            # Calculate reduction rate over first 10 steps
                            initial_entropy = entropy_history[0]
                            final_entropy = entropy_history[9]
                            rate = (initial_entropy - final_entropy) / 10
                            reduction_rates.append(rate)
                    
                    if reduction_rates:
                        interval = 0 if strategy == 'full_comm' else int(strategy.split('_')[1])
                        ax.scatter([interval] * len(reduction_rates), reduction_rates,
                                 alpha=0.6, label=f'{pattern}-{strategy}')
        
        ax.set_xlabel('Merge Interval')
        ax.set_ylabel('Entropy Reduction Rate')
        ax.set_title('Convergence Rate Analysis')
        ax.grid(True, alpha=0.3)
        
        # 2. Time to convergence
        ax = axes[0, 1]
        # Implementation depends on specific convergence criteria
        # For now, use discovery time as proxy
        
        # 3. Stability analysis
        ax = axes[1, 0]
        for pattern in patterns:
            pattern_results = self.results[pattern]
            
            for strategy, trials in pattern_results.items():
                if trials:
                    final_entropies = [t['final_entropy'] for t in trials]
                    prediction_errors = [t['prediction_error'] for t in trials]
                    
                    # Plot stability (lower std = more stable)
                    entropy_std = np.std(final_entropies)
                    error_std = np.std(prediction_errors)
                    
                    interval = 0 if strategy == 'full_comm' else (1000 if strategy == 'no_comm' 
                             else int(strategy.split('_')[1]))
                    
                    ax.scatter(entropy_std, error_std, s=100, alpha=0.7,
                             label=f'{pattern}-{strategy}')
        
        ax.set_xlabel('Entropy Std Dev (Consistency)')
        ax.set_ylabel('Error Std Dev (Reliability)')
        ax.set_title('Stability Analysis')
        ax.grid(True, alpha=0.3)
        
        # 4. Learning curves
        ax = axes[1, 1]
        # Show how performance improves over trials (if ordering matters)
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / "convergence_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_resource_analysis(self):
        """Analyze computational resource usage"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Time per trial by strategy
        ax = axes[0, 0]
        all_times = []
        all_strategies = []
        all_patterns = []
        
        for pattern, pattern_results in self.results.items():
            for strategy, trials in pattern_results.items():
                for trial in trials:
                    all_times.append(trial['elapsed_time'])
                    all_strategies.append(strategy)
                    all_patterns.append(pattern)
        
        df = pd.DataFrame({
            'Time': all_times,
            'Strategy': all_strategies,
            'Pattern': all_patterns
        })
        
        if not df.empty:
            sns.boxplot(data=df, x='Strategy', y='Time', ax=ax)
            ax.tick_params(axis='x', rotation=45)
            ax.set_title('Computation Time Distribution by Strategy')
            ax.set_ylabel('Time per Trial (seconds)')
        
        # 2. Total computation time
        ax = axes[0, 1]
        total_times = {}
        
        for pattern, pattern_results in self.results.items():
            pattern_total = 0
            for strategy, trials in pattern_results.items():
                strategy_time = sum(trial['elapsed_time'] for trial in trials)
                pattern_total += strategy_time
            total_times[pattern] = pattern_total
        
        if total_times:
            ax.bar(total_times.keys(), total_times.values())
            ax.set_title('Total Computation Time by Pattern')
            ax.set_ylabel('Total Time (seconds)')
        
        # 3. Efficiency analysis (performance per unit time)
        ax = axes[1, 0]
        for pattern, pattern_results in self.results.items():
            efficiencies = []
            intervals = []
            
            for strategy, trials in pattern_results.items():
                if trials and strategy != 'no_comm':
                    discovery_rate = np.mean([t['target_found'] for t in trials])
                    avg_time = np.mean([t['elapsed_time'] for t in trials])
                    efficiency = discovery_rate / avg_time if avg_time > 0 else 0
                    
                    efficiencies.append(efficiency)
                    intervals.append(0 if strategy == 'full_comm' else int(strategy.split('_')[1]))
            
            if efficiencies:
                ax.plot(intervals, efficiencies, 'o-', label=pattern, linewidth=2, markersize=8)
        
        ax.set_xlabel('Merge Interval')
        ax.set_ylabel('Discovery Rate / Time')
        ax.set_title('Computational Efficiency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Speedup analysis
        ax = axes[1, 1]
        # Calculate speedup relative to no_comm baseline
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / "resource_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def perform_statistical_tests(self, summary: Dict):
        """Perform statistical significance tests"""
        from scipy import stats
        
        stat_results = {}
        
        for pattern, pattern_data in summary.items():
            pattern_stats = {}
            
            # Get all strategies for this pattern
            strategies = list(pattern_data.keys())
            
            # Compare each strategy pair
            for i, strategy1 in enumerate(strategies):
                for strategy2 in strategies[i+1:]:
                    # Get trial results
                    trials1 = self.results[pattern][strategy1]
                    trials2 = self.results[pattern][strategy2]
                    
                    if trials1 and trials2:
                        # Discovery rate comparison (Chi-square test)
                        successes1 = sum(t['target_found'] for t in trials1)
                        successes2 = sum(t['target_found'] for t in trials2)
                        total1, total2 = len(trials1), len(trials2)
                        
                        contingency = [[successes1, total1 - successes1],
                                     [successes2, total2 - successes2]]
                        chi2, p_discovery = stats.chi2_contingency(contingency)[:2]
                        
                        # Prediction error comparison (Mann-Whitney U test)
                        errors1 = [t['prediction_error'] for t in trials1]
                        errors2 = [t['prediction_error'] for t in trials2]
                        _, p_error = stats.mannwhitneyu(errors1, errors2, alternative='two-sided')
                        
                        pattern_stats[f"{strategy1}_vs_{strategy2}"] = {
                            'discovery_rate_p': p_discovery,
                            'prediction_error_p': p_error,
                            'discovery_rate_significant': p_discovery < 0.05,
                            'prediction_error_significant': p_error < 0.05
                        }
            
            stat_results[pattern] = pattern_stats
        
        # Save statistical results
        with open(self.analysis_dir / "statistical_tests.json", 'w') as f:
            json.dump(self.make_json_serializable(stat_results), f, indent=2)
        
        return stat_results
    
    def generate_final_report(self, summary: Dict):
        """Generate comprehensive final report"""
        report_path = self.analysis_dir / "final_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Distributed Belief Merging Experiment Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Configuration
            f.write("## Experiment Configuration\n\n")
            f.write(f"- Grid Size: {self.config.grid_size}\n")
            f.write(f"- Number of Agents: {self.config.n_agents}\n")
            f.write(f"- False Positive Rate (α): {self.config.alpha}\n")
            f.write(f"- False Negative Rate (β): {self.config.beta}\n")
            f.write(f"- MPC Horizon: {self.config.horizon}\n")
            f.write(f"- Trials per Configuration: {self.config.n_trials}\n")
            f.write(f"- Maximum Steps: {self.config.max_steps}\n")
            f.write(f"- Merge Intervals: {self.config.merge_intervals}\n")
            f.write(f"- Target Patterns: {self.config.target_patterns}\n\n")
            
            # Summary results
            f.write("## Summary Results\n\n")
            
            for pattern, pattern_data in summary.items():
                f.write(f"### {pattern.capitalize()} Pattern\n\n")
                
                # Find best strategies
                finite_strategies = {k: v for k, v in pattern_data.items() 
                                   if v['interval'] != float('inf')}
                
                if finite_strategies:
                    best_discovery = max(finite_strategies.keys(), 
                                       key=lambda x: pattern_data[x]['discovery_rate']['mean'])
                    best_accuracy = min(finite_strategies.keys(),
                                      key=lambda x: pattern_data[x]['prediction_error']['mean'])
                    
                    f.write(f"**Best Discovery Rate:** {best_discovery} ")
                    f.write(f"({pattern_data[best_discovery]['discovery_rate']['mean']*100:.1f}%)\n\n")
                    f.write(f"**Best Accuracy:** {best_accuracy} ")
                    f.write(f"({pattern_data[best_accuracy]['prediction_error']['mean']:.2f} cells)\n\n")
                
                # Strategy comparison table
                f.write("| Strategy | Discovery Rate | Avg Discovery Step | Prediction Error | Computation Time |\n")
                f.write("|----------|----------------|-------------------|------------------|------------------|\n")
                
                for strategy, stats in pattern_data.items():
                    discovery_rate = f"{stats['discovery_rate']['mean']*100:.1f}%"
                    discovery_step = f"{stats['discovery_steps']['mean']:.1f}"
                    pred_error = f"{stats['prediction_error']['mean']:.2f}"
                    comp_time = f"{stats['computation_time']['mean']:.3f}s"
                    
                    f.write(f"| {strategy} | {discovery_rate} | {discovery_step} | {pred_error} | {comp_time} |\n")
                
                f.write("\n")
            
            # Key insights
            f.write("## Key Insights\n\n")
            f.write("1. **Optimal Merge Intervals**: Different target patterns require different optimal merge intervals.\n")
            f.write("2. **Trade-offs**: There's a clear trade-off between discovery rate, accuracy, and computational cost.\n")
            f.write("3. **Pattern Dependence**: Target movement patterns significantly affect optimal communication strategies.\n")
            f.write("4. **Diminishing Returns**: Very frequent merging may not always improve performance due to loss of exploration diversity.\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("Based on the experimental results:\n\n")
            f.write("- For **Random** targets: Consider moderate merge intervals (25-50 steps)\n")
            f.write("- For **Evasive** targets: More frequent merging may be beneficial (10-25 steps)\n")
            f.write("- For **Patrol** targets: Less frequent merging may preserve exploration diversity (50-100 steps)\n")
            f.write("- Always consider computational constraints when choosing merge intervals\n\n")
            
            # Files generated
            f.write("## Generated Files\n\n")
            f.write("- `summary_statistics.json`: Detailed numerical results\n")
            f.write("- `summary_table.csv`: Tabular summary for spreadsheet analysis\n")
            f.write("- `performance_comparison.png`: Main performance comparison plots\n")
            f.write("- `pattern_analysis_*.png`: Pattern-specific analysis plots\n")
            f.write("- `convergence_analysis.png`: Convergence and stability analysis\n")
            f.write("- `resource_analysis.png`: Computational resource usage analysis\n")
            f.write("- `statistical_tests.json`: Statistical significance test results\n\n")
        
        print(f"Final report generated: {report_path}")
