# comprehensive_comparison_updated.py
"""
Updated comprehensive comparison including Ultra Hybrid Fast model

This compares:
1. Random
2. RoundRobin  
3. ShortestQueue
4. Current_PPO
5. Ultra_Hybrid_Fast (NEW - YOUR WINNING MODEL)

Just run: python comprehensive_comparison_updated.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11


class ComprehensiveEvaluatorUpdated:
    """Runs all policies including Ultra Hybrid and generates comparison graphs"""
    
    def __init__(self, num_episodes=20, max_steps=500, num_disks=4):
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.num_disks = num_disks
        
        self.results = defaultdict(lambda: defaultdict(list))
        
    def evaluate_all_policies(self):
        """Run all policies and collect metrics"""
        from san_rl_env import GoSANSchedulerEnv
        from stable_baselines3 import PPO
        from ultra_hybrid_fast import UltraFastEnv  # NEW IMPORT
        
        policies = {
            'Random': self._random_policy,
            'RoundRobin': self._round_robin_policy(),
            'ShortestQueue': self._shortest_queue_policy,
            'Current_PPO': self._load_ppo_policy('ppo_san_rl'),
            'Ultra_Hybrid_Fast': self._load_ultra_hybrid_policy('ultra_hybrid_fast')  # NEW!
        }
        
        print("🔍 Running comprehensive evaluation...")
        print("="*70)
        
        for policy_name, policy_fn in policies.items():
            print(f"\n  ▶ Evaluating {policy_name}...")
            
            # Skip if policy couldn't be loaded
            if policy_fn is None:
                print(f"    ⚠️  Skipping {policy_name} (not available)")
                continue
            
            for ep in range(self.num_episodes):
                # Use Ultra env for Ultra model, regular env for others
                if policy_name == 'Ultra_Hybrid_Fast':
                    base_env = GoSANSchedulerEnv(num_disks=self.num_disks)
                    env = UltraFastEnv(base_env, num_disks=self.num_disks)
                else:
                    env = GoSANSchedulerEnv(num_disks=self.num_disks)
                
                try:
                    obs, _ = env.reset()
                except:
                    env.close()
                    continue
                
                ep_data = {
                    'rewards': [],
                    'latencies': [],
                    'queues': [],
                    'throughputs': [],
                    'actions': [],
                    'queue_history': []
                }
                
                for t in range(self.max_steps):
                    try:
                        action = policy_fn(obs, t)
                        obs, reward, term, trunc, info = env.step(action)
                        
                        ep_data['rewards'].append(reward)
                        ep_data['latencies'].append(info.get('last_latency', 0))
                        ep_data['actions'].append(action)
                        
                        queues = obs[:self.num_disks]
                        services = obs[self.num_disks:self.num_disks*2]
                        
                        ep_data['queues'].append(np.mean(queues))
                        ep_data['queue_history'].append(queues.copy())
                        ep_data['throughputs'].append(np.sum(services))
                        
                        if term or trunc:
                            break
                    except Exception as e:
                        print(f"    Error at step {t}: {e}")
                        break
                
                env.close()
                
                # Aggregate episode data
                if ep_data['rewards']:
                    self.results[policy_name]['episode_rewards'].append(np.sum(ep_data['rewards']))
                    self.results[policy_name]['mean_latencies'].append(np.mean(ep_data['latencies']))
                    self.results[policy_name]['mean_queues'].append(np.mean(ep_data['queues']))
                    self.results[policy_name]['mean_throughputs'].append(np.mean(ep_data['throughputs']))
                    self.results[policy_name]['latency_std'].append(np.std(ep_data['latencies']))
                    self.results[policy_name]['queue_std'].append(np.std(ep_data['queues']))
                    
                    # Store full traces for one episode
                    if ep == 0:
                        self.results[policy_name]['trace_latencies'] = ep_data['latencies'][:200]
                        self.results[policy_name]['trace_queues'] = ep_data['queue_history'][:200]
                        self.results[policy_name]['actions'] = ep_data['actions'][:200]
        
        print("\n✅ Evaluation complete!")
        self._save_results()
    
    def _random_policy(self, obs, t):
        return np.random.randint(self.num_disks)
    
    def _round_robin_policy(self):
        state = {'i': 0}
        def policy(obs, t):
            a = state['i']
            state['i'] = (state['i'] + 1) % self.num_disks
            return a
        return policy
    
    def _shortest_queue_policy(self, obs, t):
        return int(np.argmin(obs[:self.num_disks]))
    
    def _load_ppo_policy(self, path):
        try:
            from stable_baselines3 import PPO
            model = PPO.load(path)
            def policy(obs, t):
                action, _ = model.predict(obs, deterministic=True)
                return int(np.array(action).flatten()[0])
            return policy
        except Exception as e:
            print(f"    ⚠ Could not load {path}: {e}")
            return None
    
    def _load_ultra_hybrid_policy(self, path):
        """NEW: Load Ultra Hybrid Fast model"""
        try:
            from stable_baselines3 import PPO
            model = PPO.load(path)
            def policy(obs, t):
                action, _ = model.predict(obs, deterministic=True)
                return int(np.array(action).flatten()[0])
            return policy
        except Exception as e:
            print(f"    ⚠ Could not load {path}: {e}")
            print(f"    💡 Please train first: python ultra_hybrid_fast.py train")
            return None
    
    def _save_results(self):
        """Save results to JSON"""
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            else:
                return obj
        
        save_data = {}
        for policy, metrics in self.results.items():
            save_data[policy] = {k: convert_to_json_serializable(v) 
                                 for k, v in metrics.items()}
        
        with open('comprehensive_results_with_ultra.json', 'w') as f:
            json.dump(save_data, f, indent=2)
        print("💾 Results saved to comprehensive_results_with_ultra.json")
    
    def generate_all_graphs(self):
        """Generate comprehensive set of comparison graphs"""
        output_dir = Path("comprehensive_graphs_ultra")
        output_dir.mkdir(exist_ok=True)
        
        print("\n📊 Generating comprehensive graphs...")
        
        # 1. Mean Latency Comparison
        self._plot_metric_bars('mean_latencies', 'Mean Latency (seconds)', 
                               'Latency Comparison (Lower = Better)', 
                               output_dir / 'latency_comparison.png',
                               lower_is_better=True)
        
        # 2. Throughput Comparison
        self._plot_metric_bars('mean_throughputs', 'Mean Throughput (ops/sec)', 
                               'Throughput Comparison', 
                               output_dir / 'throughput_comparison.png')
        
        # 3. Queue Length Comparison
        self._plot_metric_bars('mean_queues', 'Mean Queue Length', 
                               'Queue Length Comparison (Lower = Better)', 
                               output_dir / 'queue_comparison.png',
                               lower_is_better=True)
        
        # 4. Cumulative Reward
        self._plot_metric_bars('episode_rewards', 'Cumulative Reward', 
                               'Episode Reward Comparison', 
                               output_dir / 'reward_comparison.png')
        
        # 5. Latency Stability
        self._plot_metric_bars('latency_std', 'Latency Std Dev (seconds)', 
                               'Latency Stability (Lower = More Stable)', 
                               output_dir / 'stability_comparison.png',
                               lower_is_better=True)
        
        # 6. Latency Over Time (Trace)
        self._plot_traces('trace_latencies', 'Latency (seconds)', 
                          'Latency Over Time', 
                          output_dir / 'latency_trace.png')
        
        # 7. Improvement Percentage Chart
        self._plot_improvement_chart(output_dir / 'improvement_percentages.png')
        
        # 8. Summary Table
        self._generate_summary_table(output_dir / 'summary_table.png')
        
        # 9. Box Plot - Latency Distribution
        self._plot_boxplot('mean_latencies', 'Latency (seconds)', 
                           'Latency Distribution Across Episodes', 
                           output_dir / 'latency_boxplot.png')
        
        # 10. Winner Highlight Chart
        self._plot_winner_chart(output_dir / 'winner_highlight.png')
        
        print(f"✅ All graphs saved to {output_dir}/")
    
    def _plot_metric_bars(self, metric_key, ylabel, title, filename, lower_is_better=False):
        """Generic bar plot for a metric"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        policies = list(self.results.keys())
        values = [np.mean(self.results[p][metric_key]) if self.results[p][metric_key] 
                  else 0 for p in policies]
        errors = [np.std(self.results[p][metric_key]) if self.results[p][metric_key] 
                  else 0 for p in policies]
        
        # Updated colors for 5 policies
        colors = ['#94a3b8', '#f59e0b', '#3b82f6', '#8b5cf6', '#10b981']
        bars = ax.bar(policies, values, yerr=errors, capsize=5, 
                      color=colors[:len(policies)], alpha=0.8, edgecolor='black')
        
        # Highlight best
        best_idx = np.argmin(values) if lower_is_better else np.argmax(values)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(4)
        
        # Annotations
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            symbol = '🏆' if i == best_idx else ''
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}\n{symbol}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=11)
        
        ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=20, ha='right', fontsize=11)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_traces(self, metric_key, ylabel, title, filename):
        """Line plot showing metric evolution over time"""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        colors = ['#94a3b8', '#f59e0b', '#3b82f6', '#8b5cf6', '#10b981']
        
        for idx, (policy, data) in enumerate(self.results.items()):
            if metric_key in data and data[metric_key]:
                trace = data[metric_key]
                ax.plot(trace, label=policy, color=colors[idx], 
                       linewidth=2.5, alpha=0.8)
        
        ax.set_xlabel('Time Step', fontsize=13, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9, fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_improvement_chart(self, filename):
        """Bar chart showing % improvement over baseline"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        baseline_latency = np.mean(self.results['ShortestQueue']['mean_latencies'])
        
        policies = [p for p in self.results.keys() if p != 'ShortestQueue']
        improvements = []
        
        for policy in policies:
            policy_latency = np.mean(self.results[policy]['mean_latencies'])
            improvement = (baseline_latency - policy_latency) / baseline_latency * 100
            improvements.append(improvement)
        
        colors = ['#f59e0b', '#3b82f6', '#8b5cf6', '#10b981']
        bars = ax.bar(policies, improvements, color=colors[:len(policies)], 
                      alpha=0.8, edgecolor='black', linewidth=2)
        
        # Highlight best
        best_idx = np.argmax(improvements)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(4)
        
        for i, (bar, val) in enumerate(zip(bars, improvements)):
            height = bar.get_height()
            symbol = '🏆' if i == best_idx else ''
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'+{val:.1f}%\n{symbol}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=12)
        
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.set_ylabel('Improvement vs Baseline (%)', fontsize=13, fontweight='bold')
        ax.set_title('Latency Improvement Over ShortestQueue Baseline', 
                     fontsize=15, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=20, ha='right', fontsize=11)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_summary_table(self, filename):
        """Generate visual summary table"""
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.axis('tight')
        ax.axis('off')
        
        metrics = ['Mean Latency', 'Mean Throughput', 'Mean Queue', 
                   'Latency StdDev', 'Total Reward']
        
        table_data = []
        for policy in self.results.keys():
            data = self.results[policy]
            row = [
                policy,
                f"{np.mean(data['mean_latencies']):.4f}" if data['mean_latencies'] else 'N/A',
                f"{np.mean(data['mean_throughputs']):.2f}" if data['mean_throughputs'] else 'N/A',
                f"{np.mean(data['mean_queues']):.3f}" if data['mean_queues'] else 'N/A',
                f"{np.mean(data.get('latency_std', [0])):.4f}",
                f"{np.mean(data['episode_rewards']):.2f}" if data['episode_rewards'] else 'N/A'
            ]
            table_data.append(row)
        
        table = ax.table(cellText=table_data, 
                        colLabels=['Policy'] + metrics,
                        cellLoc='center', loc='center',
                        colWidths=[0.18, 0.15, 0.17, 0.15, 0.17, 0.15])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.2)
        
        # Color header
        for i in range(len(metrics) + 1):
            table[(0, i)].set_facecolor('#3b82f6')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Highlight best latency row
        latencies = [float(table_data[i][1]) if table_data[i][1] != 'N/A' 
                    else float('inf') for i in range(len(table_data))]
        best_idx = np.argmin(latencies)
        
        for col_idx in range(len(metrics) + 1):
            table[(best_idx + 1, col_idx)].set_facecolor('#10b981')
            table[(best_idx + 1, col_idx)].set_text_props(weight='bold')
        
        plt.title('Performance Summary Table', fontsize=16, fontweight='bold', pad=20)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_boxplot(self, metric_key, ylabel, title, filename):
        """Box plot showing distribution"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        data_to_plot = []
        labels = []
        
        for policy, data in self.results.items():
            if metric_key in data and data[metric_key]:
                data_to_plot.append(data[metric_key])
                labels.append(policy)
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                        notch=True, showmeans=True)
        
        colors = ['#94a3b8', '#f59e0b', '#3b82f6', '#8b5cf6', '#10b981']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=20, ha='right', fontsize=11)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_winner_chart(self, filename):
        """Special chart highlighting the winner"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        policies = list(self.results.keys())
        latencies = [np.mean(self.results[p]['mean_latencies']) for p in policies]
        
        colors = ['#94a3b8', '#f59e0b', '#3b82f6', '#8b5cf6', '#10b981']
        bars = ax.bar(policies, latencies, color=colors[:len(policies)], 
                      alpha=0.9, edgecolor='black', linewidth=2)
        
        # Highlight winner
        winner_idx = np.argmin(latencies)
        bars[winner_idx].set_edgecolor('gold')
        bars[winner_idx].set_linewidth=6
        bars[winner_idx].set_alpha(1.0)
        
        # Annotations
        for i, (bar, lat) in enumerate(zip(bars, latencies)):
            height = bar.get_height()
            if i == winner_idx:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'🏆 WINNER 🏆\n{lat:.4f}s', ha='center', va='bottom',
                       fontweight='bold', fontsize=14, color='darkgreen',
                       bbox=dict(boxstyle='round', facecolor='gold', alpha=0.7))
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{lat:.4f}s', ha='center', va='bottom',
                       fontweight='bold', fontsize=12)
        
        ax.set_ylabel('Mean Latency (seconds)', fontsize=14, fontweight='bold')
        ax.set_title('🏆 COMPREHENSIVE COMPARISON - WINNER HIGHLIGHT 🏆', 
                     fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=20, ha='right', fontsize=12)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("🏆 COMPREHENSIVE COMPARISON (WITH ULTRA HYBRID)")
    print("="*70)
    
    evaluator = ComprehensiveEvaluatorUpdated(num_episodes=20, max_steps=500)
    
    # Run evaluations
    evaluator.evaluate_all_policies()
    
    # Generate all graphs
    evaluator.generate_all_graphs()
    
    print("\n" + "="*70)
    print("🎉 COMPLETE!")
    print("="*70)
    print("\nResults saved to:")
    print("  • comprehensive_results_with_ultra.json")
    print("  • comprehensive_graphs_ultra/")
    print("\nCheck 'winner_highlight.png' to see the champion! 🏆")
    print("="*70)


if __name__ == "__main__":
    main()