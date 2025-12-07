# comprehensive_comparison.py
"""
Complete comparison suite generating 10+ graphs comparing:
- Random
- RoundRobin (baseline)
- ShortestQueue
- Current_PPO (ppo_san_rl)
- Enhanced_Hybrid (enhanced_hybrid_scheduler)

Just run: python comprehensive_comparison.py
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


class ComprehensiveEvaluator:
    """Runs all policies and generates comparison graphs"""
    
    def __init__(self, num_episodes=20, max_steps=500, num_disks=4):
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.num_disks = num_disks
        
        self.results = defaultdict(lambda: defaultdict(list))
        
    def evaluate_all_policies(self):
        """Run all policies and collect metrics"""
        from san_rl_env import GoSANSchedulerEnv
        from stable_baselines3 import PPO
        from enhanced_hybrid_scheduler import EnhancedHybridEnv

        policies = {
            'Random': self._random_policy,
            'RoundRobin': self._round_robin_policy(),
            'ShortestQueue': self._shortest_queue_policy,
            'Current_PPO': self._load_ppo_policy('ppo_san_rl'),
            'Enhanced_Hybrid': self._load_ppo_policy('enhanced_hybrid_scheduler')
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
                if policy_name == 'Enhanced_Hybrid':
                    # Use the correct environment for the model
                    base_env = GoSANSchedulerEnv(num_disks=self.num_disks)
                    env = EnhancedHybridEnv(base_env, num_disks=self.num_disks)
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

            # Print summary for this policy
            if self.results[policy_name]['episode_rewards']:
                mean_lat = np.mean(self.results[policy_name]['mean_latencies'])
                std_lat = np.std(self.results[policy_name]['mean_latencies']) if len(self.results[policy_name]['mean_latencies']) > 1 else 0
                print(f"  ✓ {policy_name}: Latency = {mean_lat:.4f}s ± {std_lat:.4f}s")

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
        """Enhanced shortest queue policy considering alive status and service rate"""
        queues = obs[:4]
        services = obs[4:8]
        alive = obs[8:12]

        valid = [i for i in range(4) if alive[i] > 0]
        if not valid:
            return 0

        best_quality = -float('inf')
        best_disk = valid[0]
        for idx in valid:
            quality = services[idx] / (queues[idx] + 0.01)
            if quality > best_quality:
                best_quality = quality
                best_disk = idx
        return best_disk
    
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

        with open('comparison_results.json', 'w') as f:
            json.dump(save_data, f, indent=2)
        print("💾 Results saved to comparison_results.json")
    
    def generate_all_graphs(self):
        """Generate comprehensive set of comparison graphs"""
        output_dir = Path("comparison_graphs")
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

        # 7. Queue Evolution Over Time
        self._plot_queue_evolution(output_dir / 'queue_evolution.png')

        # 8. Action Distribution Heatmap
        self._plot_action_heatmap(output_dir / 'action_distribution.png')

        # 9. Performance Radar Chart
        self._plot_radar_chart(output_dir / 'performance_radar.png')

        # 10. Improvement Percentage Chart
        self._plot_improvement_chart(output_dir / 'improvement_percentages.png')

        # 11. Multi-Metric Summary Table
        self._generate_summary_table(output_dir / 'summary_table.png')

        # 12. Box Plot - Latency Distribution
        self._plot_boxplot('mean_latencies', 'Latency (seconds)',
                           'Latency Distribution Across Episodes',
                           output_dir / 'latency_boxplot.png')

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

        # Annotations
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom',
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

        for i, (bar, val) in enumerate(zip(bars, improvements)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'+{val:.1f}%', ha='center', va='bottom',
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
        
        bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True,
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
    
    
    def _plot_queue_evolution(self, filename):
        """Show queue length evolution for each disk"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        colors = ['#94a3b8', '#f59e0b', '#3b82f6', '#8b5cf6', '#10b981']
        
        for disk_idx in range(self.num_disks):
            ax = axes[disk_idx]
            
            for idx, (policy, data) in enumerate(self.results.items()):
                if 'trace_queues' in data and data['trace_queues']:
                    queue_trace = [q[disk_idx] for q in data['trace_queues']]
                    ax.plot(queue_trace, label=policy, color=colors[idx], 
                           linewidth=2, alpha=0.7)
            
            ax.set_title(f'Disk {disk_idx} Queue Evolution', fontweight='bold')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Queue Length')
            ax.legend(loc='best', fontsize=8)
            ax.grid(alpha=0.3)
        
        plt.suptitle('Per-Disk Queue Evolution', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_action_heatmap(self, filename):
        """Heatmap of action distributions"""
        fig, axes = plt.subplots(1, len(self.results), figsize=(15, 4))
        
        if len(self.results) == 1:
            axes = [axes]
        
        for idx, (policy, data) in enumerate(self.results.items()):
            if 'actions' in data and data['actions']:
                actions = data['actions']
                action_counts = np.bincount(actions, minlength=self.num_disks)
                action_matrix = action_counts.reshape(1, -1)
                
                sns.heatmap(action_matrix, annot=True, fmt='d', cmap='YlGnBu',
                           ax=axes[idx], cbar=False, 
                           xticklabels=[f'Disk {i}' for i in range(self.num_disks)],
                           yticklabels=[policy])
                axes[idx].set_title(f'{policy} Actions', fontweight='bold')
        
        plt.suptitle('Action Distribution Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_radar_chart(self, filename):
        """Radar chart comparing multiple metrics"""
        from math import pi
        
        categories = ['Latency\n(inverted)', 'Throughput', 'Queue\n(inverted)', 
                      'Stability\n(inverted)', 'Reward']
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = ['#94a3b8', '#f59e0b', '#3b82f6', '#8b5cf6', '#10b981']
        
        angles = [n / len(categories) * 2 * pi for n in range(len(categories))]
        angles += angles[:1]
        
        for idx, (policy, data) in enumerate(self.results.items()):
            if not data['mean_latencies']:
                continue
            
            # Normalize metrics to 0-100 scale
            latency_score = max(0, 100 - np.mean(data['mean_latencies']) * 100)
            throughput_score = min(100, np.mean(data['mean_throughputs']) / 3)
            queue_score = max(0, 100 - np.mean(data['mean_queues']) * 20)
            stability_score = max(0, 100 - np.mean(data.get('latency_std', [0])) * 100)
            reward_score = min(100, (np.mean(data['episode_rewards']) + 50) * 2)
            
            values = [latency_score, throughput_score, queue_score, stability_score, reward_score]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=policy, color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=11, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_title('Overall Performance Profile', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("COMPREHENSIVE COMPARISON OF POLICIES")
    print("="*70)

    evaluator = ComprehensiveEvaluator(num_episodes=20, max_steps=500)

    # Run evaluations
    evaluator.evaluate_all_policies()

    # Generate all graphs
    evaluator.generate_all_graphs()

    print("\n" + "="*70)
    print("🎉 COMPLETE!")
    print("="*70)
    print("\nResults saved to:")
    print("  • comparison_results.json")
    print("  • comparison_graphs/")
    print("="*70)


if __name__ == "__main__":
    main()
