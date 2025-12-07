# interactive_detailed_analysis.py
"""
Interactive Detailed Analysis with Live Visualization
Shows popup graphs and real-time simulation of all policies
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import seaborn as sns
from san_rl_env import GoSANSchedulerEnv
from enhanced_hybrid_scheduler import EnhancedHybridEnv
from stable_baselines3 import PPO
from collections import deque
import time

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class InteractiveSimulator:
    """Real-time interactive visualization of policies"""
    
    def __init__(self, num_disks=4):
        self.num_disks = num_disks
        self.history_length = 100
        
        # Data buffers
        self.time_steps = deque(maxlen=self.history_length)
        self.latencies = {
            'Random': deque(maxlen=self.history_length),
            'Round Robin': deque(maxlen=self.history_length),
            'Shortest Queue': deque(maxlen=self.history_length),
            'PPO-RL': deque(maxlen=self.history_length),
            'Enhanced Hybrid': deque(maxlen=self.history_length),
        }
        self.queues = {policy: [deque(maxlen=self.history_length) for _ in range(num_disks)] 
                       for policy in self.latencies.keys()}
        self.actions = {policy: deque(maxlen=50) for policy in self.latencies.keys()}
        
    def run_live_simulation(self, max_steps=200):
        """Run live simulation with real-time visualization"""
        
        print("🎬 Starting Interactive Simulation...")
        print("=" * 70)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Axes
        ax_latency = fig.add_subplot(gs[0, :])
        ax_queues = [fig.add_subplot(gs[1, i]) for i in range(3)]
        ax_actions = [fig.add_subplot(gs[2, i]) for i in range(3)]
        
        # Initialize environments and policies
        policies = self._initialize_policies()
        envs = {name: self._create_env(name) for name in policies.keys()}
        observations = {name: env.reset()[0] for name, env in envs.items()}
        
        # Animation function
        def animate(frame):
            if frame >= max_steps:
                return
            
            self.time_steps.append(frame)
            
            # Step all policies
            for policy_name, policy_fn in policies.items():
                if policy_name not in envs:
                    continue
                    
                env = envs[policy_name]
                obs = observations[policy_name]
                
                try:
                    action = policy_fn(obs, frame)
                    obs, reward, term, trunc, info = env.step(action)
                    observations[policy_name] = obs
                    
                    # Record data
                    self.latencies[policy_name].append(info.get('last_latency', 0))
                    self.actions[policy_name].append(action)
                    
                    queues = obs[:self.num_disks]
                    for i in range(self.num_disks):
                        self.queues[policy_name][i].append(queues[i])
                    
                    if term or trunc:
                        obs = env.reset()[0]
                        observations[policy_name] = obs
                        
                except Exception as e:
                    print(f"Error in {policy_name}: {e}")
            
            # Update plots
            self._update_plots(ax_latency, ax_queues, ax_actions, frame)
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=max_steps, interval=50, repeat=False
        )
        
        plt.suptitle('Live Policy Comparison - Real-Time Simulation', 
                     fontsize=16, fontweight='bold')
        plt.show()
        
        # Close environments
        for env in envs.values():
            env.close()
        
        print("\n✅ Simulation complete!")
        self._print_summary()
    
    def _initialize_policies(self):
        """Initialize all policies"""
        policies = {}
        
        # Random
        policies['Random'] = lambda obs, t: np.random.randint(self.num_disks)
        
        # Round Robin
        rr_state = {'i': 0}
        def rr_policy(obs, t):
            a = rr_state['i']
            rr_state['i'] = (rr_state['i'] + 1) % self.num_disks
            return a
        policies['Round Robin'] = rr_policy
        
        # Shortest Queue
        def sq_policy(obs, t):
            queues = obs[:self.num_disks]
            services = obs[self.num_disks:self.num_disks*2]
            alive = obs[self.num_disks*2:self.num_disks*3]
            
            valid = [i for i in range(self.num_disks) if alive[i] > 0]
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
        policies['Shortest Queue'] = sq_policy
        
        # PPO-RL
        try:
            model_ppo = PPO.load("ppo_san_rl")
            policies['PPO-RL'] = lambda obs, t: int(np.array(
                model_ppo.predict(obs, deterministic=True)[0]
            ).flatten()[0])
        except:
            print("⚠️  PPO-RL model not found, skipping")
        
        # Enhanced Hybrid
        try:
            model_hybrid = PPO.load("enhanced_hybrid_scheduler")
            policies['Enhanced Hybrid'] = lambda obs, t: int(np.array(
                model_hybrid.predict(obs, deterministic=True)[0]
            ).flatten()[0])
        except:
            print("⚠️  Enhanced Hybrid model not found, skipping")
        
        return policies
    
    def _create_env(self, policy_name):
        """Create environment for policy"""
        base_env = GoSANSchedulerEnv(num_disks=self.num_disks)
        if policy_name == 'Enhanced Hybrid':
            return EnhancedHybridEnv(base_env)
        return base_env
    
    def _update_plots(self, ax_latency, ax_queues, ax_actions, frame):
        """Update all plots"""
        # Clear axes
        ax_latency.clear()
        for ax in ax_queues:
            ax.clear()
        for ax in ax_actions:
            ax.clear()
        
        # 1. Latency plot
        colors = ['gray', 'orange', 'blue', 'purple', 'green']
        for idx, (policy, latency_data) in enumerate(self.latencies.items()):
            if len(latency_data) > 0:
                ax_latency.plot(list(self.time_steps)[-len(latency_data):], 
                              list(latency_data), 
                              label=policy, 
                              linewidth=2.5, 
                              alpha=0.8,
                              color=colors[idx])
        
        ax_latency.set_xlabel('Time Step', fontweight='bold')
        ax_latency.set_ylabel('Latency (seconds)', fontweight='bold')
        ax_latency.set_title('Real-Time Latency Comparison', fontweight='bold', fontsize=12)
        ax_latency.legend(loc='upper right', framealpha=0.9)
        ax_latency.grid(True, alpha=0.3)
        ax_latency.set_ylim([0, max(0.5, max([max(list(d)) if len(d) > 0 else 0 
                                              for d in self.latencies.values()]))])
        
        # 2. Queue plots (show 3 policies)
        selected_policies = ['Shortest Queue', 'PPO-RL', 'Enhanced Hybrid']
        for ax_idx, policy in enumerate(selected_policies):
            if policy in self.queues and ax_idx < len(ax_queues):
                for disk_idx in range(self.num_disks):
                    queue_data = self.queues[policy][disk_idx]
                    if len(queue_data) > 0:
                        ax_queues[ax_idx].plot(
                            list(self.time_steps)[-len(queue_data):],
                            list(queue_data),
                            label=f'Disk {disk_idx}',
                            linewidth=2,
                            alpha=0.7
                        )
                
                ax_queues[ax_idx].set_xlabel('Time Step', fontsize=9)
                ax_queues[ax_idx].set_ylabel('Queue Length', fontsize=9)
                ax_queues[ax_idx].set_title(f'{policy} - Queue Lengths', 
                                           fontweight='bold', fontsize=10)
                ax_queues[ax_idx].legend(loc='upper right', fontsize=8)
                ax_queues[ax_idx].grid(True, alpha=0.3)
        
        # 3. Action distribution (show 3 policies)
        for ax_idx, policy in enumerate(selected_policies):
            if policy in self.actions and ax_idx < len(ax_actions):
                action_data = list(self.actions[policy])
                if len(action_data) > 0:
                    action_counts = [action_data.count(i) for i in range(self.num_disks)]
                    ax_actions[ax_idx].bar(range(self.num_disks), action_counts,
                                          color=colors[ax_idx+2], alpha=0.7,
                                          edgecolor='black')
                    
                    ax_actions[ax_idx].set_xlabel('Disk ID', fontsize=9)
                    ax_actions[ax_idx].set_ylabel('Action Count', fontsize=9)
                    ax_actions[ax_idx].set_title(f'{policy} - Disk Selection', 
                                                 fontweight='bold', fontsize=10)
                    ax_actions[ax_idx].set_xticks(range(self.num_disks))
                    ax_actions[ax_idx].grid(True, alpha=0.3, axis='y')
    
    def _print_summary(self):
        """Print simulation summary"""
        print("\n" + "=" * 70)
        print("SIMULATION SUMMARY")
        print("=" * 70)
        
        for policy, latency_data in self.latencies.items():
            if len(latency_data) > 0:
                mean_lat = np.mean(list(latency_data))
                std_lat = np.std(list(latency_data))
                print(f"{policy:<20}: Mean = {mean_lat:.4f}s, Std = {std_lat:.4f}s")
        
        print("=" * 70)


class DetailedComparison:
    """Detailed comparison with multiple popup windows"""
    
    def __init__(self):
        self.results = {}
    
    def run_detailed_analysis(self, episodes=5):
        """Run detailed analysis with multiple visualizations"""
        
        print("\n🔬 Running Detailed Analysis...")
        print("=" * 70)
        
        # Evaluate all policies
        self._evaluate_all_policies(episodes)
        
        # Show multiple popup windows
        self._show_latency_distribution()
        self._show_performance_radar()
        self._show_queue_behavior()
        self._show_action_heatmap()
        self._show_comparative_summary()
    
    def _evaluate_all_policies(self, episodes):
        """Evaluate all policies"""
        policies = {
            'Random': lambda obs, t: np.random.randint(4),
            'Round Robin': self._make_round_robin(),
            'Shortest Queue': self._shortest_queue_policy,
        }
        
        # Add RL policies
        try:
            model_ppo = PPO.load("ppo_san_rl")
            policies['PPO-RL'] = lambda obs, t: int(np.array(
                model_ppo.predict(obs, deterministic=True)[0]
            ).flatten()[0])
        except:
            pass
        
        try:
            model_hybrid = PPO.load("enhanced_hybrid_scheduler")
            policies['Enhanced Hybrid'] = lambda obs, t: int(np.array(
                model_hybrid.predict(obs, deterministic=True)[0]
            ).flatten()[0])
        except:
            pass
        
        # Evaluate each policy
        for name, policy_fn in policies.items():
            print(f"  Evaluating {name}...")
            self.results[name] = self._evaluate_policy(policy_fn, name, episodes)
    
    def _evaluate_policy(self, policy_fn, name, episodes):
        """Evaluate single policy"""
        all_latencies = []
        all_queues = []
        all_actions = []
        
        use_hybrid = (name == 'Enhanced Hybrid')
        
        for ep in range(episodes):
            base_env = GoSANSchedulerEnv(num_disks=4)
            env = EnhancedHybridEnv(base_env) if use_hybrid else base_env
            
            try:
                obs, _ = env.reset()
            except:
                env.close()
                continue
            
            ep_latencies = []
            ep_queues = []
            ep_actions = []
            
            for t in range(500):
                try:
                    action = policy_fn(obs, t)
                    obs, reward, term, trunc, info = env.step(action)
                    
                    ep_latencies.append(info.get('last_latency', 0))
                    ep_queues.append(obs[:4].copy())
                    ep_actions.append(action)
                    
                    if term or trunc:
                        break
                except:
                    break
            
            env.close()
            
            all_latencies.extend(ep_latencies)
            all_queues.extend(ep_queues)
            all_actions.extend(ep_actions)
        
        return {
            'latencies': all_latencies,
            'queues': all_queues,
            'actions': all_actions,
            'mean_latency': np.mean(all_latencies) if all_latencies else 0,
            'std_latency': np.std(all_latencies) if all_latencies else 0,
        }
    
    def _make_round_robin(self):
        state = {'i': 0}
        def policy(obs, t):
            a = state['i']
            state['i'] = (state['i'] + 1) % 4
            return a
        return policy
    
    def _shortest_queue_policy(self, obs, t):
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
    
    def _show_latency_distribution(self):
        """Show latency distribution popup"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle('Latency Distribution Analysis', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        for idx, (policy, data) in enumerate(self.results.items()):
            if idx >= len(axes):
                break
            
            latencies = data['latencies'][:500]  # First 500 steps
            if latencies:
                axes[idx].hist(latencies, bins=30, alpha=0.7, edgecolor='black')
                axes[idx].axvline(np.mean(latencies), color='red', 
                                 linestyle='--', linewidth=2, label=f'Mean: {np.mean(latencies):.4f}')
                axes[idx].set_title(policy, fontweight='bold')
                axes[idx].set_xlabel('Latency (seconds)')
                axes[idx].set_ylabel('Frequency')
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _show_performance_radar(self):
        """Show performance radar chart"""
        from math import pi
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        categories = ['Low Latency', 'Consistency', 'Load Balance', 
                     'Action Diversity', 'Queue Management']
        N = len(categories)
        
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        for policy, data in self.results.items():
            if not data['latencies']:
                continue
            
            # Calculate scores (normalized 0-1, higher is better)
            latencies = data['latencies']
            actions = data['actions']
            queues = data['queues']
            
            scores = [
                1 / (1 + np.mean(latencies)),  # Low latency
                1 / (1 + np.std(latencies)),    # Consistency
                1 / (1 + np.std([np.mean(q) for q in queues])),  # Load balance
                len(set(actions)) / 4,          # Action diversity
                1 / (1 + np.mean([np.sum(q) for q in queues]))  # Queue management
            ]
            
            scores += scores[:1]
            
            ax.plot(angles, scores, 'o-', linewidth=2, label=policy, alpha=0.7)
            ax.fill(angles, scores, alpha=0.15)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10)
        ax.set_ylim(0, 1)
        ax.set_title('Multi-Dimensional Performance Comparison', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def _show_queue_behavior(self):
        """Show queue behavior over time"""
        fig, axes = plt.subplots(len(self.results), 1, 
                                figsize=(12, 3*len(self.results)), sharex=True)
        
        if len(self.results) == 1:
            axes = [axes]
        
        for idx, (policy, data) in enumerate(self.results.items()):
            queues = data['queues'][:200]  # First 200 steps
            if queues:
                queue_array = np.array(queues)
                for disk in range(4):
                    axes[idx].plot(queue_array[:, disk], label=f'Disk {disk}', 
                                  linewidth=2, alpha=0.7)
                
                axes[idx].set_ylabel('Queue Length')
                axes[idx].set_title(f'{policy} - Queue Evolution', fontweight='bold')
                axes[idx].legend(loc='upper right')
                axes[idx].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time Step')
        fig.suptitle('Queue Length Evolution Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def _show_action_heatmap(self):
        """Show action selection heatmap"""
        fig, axes = plt.subplots(1, len(self.results), 
                                figsize=(4*len(self.results), 5))
        
        if len(self.results) == 1:
            axes = [axes]
        
        for idx, (policy, data) in enumerate(self.results.items()):
            actions = data['actions'][:200]
            if actions:
                # Create action matrix
                action_matrix = np.zeros((4, 50))
                for t, action in enumerate(actions[:200]):
                    action_matrix[action, t//4] += 1
                
                im = axes[idx].imshow(action_matrix, aspect='auto', cmap='YlOrRd')
                axes[idx].set_title(policy, fontweight='bold')
                axes[idx].set_ylabel('Disk ID')
                axes[idx].set_xlabel('Time Window')
                axes[idx].set_yticks(range(4))
                plt.colorbar(im, ax=axes[idx], label='Selection Count')
        
        fig.suptitle('Disk Selection Patterns Over Time', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def _show_comparative_summary(self):
        """Show comparative summary"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        policies = list(self.results.keys())
        
        # 1. Mean latency comparison
        latencies = [self.results[p]['mean_latency'] for p in policies]
        errors = [self.results[p]['std_latency'] for p in policies]
        axes[0, 0].bar(policies, latencies, yerr=errors, capsize=5, alpha=0.7)
        axes[0, 0].set_ylabel('Mean Latency (s)')
        axes[0, 0].set_title('Mean Latency Comparison', fontweight='bold')
        axes[0, 0].tick_params(axis='x', rotation=15)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 2. Latency boxplot
        latency_data = [self.results[p]['latencies'][:500] for p in policies]
        axes[0, 1].boxplot(latency_data, labels=policies)
        axes[0, 1].set_ylabel('Latency (s)')
        axes[0, 1].set_title('Latency Distribution', fontweight='bold')
        axes[0, 1].tick_params(axis='x', rotation=15)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Action diversity
        diversity = [len(set(self.results[p]['actions'])) / 4 for p in policies]
        axes[1, 0].bar(policies, diversity, alpha=0.7, color='green')
        axes[1, 0].set_ylabel('Diversity (0-1)')
        axes[1, 0].set_title('Action Diversity', fontweight='bold')
        axes[1, 0].set_ylim([0, 1.1])
        axes[1, 0].tick_params(axis='x', rotation=15)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Improvement percentages
        baseline_lat = self.results['Shortest Queue']['mean_latency']
        improvements = [(baseline_lat - self.results[p]['mean_latency']) / baseline_lat * 100 
                       for p in policies if p != 'Shortest Queue']
        other_policies = [p for p in policies if p != 'Shortest Queue']
        
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        axes[1, 1].bar(other_policies, improvements, alpha=0.7, color=colors)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[1, 1].set_ylabel('Improvement (%)')
        axes[1, 1].set_title('Improvement vs Baseline', fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=15)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        fig.suptitle('Comprehensive Performance Summary', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()


def main():
    """Main execution"""
    print("\n" + "=" * 70)
    print("INTERACTIVE DETAILED ANALYSIS")
    print("=" * 70)
    print("\nChoose analysis mode:")
    print("  1. Live Simulation (real-time animation)")
    print("  2. Detailed Comparison (multiple popup graphs)")
    print("  3. Both")
    print()
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        print("\n🎬 Starting Live Simulation...")
        print("Close the window to continue...")
        simulator = InteractiveSimulator(num_disks=4)
        simulator.run_live_simulation(max_steps=200)
    
    if choice in ['2', '3']:
        print("\n🔬 Running Detailed Analysis...")
        print("Multiple popup windows will appear...")
        analyzer = DetailedComparison()
        analyzer.run_detailed_analysis(episodes=5)
    
    print("\n✅ Analysis Complete!")


if __name__ == "__main__":
    main()