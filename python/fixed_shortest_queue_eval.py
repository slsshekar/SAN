# fixed_shortest_queue_eval.py
"""
Fixed implementation of Shortest Queue policy
Should perform BETTER than Random and Round Robin
"""

import numpy as np
from san_rl_env import GoSANSchedulerEnv

NUM_EPISODES = 20
MAX_STEPS = 500
NUM_DISKS = 4


def shortest_queue_policy(obs, t):
    """
    Fixed Shortest Queue: Choose disk with minimum queue length
    among ALIVE disks only (crucial fix)
    """
    queues = obs[:NUM_DISKS]
    services = obs[NUM_DISKS:NUM_DISKS*2]
    alive = obs[NUM_DISKS*2:NUM_DISKS*3]
    
    # Only consider alive disks
    valid_indices = [i for i in range(NUM_DISKS) if alive[i] > 0]
    
    if not valid_indices:
        # All disks failed - pick any (will be handled by env)
        return 0
    
    # Among alive disks, pick shortest queue
    min_queue = float('inf')
    best_disk = valid_indices[0]
    
    for idx in valid_indices:
        if queues[idx] < min_queue:
            min_queue = queues[idx]
            best_disk = idx
    
    return best_disk


def enhanced_shortest_queue_policy(obs, t):
    """
    Enhanced version: Consider both queue length AND service rate
    Quality = service_rate / (queue_length + epsilon)
    """
    queues = obs[:NUM_DISKS]
    services = obs[NUM_DISKS:NUM_DISKS*2]
    alive = obs[NUM_DISKS*2:NUM_DISKS*3]
    
    valid_indices = [i for i in range(NUM_DISKS) if alive[i] > 0]
    
    if not valid_indices:
        return 0
    
    # Pick disk with best quality score
    best_quality = -float('inf')
    best_disk = valid_indices[0]
    
    for idx in valid_indices:
        quality = services[idx] / (queues[idx] + 0.01)
        if quality > best_quality:
            best_quality = quality
            best_disk = idx
    
    return best_disk


def evaluate_policy(policy_fn, name, episodes=NUM_EPISODES):
    """Evaluate a policy"""
    all_rewards = []
    all_latencies = []
    all_queues = []
    
    for ep in range(episodes):
        env = GoSANSchedulerEnv(num_disks=NUM_DISKS)
        
        try:
            obs, _ = env.reset()
        except:
            env.close()
            continue
        
        ep_rewards = []
        ep_latencies = []
        ep_queues = []
        
        for t in range(MAX_STEPS):
            action = policy_fn(obs, t)
            obs, reward, term, trunc, info = env.step(action)
            
            ep_rewards.append(reward)
            ep_latencies.append(info.get('last_latency', 0))
            ep_queues.append(np.mean(obs[:NUM_DISKS]))
            
            if term or trunc:
                break
        
        env.close()
        
        if ep_rewards:
            all_rewards.append(np.sum(ep_rewards))
            all_latencies.append(np.mean(ep_latencies))
            all_queues.append(np.mean(ep_queues))
    
    results = {
        'mean_reward': np.mean(all_rewards),
        'mean_latency': np.mean(all_latencies),
        'mean_queue': np.mean(all_queues),
        'latency_std': np.std(all_latencies)
    }
    
    print(f"\n{name}:")
    print(f"  Mean Reward:   {results['mean_reward']:.4f}")
    print(f"  Mean Latency:  {results['mean_latency']:.4f}s")
    print(f"  Mean Queue:    {results['mean_queue']:.4f}")
    print(f"  Latency Std:   {results['latency_std']:.4f}")
    
    return results


if __name__ == "__main__":
    print("="*60)
    print("Testing Shortest Queue Policies")
    print("="*60)
    
    # Test both versions
    results_basic = evaluate_policy(shortest_queue_policy, "Shortest Queue (Basic)")
    results_enhanced = evaluate_policy(enhanced_shortest_queue_policy, "Shortest Queue (Enhanced)")
    
    print("\n" + "="*60)
    print("Comparison:")
    print("="*60)
    print(f"{'Policy':<30} {'Latency':<15} {'Reward':<15}")
    print("-"*60)
    print(f"{'Shortest Queue (Basic)':<30} {results_basic['mean_latency']:<15.4f} {results_basic['mean_reward']:<15.4f}")
    print(f"{'Shortest Queue (Enhanced)':<30} {results_enhanced['mean_latency']:<15.4f} {results_enhanced['mean_reward']:<15.4f}")
    print("="*60)