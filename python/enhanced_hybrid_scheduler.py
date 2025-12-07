# enhanced_hybrid_scheduler.py
"""
Enhanced Hybrid Predictive Scheduler for Storage Area Networks

A production-ready RL scheduler combining:
1. Adaptive workload prediction
2. Multi-objective optimization
3. Efficient architecture for fast training
"""

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces
from collections import deque


# ==================== Efficient Feature Extractor ====================
class HybridFeatureExtractor(BaseFeaturesExtractor):
    """Efficient feature extractor with attention to disk states"""
    def __init__(self, observation_space, num_disks=4):
        super().__init__(observation_space, features_dim=128)
        
        state_dim = observation_space.shape[0]
        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
    
    def forward(self, observations):
        return self.encoder(observations)


# ==================== Enhanced Environment Wrapper ====================
class EnhancedHybridEnv(gym.Wrapper):
    """Enhanced environment with statistical features and multi-objective rewards"""
    def __init__(self, env, num_disks=4):
        super().__init__(env)
        self.num_disks = num_disks
        
        # History buffers for statistics
        self.recent_latencies = deque(maxlen=10)
        self.recent_queues = deque(maxlen=10)
        
        # Enhanced observation space
        base_dim = env.observation_space.shape[0]
        extra_dim = 4  # avg_queue, min_queue, avg_latency, latency_trend
        
        self.observation_space = spaces.Box(
            low=-1e6, high=1e6, shape=(base_dim + extra_dim,), dtype=np.float32
        )
        
        self.step_count = 0
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.recent_latencies.clear()
        self.recent_queues.clear()
        self.step_count = 0
        return self._enhance_obs(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        latency = info.get('last_latency', 0)
        queues = obs[:self.num_disks]
        
        self.recent_latencies.append(latency)
        self.recent_queues.append(np.mean(queues))
        self.step_count += 1
        
        # Multi-objective reward
        enhanced_reward = self._compute_multiobjective_reward(obs, info, action)
        
        return self._enhance_obs(obs), enhanced_reward, terminated, truncated, info
    
    def _enhance_obs(self, base_obs):
        """Add statistical features to observation"""
        queues = base_obs[:self.num_disks]
        
        # Queue statistics
        avg_queue = np.mean(queues)
        min_queue = np.min(queues)
        
        # Latency statistics
        if len(self.recent_latencies) > 0:
            avg_latency = np.mean(list(self.recent_latencies))
            
            # Latency trend
            if len(self.recent_latencies) >= 3:
                recent_3 = list(self.recent_latencies)[-3:]
                latency_trend = recent_3[-1] - recent_3[0]
            else:
                latency_trend = 0
        else:
            avg_latency = 0
            latency_trend = 0
        
        extra_features = np.array([avg_queue, min_queue, avg_latency, latency_trend])
        enhanced_obs = np.concatenate([base_obs, extra_features])
        
        return enhanced_obs.astype(np.float32)
    
    def _compute_multiobjective_reward(self, obs, info, action):
        """
        Multi-objective reward balancing:
        1. Latency minimization (primary)
        2. Smart disk selection
        3. Consistency bonus
        4. Queue management
        """
        latency = info.get('last_latency', 0)
        
        # 1. Latency penalty (primary objective)
        if latency > 0.2:
            latency_penalty = -10.0 * latency
        else:
            latency_penalty = -5.0 * latency
        
        # 2. Smart disk selection
        queues = obs[:self.num_disks]
        services = obs[self.num_disks:self.num_disks*2]
        alive = obs[self.num_disks*2:self.num_disks*3]
        
        if action < len(queues) and alive[action] > 0:
            disk_quality = services[action] / (queues[action] + 0.01)
            selection_bonus = 0.5 * disk_quality
        else:
            selection_bonus = -2.0
        
        # 3. Consistency bonus
        if len(self.recent_latencies) >= 5:
            recent_avg = np.mean(list(self.recent_latencies)[-5:])
            if recent_avg < 0.13:
                consistency_bonus = 0.3
            elif recent_avg < 0.15:
                consistency_bonus = 0.1
            else:
                consistency_bonus = 0
        else:
            consistency_bonus = 0
        
        # 4. Queue penalty
        total_queue = np.sum(queues)
        queue_penalty = -0.02 * total_queue
        
        total_reward = (
            latency_penalty +
            selection_bonus +
            consistency_bonus +
            queue_penalty
        )
        
        return total_reward


# ==================== Training ====================
def train_enhanced_hybrid(total_timesteps=200_000, save_name="enhanced_hybrid_scheduler"):
    """Train the enhanced hybrid scheduler"""
    from san_rl_env import GoSANSchedulerEnv
    
    print("🚀 Training Enhanced Hybrid Scheduler")
    print("="*60)
    print("⚡ Multi-objective optimization")
    print("⚡ Efficient architecture")
    print(f"⚡ Timesteps: {total_timesteps:,}")
    print("="*60)
    
    base_env = GoSANSchedulerEnv(num_disks=4)
    env = EnhancedHybridEnv(base_env, num_disks=4)
    
    policy_kwargs = dict(
        features_extractor_class=HybridFeatureExtractor,
        features_extractor_kwargs=dict(num_disks=4),
        net_arch=[dict(pi=[128, 64], vf=[128, 64])],
        activation_fn=nn.ReLU
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=5e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device='auto'
    )
    
    print(f"\n⏱️  Training for {total_timesteps:,} timesteps...\n")
    
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    model.save(save_name)
    print(f"\n✅ Training complete! Model saved as {save_name}.zip")
    
    return model


# ==================== Evaluation ====================
def evaluate_enhanced_hybrid(model_path="enhanced_hybrid_scheduler", episodes=20):
    """Evaluate the enhanced hybrid scheduler"""
    from san_rl_env import GoSANSchedulerEnv
    
    print(f"\n🔍 Evaluating {model_path}...")
    
    base_env = GoSANSchedulerEnv(num_disks=4)
    env = EnhancedHybridEnv(base_env, num_disks=4)
    
    model = PPO.load(model_path)
    
    all_latencies = []
    all_rewards = []
    all_queues = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        ep_latencies = []
        ep_reward = 0
        ep_queues = []
        
        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            
            ep_latencies.append(info.get('last_latency', 0))
            ep_reward += reward
            
            queues = obs[:4]
            ep_queues.append(np.mean(queues))
            
            if term or trunc:
                break
        
        all_latencies.append(np.mean(ep_latencies))
        all_rewards.append(ep_reward)
        all_queues.append(np.mean(ep_queues))
    
    env.close()
    
    results = {
        'mean_latency': np.mean(all_latencies),
        'latency_std': np.std(all_latencies),
        'mean_reward': np.mean(all_rewards),
        'mean_queue': np.mean(all_queues),
        'latency_p95': np.percentile(all_latencies, 95),
        'latency_p99': np.percentile(all_latencies, 99)
    }
    
    print("\n" + "="*60)
    print("📊 Enhanced Hybrid Results:")
    print("="*60)
    for key, val in results.items():
        print(f"  {key:25s}: {val:.4f}")
    print("="*60)
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            train_enhanced_hybrid(total_timesteps=200_000)
        elif sys.argv[1] == "eval":
            evaluate_enhanced_hybrid()
    else:
        print("Usage:")
        print("  python enhanced_hybrid_scheduler.py train")
        print("  python enhanced_hybrid_scheduler.py eval")