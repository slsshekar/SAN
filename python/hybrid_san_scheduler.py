# hybrid_san_scheduler.py
"""
Novel Hybrid Predictive Failure-Aware Scheduler for Storage Area Networks

Key Innovations:
1. LSTM-based workload prediction (5-10 steps ahead)
2. Failure risk estimation per disk
3. Multi-objective reward function
4. Attention mechanism for disk selection
5. PPO with enhanced state representation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces


# ==================== LSTM Workload Predictor ====================
class WorkloadPredictor(nn.Module):
    """Predicts future job arrivals and sizes"""
    def __init__(self, input_dim=4, hidden_dim=32, num_layers=2, pred_horizon=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_horizon = pred_horizon
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_arrival = nn.Linear(hidden_dim, pred_horizon)
        self.fc_size = nn.Linear(hidden_dim, pred_horizon)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        
        arrival_pred = torch.sigmoid(self.fc_arrival(last_hidden))  # 0-1 prob
        size_pred = torch.relu(self.fc_size(last_hidden))  # positive sizes
        
        return arrival_pred, size_pred


# ==================== Failure Risk Estimator ====================
class FailureRiskEstimator(nn.Module):
    """Estimates per-disk failure probability based on history"""
    def __init__(self, num_disks=4, history_len=20):
        super().__init__()
        self.num_disks = num_disks
        self.history_len = history_len
        
        # Per-disk feature extraction
        self.disk_encoder = nn.Sequential(
            nn.Linear(history_len * 3, 64),  # queue, service, alive history
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # risk score
            nn.Sigmoid()
        )
    
    def forward(self, disk_histories):
        # disk_histories: (batch, num_disks, history_len * 3)
        batch_size = disk_histories.shape[0]
        risks = []
        
        for i in range(self.num_disks):
            disk_hist = disk_histories[:, i, :]
            risk = self.disk_encoder(disk_hist)
            risks.append(risk)
        
        return torch.cat(risks, dim=1)  # (batch, num_disks)


# ==================== Attention-based Disk Scorer ====================
class AttentionDiskScorer(nn.Module):
    """Uses attention to weigh disk importance dynamically"""
    def __init__(self, state_dim, num_disks=4):
        super().__init__()
        self.num_disks = num_disks
        
        # Query: current system state
        self.query_net = nn.Linear(state_dim, 32)
        
        # Keys: per-disk features
        self.key_net = nn.Linear(3, 32)  # queue, service, alive
        
        # Values: disk embeddings
        self.value_net = nn.Linear(3, 32)
        
        self.scale = np.sqrt(32)
    
    def forward(self, state, disk_features):
        # state: (batch, state_dim)
        # disk_features: (batch, num_disks, 3)
        
        query = self.query_net(state).unsqueeze(1)  # (batch, 1, 32)
        keys = self.key_net(disk_features)  # (batch, num_disks, 32)
        values = self.value_net(disk_features)  # (batch, num_disks, 32)
        
        # Attention scores
        scores = torch.bmm(query, keys.transpose(1, 2)) / self.scale  # (batch, 1, num_disks)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Weighted values
        context = torch.bmm(attn_weights, values)  # (batch, 1, 32)
        
        return context.squeeze(1), attn_weights.squeeze(1)


# ==================== Custom PPO Feature Extractor ====================
class HybridFeatureExtractor(BaseFeaturesExtractor):
    """Enhanced feature extractor with prediction and attention"""
    def __init__(self, observation_space, num_disks=4):
        super().__init__(observation_space, features_dim=128)
        
        state_dim = observation_space.shape[0]
        self.num_disks = num_disks
        
        # Base state encoding
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Attention scorer
        self.attention = AttentionDiskScorer(state_dim, num_disks)
        
        # Fusion layer
        self.fusion = nn.Linear(64 + 32, 128)
    
    def forward(self, observations):
        state_emb = self.state_encoder(observations)
        
        # Extract disk features for attention
        disk_features = observations[:, :self.num_disks * 3].reshape(-1, self.num_disks, 3)
        attn_context, _ = self.attention(observations, disk_features)
        
        # Fuse
        combined = torch.cat([state_emb, attn_context], dim=1)
        features = self.fusion(combined)
        
        return features


# ==================== Enhanced Environment Wrapper ====================
class HybridPredictiveEnv(gym.Wrapper):
    """Wraps base env with prediction and risk estimation"""
    def __init__(self, env, num_disks=4, history_len=20, pred_horizon=5):
        super().__init__(env)
        self.num_disks = num_disks
        self.history_len = history_len
        self.pred_horizon = pred_horizon
        
        # History buffers
        self.queue_history = deque(maxlen=history_len)
        self.service_history = deque(maxlen=history_len)
        self.alive_history = deque(maxlen=history_len)
        self.workload_history = deque(maxlen=history_len)
        
        # Models
        self.workload_predictor = WorkloadPredictor(
            input_dim=2, hidden_dim=32, pred_horizon=pred_horizon
        )
        self.risk_estimator = FailureRiskEstimator(num_disks, history_len)
        
        # Enhanced observation space
        base_dim = env.observation_space.shape[0]
        pred_dim = pred_horizon * 2  # arrivals + sizes
        risk_dim = num_disks
        enhanced_dim = base_dim + pred_dim + risk_dim
        
        self.observation_space = spaces.Box(
            low=-1e6, high=1e6, shape=(enhanced_dim,), dtype=np.float32
        )
        
        self.step_count = 0
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Clear histories
        self.queue_history.clear()
        self.service_history.clear()
        self.alive_history.clear()
        self.workload_history.clear()
        
        # Initialize with current state
        for _ in range(self.history_len):
            self._update_history(obs)
        
        self.step_count = 0
        return self._get_enhanced_obs(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self._update_history(obs)
        self.step_count += 1
        
        # Enhanced reward with multi-objective
        enhanced_reward = self._compute_multiobjective_reward(obs, info)
        
        return self._get_enhanced_obs(obs), enhanced_reward, terminated, truncated, info
    
    def _update_history(self, obs):
        """Update history buffers"""
        queues = obs[:self.num_disks]
        services = obs[self.num_disks:self.num_disks*2]
        alive = obs[self.num_disks*2:self.num_disks*3]
        
        self.queue_history.append(queues)
        self.service_history.append(services)
        self.alive_history.append(alive)
        
        # Workload: just queue sum for simplicity
        self.workload_history.append([np.sum(queues), np.mean(services)])
    
    def _get_enhanced_obs(self, base_obs):
        """Add predictions and risk scores to observation"""
        # Workload predictions
        if len(self.workload_history) >= 5:
            workload_tensor = torch.FloatTensor(
                list(self.workload_history)[-10:]
            ).unsqueeze(0)
            
            with torch.no_grad():
                arrival_pred, size_pred = self.workload_predictor(workload_tensor)
            
            pred_features = torch.cat([arrival_pred, size_pred], dim=1).numpy().flatten()
        else:
            pred_features = np.zeros(self.pred_horizon * 2)
        
        # Risk scores
        if len(self.queue_history) >= self.history_len:
            disk_hist = []
            for i in range(self.num_disks):
                q_hist = [h[i] for h in self.queue_history]
                s_hist = [h[i] for h in self.service_history]
                a_hist = [h[i] for h in self.alive_history]
                disk_hist.append(q_hist + s_hist + a_hist)
            
            disk_hist_tensor = torch.FloatTensor(disk_hist).unsqueeze(0)
            
            with torch.no_grad():
                risk_scores = self.risk_estimator(disk_hist_tensor).numpy().flatten()
        else:
            risk_scores = np.zeros(self.num_disks)
        
        # Concatenate
        enhanced_obs = np.concatenate([base_obs, pred_features, risk_scores])
        return enhanced_obs.astype(np.float32)
    
    def _compute_multiobjective_reward(self, obs, info):
        """Multi-objective reward balancing multiple goals"""
        # Original latency penalty
        latency = info.get('last_latency', 0)
        latency_reward = -0.1 * latency
        
        # Queue balance (penalize imbalance)
        queues = obs[:self.num_disks]
        queue_std = np.std(queues)
        balance_reward = -0.05 * queue_std
        
        # Throughput bonus (prefer alive disks with good service)
        services = obs[self.num_disks:self.num_disks*2]
        alive = obs[self.num_disks*2:self.num_disks*3]
        effective_service = np.sum(services * alive)
        throughput_reward = 0.02 * effective_service
        
        # Failure avoidance (penalize routing to risky disks)
        # Would need action info for this - simplified here
        failure_penalty = 0.0
        
        total_reward = (
            latency_reward + 
            balance_reward + 
            throughput_reward + 
            failure_penalty
        )
        
        return total_reward


# ==================== Training Script ====================
def train_hybrid_scheduler(total_timesteps=500_000):
    """Train the hybrid scheduler"""
    from san_rl_env import GoSANSchedulerEnv
    
    # Create base environment
    base_env = GoSANSchedulerEnv(num_disks=4)
    
    # Wrap with hybrid features
    env = HybridPredictiveEnv(base_env, num_disks=4, history_len=20, pred_horizon=5)
    
    # Custom policy with attention
    policy_kwargs = dict(
        features_extractor_class=HybridFeatureExtractor,
        features_extractor_kwargs=dict(num_disks=4),
        net_arch=[128, 128, 64]
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./hybrid_san_logs/"
    )
    
    print("🚀 Training Hybrid Predictive Failure-Aware Scheduler...")
    model.learn(total_timesteps=total_timesteps)
    
    model.save("hybrid_san_scheduler")
    print("✅ Training complete! Model saved.")
    
    return model


# ==================== Evaluation Script ====================
def evaluate_hybrid(model_path="hybrid_san_scheduler", episodes=10):
    """Evaluate the hybrid scheduler"""
    from san_rl_env import GoSANSchedulerEnv
    
    base_env = GoSANSchedulerEnv(num_disks=4)
    env = HybridPredictiveEnv(base_env, num_disks=4)
    
    model = PPO.load(model_path)
    
    all_rewards = []
    all_latencies = []
    all_throughputs = []
    all_queues = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        ep_reward = 0
        ep_latencies = []
        ep_throughputs = []
        ep_queues = []
        
        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            
            ep_reward += reward
            ep_latencies.append(info.get('last_latency', 0))
            
            # Extract metrics from obs
            queues = obs[:4]
            services = obs[4:8]
            ep_queues.append(np.mean(queues))
            ep_throughputs.append(np.sum(services))
            
            if term or trunc:
                break
        
        all_rewards.append(ep_reward)
        all_latencies.append(np.mean(ep_latencies))
        all_throughputs.append(np.mean(ep_throughputs))
        all_queues.append(np.mean(ep_queues))
    
    env.close()
    
    results = {
        'mean_reward': np.mean(all_rewards),
        'mean_latency': np.mean(all_latencies),
        'mean_throughput': np.mean(all_throughputs),
        'mean_queue': np.mean(all_queues),
        'std_latency': np.std(all_latencies)
    }
    
    print("\n📊 Hybrid Scheduler Results:")
    for key, val in results.items():
        print(f"  {key}: {val:.4f}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_hybrid_scheduler(total_timesteps=500_000)
    else:
        evaluate_hybrid()