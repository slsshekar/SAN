import numpy as np
from stable_baselines3 import PPO
from san_rl_env import GoSANSchedulerEnv


def eval_rl(model_path, episodes=5, steps=500):
    model = PPO.load(model_path)
    rewards = []
    latencies = []

    for ep in range(episodes):
        env = GoSANSchedulerEnv(num_disks=4)
        obs, _ = env.reset()
        ep_r = 0.0
        ep_lat = []

        for t in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, _, _, info = env.step(action)
            ep_r += r
            ep_lat.append(info.get("avg_latency", 0.0))

        env.close()
        rewards.append(ep_r)
        latencies.append(np.mean(ep_lat))

    return float(np.mean(rewards)), float(np.mean(latencies))


if __name__ == "__main__":
    mean_r, mean_lat = eval_rl("ppo_go_san")
    print("PPO RL -> Mean reward:", mean_r, "| Mean latency:", mean_lat)
