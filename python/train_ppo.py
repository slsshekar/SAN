import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from san_rl_env import GoSANSchedulerEnv


def make_env():
    # Single environment because Go server is single-port
    return GoSANSchedulerEnv(num_disks=4)


if __name__ == "__main__":
    env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
    )

    model.learn(total_timesteps=300_000)
    model.save("ppo_san_rl")

    env.close()
    print("✅ Training complete, model saved as ppo_san_rl.zip")
