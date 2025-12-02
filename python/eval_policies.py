import numpy as np
from san_rl_env import GoSANSchedulerEnv
from stable_baselines3 import PPO

NUM_EPISODES = 10
MAX_STEPS = 500
NUM_DISKS = 4


def rollout_policy(name, policy_fn, episodes=NUM_EPISODES, max_steps=MAX_STEPS):
    """
    Run multiple episodes for a given scheduling policy.
    Returns: (mean_reward, mean_latency)
    """
    rewards_per_ep = []
    latencies_per_ep = []

    for ep in range(episodes):
        env = GoSANSchedulerEnv(num_disks=NUM_DISKS)

        try:
            obs, _ = env.reset()
        except Exception as e:
            print(f"[{name}] Episode {ep}: reset failed:", e)
            env.close()
            continue

        ep_rewards = []
        ep_latencies = []

        for t in range(max_steps):
            try:
                action = policy_fn(obs, t)
            except Exception as e:
                print(f"[{name}] Episode {ep}: action failed:", e)
                break

            try:
                obs, r, terminated, truncated, info = env.step(action)
            except Exception as e:
                print(f"[{name}] Episode {ep}: step failed:", e)
                break

            ep_rewards.append(r)

            # last_latency is reported by Go server in metrics
            if "last_latency" in info:
                ep_latencies.append(info["last_latency"])

            if terminated or truncated:
                break

        env.close()

        if ep_rewards:
            rewards_per_ep.append(np.mean(ep_rewards))
        if ep_latencies:
            latencies_per_ep.append(np.mean(ep_latencies))

    if rewards_per_ep:
        mean_r = float(np.mean(rewards_per_ep))
    else:
        mean_r = float("nan")

    if latencies_per_ep:
        mean_l = float(np.mean(latencies_per_ep))
    else:
        mean_l = float("nan")

    return mean_r, mean_l


# ------------------ Baseline policies ------------------ #

def random_policy(obs, t):
    """Uniformly random disk."""
    return np.random.randint(NUM_DISKS)


def round_robin_policy_factory():
    """Returns a stateful round-robin policy closure."""
    state = {"i": 0}

    def policy(obs, t):
        a = state["i"]
        state["i"] = (state["i"] + 1) % NUM_DISKS
        return a

    return policy


def shortest_queue_policy(obs, t):
    """
    Queues are stored in the first NUM_DISKS entries of the state vector.
    Choose disk with minimum queue.
    """
    queues = np.array(obs[:NUM_DISKS])
    return int(np.argmin(queues))


# ------------------ Main script ------------------ #

if __name__ == "__main__":
    print("🔹 Evaluating baselines...")

    rr_policy = round_robin_policy_factory()

    r_rand, l_rand = rollout_policy("Random", random_policy)
    r_rr, l_rr = rollout_policy("RoundRobin", rr_policy)
    r_sq, l_sq = rollout_policy("ShortestQ", shortest_queue_policy)

    print(f"Random       -> mean_reward = {r_rand:.6f}, mean_latency = {l_rand:.6f}")
    print(f"RoundRobin   -> mean_reward = {r_rr:.6f}, mean_latency = {l_rr:.6f}")
    print(f"ShortestQ    -> mean_reward = {r_sq:.6f}, mean_latency = {l_sq:.6f}")

    print("\n🔹 Evaluating PPO RL policy...")

    try:
        model = PPO.load("ppo_san_rl")
    except Exception as e:
        print("Could not load PPO model:", e)
    else:

        def ppo_policy(obs, t):
            action, _ = model.predict(obs, deterministic=True)
            arr = np.array(action).astype(int)
            if arr.ndim == 0:
                return int(arr)
            else:
                return int(arr.flatten()[0])

        r_ppo, l_ppo = rollout_policy("PPO_RL", ppo_policy)
        print(f"PPO_RL       -> mean_reward = {r_ppo:.6f}, mean_latency = {l_ppo:.6f}")
