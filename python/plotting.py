import matplotlib.pyplot as plt
import numpy as np

# =====================
#  ENTER YOUR VALUES HERE
# =====================

# Replace these with your actual evaluation results
policies = ["Random", "RoundRobin", "ShortestQ", "PPO_RL"]

# Example values from your output (update if newer values appear later)
mean_latency = [0.397090, 0.155014, 0.651744, 0.099886]  # lower is better
mean_reward = [-0.041872, -0.017235, -0.067436, -0.011246]  # higher (less negative) is better

# Optional: add more metrics if available from advanced_eval.py
# throughput = [...]
# cumulative_latency = [...]
# queue_sum = [...]

# =====================
#  PLOTTING
# =====================

def plot_metric(values, ylabel, title, filename, invert=False):
    """
    Plot a bar graph for given metric.
    invert=True means lower value is better (we visually swap here).
    """
    values_to_plot = -np.array(values) if invert else values

    plt.figure(figsize=(7, 4))
    bars = plt.bar(policies, values_to_plot, alpha=0.7)

    # Annotate bars with original values
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{val:.4f}', ha='center', va='bottom')

    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"📁 Saved: {filename}")

    plt.show()


# Plot latency (lower is better)
plot_metric(mean_latency, "Mean Latency (s)", "Latency Comparison", "latency_comparison.png", invert=True)

# Plot reward (higher is better)
plot_metric(mean_reward, "Mean Reward", "Reward Comparison", "reward_comparison.png")

# If you add more metrics like throughput, queue sum etc.:
# plot_metric(throughput, "Throughput", "Throughput Comparison", "throughput_comparison.png")
# plot_metric(queue_sum, "Final Queue Sum", "Queue Length Comparison", "queue_comparison.png", invert=True)
# plot_metric(cumulative_latency, "Total Latency", "Cumulative Latency", "cum_latency_comparison.png", invert=True)

print("\n🎉 All graphs generated successfully!")
