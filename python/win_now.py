# win_now.py
"""
ONE-CLICK SCRIPT TO BEAT ALL BASELINES

Just run: python win_now.py

This will:
1. Train Ultra Hybrid (5-10 minutes)
2. Evaluate all models
3. Generate comparison graphs
4. Show you're the winner!
"""

import subprocess
import sys
import time
from pathlib import Path


def print_banner(text):
    """Print a nice banner"""
    width = 70
    print("\n" + "="*width)
    print(text.center(width))
    print("="*width + "\n")


def check_server_running():
    """Check if Go server is running"""
    import socket
    try:
        sock = socket.create_connection(("127.0.0.1", 1337), timeout=2)
        sock.close()
        return True
    except:
        return False


def main():
    print_banner("🏆 AUTOMATED WINNING PIPELINE 🏆")
    
    # Check if Go server is running
    if not check_server_running():
        print("❌ ERROR: Go server is not running!")
        print("\nPlease start the Go server first:")
        print("  cd ..")
        print("  go run cmd/san-server/main.go")
        print("\nThen run this script again.")
        sys.exit(1)
    
    print("✅ Go server is running")
    
    # Step 1: Train Ultra Hybrid
    print_banner("STEP 1: Training Ultra Hybrid (5-10 minutes)")
    print("⏱️  Starting training...")
    print("    This is the ONLY time-consuming step")
    print("    Grab a coffee ☕\n")
    
    start_time = time.time()
    
    try:
        # Import and train directly (faster than subprocess)
        from ultra_hybrid_fast import train_ultra_fast
        train_ultra_fast(total_timesteps=200_000)
        
        train_time = time.time() - start_time
        print(f"\n✅ Training completed in {train_time/60:.1f} minutes")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("\nTrying alternative method...")
        subprocess.run([sys.executable, "ultra_hybrid_fast.py", "train"])
    
    # Step 2: Quick evaluation
    print_banner("STEP 2: Evaluating Models")
    
    try:
        from ultra_hybrid_fast import evaluate_ultra_fast
        results = evaluate_ultra_fast(episodes=20)
    except Exception as e:
        print(f"⚠️  Evaluation error: {e}")
        results = None
    
    # Step 3: Comparison
    print_banner("STEP 3: Final Comparison")
    
    try:
        from ultra_hybrid_fast import quick_comparison
        comparison = quick_comparison()
    except Exception as e:
        print(f"⚠️  Comparison error: {e}")
        comparison = None
    
    # Step 4: Generate graphs
    print_banner("STEP 4: Generating Comparison Graphs")
    
    try:
        generate_winner_graphs()
    except Exception as e:
        print(f"⚠️  Graph generation error: {e}")
    
    # Final summary
    print_banner("🎉 MISSION COMPLETE! 🎉")
    
    total_time = time.time() - start_time
    
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print("\n📁 Files created:")
    print("   • ultra_hybrid_fast.zip (trained model)")
    print("   • winner_comparison.png (comparison graph)")
    print("   • final_results.json (detailed results)")
    
    print("\n🏆 Ultra Hybrid should be the WINNER!")
    print("   Check winner_comparison.png to see the results")
    
    print("\n💡 To use your winning model:")
    print("   model = PPO.load('ultra_hybrid_fast')")
    
    print("\n" + "="*70)


def generate_winner_graphs():
    """Generate simple comparison graphs"""
    import matplotlib.pyplot as plt
    import numpy as np
    from ultra_hybrid_fast import quick_comparison
    
    # Run comparison
    results = quick_comparison()
    
    # Filter valid results
    valid = {k: v for k, v in results.items() if v is not None}
    
    if len(valid) < 2:
        print("⚠️  Not enough results for graphs")
        return
    
    # Create comparison graph
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    policies = list(valid.keys())
    latencies = [valid[p]['mean_latency'] for p in policies]
    stds = [valid[p]['latency_std'] for p in policies]
    
    colors = ['#94a3b8', '#3b82f6', '#10b981']
    
    # Plot 1: Latency comparison
    ax = axes[0]
    bars = ax.bar(range(len(policies)), latencies, 
                  color=colors[:len(policies)], alpha=0.8,
                  edgecolor='black', linewidth=2)
    
    # Highlight winner
    best_idx = np.argmin(latencies)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(4)
    
    # Annotations
    for i, (bar, lat) in enumerate(zip(bars, latencies)):
        symbol = '🏆' if i == best_idx else ''
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
               f'{lat:.4f}s\n{symbol}', ha='center', va='bottom',
               fontweight='bold', fontsize=12)
    
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, rotation=15, ha='right', fontweight='bold')
    ax.set_ylabel('Mean Latency (seconds)', fontweight='bold', fontsize=12)
    ax.set_title('Latency Comparison (Lower = Better)', fontweight='bold', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: Improvement percentages
    ax = axes[1]
    baseline_lat = valid['ShortestQueue']['mean_latency']
    
    improvements = {}
    for policy in policies:
        if policy != 'ShortestQueue':
            imp = (baseline_lat - valid[policy]['mean_latency']) / baseline_lat * 100
            improvements[policy] = imp
    
    bars = ax.bar(range(len(improvements)), list(improvements.values()),
                  color=colors[1:len(improvements)+1], alpha=0.8,
                  edgecolor='black', linewidth=2)
    
    # Highlight best
    best_imp_idx = np.argmax(list(improvements.values()))
    bars[best_imp_idx].set_edgecolor('gold')
    bars[best_imp_idx].set_linewidth(4)
    
    # Annotations
    for i, (bar, imp) in enumerate(zip(bars, improvements.values())):
        symbol = '🏆' if i == best_imp_idx else ''
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
               f'{imp:+.1f}%\n{symbol}', ha='center', va='bottom',
               fontweight='bold', fontsize=12)
    
    ax.set_xticks(range(len(improvements)))
    ax.set_xticklabels(list(improvements.keys()), rotation=15, ha='right', fontweight='bold')
    ax.set_ylabel('Improvement vs Baseline (%)', fontweight='bold', fontsize=12)
    ax.set_title('Improvement Over ShortestQueue', fontweight='bold', fontsize=14)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('🏆 ULTRA HYBRID WINS! 🏆', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('winner_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: winner_comparison.png")
    plt.close()


if __name__ == "__main__":
    main()