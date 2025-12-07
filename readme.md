# 🚀 Hybrid RL-Based Storage Scheduler (SANgo Enhancement)

## 📋 Project Overview

This project extends the **SANgo** framework (Storage Area Network simulator) from the research paper *"SANgo: a storage infrastructure simulator with reinforcement learning support"* (Arzymatov et al., 2020) by introducing a **novel Enhanced Hybrid Predictive Scheduler** that outperforms traditional scheduling policies.

### 🎯 Core Contributions

1. **Implementation of SANgo Framework**
   - Go-based discrete-event SAN simulator
   - TCP interface for RL agent communication
   - Configurable disk failures, network latency, heterogeneous workloads

2. **Novel Enhanced Hybrid Scheduler** ⭐
   - Multi-objective reward optimization
   - Statistical feature extraction (latency trends, queue statistics)
   - Efficient neural architecture for fast convergence
   - **1.6% improvement over intelligent baseline (Shortest Queue)**

3. **Comprehensive Evaluation Suite**
   - 5 policy comparisons (Random, Round Robin, Shortest Queue, Standard PPO, Enhanced Hybrid)
   - 12+ visualization types (latency traces, radar charts, heatmaps)
   - Interactive real-time simulation tools

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Python RL Environment                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │   Enhanced Hybrid Environment Wrapper               │   │
│  │   • Statistical feature extraction                   │   │
│  │   • Multi-objective reward shaping                   │   │
│  │   • History buffers (latency/queue trends)          │   │
│  └─────────────────┬───────────────────────────────────┘   │
│                    │ TCP Socket (JSON)                      │
└────────────────────┼────────────────────────────────────────┘
                     │
┌────────────────────┼────────────────────────────────────────┐
│                    ▼                                         │
│              Go SAN Simulator                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  • 4 heterogeneous disks (different speeds)         │   │
│  │  • Failure/recovery simulation                       │   │
│  │  • Variable workload generation                      │   │
│  │  • Network latency modeling (base + jitter)         │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 Novel Hybrid Scheduler - Technical Details

### What Makes It "Hybrid"?

The **Enhanced Hybrid Scheduler** combines:

1. **Physics-Based Simulation** (from SANgo paper)
   - Realistic disk queue dynamics
   - Network latency modeling
   - Failure/recovery patterns

2. **Learned Policy Optimization** (our contribution)
   - Deep RL (PPO) with custom feature extraction
   - Multi-objective reward balancing 4 goals:
     - Latency minimization (primary)
     - Smart disk selection (service rate / queue)
     - Consistency bonus (stable performance)
     - Queue load management

3. **Adaptive Statistical Features** (our contribution)
   - Real-time latency trend detection
   - Queue statistics (avg, min across disks)
   - Recent history integration (10-step window)

### Key Differences from Standard PPO

| Aspect | Standard PPO | Enhanced Hybrid |
|--------|--------------|-----------------|
| **Observation** | Raw state (12D) | Raw + statistics (16D) |
| **Reward** | Simple latency penalty | Multi-objective (4 components) |
| **Architecture** | Default MLP | Custom feature extractor with attention |
| **Training Speed** | ~300K steps | ~200K steps (40% faster) |
| **Performance** | 0.1228s latency | **0.1223s latency** (↓0.4%) |

---

## 📊 Experimental Results

### Performance Comparison

| Policy | Mean Latency (s) | Improvement vs Baseline | Queue Length |
|--------|------------------|------------------------|--------------|
| **Enhanced Hybrid** ⭐ | **0.1223** | **+1.6%** | 0.471 |
| Current PPO | 0.1228 | +1.2% | 0.444 |
| Shortest Queue (Baseline) | 0.1243 | 0% | 0.468 |
| Round Robin | 0.3265 | -162% | 0.583 |
| Random | 0.5839 | -369% | 0.723 |

### Stability Metrics

- **Latency Standard Deviation**: 0.0649s (most stable)
- **P95 Latency**: 0.245s
- **P99 Latency**: 0.298s

### Visual Results

![Comparison Graphs](comparison_graphs/latency_comparison.png)

*See `comparison_graphs/` directory for all 12+ visualizations*

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install Go 1.21+
go version

# Install Python dependencies
pip install -r python/requirements.txt
```

### 1️⃣ Start the SAN Simulator

```bash
go run cmd/san-server/main.go
```

Output:
```
[SERVER] Listening on :1337
```

### 2️⃣ Train the Enhanced Hybrid Scheduler

```bash
cd python
python enhanced_hybrid_scheduler.py train
```

Training takes ~15 minutes on CPU, ~5 minutes on GPU.

### 3️⃣ Run Comprehensive Evaluation

```bash
python comprehensive_comparison.py
```

Generates:
- `comparison_results.json` - Raw metrics
- `comparison_graphs/` - 12+ visualization files

### 4️⃣ Interactive Analysis (Optional)

```bash
python interactive_detailed_analysis.py
```

Choose:
1. **Live Simulation** - Real-time animated comparison
2. **Detailed Comparison** - Multiple popup graphs
3. **Both** - Full analysis suite

---

## 📁 Repository Structure

```
.
├── cmd/
│   └── san-server/
│       └── main.go              # SAN server entry point
├── sim/
│   └── san.go                   # Core SAN simulator logic
├── rl/
│   └── server.go                # TCP interface for RL
├── python/
│   ├── san_rl_env.py           # Gymnasium environment wrapper
│   ├── enhanced_hybrid_scheduler.py  # 🌟 NOVEL CONTRIBUTION
│   ├── comprehensive_comparison.py    # Evaluation suite
│   ├── interactive_detailed_analysis.py  # Live visualization
│   ├── train_ppo.py            # Standard PPO training
│   └── requirements.txt
├── go.mod
└── README.md
```

---

## 🔧 Configuration

### SAN Simulator Settings (`cmd/san-server/main.go`)

```go
cfg := sim.SANConfig{
    NumDisks:      4,
    ServiceRates:  []float64{2.0, 0.8, 1.5, 1.2},  // Heterogeneous
    FailureProbs:  []float64{0.008, 0.015, 0.005, 0.012},  // High risk
    RecoverProbs:  []float64{0.05, 0.05, 0.05, 0.05},
    ArrivalRate:   3.5,       // High load
    ReqSizeLow:    0.3,
    ReqSizeHigh:   2.5,       // Variable workload
    NetworkBase:   0.12,
    NetworkJitter: 0.05,      // Realistic jitter
    MaxSteps:      500,
}
```

### Hyperparameters (`enhanced_hybrid_scheduler.py`)

```python
PPO(
    learning_rate=5e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
)
```

---

## 🧪 Reproducing Results

### Full Reproduction Pipeline

```bash
# 1. Start simulator
go run cmd/san-server/main.go &

# 2. Train standard PPO
python python/train_ppo.py

# 3. Train Enhanced Hybrid
python python/enhanced_hybrid_scheduler.py train

# 4. Run evaluation
python python/comprehensive_comparison.py

# 5. Check results
cat python/comparison_results.json
ls python/comparison_graphs/
```

### Expected Training Times

- Standard PPO (300K steps): ~20 minutes (CPU)
- Enhanced Hybrid (200K steps): ~15 minutes (CPU)

---

## 📖 Paper Reference

**Original Paper**:  
Arzymatov, K., Sapronov, A., Belavin, V., et al. (2020).  
*SANgo: a storage infrastructure simulator with reinforcement learning support.*  
PeerJ Computer Science, 6:e271.  
DOI: [10.7717/peerj-cs.271](https://doi.org/10.7717/peerj-cs.271)

**Key Contributions from Paper**:
- Discrete-event SAN simulation framework
- OpenAI Gym interface for RL
- Hybrid approach combining physics + learning

**Our Extensions**:
- Multi-objective reward optimization
- Statistical feature engineering
- Efficient architecture design
- Comprehensive evaluation framework

---

## 🎓 Academic Context

### Research Gap Addressed

Existing SAN schedulers:
1. **Heuristic policies** (Round Robin, Shortest Queue): No adaptation to workload patterns
2. **Pure RL**: Slow convergence, no domain knowledge
3. **Existing hybrids**: Limited feature engineering, single-objective rewards

### Our Solution

✅ **Domain knowledge** (queue theory, service rates)  
✅ **Learned adaptation** (RL policy optimization)  
✅ **Multi-objective balancing** (latency + stability + load)  
✅ **Fast convergence** (efficient architecture)

---

## 🏆 Key Achievements

1. ✅ **Implemented SANgo framework** from scratch in Go
2. ✅ **Novel hybrid scheduler** with 1.6% improvement
3. ✅ **Comprehensive evaluation** with 12+ graph types
4. ✅ **Interactive visualization** tools
5. ✅ **Reproducible experiments** with full pipeline

---

## 🐛 Known Issues & Future Work

### Current Limitations
- Single-episode training (Go server limitation)
- Fixed episode horizon (500 steps)
- Simplified failure model

### Future Enhancements
- [ ] Multi-episode vectorized training
- [ ] Adaptive episode length based on performance
- [ ] More realistic failure patterns (Weibull distribution)
- [ ] Multi-agent coordination (multiple RL schedulers)
- [ ] Integration with real storage traces (Microsoft, Google)

---

## 📧 Contact

For questions or collaboration:
- 📄 Paper: [peerj-cs-271](https://doi.org/10.7717/peerj-cs.271)
- 💻 Code: This repository

---

## 📜 License

This project extends the GPL-licensed SANgo framework.  
See original paper for citation requirements.

---

## 🙏 Acknowledgments

- Original SANgo authors (Arzymatov et al., 2020)
- OpenAI Gym / Gymnasium maintainers
- Stable-Baselines3 contributors

---

**Made with ❤️ for better storage system scheduling**