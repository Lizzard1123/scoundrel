<div align="center">

<img src="images/scoundrel_banner.png" alt="Scoundrel Banner" width="100%" style="border-radius: 6px; box-shadow: 0 6px 20px rgba(0,0,0,0.4); border: 2px solid #8b5a2b;"/>

**A Python implementation of the Scoundrel card game**

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Version](https://img.shields.io/badge/version-0.1.0-orange.svg)](setup.py)

</div>

---

## 📋 Table of Contents

- [The Game](#-the-game)
- [Quick Start](#-quick-start)
- [Setup](#-setup)
- [Agent Approaches](#-agent-approaches)
- [Development](#-development)

---

## 🎮 The Game

Scoundrel is a dungeon-crawling card game where players navigate through rooms, collect cards, and battle monsters.

<div align="center">

![Terminal UI](images/tui.png)

**[📖 Official Scoundrel Rules PDF](http://www.stfj.net/art/2011/Scoundrel.pdf)**

</div>

## 🎯 Playing the Game

```bash
play [--seed SEED]
```

Play interactively in the terminal. Use `--seed` for deterministic deck shuffling (same seed = same game sequence).

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Lizzard1123/scoundrel.git
cd scoundrel

# Create conda environment
conda env create -f environment.yml
conda activate scoundrel

# Install package
pip install -e .

# Play the game
play
```

---

## ⚙️ Setup

### Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate scoundrel
pip install -e .
```

---

## 🤖 Agent Approaches

This implementation includes two AI agent approaches for playing Scoundrel:

### 🌳 MCTS Agent

Monte Carlo Tree Search agent with parallelization support for high-performance gameplay.

**Features:**
- ⚡ Parallel simulation workers
- 🧠 Transposition table caching
- 📊 Performance visualization tools
- 🎯 Configurable exploration parameters

**Console Scripts:**

| Command | Description |
|---------|-------------|
| `mcts` | Watch the MCTS agent play interactively |
| `mcts-eval` | Evaluate MCTS agent performance |
| `mcts-plot` | Visualize MCTS episode performance |

**Usage Examples:**

```bash
# Interactive gameplay
mcts --num-simulations 1000000 --num-workers 8

# Performance evaluation
mcts-eval --num-games 10 --verbose

# Episode visualization
mcts-plot --num-simulations 1000000 --batch 5 --confidence
```

<div align="center">

![MCTS Interactive Visualizer](images/mcts_tui.png)

*Interactive MCTS visualizer showing real-time gameplay*

</div>

<div align="center">

![MCTS Action Outlook](images/mcts_graph.png)

*Average outlook of the action picked at each turn*

</div>

<div align="center">

![MCTS Batch Variability](images/mcts_batch_graph.png)

*MCTS performance variability across multiple runs (6 out of 10 runs won)*

</div>

### 🎯 MCTS Data Collection (AlphaGo Style)

MCTS data collection pipeline for training AlphaGo-style neural network agents with supervised learning from expert gameplay.

<div align="center">

![MCTS Data Distribution](images/mcts_data.png)

*MCTS data collection showing game statistics and performance distribution*

</div>

**Data Collection Results:**

```python
Games: 5041

Statistics:
  Wins: 1316 (26.11%)
  Average score: -21.45
  Best score: 30
  Worst score: -188
  Average turns per game: 42.8
  Total turns: 215803
```

### 🧠 RL Agent

Reinforcement learning agent using a Transformer-based architecture with PPO training.

**Features:**
- 🔄 Transformer encoder for sequence planning
- 🎯 MLP for immediate tactical decisions
- 📈 TensorBoard integration
- 💾 Checkpoint management

**Training:**

```bash
cd scoundrel/rl/transformer_mlp/scripts
./train.sh
```

---

## 🛠️ Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black .
isort .
```

### Project Structure

```
scoundrel/
├── game/          # Core game logic
├── models/        # Game state models
├── rl/            # AI agents
│   ├── mcts/      # MCTS implementation
│   └── transformer_mlp/  # RL agent
└── ui/            # Terminal UI
```

---

<div align="center">

**Made with ❤️ for card game enthusiasts**

</div>
