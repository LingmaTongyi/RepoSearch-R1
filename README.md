<h1 style="text-align: center;">RepoSearch-R1: A RepoQA-Agent based on Reinforcement Learning Driven by Monte-carlo Tree Search</h1>

**RepoSearch-R1** is a cold-start agentic reinforcement learning framework that enables large language models to efficiently navigate and extract information from complex code repositories through multi-turn tool interactions. Built on the verl framework, it integrates Monte Carlo Tree Search (MCTS) into the Group Relative Policy Optimization (GRPO) pipeline to generate diverse, high-quality reasoning trajectories via self-training without requiring model distillation or external supervision.

This implementation is based on the research paper: **"RepoSearch-R1: A RepoQA-Agent based on Reinforcement Learning Driven by Monte-carlo Tree Search"**

## Key Innovations

RepoSearch-R1 addresses the limitations of existing approaches through several key innovations:

- **🎯 Cold-Start Training**: Eliminates the need for costly distillation from larger LLMs, addressing data compliance concerns in enterprise environments

- **🌳 MCTS-Guided Exploration**: Integrates Monte Carlo Tree Search into the GRPO pipeline for systematic exploration of diverse reasoning trajectories

- **🛠️ Specialized Tool Suite**: Five repository exploration tools designed for semantic understanding and efficient code navigation

## Performance Results

Comprehensive evaluation on repository question-answering tasks demonstrates significant improvements:

- **16.0%** enhancement over no-retrieval methods
- **10.24%** improvement over iterative retrieval methods  
- **33%** increase in training efficiency compared to general agentic RL approaches
- Maintains robust exploration diversity and answer completeness across repository-level reasoning tasks

## MCTS Framework

The RepoSearch-R1 framework consists of three main stages:

### 1. MCTS-Guided Rollout
- **Selection**: Navigate from root to leaf using exploration-decay UCT formula
- **Expansion**: Generate child nodes with self-critique mechanism
- **Simulation**: Complete rollout using current policy until terminal state
- **Backpropagation**: Update node values with reward calculations

### 2. Trajectory Selection and Reward Computation
- Multiple rollout trajectories containing thought-action-observation cycles
- LLM-as-a-judge answer quality assessment combined with intermediate process rewards
- Selection of most promising exploration paths for training

### 3. Advantage Computation and GRPO Training
- Group-based advantage estimation with relative quality evaluation
- KL-free Group Relative Policy Optimization for policy updates
- Self-training without external supervision or distilled data

## RepoQA-Agent Tools

The framework includes five specialized tools for repository exploration:

| Tool Name | Parameters | Description |
|-----------|------------|-------------|
| `review_file` | `file_path`, `start_lineno`, `end_lineno` | Review code in a specific file from start to end line |
| `search_keyword_in_folder` | `keyword`, `folder_path` | Search for a keyword in all files within a folder |
| `list_files_in_folder` | `folder_path` | List all files and subdirectories in a folder |
| `search_symbol_in_file` | `symbol`, `file_path` | Search for code symbols (functions, variables) in a file |
| `search_file_in_folder` | `file_name`, `folder_path` | Search for specific files in subdirectories |


## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA-compatible GPU (recommended)
- Access to a language model (Qwen, Llama, etc.)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/mcts-rollout-verl-json_tool.git
cd mcts-rollout-verl-json_tool

# Install dependencies
pip install -r requirements.txt

# Install verl framework
pip install verl
```

### Quick Start

1. **Prepare your repository dataset** following the CoReQA format
2. **Configure MCTS parameters** in your training config:

```yaml
actor_rollout_ref:
  rollout:
    mcts_agent:
      num_rollouts: 40
      exploration_weight: 2.0
      max_depth_allowed: 10
      max_children_num: 2
      enable_reflection: true
```

3. **Run RepoSearch-R1 training**:

```bash
# Example training command
python examples/repoqa/run_qwen3-8b_repoqa_bash_tool_agent_swanlab.sh
```

## Implementation Details

### Core Components

- **`verl/experimental/agent_loop/mcts_tool_agent_loop.py`**: Main MCTS implementation with UCT selection and self-critique mechanisms
- **`verl/tools/repoqa_tool.py`**: Repository exploration tools with LLM-as-a-judge reward calculation
- **`examples/repoqa/`**: Training scripts and configuration files for repository QA tasks

### MCTS Configuration

Key parameters for tuning the MCTS behavior:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_rollouts` | 40 | Number of MCTS rollout iterations |
| `exploration_weight` | 2.0 | Initial UCT exploration weight (w₀) |
| `max_depth_allowed` | 10 | Maximum tree depth for exploration |
| `max_children_num` | 2 | Number of children generated per node |
| `enable_reflection` | true | Enable self-critique mechanism |

