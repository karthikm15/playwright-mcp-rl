# Architecture Overview

## 1. System Overview

### Core Components
- **Environment wrapper** (`env/browser_env.py`): Interfaces with Playwright MCP server
- **Observation and action representation**: Placeholder - state from browser, discrete actions
- **Policy network** (`models/policy.py`): Transformer-based model (placeholder architecture)
- **Behavior cloning trainer** (`training/bc_trainer.py`): Supervised learning on demos
- **PPO trainer** (`training/ppo_trainer.py`): Policy gradient training
- **Rollout storage** (`training/rollout_buffer.py`): Stores trajectories
- **Evaluation scripts** (`scripts/evaluate.py`): Test policy on fixed tasks

## 2. Environment Design

### Interface
- `reset()`: Navigate to URL, return initial state
- `step(action)`: Execute action, return (state, reward, done, info)

### State
- Placeholder: Browser DOM representation or screenshot

### Actions
- Placeholder: Discrete action space (e.g., click field, type text, submit)

### Rewards
- Success: +1.0 on task completion
- Failure: -1.0 on timeout/error
- Step penalty: -0.01 per step

### Task Definition
- URL, field selectors, success condition (stored in `data/tasks/`)

## 3. Policy and Value Networks

### Architecture
- Single transformer-based model in `models/policy.py`
- Encoder: Transformer (placeholder)
- Policy head: Linear layer → action logits
- Value head: Linear layer → value estimate
- Output: (action_logits, value_estimate)

## 4. Training Stages

### Behavior Cloning
```
1. Load expert demonstrations
2. For each demo trajectory:
   - Compute cross-entropy loss: policy(obs) vs expert actions
   - Backprop and update policy
```

### PPO
```
1. Collect rollouts: policy interacts with environment
2. Compute advantages using GAE
3. For multiple epochs:
   - Compute clipped PPO objective
   - Update policy and value network
```

## 5. Evaluation

- Run policy on fixed test pages from `data/tasks/`
- Record success rate (completed / total)
- Minimal logging to console/file

## 6. File Structure

```
project_root/
  env/
    browser_env.py          # Environment wrapper
  models/
    policy.py               # Transformer policy (placeholder)
  training/
    bc_trainer.py           # Behavior cloning
    ppo_trainer.py          # PPO trainer
    rollout_buffer.py       # Trajectory storage
  data/
    demos/                  # Expert demonstrations
    tasks/                  # Task JSON definitions
  scripts/
    run_bc.py               # BC training script
    run_ppo.py              # PPO training script
    evaluate.py             # Evaluation script
  configs/
    default_bc.yaml         # BC hyperparameters
    default_ppo.yaml        # PPO hyperparameters
  utils/
    logging.py              # Logging utilities
    serialization.py        # Model/data I/O
```

## 7. Implementation Order

1. **Environment wrapper**: Minimal `BrowserEnv` with `reset()` and `step()`
2. **Model placeholder**: Basic `PolicyNetwork` class structure
3. **Rollout buffer**: Data structure for storing trajectories
4. **Behavior cloning**: Supervised training loop on demos
5. **PPO trainer**: Policy gradient update logic
6. **Evaluation**: Test script with success rate metric

