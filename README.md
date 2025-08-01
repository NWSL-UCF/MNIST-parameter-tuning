# MNIST Parameter Tuning

This repository explores **hyperparameter tuning** for a **digit classification model** using the **MNIST dataset** (Modified National Institute of Standards and Technology). MNIST is a benchmark dataset composed of 70,000 handwritten digit images (60,000 for training and 10,000 for testing), commonly used in evaluating computer vision models.

The goal is to systematically evaluate how different model configurations affect training performance and accuracy.

---

## 🔧 Hyperparameters Tuned

We tune the following hyperparameters:

- **Number of Epochs**: `[1, 2, 4, 8, 16, 32]`
- **Optimizer**: `["adam", "sgd", "lbfgs"]`
- **Number of Hidden Layers**: `[1, 2, 4, 8, 16]`
- **Nodes per Hidden Layer**: `[10, 20, 30, 40, 50]`
- **Batch Size**: `[4, 8, 16, 32, 64, 128]`

This results in a total of **2,700 unique combinations**.

---

## Code Overview

### `main_grid.py` – Grid Search Runner (All-in-One Mode)

This script:
- Generates all possible hyperparameter combinations
- Trains and tests the model for each combination
- Saves logs and metrics (loss, accuracy, training time) under the `results/` directory
- Useful for running the entire sweep on a **single machine**

#### Example result structure for each combination:

```
results/
└── comb_123/
    ├── train.csv
    ├── test.csv
    └── params.json
```

This approach works on a single machine but becomes increasingly time-consuming for large-scale sweeps. While multi-processing (e.g., using 16–32 processes for 16–32 physical cores) can speed things up, you are still limited by your hardware’s capacity. For example, if your machine has 16 physical cores, running more than 16 parallel processes is not advisable. As a result, this method restricts you to running the entire experiment on a single machine, even if you have access to multiple machines that could help accelerate the task.

---

### `main.py` – Atomic Job Runner (One Combination at a Time)

This script runs **one specific configuration** at a time. It’s ideal for **distributed execution across multiple machines** or parallel processing via job servers.

Each process is responsible for:
- Training with a single hyperparameter combination
- Evaluating test accuracy
- Logging the result to a user-defined folder

#### Example usage:

```bash
python main.py \
    --epochs 4 \
    --optimizer adam \
    --hidden_layers 2 \
    --nodes_per_layer 10 \
    --batch_size 32 \
    --base_path ./experiment_runs/comb_42
```

This is especially useful when scaling across a **centralized job server**, where:
- A **master node** handles task scheduling
- Multiple **worker nodes** run `main.py` with assigned hyperparameter combinations
- Running using multiple physical machines has been discussed in job-distributor repository. 

---

## File Structure

```
MNIST-parameter-tuning/
├── main_grid.py         # All-in-one grid search over all parameter combinations
├── main.py              # CLI-based runner for one specific configuration
├── data_loader.py       # MNIST dataset loader (train/test)
├── model_utils.py       # MLP builder and optimizer selector
├── train.py             # Training loop with logging
└── test.py              # Model evaluation on test set
```

---

## Logs & Outputs

Each training run stores:

- `train.csv`: Per-iteration loss and elapsed time
- `test.csv`: Final test accuracy and test time
- `params.json`: Parameter configuration for the run

These logs are helpful for visualizing trends and selecting the best-performing hyperparameter set.

---
