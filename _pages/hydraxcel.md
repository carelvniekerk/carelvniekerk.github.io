---
layout: archive
title: "HydraXcel"
permalink: /hydraxcel/
author_profile: false
redirect_from:
  - /hydraxcel
---

# HydraXcel: Accelerate Your Experiments with Hydra and Accelerate

[![PyPI version](https://img.shields.io/pypi/v/hydraxcel.svg)](https://pypi.org/project/hydraxcel/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Build](https://img.shields.io/github/actions/workflow/status/carelvniekerk/HydraXcel/ci.yml?branch=main)](https://github.com/carelvniekerk/HydraXcel/actions)

**HydraXcel** is a configuration-driven deep learning experiment launcher that combines the power of [Facebook Hydra](https://hydra.cc/) for configuration management, HuggingFace **Accelerate** for effortless multi-GPU/distributed training, and the UV workflow for streamlined execution. Its goal is to make running fast, reproducible ML experiments simple and consistent. With HydraXcel, you can define your experiment setup declaratively (via YAML files or Python dataclasses), automatically log results to MLflow or Weights & Biases, and launch training on one or many GPUs with minimal code changes.

## Key Features

- **Hydra-Based Configuration:** Use either **YAML configuration files** or **Python dataclass** configs to define all experiment parameters. HydraXcel wraps your `main` function to parse configs and set up output directories automatically, so each run’s results are organized in a timestamped folder (e.g. `outputs/<job_name>/<param_name>=<value>/<timestamp>`). You get Hydra’s full flexibility in overriding configs via the command line for sweeps, hyperparameter tuning, etc.

- **Accelerate Integration for Multi-GPU:** HydraXcel integrates smoothly with HuggingFace Accelerate. The **`launch`** function takes care of spawning distributed training jobs with Accelerate. This allows your training script to run on multiple GPUs, TPUs, or even multi-machine setups without manual boilerplate. You write your code as if it runs on a single device, and Accelerate (via HydraXcel’s launcher) handles device placement, synchronization, and distributed execution.

- **Seamless Experiment Logging:** HydraXcel can initialize experiment tracking with **Weights & Biases (wandb)** or **MLflow** with one argument. By setting `logging_platform` to `LoggingPlatform.WANDB` or `MLFLOW`, HydraXcel will automatically: (a) initialize a wandb run or MLflow run at the start of your experiment, (b) log the entire Hydra config (all hyperparameters) to the tracking service, and (c) capture system environment info for reproducibility (e.g. git commit, Python version, CUDA devices).

- **Robust Output & Error Handling:** The toolkit sets up Python logging (using `colorlog` for nice console output) and integrates Hydra’s logging configuration. It also installs a global exception hook so that if your code crashes, the exception is logged with traceback to help debugging. System information (hostname, virtual environment details, etc.) is logged at startup for each run.

- **UV-Friendly Launching:** HydraXcel is designed to work well with the **UV** workflow. Its launcher allows you to define **“uv run” scripts** for your experiments easily. For example, your `pyproject.toml` can define an entry point for training and one for launching an MLflow server:

  ```toml
  [project.scripts]
  train = "project_name.scripts:train_command"
  mlflow_server = "hydraxcel.logging:run_mlflow_server"
  ```

With this setup, running experiments or servers is as easy as typing:  

```bash
uv run train
uv run mlflow_server
```

- **Future-Proof Launcher Architecture:** HydraXcel anticipates extending to Hydra Launchers (Slurm, Azure ML, etc.).

## Installation

HydraXcel can be included in your project as a git submodule/source. It is suggested to use `uv` to manage the dependency. In this way `hydraxcel` is added to the dependencies in the `pyproject.toml` together with the following source information.

```toml
[tool.uv.sources]
hydraxcel = { git = "https://github.com/carelvniekerk/HydraXcel" }
```

*(Requires Python 3.13+; currently version 0.1.0a).*

## Usage Guide

1. **Define your configuration** – either as a dataclass in Python or as YAML files.  
2. **Wrap your main function** with HydraXcel’s decorator (`hydraxcel_main`).  
3. **Run your experiment** with Python or UV CLI.  
4. **(Optional) Use `launch` for distributed training with Accelerate.**  
5. **(Optional) Launch MLflow tracking server.**

### 1. Using a Dataclass Config

```python
# train.py
from dataclasses import dataclass
from hydraxcel import hydraxcel_main, LoggingPlatform

@dataclass
class ModelConfig:
    type: str = "ResNet50"
    pretrained: bool = True

@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    model: ModelConfig = ModelConfig()

@hydraxcel_main(
    "MyProject",
    config_class=TrainConfig,
    output_dir_keys=["batch_size"],
    logging_platform=LoggingPlatform.WANDB,
)
def main(cfg):
    print(f"Training for {cfg.epochs} epochs on batch size {cfg.batch_size}...")
```

- `@hydraxcel_main` registers the config, parses overrides, sets up logging, and creates structured output directories.  
- Logging is automatically configured (to WandB, MLflow, or local logs).  
- **Note:** `hydraxcel_main` does *not* initialise an `Accelerator`. Use the `launch` function if you need distributed training.

**Run examples:**
```bash
uv run train
uv run train learning_rate=0.0005 batch_size=64 model.type="ResNet101"
uv run train -m epochs=5,10,20
```

*(Using the train script defined in the `pyproject.toml`.)*

### 2. Using YAML Configs

```yaml
# conf/train.yaml
epochs: 10
batch_size: 32
learning_rate: 0.001
model:
  type: "ResNet50"
  pretrained: true
```

```python
# train.py
from hydraxcel import hydraxcel_main

@hydraxcel_main(
    "MyProject",
    hydra_configs_dir="conf",
    output_dir_keys=["model.type"],
    logging_platform="mlflow",
)
def main(cfg):
    print(cfg)
```

Run:
```bash
python train.py
python train.py learning_rate=0.0005 model.type="ResNet101"
```

### 3. Launching with Accelerate

For distributed training, use HydraXcel’s `launch`:

```python
# scripts/__init__.py
from pathlib import Path
from hydraxcel import launch

SCRIPTS_DIR = Path(__file__).parent

launch_train = launch(
    SCRIPTS_DIR / "train.py",
    config_name="accelerate",
)
```

Add in `pyproject.toml`:
```toml
[project.scripts]
myproject-train = "myproject.scripts:launch_train"
```

Run with UV:
```bash
uv run myproject-train -- accelerate.num_processes=4
```

HydraXcel provides Accelerate configs (`accelerate.yaml`, presets for GPU, FP16, etc.) in `examples/configs`. Dataclasses can also be used to configure accelerate.

### 4. MLflow Tracking Server

Expose the built-in MLflow server runner:

```toml
[project.scripts]
mlflow_server = "hydraxcel.logging:run_mlflow_server"
```

Run:
```bash
uv run mlflow_server
```

Server at [http://127.0.0.1:5000](http://127.0.0.1:5000). Override with:
```bash
uv run mlflow_server host=0.0.0.0 port=8080
```

## License

HydraXcel is released under the **Apache License 2.0**. This permissive licence allows free academic and commercial use with attribution, aligning with Hydra and HuggingFace projects.

---

*HydraXcel is a work in progress (v0.1.0a). This README currently serves as the main documentation. Contributions welcome!* 🚀

