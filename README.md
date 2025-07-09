# Humanoid RL: Reinforcement Learning for Humanoid

<div align="center">

![Humanoid Walking](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows%20%7C%20macOS-lightgrey)

**Train and deploy your own humanoid using state-of-the-art reinforcement learning techniques**

[Quick Start](#quick-start) • [Documentation](#documentation) • [Examples](#examples) • [Troubleshooting](#troubleshooting)

</div>

---

## Overview

This repository provides a comprehensive framework for training humanoid robots using reinforcement learning. Built on top of cutting-edge research in robotics and AI, it enables you to:

- **Train humanoid walking policies** from scratch using PPO (Proximal Policy Optimization)
- **Deploy trained models** on real robots using the kinfer platform
- **Visualize and analyze** training progress with TensorBoard integration
- **Simulate and test** policies in realistic environments

The framework is designed to be both research-friendly and production-ready, making it suitable for academic research, industrial applications, and educational purposes.

## Features

- 🚀 **Fast Training**: Achieve walking behavior in ~80 training steps (30-60 minutes on modern GPUs)
- 🎯 **Optimized Performance**: Pre-tuned hyperparameters for different GPU configurations
- 📊 **Comprehensive Monitoring**: Real-time training visualization with TensorBoard
- 🤖 **Real Robot Deployment**: Export trained models to kinfer format for hardware deployment
- 🔧 **Flexible Configuration**: Extensive command-line options for customization
- 📱 **Interactive Visualization**: Built-in viewer for policy evaluation and debugging

## Quick Start

### Prerequisites

- **Python**: 3.11 or later
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **OS**: Linux, Windows, or macOS

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<YOUR_USERNAME>/humanoid_rl.git
   cd humanoid_rl
   ```

2. **Set up Python environment**
   ```bash
   # Using conda (recommended)
   conda create -n humanoid_rl python=3.11
   conda activate humanoid_rl
   
   # Or using venv
   python -m venv humanoid_rl_env
   source humanoid_rl_env/bin/activate  # On Windows: humanoid_rl_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   
   # For GPU acceleration (necessary)
   pip install 'jax[cuda12]'
   
   # Verify GPU installation
   python -c "import jax; print(f'JAX backend: {jax.default_backend()}')" # should return "gpu"
   ```

### Training Your First Policy

1. **Start training**
   ```bash
   python -m train
   ```
   
   The training will begin automatically. You should see walking behavior emerge within approximately 80 training steps.

2. **Monitor progress**
   - TensorBoard logs are automatically generated
   - Click the TensorBoard link in the terminal to view real-time metrics
   - Training videos and performance graphs are available

3. **Stop training**
   - Press `Ctrl+C` to stop training at any time
   - Or set `max_steps` parameter to limit training duration

### GPU Configuration

For optimal performance, adjust parameters based on your GPU:

| GPU Model | `num_envs` | `batch_size` | Expected Training Time |
|-----------|------------|--------------|----------------------|
| RTX 4090  | 2048       | 256          | ~30 minutes          |
| RTX 4070  | 1024       | 128          | ~60 minutes          |
| RTX 3080  | 1024       | 128          | ~90 minutes          |

Example with custom parameters:
```bash
python -m train num_envs=2048 batch_size=256 max_steps=1000
```

## Documentation

### Command Line Interface

View all available options:
```bash
python -m train --help
```

### Key Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `num_envs` | Number of parallel environments | 1024 | 256-4096 |
| `batch_size` | Training batch size | 128 | 64-512 |
| `max_steps` | Maximum training steps | ∞ | 1-1000000 |
| `learning_rate` | Learning rate | 3e-4 | 1e-5-1e-3 |

### Training Visualization

Monitor your training progress:

```bash
# View TensorBoard logs
tensorboard --logdir humanoid_walking_task

# Or access via web browser at http://localhost:6006 or any provided links
```

### Model Evaluation

1. **Interactive visualization**
   ```bash
   python -m train run_mode=view
   ```

2. **Load and evaluate a checkpoint**
   ```bash
   python -m train run_mode=view load_from_ckpt_path=humanoid_walking_task/run_01/checkpoints/ckpt.bin
   ```

3. **Convert to deployment format**
   ```bash
   python -m convert /path/to/checkpoint.bin /path/to/model.kinfer
   ```

## Examples

### Using Pre-trained Models

Load and continue training from a pre-trained checkpoint:
```bash
python -m train load_from_ckpt_path=assets/ckpt.bin
```

Visualize a pre-trained model:
```bash
python -m train load_from_ckpt_path=assets/ckpt.bin run_mode=view
```

### Advanced Training Scenarios

For more complex locomotion tasks with advanced reward tuning, see the [kbot-joystick](https://github.com/kscalelabs/kbot-joystick) example, which demonstrates:
- Multi-objective reward functions
- Joystick control integration
- Advanced policy architectures

### Deployment

1. **Convert trained model**
   ```bash
   python -m convert humanoid_walking_task/run_001/checkpoints/ckpt.bin model.kinfer
   ```

2. **Simulate deployment**
   ```bash
   kinfer-sim model.kinfer kbot --start-height 1.2 --save-video output.mp4
   ```

3. **Deploy to real robot**
   - Follow the [kinfer documentation](https://docs.kscale.dev/docs/k-infer) for hardware deployment
   - Ensure proper safety protocols are in place

## Troubleshooting

### Common Issues

**JAX not detecting GPU**
```bash
# Verify CUDA installation
nvidia-smi
python -c "import jax; print(jax.devices())"
```

**OpenGL/MESA graphics errors**
```bash
# Error: libGL error: MESA-LOADER: failed to open iris
# Solution: Set LD_PRELOAD environment variable
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6:/usr/lib/x86_64-linux-gnu/libGLEW.so"
```

**Cython compilation errors**
```bash
# Error: Cython.Compiler.Errors.CompileError: /home/archer/.mujoco/mujoco-py/mujoco_py/cymj.pyx
# Solution: Install compatible Cython version
pip install "cython<3"
```

**Out of memory errors**
- Reduce `num_envs` and `batch_size`
- Close other GPU-intensive applications
- Consider using gradient accumulation

**Training not converging**
- Check learning rate settings
- Verify reward function parameters
- Ensure proper environment setup

### Performance Optimization

- **Memory usage**: Monitor with `nvidia-smi` during training
- **CPU utilization**: Ensure sufficient CPU cores for environment simulation
- **Disk space**: TensorBoard logs can grow large over time

### Getting Help

- **Documentation**: [ksim documentation](https://docs.kscale.dev/docs/ksim#/)
- **Community**: [k-scale Discord](https://url.kscale.dev/discord)
- **Issues**: Create an issue on this repository

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We would like to express our sincere gratitude to [k-scale labs](https://kscale.dev/) for their groundbreaking work in humanoid robotics and reinforcement learning. This project builds upon their innovative research and open-source contributions to the robotics community. Their dedication to advancing the field of humanoid locomotion and making these technologies accessible to researchers and developers worldwide has been invaluable.

For more information about k-scale labs and their work, visit [https://kscale.dev/](https://kscale.dev/) or join their [Discord community](https://url.kscale.dev/discord).

---
