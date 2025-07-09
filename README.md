<div align="center">
<h1>Train Your Humanoids with RL</h1>
<p>Train and deploy your own humanoid robot controller</p>

</div>

### Training on your own GPU

```bash
git clone git@github.com:<YOUR USERNAME>/humanoid_rl.git
cd humanoid_rl
```

4. Create a new Python environment (we require Python 3.11 or later and recommend using [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html))
5. Install the package with its dependencies:

```bash
pip install -r requirements.txt
pip install 'jax[cuda12]'  # If using GPU machine, install JAX CUDA libraries
python -c "import jax; print(jax.default_backend())" # Should print "gpu"
```

6. Train a policy:
  - Your robot should be walking within ~80 training steps, which takes 30 minutes on an RTX 4090 GPU and 60 minutes on an RTX 4070.
  - For 4090, use num_envs=2048, batch_size=256. For 4070, use num_envs=1024, batch_size=128.
  - Training runs indefinitely, unless you set the `max_steps` argument. You can also use `Ctrl+C` to stop it.
  - Click on the TensorBoard link in the terminal to visualize the current run's training logs and videos.
  - List all the available arguments with `python -m train --help`.
```bash
python -m train
```
```bash
# You can override default arguments like this
python -m train max_steps=100
```
7. To see the TensorBoard logs for all your runs the command or just click the link:
```bash
tensorboard --logdir humanoid_walking_task
```
8. To view your trained checkpoint in the interactive viewer:
- Use the mouse to move the camera around
- Hold `Ctrl` and double click to select a body on the robot, and then left or right click to apply forces to it.
```bash
python -m train run_mode=view load_from_ckpt_path=humanoid_walking_task/run_<number>/checkpoints/ckpt.bin
```

9. Convert your trained checkpoint to a `kinfer` model, which can be deployed on a real robot:

```bash
python -m convert /path/to/ckpt.bin /path/to/model.kinfer
```

10. Visualize the converted model in [`kinfer-sim`](https://docs.kscale.dev/docs/k-infer):

```bash
kinfer-sim assets/model.kinfer kbot --start-height 1.2 --save-video video.mp4
```

## Troubleshooting

If you encounter issues, please consult the [ksim documentation](https://docs.kscale.dev/docs/ksim#/) or reach out to k-scale on [Discord](https://url.kscale.dev/discord).

## Tips and Tricks

To see all the available command line arguments, use the command:

```bash
python -m train --help
```

To visualize running your model without using `kinfer-sim`, use the command:

```bash
python -m train run_mode=view
```

To see an example of a locomotion task with more complex reward tuning, see [kbot-joystick](https://github.com/kscalelabs/kbot-joystick) task which was generated from their template. It also contains a pretrained checkpoint that you can initialize training from by running

```bash
python -m train load_from_ckpt_path=assets/ckpt.bin
```

You can also visualize the pre-trained model by combining these two commands:

```bash
python -m train load_from_ckpt_path=assets/ckpt.bin run_mode=view
```

## Acknowledgments

We would like to express our sincere gratitude to [k-scale labs](https://kscale.dev/) for their groundbreaking work in humanoid robotics and reinforcement learning. This project builds upon their innovative research and open-source contributions to the robotics community. Their dedication to advancing the field of humanoid locomotion and making these technologies accessible to researchers and developers worldwide has been invaluable.

For more information about k-scale labs and their work, visit [https://kscale.dev/](https://kscale.dev/) or join their [Discord community](https://url.kscale.dev/discord).