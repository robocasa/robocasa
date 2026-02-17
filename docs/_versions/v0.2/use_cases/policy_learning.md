# Policy Learning

Our [policy learning code](https://github.com/ARISE-Initiative/robomimic/tree/robocasa) provides official support for training and evaluating polices. This repo is a branch of robomimic, with modifications to train on RoboCasa datasets.

-------
## Installation
After installing the RoboCasa codebase, follow the instructions below:
```
git clone https://github.com/ARISE-Initiative/robomimic -b robocasa
cd robomimic
pip install -e .
```

-------
## Training
There are a number of algorithms to choose from. We offer official support for BC-Transformer. Users can also adapt the code to run Diffusion Policy, ACT, etc.

Before training, download datasets, see instructions [here](https://github.com/robocasa/robocasa?tab=readme-ov-file#datasets).

Each algorithm has its own config generator script. For example for BC-Transformer policy run:
```
python robomimic/scripts/config_gen/bc_xfmr_gen.py --name <experiment-name>
```
You can modify this file accordingly to train on datasets of your choice.

<div class="admonition note">
<p class="admonition-title">Debugging</p>

You can add --debug to generate small runs for debugging.

</div>

Running this script will generate training run commands. You can use this script for generating a single run or multiple (for comparing settings and hyperparameter tuning).
After running this script you just need to run the command(s) outputted.

Want to learn how to set your own config values and sweep them? Read this short [tutorial section](https://robomimic.github.io/docs/tutorials/hyperparam_scan.html#step-3-set-hyperparameter-values).

### Loading model checkpoint weights
Want to intialize your model with weights from a previous model checkpoint? Set the checkpoint path under `experiment.ckpt_path` in the config.

### Logging and viewing results
Read this short [tutorial page](https://robomimic.github.io/docs/tutorials/viewing_results.html).

-------
## Evaluation
We provide a script for running evaluations on existing model checkpoints:
```
python robomimic/scripts/config_gen/eval_ckpt.py --ckpt <ckpt-path> --name <experiment-name>
```
This will generate a command which you can subsequently run. 

<div class="admonition note">
<p class="admonition-title">Debugging</p>

You can add --debug to generate small runs for debugging.

</div>
