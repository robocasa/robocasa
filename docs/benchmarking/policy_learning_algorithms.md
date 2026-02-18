# Policy Learning Algorithms

We provide official support for benchmarking the following policy learning algorithms: [Diffusion Policy](https://github.com/robocasa-benchmark/diffusion_policy), [Openpi](https://github.com/robocasa-benchmark/openpi), and [GR00T](https://github.com/robocasa-benchmark/Isaac-GR00T).

-------
## Diffusion Policy
We fork the official Diffusion Policy code base, hosted at [https://github.com/robocasa/diffusion_policy_dev](https://github.com/robocasa/diffusion_policy_dev).
### Recommended system specs
For training we recommend a GPU with at least 24 Gb of memory, but 48 Gb+ is prefered.
For inference we recommend a GPU with at least 8 Gb of memory.

### Installation
```
git clone https://github.com/robocasa-benchmark/diffusion_policy
cd diffusion_policy
pip install -e .
```

### Key files
- Training: [train.py](https://github.com/robocasa-benchmark/diffusion_policy/blob/main/train.py)
- Evaluation: [eval_robocasa.py](https://github.com/robocasa-benchmark/diffusion_policy/blob/main/eval_robocasa.py)

### Experiment workflow
```
# train model
python train.py \
--config-name=train_diffusion_transformer_bs192 \
task=robocasa/<dataset-soup>

# Evaluate model
python eval_robocasa.py \
--checkpoint <checkpoint-path> \
--task_set <task-set> \
--split <split>

# Report evaluation results
python diffusion_policy/scripts/get_eval_stats.py \
--dir <outputs-dir>
```


-------
## Openpi
We fork the official Openpi code base, hosted at [https://github.com/robocasa-benchmark/openpi](https://github.com/robocasa-benchmark/openpi). Our fork support training for **pi0**.

### Recommended system specs
For training we recommend a GPU with at least 80 Gb of memory (H100, H200, etc).
For inference we recommend a GPU with at least 8 Gb of memory.


### Installation
```
git clone https://github.com/robocasa-benchmark/openpi
cd openpi
pip install -e .
pip install -e packages/openpi-client/
```

### Key files
- Training: [scripts/train.py](https://github.com/robocasa-benchmark/openpi/blob/main/scripts/train.py)
- Evaluation: [scripts/serve_policy.py](https://github.com/robocasa-benchmark/openpi/blob/main/scripts/serve_policy.py) and [examples/robocasa/main.py](https://github.com/robocasa-benchmark/openpi/blob/main/examples/robocasa/main.py)
- Setting up configs: [src/openpi/training/config.py](https://github.com/robocasa-benchmark/openpi/blob/main/src/openpi/training/config.py)

### Experiment workflow
```
# train model
XLA_PYTHON_CLIENT_MEM_FRACTION=1.0 python scripts/train.py \
<dataset-soup> \
--exp-name=<exp-name>

# evaluate model
# part a: start inference server
python scripts/serve_policy.py \
--port=8000 policy:checkpoint \
--policy.config=<dataset-soup> \
--policy.dir=<checkpoint-path>

# part b: run evals on server
python examples/robocasa/main.py \
--args.port 8000 \
--args.task_set <task-set> \
--args.split <split> \
--args.log_dir <checkpoint-path>

# report evaluation results
python examples/robocasa/get_eval_stats.py \
--dir <checkpoint-path>
```

-------
## GR00T
We fork the official GR00T code base, hosted at [https://github.com/robocasa-benchmark/Isaac-GR00T](https://github.com/robocasa-benchmark/Isaac-GR00T). Our fork supports training for **GR00T N1.5**.

### Recommended system specs
For training we recommend a GPU with at least 80 Gb of memory (H100, H200, etc).
For inference we recommend a GPU with at least 8 Gb of memory.


### Installation
```
git clone https://github.com/robocasa-benchmark/Isaac-GR00T
cd groot
pip install -e .
```

### Key files
- Training: [scripts/gr00t_finetune.py](https://github.com/robocasa-benchmark/Isaac-GR00T/blob/main/scripts/gr00t_finetune.py)
- Evaluation: [scripts/run_eval.py](https://github.com/robocasa-benchmark/Isaac-GR00T/blob/main/scripts/run_eval.py)

### Experiment workflow
```
# train model
python scripts/gr00t_finetune.py \
--output-dir <experiment-path> \
--dataset_soup <dataset-soup> \
--max_steps <num-training-steps>

# evaluate model
python scripts/run_eval.py \
--model_path <checkpoint-path> \
--task_set <task-set> \
--split <split>

# report evaluation results
python gr00t/eval/get_eval_stats.py \
--dir <checkpoint-path>
```
