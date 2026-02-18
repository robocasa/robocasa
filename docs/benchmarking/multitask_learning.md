<!-- Add class so secondary (right) sidebar is hidden -->
<script>document.body.classList.add("rc-multitask-learning");</script>

# Multi-Task Learning

In the multi-task learning benchmark, we study training on multi-task pretraining datasets.
We do policy learning on the [Human Pretraining Datasets](../datasets/datasets_overview.html#human-datasets), which data across 300 tasks, comprising 65 atomic tasks and 235 composite tasks.
For each task, we provide 100 task demonstrations per task, resulting in **482 hours of total data**.

-------
<!-- ## Benchmark results and checkpoints

We provide support for benchmarking across Diffusion Policy, Openpi, and GR00T N1.5. Here is a summary of our benchmarking results **(average task success rate, in %)**. We share the model checkpoints for reference.

<table class="docutils rc-benchmark-table">
  <thead>
    <tr>
      <th><strong>Task Split</strong></th>
      <th><strong>Diffusion Policy</strong></th>
      <th><strong>π<span class="rc-pi-subnum">0</span></strong></th>
      <th><strong>GR00T N1.5</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code class="rc-benchmark-split rc-benchmark-atomic">Atomic-Seen</code></td>
      <td>15.7%</td>
      <td>34.6%</td>
      <td>43.0%</td>
    </tr>
    <tr>
      <td><code class="rc-benchmark-split rc-benchmark-comp-seen">Composite-Seen</code></td>
      <td>0.2%</td>
      <td>6.1%</td>
      <td>9.6%</td>
    </tr>
    <tr>
      <td><code class="rc-benchmark-split rc-benchmark-comp-unseen">Composite-Unseen</code></td>
      <td>1.25%</td>
      <td>1.1%</td>
      <td>4.4%</td>
    </tr>
    <tr>
      <td><strong>Average</strong></td>
      <td>6.1%</td>
      <td>14.8%</td>
      <td>20.0%</td>
    </tr>
    <tr>
      <td><strong>Model Checkpoint</strong></td>
      <td><a href="">TBD</a></td>
      <td><a href="">TBD</a></td>
      <td><a href="https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/gr00t_n1-5/multitask_learning/checkpoint-120000">Link</a></td>
    </tr>
  </tbody>
</table>

------- -->
## Benchmark instructions

### Diffusion Policy

#### Guidelines
* We use a batch size of 192 and train for 250k steps
* We evaluate the model in **pretrain** scenes

#### Train model
```
python train.py \
--config-name=train_diffusion_transformer_bs192 \
task=robocasa/pretrain_human300
```

#### Evaluate model
```
python eval_robocasa.py \
--checkpoint <checkpoint-path> \
--task_set atomic_seen composite_seen composite_unseen \
--split pretrain
```

#### Report evaluation results
```
python diffusion_policy/scripts/get_eval_stats.py \
--dir <outputs-dir>
```

-------
### Openpi

#### Guidelines
* We use a batch size of 64 and train for 75k steps
* We evaluate the model in **pretrain** scenes

#### Train model
```
XLA_PYTHON_CLIENT_MEM_FRACTION=1.0 python scripts/train.py \
pi0_robocasa_pretrain_human300 \
--exp-name=multitask_learning
```

#### Evaluate model
```
# part a: start inference server
python scripts/serve_policy.py \
--port=8000 policy:checkpoint \
--policy.config=pi0_robocasa_pretrain_human300 \
--policy.dir=expdata/pi0_robocasa_pretrain_human300/multitask_learning/75000

# part b: run evals on server
python examples/robocasa/main.py \
--args.port 8000 \
--args.task_set atomic_seen composite_seen composite_unseen \
--args.split pretrain \
--args.log_dir expdata/pi0_robocasa_pretrain_human300/multitask_learning/75000
```

#### Report evaluation results
```
python examples/robocasa/get_eval_stats.py \
--dir expdata/pi0_robocasa_pretrain_human300/multitask_learning/75000
```

-------
### GR00T

#### Guidelines
* We use a batch size of 128 and train for 120k steps
* We evaluate the model in **pretrain** scenes

#### Train model
```
python scripts/gr00t_finetune.py \
--output-dir expdata/multitask_learning \
--dataset_soup pretrain_human300 \
--max_steps 120000
```

#### Evaluate model
```
python scripts/run_eval.py \
--model_path expdata/multitask_learning/checkpoint-120000 \
--task_set atomic_seen composite_seen composite_unseen \
--split pretrain
```

#### Report evaluation results
```
python gr00t/eval/get_eval_stats.py \
--dir expdata/multitask_learning/checkpoint-120000
```