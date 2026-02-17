# Lifelong Learning

In the lifelong learning benchmark, we study learning tasks over a sequence of phases.
We do policy learning on a subset of the [Human Pretraining Datasets](../datasets/datasets_overview.html#human-datasets), over 125 task total.
We begin with learning on simple atomic tasks invovling the invokation of a single skill, and progressively learn longer horizon tasks which involve invoking a longer sequence of skills.
Specifically we learn over four phases:
- **Phase 1**: tasks with 1 stage, i.e., atomic tasks (we use all 65 atomic tasks; see [here](https://github.com/robocasa/robocasa/blob/main/robocasa/utils/dataset_registry.py) for the list of tasks)
- **Phase 2**: tasks with 2-3 stages (we use 20 tasks total; see [here](https://github.com/robocasa/robocasa/blob/main/robocasa/utils/dataset_registry.py) for the list of tasks)
- **Phase 3**: tasks with 4-5 stages (we use 20 tasks total; see [here](https://github.com/robocasa/robocasa/blob/main/robocasa/utils/dataset_registry.py) for the list of tasks)
- **Phase 4**: tasks with 6+ stages (we use 20 tasks total; see [here](https://github.com/robocasa/robocasa/blob/main/robocasa/utils/dataset_registry.py) for the list of tasks)

After each phase, we evaluate the policy on all of the tasks learned from the first phase through the latest phase.
The goal is to learn tasks in new phases effectively while retaining knowledge about all previous phases.

-------
<!-- ## Benchmark results and checkpoints
We have performed the lifelong learning benchmark using GR00T N1.5. Here is a summary of our benchmarking results **(average task success rate, in %)**. We share the model checkpoints for reference.

<table class="docutils rc-benchmark-table">
  <thead>
    <tr>
      <th><strong>Phase</strong></th>
      <th><strong>Atomic Tasks</strong></th>
      <th><strong>2-3 Stage Tasks</strong></th>
      <th><strong>4-5 Stage Tasks</strong></th>
      <th><strong>6+ Stage Tasks</strong></th>
      <th><strong>Checkpoint</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Phase1</td>
      <td>41.5</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td><a href="https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/gr00t_n1-5/lifelong_learning/phase1/checkpoint-100000">Link</a></td>
    </tr>
    <tr>
      <td>Phase2</td>
      <td>13.9</td>
      <td>24.5</td>
      <td>-</td>
      <td>-</td>
      <td><a href="https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/gr00t_n1-5/lifelong_learning/phase2/checkpoint-60000">Link</a></td>
    </tr>
    <tr>
      <td>Phase3</td>
      <td>13.9</td>
      <td>4.8</td>
      <td>11.3</td>
      <td>-</td>
      <td><a href="https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/gr00t_n1-5/lifelong_learning/phase3/checkpoint-60000">Link</a></td>
    </tr>
    <tr>
      <td>Phase4</td>
      <td>10.6</td>
      <td>1.7</td>
      <td>2.7</td>
      <td>4.3</td>
      <td><a href="https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/gr00t_n1-5/lifelong_learning/phase4/checkpoint-60000">Link</a></td>
    </tr>
  </tbody>
</table>

------- -->
## Benchmark instructions

### GR00T

#### Guidelines
- We use a batch size of 128
- For the initial phase of training (phase 1), we train for 100k steps
- For all subsequent phases of training (phase 2, 3, 4) we train for 60k steps
- We always evaluate the models in **pretrain** scenes

#### Train model
```
# phase 1 initial training
python scripts/gr00t_finetune.py \
--output-dir expdata/lifelong_learning/phase1 \
--dataset_soup lifelong_learning_phase1 \
--max_steps 100000

# phase 2 fine-tuning
python scripts/gr00t_finetune.py \
--output-dir expdata/lifelong_learning/phase2 \
--base_model_path expdata/lifelong_learning/phase1/checkpoint-100000 \
--dataset_soup lifelong_learning_phase2 \
--max_steps 60000

# phase 3 fine-tuning
python scripts/gr00t_finetune.py \
--output-dir expdata/lifelong_learning/phase3 \
--base_model_path expdata/lifelong_learning/phase2/checkpoint-60000 \
--dataset_soup lifelong_learning_phase3 \
--max_steps 60000

# phase 4 fine-tuning
python scripts/gr00t_finetune.py \
--output-dir expdata/lifelong_learning/phase4 \
--base_model_path expdata/lifelong_learning/phase3/checkpoint-60000 \
--dataset_soup lifelong_learning_phase4 \
--max_steps 60000
```

#### Evaluate model
```
# evaluate model after phase1
python scripts/run_eval.py \
--model_path expdata/lifelong_learning/phase1/checkpoint-100000 \
--task_set lifelong_learning_phase1 \
--split pretrain

# evaluate model after phase2
python scripts/run_eval.py \
--model_path expdata/lifelong_learning/phase2/checkpoint-60000 \
--task_set lifelong_learning_phase1 lifelong_learning_phase2 \
--split pretrain

# evaluate model after phase3
python scripts/run_eval.py \
--model_path expdata/lifelong_learning/phase3/checkpoint-60000 \
--task_set lifelong_learning_phase1 lifelong_learning_phase2 lifelong_learning_phase3 \
--split pretrain

# evaluate model after phase4
python scripts/run_eval.py \
--model_path expdata/lifelong_learning/phase4/checkpoint-60000 \
--task_set lifelong_learning_phase1 lifelong_learning_phase2 lifelong_learning_phase3 lifelong_learning_phase4 \
--split pretrain
```

#### Report evaluation results
```
python gr00t/eval/get_eval_stats.py \
--dir <your-ckpt>
```