# Overview of Benchmarking
We evaluate learning across three distinct settings:
- **Multi-task learning**: studies large-scale pretraining from diverse human demonstrations across many tasks
- **Foundation model learning** builds on this by fine-tuning pretrained models to master specific target tasks, including both seen and unseen task compositions.
- **Lifelong learning**: further generalizes this setting by requiring models to continually acquire more complex, long-horizon tasks over time while retaining competence on previously learned tasks.

We support numerous policy learning algorithms, please review the [policy learning page](./policy_learning_algorithms.html) for an overview.

## Training and evaluation protocols
Our benchmark assumes understanding the conventions and terminology set in our datasets. Before proceeding, please review the [datasets page](../datasets/datasets_overview.html) for an overview.

### Training
Most experiments involve training on a collection of individual datasets, which we term **dataset soup**.
A dataset soup simply specifies a list of dataset metadata, and embedded in this metadata is the path specifying the location to the dataset on disk. See the [dataset page](../datasets/using_datasets.html#training-beyond-a-single-dataset) for additional details. For each benchmarking experiment we specify the specific dataset soup to train on.

### Evaluation

<div class="admonition note">
<p class="admonition-title">Horizon update (v1.0.1)</p>

As of v1.0.1, all task horizon lengths have been increased by 1.5x for consistency. Please update to the latest version of RoboCasa for running evals.

</div>

We evaluate trained models on one or several tasks.
To specify which tasks, you need to specify two variables:
- **task set**: Which set of tasks to evaluate on. Typically, this corresponds to the same tasks represented in the dataset soup used to trained the model, but can be any arbitrary set of tasks. The list of registered task sets is specified in [dataset_registry.py](https://github.com/robocasa/robocasa/blob/main/robocasa/utils/dataset_registry.py).
- **split**: This specifies which evaluation split to run evaluations are. There are two options: `pretrain` and `target`. The `pretrain` split features 2500 kitchen scenes and all of the pretraining objects, which the `target` split features a disjoint set of 10 target kitchen scenes and a disjoint set of target objects.

For each benchmarking experiment we specify the specific task set and split to evaluate on.
For all experiments, we randomly sample 50 scenarios to run evaluation rollouts on.
For each rollout, we recieve a binary task success signal for whether the task is successfully completed.
We report average task success rates across these 50 rollouts. 

#### Matching initial scenes across policies

Passing the same `seed` when creating two environments does not guarantee that
episode *i* starts from the same scene in both runs. The environment uses a
random-number stream across resets, and actions in an earlier rollout can
consume it. Different policies may therefore reach later resets with different
random states.

For a paired comparison, supply an explicit seed at **every** reset of the raw
RoboCasa / robosuite kitchen environment:

```py
import numpy as np

base_seed = 7
for episode_index in range(50):
    episode_seed = int(
        np.random.SeedSequence([base_seed, episode_index]).generate_state(1)[0]
    )
    obs = env.reset(episode_seed=episode_seed)
    # Run this policy's episode here.
```

Use the same task, environment configuration, software and assets, and
`episode_seed` mapping for every policy. Record the per-episode seeds with the
results. `episode_seed` is optional: plain `env.reset()` retains the historical
continuous-stream behavior. This controls initial-scene sampling, not the
policy's inference noise or differences caused by later actions.
