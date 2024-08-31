## Basic Usage

The following script shows how to create evaluation environments and to run random rollouts:

```
from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS
from robocasa.utils.dataset_registry import SINGLE_STAGE_TASK_DATASETS, MULTI_STAGE_TASK_DATASETS
from robocasa.utils.dataset_registry import get_ds_path
from robocasa.utils.env_utils import create_env, run_random_rollouts

import numpy as np

"""
Select a random task (that comes with an accompanying dataset) to run rollouts for.
Alternatively, sample *any* kitchen task in RoboCasa by replacing the following line with
env_name = np.random.choice(list(ALL_KITCHEN_ENVIRONMENTS))
"""
env_name = np.random.choice(
    list(SINGLE_STAGE_TASK_DATASETS) + list(MULTI_STAGE_TASK_DATASETS)
)

# seed environment as needed. set seed=None to run unseeded
env = create_env(env_name=env_name, seed=0)

# run rollouts with random actions and save video
info = run_random_rollouts(
    env, num_rollouts=3, num_steps=100, video_path="/tmp/test.mp4"
)
print(info)
```

Separately we provide tools to run policy rollouts within robomimic. See the [policy learning page](../use_cases/policy_learning.html) for additional details.