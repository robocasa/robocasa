## Basic Usage

### Gym interface
We provide a gym-like interface to run rollouts:
<div class="admonition warning">
<p class="admonition-title">Attention Mac users!</p>
Mac users who wish to run this example need to prepend the “python” command with “mj”: mjpython
</div>

```py
from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS
from robocasa.utils.env_utils import create_env, run_random_rollouts
import numpy as np

# choose random task
env_name = np.random.choice(list(ALL_KITCHEN_ENVIRONMENTS))

env = create_env(
    env_name=env_name,
    render_onscreen=True,
    seed=0, # set seed=None to run unseeded
)

# reset the environment
env.reset()

# get task language
lang = env.get_ep_meta()["lang"]
print("Instruction:", lang)

for i in range(500):
    action = np.random.randn(*env.action_spec[0].shape) * 0.1
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display
```

### Offscreen rollouts
You can also run rollouts and save videos:
```py
from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS
from robocasa.utils.env_utils import create_env, run_random_rollouts
import numpy as np

# choose random task
env_name = np.random.choice(list(ALL_KITCHEN_ENVIRONMENTS))

env = create_env(
    env_name=env_name,
    render_onscreen=False,
    seed=0, # set seed=None to run unseeded
)

# run rollouts with random actions and save video
info = run_random_rollouts(
    env, num_rollouts=3, num_steps=100, video_path="/tmp/test.mp4"
)
print(info)
```

Separately we provide tools to run policy rollouts within robomimic. See the [policy learning page](../use_cases/policy_learning.html) for additional details.