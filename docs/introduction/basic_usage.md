# Basic Usage

<div class="admonition warning">
<p class="admonition-title">Attention Mac users!</p>

For these scripts, Mac users need to prepend the "python" command with "mj": `mjpython ...`
</div>

-------
### Gym Interface

The following script shows how to create a gym environment and to run random rollouts:

```py
import gymnasium as gym
import robocasa
from robocasa.utils.env_utils import run_random_rollouts

env = gym.make(
    "robocasa/PickPlaceCounterToCabinet",
    split="pretrain", # use 'pretrain' or 'target' kitchen scenes and objects
    seed=0 # seed environment as needed. set seed=None to run unseeded
)

# run rollouts with random actions and save video
run_random_rollouts(
    env, num_rollouts=3, num_steps=100, video_path="/tmp/test.mp4"
)
```

-------
### Explore kitchen layouts and styles
Explore kitchen layouts (G-shaped, U-shaped, etc) and kitchen styles (mediterranean, industrial, etc):
```
python -m robocasa.demos.demo_kitchen_scenes
```

-------
### Play back sample demonstrations of tasks
Select a task and the script will render a human collected demonstration from our [datasets](../datasets/datasets_overview.html). Rendering can either be on-screen or off-screen as a video:
```
python -m robocasa.demos.demo_tasks
```

<div class="admonition note">
<p class="admonition-title">Off-screen rendering</p>

You can render videos of the demos off-screen by adding the `--render_offscreen` flag. Mac users will need to run the command as `python` instead of `mjpython`.

</div>

-------
### Explore library of 3200+ objects
View and interact with both human-designed and AI-generated objects. Can also specify the path to a custom mjcf file. If no mjcf path specified, the script will show a random object:
```
python -m robocasa.demos.demo_objects
```
Note: by default this demo shows objaverse objects. To view AI-generated objects, add the flag `--obj_types aigen`.

-------
### Teleoperate the robot
Control the robot directly, either through a keyboard controller or spacemouse. This script renders the robot semi-translucent in order to minimize occlusions and enable better visibility. Can specify environment, layout, and style via arguments to the script. Collected demonstrations will not be saved 
```
python -m robocasa.demos.demo_teleop
```
Note: If using spacemouse: you may need to modify the product ID to your appropriate model, setting `SPACEMOUSE_PRODUCT_ID` in `robocasa/macros_private.py`.

## Other useful sections
 - [Overview of Datasets](../datasets/datasets_overview.html)
 - [Overview of Atomic Tasks](../tasks/atomic_tasks.html)
 - [Overview of Composite Tasks](../tasks/composite_tasks.html)
 - [Overview of Scenes](../assets/scenes.html)
 - [Policy Learning Algorithms](../benchmarking/policy_learning_algorithms.html)