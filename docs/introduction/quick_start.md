# Quick Start

<div class="admonition warning">
<p class="admonition-title">Attention Mac users!</p>

For these scripts, Mac users need to prepend the "python" command with "mj": `mjpython ...`
</div>

-------
### Explore kitchen layouts and styles
Explore kitchen layouts (G-shaped, U-shaped, etc) and kitchen styles (mediterranean, industrial, etc):
```
python -m robocasa.demos.demo_kitchen_scenes
```

-------
### Play back sample demonstrations of tasks
Select a task and the script will render a human collected demonstration from our [datasets](../use_cases/datasets.html). Rendering can either be on-screen or off-screen as a video:
```
python -m robocasa.demos.demo_tasks
```

<div class="admonition note">
<p class="admonition-title">Off-screen rendering</p>

You can render videos of the demos off-screen by adding the `--render_offscreen` flag. Mac users will need to run the command as `python` instead of `mjpython`.

</div>

-------
### Explore library of 2500+ objects
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
 - [Basic Usage](../use_cases/basic_usage.html)
 - [Downloading Datasets](../use_cases/downloading_datasets.html)
 - [Using Datasets](../use_cases/using_datasets.html)
 - [Overview of Atomic Tasks](../tasks_scenes_assets/atomic_tasks.html)
 - [Overview of Composite Tasks](../tasks_scenes_assets/composite_tasks.html)
 - [Training Policies on Datasets](../use_cases/policy_learning.html)