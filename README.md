# RoboCasa: Large-Scale Simulation of Household Tasks for Generalist Robots
<!-- ![alt text](https://github.com/UT-Austin-RPL/maple/blob/web/src/overview.png) -->
<img src="docs/images/robocasa-banner.png" width="100%" />

This is the official RoboCasa codebase. Please refer to the accompanying [paper]() and [project website](https://robocasa.github.io/robocasa-web-dev/) for additional information.

-------
## Installation
1. Set up conda environment: `conda create -c conda-forge -n robocasa python=3.9`
2. Activate conda environment: `conda activate robocasa`
3. Clone and setup robosuite-dev (**important: use the robocasa_v0.5 branch!**): `git clone https://github.com/ARISE-Initiative/robosuite-dev -b robocasa_v0.5; cd robosuite-dev; pip install -e .`
4. Clone and setup this repo: `git clone https://github.com/robocasa/robocasa; cd robocasa; source setup.sh`

-------
## Quick start
Run demo script: `python -m robocasa.demos.demo_kitchens`

**(Mac users:, you need to preprend the "python" command with "mj": `mjpython ...`)**

Please note the following:
- If using the keyboard device for control, you must make sure the mujoco window isn't the "active" window, otherwise the mujoco viewer keybindings will interfere. Press a background window (eg. desktop) and then proceed.
- If using a non-default spacemouse model, you'll need to find the correct product id and set `SPACEMOUSE_PRODUCT_ID` in `robocasa/macros_private.py`
