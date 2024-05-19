# RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots
<!-- ![alt text](https://github.com/UT-Austin-RPL/maple/blob/web/src/overview.png) -->
<img src="docs/images/robocasa-banner.png" width="100%" />

This is the official RoboCasa codebase. Please refer to the accompanying [paper]() and [project website](https://robocasa.ai) for additional information.

-------
## Installation
RoboCasa works across all major computing platforms. The easiest way to set up is through the [Anaconda](https://www.anaconda.com/) package management system. Follow the instructions below to install:
1. Set up conda environment:

   ```sh
   conda create -c conda-forge -n robocasa python=3.9
   ```
2. Activate conda environment:
   ```sh
   conda activate robocasa
   ```
3. Clone and setup robosuite dependency (**important: use the robocasa_v0.1 branch!**):

   ```sh
   git clone https://github.com/ARISE-Initiative/robosuite -b robocasa_v0.1
   cd robosuite
   pip install -e .
   ```
4. Clone and setup this repo:

   ```sh
   cd .. # go back to previous directory
   git clone https://github.com/robocasa/robocasa
   cd robocasa
   pip install -e .
   ```
5. Install the package and download assets:
   ```sh
   conda install -c numba numba -y
   python robocasa/scripts/download_kitchen_assets.py   # Caution: Assets to be downloaded are around 5GB.
   python robocasa/scripts/setup_macros.py              # Set up system variables.
   ```

-------
## Quick start
Choose among following demo scripts:
- Explore kitchen layouts and styles: `python -m robocasa.demos.demo_kitchens`
- Play back sample demonstrations of tasks: `python -m robocasa.demos.demo_tasks`
- Collect your own demonstrations of tasks: `python -m robocasa.demos.demo_teleop`
- Explore library of 2500+ objects: `python -m robocasa.demos.demo_objects`

**(Mac users: prepend the "python" command with "mj": `mjpython ...`)**

Note: if using spacemouse: you may need to modify the product ID to your appropriate model, setting `SPACEMOUSE_PRODUCT_ID` in `robocasa/macros_private.py`

-------
## Key files
- `robocasa/demos/`: interactive scripts to explore tasks and environments (see quick start section for details)
- `robocasa/environments/kitchen/`: all implementations for kitchen tasks
  - `kitchen.py`: base class from which all kitchen tasks extend from
  - `single_stage/`: implementations of all single-stage tasks
  - `multi_stage/`: implementations of all multi-stage (aka "composite") tasks
- `robocasa/scripts/`: utility scripts
  - `collect_demos.py`: collect demonstration trajectories for any task and environment
  - `download_kitchen_assets.py`: downloads all assets. called automatically during setup in `setup.sh`
- `robocasa/models/assets/`: assets and implementations for objects and environment fixtures
  - `assets/objects/`: raw assets for all objects
  - `assets/fixtures/`: raw assets for all environment fixtures
  - `assets/kitchen_layouts/`: blueprints for kitchen layouts and designs
  - `objects/kitchen_objects.py`: registry for all object categories and groups
  - `objects/fixtures/`: implementations of all fixture classes
 
-------
## Citation
```
@inproceedings{robocasa2024,
  title={RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots},
  author={Soroush Nasiriany and Abhiram Maddukuri and Lance Zhang and Adeet Parikh and Aaron Lo and Abhishek Joshi and Ajay Mandlekar and Yuke Zhu},
  booktitle={Robotics: Science and Systems},
  year={2024}
}
```
