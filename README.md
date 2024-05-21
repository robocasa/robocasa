# RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots
<!-- ![alt text](https://github.com/UT-Austin-RPL/maple/blob/web/src/overview.png) -->
<img src="docs/images/robocasa-banner.jpg" width="100%" />

This is the official RoboCasa codebase. Please refer to the accompanying [paper](https://robocasa.ai/robocasa-paper.pdf) and [project website](https://robocasa.ai) for additional information.

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
   cd ..
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
**(Mac users: for these scripts, prepend the "python" command with "mj": `mjpython ...`)**

### Explore kitchen layouts and styles
Explore kitchen layouts (G-shaped, U-shaped, etc) and kitchen styles (mediterranean, industrial, etc):
```
python -m robocasa.demos.demo_kitchen_scenes
```

### Play back sample demonstrations of tasks
Select a task and play back a sample demonstration for the selected task:
```
python -m robocasa.demos.demo_tasks
```

### Explore library of 2500+ objects
View and interact with both human-designed and AI-generated objects:
```
python -m robocasa.demos.demo_objects
```
Note: by default this demo shows objaverse objects. To view AI-generated objects, add the flag `--obj_types aigen`.

### Teleoperate the robot
Control the robot directly, either through a keyboard controller or spacemouse. This script renders the robot semi-translucent in order to minimize occlusions and enable better visibility.
```
python -m robocasa.demos.demo_teleop
```
Note: If using spacemouse: you may need to modify the product ID to your appropriate model, setting `SPACEMOUSE_PRODUCT_ID` in `robocasa/macros_private.py`.

-------
## Datasets
RoboCasa comes with a large selection of demonstrations to faciliate training agents. This includes human trajectories and machine-generated trajectories from [MimicGen]([https://www.anaconda.com/](https://mimicgen.github.io/)). At this time we are releasing human datasets and plan to release MimicGen datasets in the near future.

Here are a few ways to download the datasets:
```
python -m robocasa.scripts.download_datasets                                            # downloads all datasets
python -m robocasa.scripts.download_datasets --tasks PnPCounterToCab ArrangeVegetables  # downloads datasets for specific tasks
python -m robocasa.scripts.download_datasets --overwrite                                # overwrites existing datasets
python -m robocasa.scripts.download_datasets --ds_types human_raw                       # lite download: download human datasets without images
```

By default, all datasets are stored under `datasets/` in the root robocasa directory.

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
  - `download_datasets.py`: downloads datasets (see datasets section for details)
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
