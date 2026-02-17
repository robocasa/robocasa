# Codebase Overview

Here is an outline of prominent components of the codebase:
- `robocasa/demos/`: interactive scripts to explore tasks and environments (see [basic usage](../introduction/basic_usage.html) section for details)
- `robocasa/environments/kitchen/`: all implementations for kitchen tasks
  - `kitchen.py`: base class from which all kitchen tasks extend from
  - `atomic/`: implementations of all atomic tasks
  - `composite/`: implementations of all composite tasks
- `robocasa/scripts/`: utility scripts
  - `download_kitchen_assets.py`: downloads all assets. called automatically during setup in setup.sh
  - `download_datasets.py`: downloads datasets (see datasets section for details)
  - `collect_demos.py`: collect demonstration trajectories for any task and environment
- `robocasa/utils/`: utilities
  - `dataset_registry.py`: registry of all datasets (see [datasets overview page](../datasets/datasets_overview.html) for more details) 
- `robocasa/models/`: assets and implementations for objects, fixtures, and scenes
  - `objects/kitchen_objects.py`: registry for all object categories and groups
  - `assets/objects/`: raw assets for all objects
  - `fixtures/`: implementations of all fixture classes
  - `assets/fixtures/`: raw assets for all environment fixtures
    - `assets/fixtures/fixture_registry/`: registry for all fixture assets
  - `scenes/`: implementations for constructing kitchen scenes
    - `scene_registry.py`: registry of all kitchen scenes, including layouts and styles
  - `assets/scenes/`: blueprints for kitchen scenes, including layouts and styles