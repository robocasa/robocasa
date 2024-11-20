# Codebase Overview

Here is an outline of prominent components of the codebase:
- `robocasa/demos/`: interactive scripts to explore tasks and environments (see [quick start](../introduction/quick_start.html) section for details)
- `robocasa/environments/kitchen/`: all implementations for kitchen tasks
  - `kitchen.py`: base class from which all kitchen tasks extend from
  - `single_stage/`: implementations of all single-stage (aka "atomic") tasks
  - `multi_stage/`: implementations of all multi-stage (aka “composite”) tasks
- `robocasa/scripts/`: utility scripts
  - `download_kitchen_assets.py`: downloads all assets. called automatically during setup in setup.sh
  - `download_datasets.py`: downloads datasets (see datasets section for details)
  - `collect_demos.py`: collect demonstration trajectories for any task and environment
- `robocasa/utils/`: utilities
  - `dataset_registry.py`: registry of all datasets (see [using datasets](../use_cases/using_datasets.html) for more details)
- `robocasa/models/`: assets and implementations for objects, fixtures, and scenes
  - `objects/kitchen_objects.py`: registry for all object categories and groups
  - `assets/objects/`: raw assets for all objects
  - `fixtures/`: implementations of all fixture classes
  - `assets/fixtures/`: raw assets for all environment fixtures
    - `assets/fixtures/fixture_registry/`: registry for all fixture assets
  - `scenes/`: implementations for constructing kitchen scenes
    - `scene_registry.py`: registry of all kitchen scenes, including layouts and styles
  - `assets/scenes/`: blueprints for kitchen scenes, including layouts and styles