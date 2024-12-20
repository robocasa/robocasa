## Generating Data with MimicGen

Given a modest number of source demonstrations, we offer the ability to syntehsize new trajectories with [MimicGen](https://mimicgen.github.io/).

### Installation
1. Clone and setup MimicGen under the `experimental/robocasa` branch:

    ```sh
    git clone https://github.com/NVlabs/mimicgen -b experimental/robocasa
    cd mimicgen
    pip install -e .
    ```

2. Setup the robomimic dependency: 
    ```sh
    cd ..
    git clone https://github.com/ARISE-Initiative/robomimic -b robocasa
    cd robomimic
    pip install -e .
    ```

### Generate data for existing tasks
We have set up MimicGen for 24/25 existing atomic tasks. Generating demonstration datasets for these tasks is a 3 step process, which we illustrate for the `PnPCounterToSink` task:

(Note: these commands assume that datasets are stored under `~/robocasa/datasets`)

1. Extract subtask meta data:
    ```sh
    python mimicgen/scripts/prepare_src_dataset.py --dataset ~/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToSink/2024-04-25/demo.hdf5
    ```
2. Generate demonstartions:
    ```sh
    python mimicgen/scripts/generate_dataset.py --config mimicgen/exps/templates/robocasa/single_stage/kitchen_pnp/PnPCounterToSink.json
    ```
3. Extract image observations.
    You can then extract image datasets following the dataset extraction tutorial on [this guide](../use_cases/creating_datasets.html).

### Set up MimicGen for new tasks
Please follow the official MimicGen documentation [tutorial page](https://mimicgen.github.io/docs/tutorials/datagen_custom.html).
