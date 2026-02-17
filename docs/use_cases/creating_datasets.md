# Creating Datasets

## Collecting human demonstrations
You can use the collect_demos.py script to collect demonstrations:
```
python robocasa/scripts/collect_demos.py --env <env-name>
```

<div class="admonition note">
<p class="admonition-title">Attention Mac users</p>

Mac users must prepend this script with `mj`, ie. `mjpython`

</div>

You can either control the robot with a [spacemouse](https://3dconnexion.com/us/product/spacemouse-compact/) (`--device spacemouse`) or the keyboard (`--device keyboard`). A spacemouse is recommended.

This will save a raw dataset. Follow the steps in the next section to extract image observations and convert the dataset to [lerobot](https://github.com/huggingface/lerobot) format for policy learning.

## Extracting observations
To extract observations from an existing dataset, you can run:
```
 python robocasa/scripts/dataset_scripts/convert_hdf5_lerobot.py --raw_dataset_path <hdf5-ds-path>
```
This script will generate a new dataset subfolder named `lerobot` in the same directory as `--raw_dataset_path`. For more information on the generated dataset structure, please refer to the [Using Datasets](../datasets/using_datasets.md) guide.

<div class="admonition note">
<p class="admonition-title">Image resolution</p>

You can override the default image resolution by setting the flags `--camera_height` and `--camera_width`.

</div>
