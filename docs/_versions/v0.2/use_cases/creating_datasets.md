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

This will save a raw dataset. Follow the steps in the next section to extract image observations for this dataset.

## Extracting observations
To extract observations from an existing dataset, you can run:
```
OMP_NUM_THREADS=1 MPI_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python robocasa/scripts/dataset_states_to_obs.py --dataset <ds-path>
```
This script will generate a new dataset with the suffix `_im128.hdf5` in the same directory as `--dataset`.

<div class="admonition note">
<p class="admonition-title">Image resolution</p>

You can override the default image resolution by setting the flags `--camera_height` and `--camera_width`.

</div>

<div class="admonition note">
<p class="admonition-title">Visual randomization</p>

You can add the flag `--generative_textures` to render images with AI-generated environment textures, and `--randomize_cameras` to randomize camera viewpoints for rendering.

</div>

<div class="admonition warning">
<p class="admonition-title">Warning!</p>
This observation extraction script may drop a few demonstrations due to subprocesses failing. To minimize this issue you can run the script with the flag --num_procs 1
</div>