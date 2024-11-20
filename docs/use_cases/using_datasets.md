## Using and Inspecting Datasets

### Dataset registry
You can retrieve dataset paths from the [dataset registry](https://github.com/robocasa/robocasa/blob/main/robocasa/utils/dataset_registry.py) as follows:

```py
from robocasa.utils.dataset_registry import SINGLE_STAGE_TASK_DATASETS, MULTI_STAGE_TASK_DATASETS
from robocasa.utils.dataset_registry import get_ds_path

# iterate through all atomic and composite tasks
for task in list(SINGLE_STAGE_TASK_DATASETS) + list(MULTI_STAGE_TASK_DATASETS):
    print(f"Datasets for {task}")
    human_path, ds_meta = get_ds_path(task=task, ds_type="human_im", return_info=True)  # human dataset path
    horizon = ds_meta["horizon"]                                      # get suggested for dataset
    mg_path, ds_meta = get_ds_path(task=task, ds_type="mg_im", return_info=True)        # MimicGen dataset path
    print(f"Human ds: {human_path}")
    print(f"MimicGen ds: {mg_path}")
    print(f"Dataset horizon:", horizon)
    print()
```

### Basic usage
Here is an example script to access dataset elements:
```py
import h5py
import json

f = h5py.File(INSERT_DATASET_PATH)
demo = f["data"]["demo_5"]                        # access demo 5
obs = demo["obs"]                                 # obervations across all timesteps
left_img = obs["robot0_agentview_left_image"][:]  # get left camera images in numpy format
ep_meta = json.loads(demo.attrs["ep_meta"])       # get meta data for episode
lang = ep_meta["lang"]                            # get language instruction for episode
f.close()
```

### Inspecting and visualizing datasets

To get dataset meta data (filter keys, env args) and statistics (object, task language, scenes):
```
python robocasa/scripts/get_dataset_info.py --dataset <ds-path>
```

To visualize a dataset and save a video:
```
python robocasa/scripts/playback_dataset.py --n 10 --dataset <ds-path>
```
This will save a video of 10 random demonstrations in the same path as the dataset. You can play the full dataset by removing the `--n` flag.

### Dataset structure
RoboCasa datasets follow the convention of robomimic `.hdf5` files. Here is an overview of important elements of each dataset:

`|__data`: env meta data and all demos <br>
`   |__env_args` (attribute): meta data infromation about the dataset task<br>
`   |__demo_<n>`: data for demo n <br>
`      |__model_file` (attribute): the xml string corresponding to the MJCF MuJoCo model<br>
`      |__ep_meta` (attribute): episode meta data (task langage, scene info, object info, etc)<br>
`      |__actions`: environment actions, ordered by time. Shape (N, A) where N is the<br>
`      `length of the trajectory, and A is the action space dimension<br>
`      |__action_dict`: dictionary that splits actions by fine-grained components,<br>
`      `eg. position, rotation, gripper, etc<br>
`      |__obs`: dictionary of observation keys, including images, proprioception, etc.<br>
`      |__states`: flattened raw low-level MuJoCo states, ordered by time. Used to replay demos <br>
`      `Not to be used for policy learning! <br>
`|__mask`: contains filter key meta data to split the dataset in different ways