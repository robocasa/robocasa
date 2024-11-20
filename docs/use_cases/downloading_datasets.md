# Downloading Datasets

RoboCasa comes with a large selection of demonstrations to faciliate training agents. We provide:
- **human demonstrations**: across all 25 atomic tasks and a subset of composite tasks, with 50 demonstrations per task.
- **machine-generated demonstrations** from [MimicGen](https://mimicgen.github.io/): across 24 atomic tasks, with 3000 demonstrations per task.

<div class="admonition note">
<p class="admonition-title">Dataset storage location</p>

By default, all datasets are stored under `datasets/` in the root robocasa directory. You can change the location for datasets by setting `DATASET_BASE_PATH` in `robocasa/macros_private.py`.

</div>

Here are a few ways to download the datasets:
```
# downloads all human datasets with images
python -m robocasa.scripts.download_datasets --ds_types human_im

# lite download: download human datasets without images
python -m robocasa.scripts.download_datasets --ds_types human_raw

# downloads all MimicGen datasets with images
python -m robocasa.scripts.download_datasets --ds_types mg_im
```

Additionally you can specify the following optional arguments:
```
--tasks <tasks>: downloads datasets for specific tasks, eg, --tasks PnPCounterToCab ArrangeVegetables
--overwrite: overwrites existing datasets
```
