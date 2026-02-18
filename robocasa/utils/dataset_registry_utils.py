import os
from collections import OrderedDict
from copy import deepcopy
import numpy as np
from pathlib import Path

import robocasa
import robocasa.macros as macros


def get_ds_meta(task, split, source="human", demo_fraction=1.0):
    from robocasa.utils.dataset_registry import (
        ATOMIC_TASK_DATASETS,
        COMPOSITE_TASK_DATASETS,
    )

    meta = {}

    assert split in ["pretrain", "target", "real"]

    if task in ATOMIC_TASK_DATASETS:
        ds_config = ATOMIC_TASK_DATASETS[task]
    elif task in COMPOSITE_TASK_DATASETS:
        ds_config = COMPOSITE_TASK_DATASETS[task]
    else:
        raise ValueError

    if source in ["human", "human_cotraining_cams"]:
        folder = ds_config.get(split, {}).get("human_path")
        if split == "pretrain":
            num_total_demos = 100
        elif split == "target":
            num_total_demos = 500
        elif split == "real":
            if task in ATOMIC_TASK_DATASETS:
                num_total_demos = 30
            elif task in COMPOSITE_TASK_DATASETS:
                num_total_demos = 50
            else:
                raise ValueError
        else:
            raise ValueError
    elif source == "mg":
        folder = ds_config.get(split, {}).get("mg_path")
        if task == "OpenCabinet":
            num_total_demos = 5000
        else:
            num_total_demos = 10000
    elif source == "mg_5x5":
        folder = ds_config.get(split, {}).get("mg_5x5_path")
        if task == "OpenCabinet":
            num_total_demos = 5000
        else:
            num_total_demos = 10000
    elif source == "mg_5x1":
        folder = ds_config.get(split, {}).get("mg_5x1_path")
        if task == "OpenCabinet":
            num_total_demos = 5000
        else:
            num_total_demos = 10000
    else:
        raise ValueError

    if source.endswith("cotraining_cams"):
        fname = "lerobot_cotraining_cams"
    else:
        fname = "lerobot"

    # if dataset type is not registered, return None
    if folder is None:
        return None

    if macros.DATASET_BASE_PATH is None:
        ds_base_path = os.path.join(
            Path(robocasa.__path__[0]).parent.absolute(), "datasets"
        )
    else:
        ds_base_path = macros.DATASET_BASE_PATH

    meta["path"] = os.path.join(ds_base_path, folder, fname)
    if "download_links" in ds_config:
        meta["url"] = ds_config["download_links"][source]
    meta["horizon"] = ds_config["horizon"]
    num_sampled_demos = int(num_total_demos * demo_fraction)
    meta["filter_key"] = f"{num_sampled_demos}_demos"
    meta["task"] = task
    meta["split"] = split
    meta["source"] = source
    return meta


def get_ds_path(task, source, split="pretrain", return_info=False):
    """
    Convenience wrapper to get a dataset path from the registry.

    Args:
        task (str): task name (must exist in ATOMIC_TASK_DATASETS / COMPOSITE_TASK_DATASETS)
        source (str): "human" or "mg" (use "mimicgen" as alias for "mg")
        split (str): "pretrain", "target", or "real". Default "pretrain".
        return_info (bool): if True, return (path, meta). Otherwise just path.

    Returns:
        str | None OR (str | None, dict): Dataset path (or None if unregistered).
    """
    if source == "mimicgen":
        source = "mg"
    if source not in ("human", "mg") or split not in ("pretrain", "target", "real"):
        return (None, {}) if return_info else None

    try:
        meta = get_ds_meta(task=task, split=split, source=source)
    except Exception:
        meta = None

    path = None if meta is None else meta.get("path")
    if return_info:
        return path, (meta or {})
    return path


def get_ds_soup(split, task_set, source, demo_fraction=1.0):
    from robocasa.utils.dataset_registry import (
        ATOMIC_TASK_DATASETS,
        COMPOSITE_TASK_DATASETS,
        TASK_SET_REGISTRY,
    )

    assert split in ["pretrain", "target", "real"]
    assert task_set in TASK_SET_REGISTRY
    assert source in [
        "all",
        "human",
        "human_cotraining_cams",
        "mg",
        "mg_5x5",
        "mg_5x1",
    ]

    # get the list of tasks
    task_list = TASK_SET_REGISTRY[task_set]

    if source == "human":
        source_list = ["human"]
    elif source == "human_cotraining_cams":
        source_list = ["human_cotraining_cams"]
    elif source == "mg":
        source_list = ["mg"]
    elif source == "all":
        source_list = ["human", "mg"]
    elif source == "mg_5x5":
        source_list = ["mg_5x5"]
    elif source == "mg_5x1":
        source_list = ["mg_5x1"]
    else:
        raise ValueError

    ds_soup_list = []
    for task in task_list:
        for source in source_list:
            ds_meta = get_ds_meta(
                task=task,
                split=split,
                source=source,
                demo_fraction=demo_fraction,
            )
            if ds_meta is not None:
                ds_soup_list.append(ds_meta)

    return ds_soup_list


def add_cotraining_weights(soup, real_weight=0.15, dc_weight=0.3, non_dc_weight=0.55):
    """
    takes a soup as input, adds weights (in place), returns the same soup
    """
    real_indices = []
    real_tasks = []
    dc_indices = []
    dc_tasks = {}

    sim_indices = []

    # first go through and find real tasks
    for i, ds_cfg in enumerate(soup):
        if ds_cfg["split"] == "real":
            real_indices.append(i)
            real_tasks.append(ds_cfg["task"])

    # then go through and find dc tasks
    for i, ds_cfg in enumerate(soup):
        if ds_cfg["task"] in real_tasks and ds_cfg["split"] != "real":
            if ds_cfg["task"] not in dc_tasks:
                dc_tasks[ds_cfg["task"]] = []
            dc_tasks[ds_cfg["task"]].append((i, ds_cfg["split"]))

    # if dc tasks are repeated, keep the one where split is target
    for task, indices in dc_tasks.items():
        if len(indices) > 1:
            for idx, split in indices:
                if split == "target":
                    dc_indices.append(idx)
        else:
            dc_indices.append(indices[0][0])

    # finally, everything else is sim
    for i in range(len(soup)):
        if i not in real_indices and i not in dc_indices:
            sim_indices.append(i)

    # make sure indices are disjoint
    assert len(set(real_indices).intersection(set(dc_indices))) == 0
    assert len(set(real_indices).intersection(set(sim_indices))) == 0
    assert len(set(dc_indices).intersection(set(sim_indices))) == 0
    assert len(real_indices) + len(dc_indices) + len(sim_indices) == len(soup)
    # finally, assign weights
    weights = np.zeros((len(soup),), dtype=np.float32)
    if len(real_indices) > 0:
        real_weight_per_task = real_weight / len(real_indices)
        for i in real_indices:
            weights[i] = real_weight_per_task
    if len(dc_indices) > 0:
        dc_weight_per_task = dc_weight / len(dc_indices)
        for i in dc_indices:
            weights[i] = dc_weight_per_task
    if len(sim_indices) > 0:
        sim_weight_per_task = non_dc_weight / len(sim_indices)
        for i in sim_indices:
            weights[i] = sim_weight_per_task

    # normalize weights
    weights = weights / weights[0]
    for i in range(len(soup)):
        soup[i]["ds_weight"] = float(weights[i])

    soup = [meta for meta in soup if meta.get("ds_weight", -1.0) != 0.0]

    return soup


def get_task_horizon(task):
    from robocasa.utils.dataset_registry import (
        ATOMIC_TASK_DATASETS,
        COMPOSITE_TASK_DATASETS,
    )

    if task in ATOMIC_TASK_DATASETS:
        ds_config = ATOMIC_TASK_DATASETS[task]
    elif task in COMPOSITE_TASK_DATASETS:
        ds_config = COMPOSITE_TASK_DATASETS[task]
    else:
        raise ValueError

    return ds_config["horizon"]
