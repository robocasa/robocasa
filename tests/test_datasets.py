import os
import numpy as np
import tqdm
import traceback
import argparse
import h5py
import json
from collections import OrderedDict
from termcolor import colored


def check_dataset(ds_path):
    datased_passed = True

    f = h5py.File(ds_path)
    obj_cat_counts = {}
    layout_counts = {}
    style_counts = {}
    langs = []

    num_demos_invalid = 0
    for ep in f["data"]:
        acs = f["data"][ep]["actions"][:]
        if np.any(np.min(acs, axis=0) < -1.0) or np.any(np.max(acs, axis=0) > 1.0):
            print(
                colored(
                    f"[WARNING] demo {ep} actions out of bounds for dims:",
                    np.where(np.max(np.abs(acs), axis=0) > 1.0)[0],
                    "magenta",
                )
            )
            datased_passed = False
            num_demos_invalid += 1

        ep_meta = json.loads(f["data/{}".format(ep)].attrs["ep_meta"])
        langs.append(ep_meta["lang"])
        obj_cfgs = ep_meta["object_cfgs"]
        cat = None
        for cfg in obj_cfgs:
            if cfg["name"] == "obj":
                cat = cfg["info"]["cat"]
                break
        if cat not in obj_cat_counts:
            obj_cat_counts[cat] = 0
        obj_cat_counts[cat] += 1

        layout_id = ep_meta["layout_id"]
        style_id = ep_meta["style_id"]
        if layout_id not in layout_counts:
            layout_counts[layout_id] = 0
        if style_id not in style_counts:
            style_counts[style_id] = 0
        layout_counts[layout_id] += 1
        style_counts[style_id] += 1

    obj_cat_counts = OrderedDict(sorted(obj_cat_counts.items()))
    layout_counts = OrderedDict(sorted(layout_counts.items()))
    style_counts = OrderedDict(sorted(style_counts.items()))

    num_demos = len(f["data"])

    print("obj cat counts:", json.dumps(obj_cat_counts, indent=4))

    print("layout_counts:", json.dumps(layout_counts, indent=4))
    num_layouts = len(layout_counts.keys())
    min_demos_per_layout = num_demos // (num_layouts * 2)
    for layout, count in layout_counts.items():
        if count < min_demos_per_layout:
            datased_passed = False
            print(
                colored(
                    f"[WARNING] low count for layout={layout}, found {count}", "magenta"
                )
            )

    print("style_counts:", json.dumps(style_counts, indent=4))

    print("num unique lang instructions:", len(set(langs)))
    print("\n")

    return datased_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset (or folder containing demo.hdf5s)",
    )
    args = parser.parse_args()

    if os.path.isdir(args.dataset):
        ds_path_list = []
        for root, dirs, files in os.walk(args.dataset):
            for file in files:
                if file == "demo.hdf5":
                    ds_path_list.append(os.path.join(root, file))
    else:
        ds_path_list = [args.dataset]

    num_passed = 0
    for ds_i, ds_path in enumerate(ds_path_list):
        print(
            colored(f"[{ds_i+1}/{len(ds_path_list)}] Inspecting: {ds_path}", "yellow")
        )
        datased_passed = check_dataset(ds_path=ds_path)
        num_passed += int(datased_passed)

    print()
    print(
        colored(
            f"Number of passed datasets: {num_passed} / {len(ds_path_list)}", "yellow"
        )
    )


def test_make_demo_ids_contiguous_preserves_episode_and_filter_metadata(tmp_path):
    from robocasa.utils.robomimic.robomimic_dataset_utils import (
        make_demo_ids_contiguous,
    )

    dataset = tmp_path / "demos.hdf5"
    with h5py.File(dataset, "w") as f:
        data = f.create_group("data")
        for index in (0, 2, 5):
            demo = data.create_group(f"demo_{index}")
            demo.attrs["original_id"] = index
            obs = demo.create_group("obs")
            obs.attrs["camera"] = "agentview"
            image = obs.create_dataset(
                "image", data=np.full((2, 3), index), compression="gzip"
            )
            image.attrs["units"] = "pixels"
        train = f.create_dataset(
            "mask/train", data=np.array(["demo_2", "demo_5", "demo_1"], dtype="S")
        )
        train.attrs["split"] = "train"
        f.create_dataset("mask/valid", data=np.array(["demo_0"], dtype="S"))
    make_demo_ids_contiguous(dataset)
    with h5py.File(dataset, "r") as f:
        assert set(f["data"]) == {"demo_0", "demo_1", "demo_2"}
        assert f["mask/train"][:].tolist() == [b"demo_2", b"demo_1"]
        assert f["mask/valid"][:].tolist() == [b"demo_0"]
        assert f["mask/train"].attrs["split"] == "train"
        assert f["data/demo_1"].attrs["original_id"] == 5
        assert f["data/demo_1/obs"].attrs["camera"] == "agentview"
        assert f["data/demo_1/obs/image"].attrs["units"] == "pixels"
        assert f["data/demo_1/obs/image"].compression == "gzip"
        assert np.all(f["data/demo_1/obs/image"][:] == 5)
    # A second conversion must leave both IDs and filters unchanged.
    make_demo_ids_contiguous(dataset)
    with h5py.File(dataset, "r") as f:
        assert f["mask/train"][:].tolist() == [b"demo_2", b"demo_1"]


def test_make_demo_ids_contiguous_without_filter_masks(tmp_path):
    from robocasa.utils.robomimic.robomimic_dataset_utils import (
        make_demo_ids_contiguous,
    )

    dataset = tmp_path / "demos.hdf5"
    with h5py.File(dataset, "w") as f:
        f.create_dataset("data/demo_3/actions", data=np.ones((2, 7)))
    make_demo_ids_contiguous(dataset)
    with h5py.File(dataset, "r") as f:
        assert list(f["data"]) == ["demo_0"]
        assert "mask" not in f
        assert np.array_equal(f["data/demo_0/actions"][:], np.ones((2, 7)))
