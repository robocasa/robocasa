import objaverse
from tqdm import tqdm

import json
import os


def load_metadata(metadata_path):
    with open(metadata_path, "r") as f:
        objects = json.load(f)
    return objects


def save_metadata(metadata, save_path):
    with open(save_path, "w") as f:
        json.dump(metadata, f)


def download_metadata(save_path, target_cats, exists_ok=True):
    if os.path.exists(save_path) and not exists_ok:
        raise FileExistsError("Metadata data already exists")
    target_cats = set(target_cats)

    print("Downloading objaverse uids and annotations")
    uids = objaverse.load_uids()
    print("Number of objects:", len(uids))
    annotations = objaverse.load_annotations(uids)

    metadata = list()
    for annot in tqdm(annotations.values(), ncols=100):
        obj_cats = {cat["name"] for cat in annot["categories"]}
        if not target_cats.union(obj_cats):
            continue

        obj_meta = dict(
            (k, annot[k]) for k in ("name", "uid", "embedUrl", "vertexCount")
        )
        obj_meta["tags"] = [tag["name"] for tag in annot["tags"]]
        metadata.append(obj_meta)
    save_metadata(metadata, save_path)


if __name__ == "__main__":
    download_metadata(
        save_path="food_drink_furniture.json",
        target_cats=["food-drink", "furniture-home"],
        exists_ok=False,
    )
