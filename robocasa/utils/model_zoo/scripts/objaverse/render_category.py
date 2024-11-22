from metadata import load_metadata

import objaverse
import numpy as np
import spacy
import trimesh
from trimesh.transformations import rotation_matrix

from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm
import argparse
import time
import json
import os
import io


nlp = spacy.load("en_core_web_md")
img_resolution = np.array((640, 480))


def clear_directory(directory):
    for f in os.listdir(directory):
        os.remove(os.path.join(directory, f))


def add_text(img, obj_meta):
    text = (
        "uid:".ljust(10)
        + obj_meta["uid"]
        + "\n"
        + "name:".ljust(10)
        + obj_meta["name"]
        + "\n"
        + "vertices:".ljust(10)
        + str(obj_meta["vertexCount"])
        + "\n"
        + "url:".ljust(10)
        + obj_meta["embedUrl"]
        + "\n"
        + "save_path".ljust(10)
        + obj_meta["save_path"]
    )

    font = ImageFont.truetype("assets/JetBrainsMono-Medium.ttf", 15)
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), text, (0, 0, 0), font=font)


def concat_images(images):
    dst = Image.new("RGB", tuple(img_resolution * 2))
    dst.paste(images[0], (0, 0))
    dst.paste(images[1], (img_resolution[0], 0))
    dst.paste(images[2], (0, img_resolution[1]))
    dst.paste(images[3], tuple(img_resolution))
    return dst


def render_object(obj, obj_meta):
    transformations = [
        rotation_matrix(0, [1, 0, 0]),
        rotation_matrix(np.pi / 2, [1, 0, 0]),
        rotation_matrix(-np.pi / 2, [1, 0, 0]),
        rotation_matrix(np.pi / 2, [0, 1, 0]),
    ]
    imgs = list()
    for rot_matrix in transformations:
        start = time.time()
        scene = trimesh.load(obj).apply_transform(rot_matrix)
        data = scene.save_image(resolution=img_resolution / 2)
        img = Image.open(io.BytesIO(data))
        if np.array(img).mean() == 255:
            return None
        if time.time() - start > args.timeout_limit:
            return None
        imgs.append(img)
    combined = concat_images(imgs)
    add_text(combined, obj_meta)
    return combined


def get_selections(keyword, file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    annotations = list()

    idx = lines.index(keyword + "\n") + 1
    while idx < len(lines) and lines[idx] != "\n":
        uid = lines[idx].strip()
        annotation = objaverse.load_annotations([uid])[uid]
        annotations.append(annotation)
        idx += 1
    return annotations


def get_matches(keyword, metadata, threshold=0.2, max_objects=None):
    matches = list()
    keyword = " ".join(keyword.lower().split("_"))

    for obj_meta in metadata:
        # selection criterion might need to be adjusted in the future
        if keyword in obj_meta["name"].lower():
            matches.append(obj_meta)
            continue

        # this might include packs of multiple objects
        for tag in obj_meta["tags"]:
            if "_".join(keyword.split()) == tag.lower():
                matches.append(obj_meta)
                break

    # calculate similarity scores based on word embeddings
    keyword_embed = nlp(keyword)
    matches_scores = [nlp(obj_meta["name"].lower()) for obj_meta in matches]
    matches_scores = [keyword_embed.similarity(name) for name in matches_scores]
    matches_rank = np.argsort(matches_scores)[::-1][:max_objects]

    # only return objects with score >= threshold
    ret = list()
    for idx in matches_rank:
        if matches_scores[idx] >= threshold:
            ret.append(matches[idx])
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # add multi-word arguments with underline in between words!
    parser.add_argument("keyword", type=str)
    parser.add_argument("--save_folder", type=str, default=None)
    parser.add_argument("--max_vertices", type=int, default=300000)
    parser.add_argument("--timeout_limit", type=int, default=3)
    parser.add_argument("--max_objects", type=int, default=None)
    parser.add_argument("--manual_selection_file", type=str, default=None)
    args = parser.parse_args()

    if args.save_folder is None:
        args.save_folder = os.path.join("temp", "_".join(args.keyword.lower().split()))
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    else:
        clear_directory(args.save_folder)
    print("Saving renders at:", args.save_folder)

    if args.manual_selection_file is None:
        matches = get_matches(
            args.keyword,
            metadata=load_metadata("food_drink_furniture.json"),
            threshold=0.2,
            max_objects=None,
        )
    else:
        matches = get_selections(args.keyword, args.manual_selection_file)
    print("Matches found:", len(matches))

    skipped = list()
    rendered = list()
    if args.max_objects is not None:
        skipped.extend(matches[args.max_objects :])
        matches = matches[: args.max_objects]

    for i, obj_meta in tqdm(enumerate(matches), ncols=100, total=len(matches)):
        save_path = os.path.join(args.save_folder, str(i) + ".png")
        uid = obj_meta["uid"]

        # skip if object contains too many vertices
        if (
            args.max_vertices is not None
            and obj_meta["vertexCount"] > args.max_vertices
        ):
            skipped.append(obj_meta)
            continue

        obj = list(objaverse.load_objects([uid]).values())[0]
        obj_meta["save_path"] = obj
        img = render_object(obj, obj_meta)

        # render_object skips an object if rendering takes too long or returns blank image
        if img is None:
            skipped.append(obj_meta)
        else:
            rendered.append(obj_meta)
            img.save(save_path)

    # save records of objects
    print("Objects skipped:", len(skipped))
    with open(os.path.join(args.save_folder, "skipped.json"), "w") as f:
        skipped.insert(0, len(skipped))
        f.writelines(json.dumps(skipped, indent=4))
    with open(os.path.join(args.save_folder, "rendered.json"), "w") as f:
        rendered.insert(0, len(rendered))
        f.writelines(json.dumps(rendered, indent=4))
    with open(os.path.join(args.save_folder, "matches.json"), "w") as f:
        f.writelines(json.dumps(matches, indent=4))
