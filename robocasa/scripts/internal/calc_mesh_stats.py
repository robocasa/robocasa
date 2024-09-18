"""
Calculate the asset statistics
"""

import os
from tqdm import tqdm
import trimesh
import numpy as np
import random

import robocasa

FIXTURES_PATH = os.path.join(robocasa.__path__[0], "models/assets/fixtures")
OBJECTS_PATH = os.path.join(robocasa.__path__[0], "models/assets/objects/objaverse")


def search_paths_with_name(folder_path, name):
    matching_paths = []

    for root, dirs, files in os.walk(folder_path):
        if "__MACOSX" in root:
            continue

        for file in files:
            if file == name:
                matching_paths.append(os.path.join(root, file))

        for dir in dirs:
            if name in dir:
                matching_paths.append(os.path.join(root, dir))

    return matching_paths


def count_and_report_stats(base_path):
    all_model_vert_counts = []
    all_model_face_counts = []

    for path in tqdm(base_path):
        vert_count = 0
        face_count = 0
        for file in os.listdir(path):
            if file.endswith(".obj"):
                # load mesh
                mesh = trimesh.load_mesh(os.path.join(path, file))
                vert_count += mesh.vertices.shape[0]
                face_count += mesh.faces.shape[0]

        if vert_count == 0 or face_count == 0:
            continue

        all_model_vert_counts.append(vert_count)
        all_model_face_counts.append(face_count)

    vert_percentile_stats = [
        np.percentile(all_model_vert_counts, p) for p in [1, 25, 50, 75, 99]
    ]
    face_percentile_stats = [
        np.percentile(all_model_face_counts, p) for p in [1, 25, 50, 75, 99]
    ]

    print("Vertex stats:", vert_percentile_stats)
    print("Face stats:", face_percentile_stats)


all_fixture_paths = search_paths_with_name(FIXTURES_PATH, "visuals")
all_object_paths = search_paths_with_name(OBJECTS_PATH, "visual")

# print("Getting stats for fixtures...")
# count_and_report_stats(all_fixture_paths)
# print()

print("Getting stats for objects...")
count_and_report_stats(random.sample(all_object_paths, 200))
print()
