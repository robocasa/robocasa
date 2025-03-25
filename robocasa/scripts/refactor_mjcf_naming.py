"""
A script to refactor naming conventions in mjcf xmls files.
Strip out the name of the model from the mjcf itself, and rename all asset files with the model name
"""

import argparse
import xml.etree.ElementTree as ET
from robosuite.utils.mjcf_utils import (
    array_to_string,
    find_elements,
    find_parent,
    string_to_array,
    xml_path_completion,
)
import os
import numpy as np
from lxml import etree
import shutil
from tqdm import tqdm

import robocasa
from robocasa.scripts.prettify_xmls import prettify_xmls


def strip_out_keyword(mjcf_path, keyword, verbose=False, dry_run=False):
    modified = False

    tree = etree.parse(mjcf_path)
    root = tree.getroot()
    all_elements = list(root.iter())

    for element in all_elements:
        for tag_name in ["name", "texture", "material", "joint", "mesh", "file"]:
            tag_val = element.get(tag_name)
            if tag_val is None:
                continue
            if not keyword in tag_val:
                continue

            new_tag_val = tag_val.replace(keyword, "")
            if verbose:
                print(tag_name, tag_val, "->", new_tag_val)
            if not dry_run:
                element.set(tag_name, new_tag_val)
            modified = True

            if tag_name == "file":
                old_asset_path = os.path.join(os.path.dirname(mjcf_path), tag_val)
                new_asset_path = os.path.join(os.path.dirname(mjcf_path), new_tag_val)
                # print(old_asset_path)
                # print(new_asset_path)
                # print()
                if not dry_run:
                    if os.path.exists(old_asset_path):
                        os.rename(old_asset_path, new_asset_path)

    if not dry_run:
        tree.write(mjcf_path, encoding="utf-8")

    return modified


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        help="directory containing fixtures",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
    )
    args = parser.parse_args()

    directory = args.directory
    if directory is None:
        directory = os.path.join(robocasa.__path__[0], "models/assets/fixtures")

    xml_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".xml"):
                full_path = os.path.join(root, file)
                if full_path.endswith(
                    "assets/fixtures/counters/counter/model.xml"
                ) or full_path.endswith(
                    "assets/fixtures/counters/counter_with_opening/model.xml"
                ):
                    # make an exception for counters, as we want to keep the existing naming conventions
                    continue
                xml_paths.append(full_path)

    modified_paths = []
    for path in tqdm(xml_paths):
        keyword = os.path.basename(os.path.dirname(path)) + "_"
        modified = strip_out_keyword(path, keyword, dry_run=args.dry_run)

        if keyword.startswith("StandMixer"):
            ### special case for FoodMixer being renamed to StandMixer ###
            new_prefix = keyword.replace("StandMixer", "FoodMixer")
            modified = modified or strip_out_keyword(
                path, new_prefix, dry_run=args.dry_run
            )
        if keyword.startswith("Oven"):
            ### special case for WallStackOven being renamed to Oven ###
            new_prefix = keyword.replace("Oven", "WallStackOven")
            modified = modified or strip_out_keyword(
                path, new_prefix, dry_run=args.dry_run
            )

        if modified:
            prettify_xmls(filepath=path)
            modified_paths.append(path)

    print(f"Num modified files: {len(modified_paths)}")
    print("The following paths were modified:")
    for path in modified_paths:
        print(path)
