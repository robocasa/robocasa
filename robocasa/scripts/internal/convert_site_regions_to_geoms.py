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

import robocasa
from robocasa.scripts.internal.prettify_xmls import prettify_xmls


def refactor_xml_regions(old_path, new_path, remove_old_sites=True):
    tree = etree.parse(old_path)
    root = tree.getroot()

    all_scanned_sites = find_elements(
        root,
        tags="site",
        return_first=False,
    )
    if all_scanned_sites is None:
        return

    # Create a child-to-parent mapping
    parent_map = {child: parent for parent in root.iter() for child in parent}

    region_sites_dict = {}
    for site in all_scanned_sites:
        name = site.get("name")
        if name is None:
            continue

        if not any([name.endswith(postfix) for postfix in ["p0", "px", "py", "pz"]]):
            continue

        postfix = name.split("_")[-1]
        prefix = "_".join(name.split("_")[:-1])

        if prefix not in region_sites_dict:
            region_sites_dict[prefix] = {}
        region_sites_dict[prefix][postfix] = site

    for site_name, site_dict in region_sites_dict.items():
        p0 = string_to_array(site_dict["p0"].get("pos"))
        px = string_to_array(site_dict["px"].get("pos"))
        py = string_to_array(site_dict["py"].get("pos"))
        pz = string_to_array(site_dict["pz"].get("pos"))

        pos = [
            np.mean([p0[0], px[0]]),
            np.mean([p0[1], py[1]]),
            np.mean([p0[2], pz[2]]),
        ]
        size = [
            (px[0] - p0[0]) / 2,
            (py[1] - p0[1]) / 2,
            (pz[2] - p0[2]) / 2,
        ]
        if size == [0.0, 0.0, 0.0]:
            size = [0.0001, 0.0001, 0.0001]

        region_name = f"reg_{site_name}"
        new_element = etree.Element("geom")
        new_element.set("name", region_name)
        new_element.set("conaffinity", "0")
        new_element.set("contype", "0")
        new_element.set("group", "1")
        new_element.set("type", "box")
        new_element.set("size", array_to_string(size))
        new_element.set("pos", array_to_string(pos))
        new_element.set("rgba", "0 0 1 0")

        parent = parent_map.get(site_dict["p0"])
        index = list(parent).index(site_dict["p0"])  # Find the index of 'child2'
        parent.insert(index, new_element)  # Insert 'new_child' at the same position
        if remove_old_sites:
            parent.remove(site_dict["p0"])  # Remove 'child2'
            parent.remove(site_dict["px"])  # Remove 'child2'
            parent.remove(site_dict["py"])  # Remove 'child2'
            parent.remove(site_dict["pz"])  # Remove 'child2'

    tree.write(new_path, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        help="directory containing fixtures",
    )
    args = parser.parse_args()

    directory = args.directory
    if directory is None:
        directory = os.path.join(robocasa.__path__[0], "models/assets/fixtures")

    xml_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".xml") and not file.endswith("_orig.xml"):
                xml_paths.append(os.path.join(root, file))

    for path in xml_paths:
        orig_path = path[:-4] + "_orig.xml"
        if not os.path.exists(orig_path):
            shutil.copy(path, orig_path)

        remove_old_sites = True
        model_name = path.split("/")[-1]
        if model_name == "drawer.xml":
            remove_old_sites = False
        refactor_xml_regions(orig_path, path, remove_old_sites=remove_old_sites)
        prettify_xmls(filepath=path)
