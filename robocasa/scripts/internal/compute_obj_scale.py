import numpy as np
from copy import deepcopy
import pathlib
import os
import math
import random
import xml.etree.ElementTree as ET

from robosuite.utils.mjcf_utils import find_elements, string_to_array

from robocasa.models.objects.kitchen_objects import OBJ_CATEGORIES



def get_obj_size(path):
    tree = ET.parse(path)
    root = tree.getroot()
    bottom = string_to_array(find_elements(root=root, tags="site", attribs={"name": "bottom_site"}).get("pos"))
    top = string_to_array(find_elements(root=root, tags="site", attribs={"name": "top_site"}).get("pos"))
    horizontal_radius = string_to_array(find_elements(root=root, tags="site", attribs={"name": "horizontal_radius_site"}).get("pos"))
    obj_size = np.array([horizontal_radius[0] * 2, horizontal_radius[1] * 2, top[2] - bottom[2]])
    return np.array(obj_size)


for cat in OBJ_CATEGORIES:
    # if "objaverse" not in OBJ_CATEGORIES[cat]:
    #     continue
    if "aigen" not in OBJ_CATEGORIES[cat]:
        continue
    
    # objaverse_paths = OBJ_CATEGORIES[cat]["objaverse"].mjcf_paths
    # objaverse_sizes = []
    # for path in objaverse_paths:
    #     obj_size = get_obj_size(path)
    #     scale = OBJ_CATEGORIES[cat]["objaverse"].scale
    #     obj_size = obj_size * scale
    #     objaverse_sizes.append(obj_size)

    aigen_paths = OBJ_CATEGORIES[cat]["aigen"].mjcf_paths
    aigen_sizes = []
    for path in aigen_paths:
        obj_size = get_obj_size(path)
        # scale = OBJ_CATEGORIES[cat]["aigen"].scale
        # obj_size = obj_size * scale
        aigen_sizes.append(obj_size)
    
    print(cat)
    # print("objaverse:", np.mean(objaverse_sizes, axis=0))
    print("aigen:", np.mean(aigen_sizes, axis=0))
    # print("scale:", np.mean(objaverse_sizes, axis=0) / np.mean(aigen_sizes, axis=0))
    print()