#!/usr/bin/env python3

import os
import shutil
from distutils.dir_util import copy_tree
from pathlib import Path
import argparse
import trimesh
import robocasa
import robocasa.utils.model_zoo.mjcf_gen_utils as MJCFGenUtils
import robocasa.utils.model_zoo.file_utils as FileUtils
import robocasa.utils.model_zoo.parser_utils as ParserUtils
import robocasa.utils.model_zoo.log_utils as LogUtils
import xml.etree.ElementTree as ET


# ------------------------------------------------------------------ #
# 1. safer: wipe the old collision folder before VHACD
# ------------------------------------------------------------------ #
def maybe_decompose(raw_obj, coll_path, args):
    if Path(coll_path).exists():
        shutil.rmtree(coll_path)  # <-- os.removedirs fails unless empty
    MJCFGenUtils.decompose_convex(  # always (re)build new hulls
        Path(raw_obj),
        Path(coll_path),
        max_output_convex_hulls=args.vhacd_max_output_convex_hulls,
        voxel_resolution=args.vhacd_voxel_resolution,
        volume_error_percent=args.vhacd_volume_error_percent,
        max_hull_vert_count=args.vhacd_max_hull_vert_count,
    )


# ------------------------------------------------------------------ #
# 2. replace every old collision mesh + geom in model.xml
# ------------------------------------------------------------------ #
def update_xml_with_vhacd(model_path):
    """
    Replace existing collision assets + geoms in <model.xml>
    with every *.obj sitting in  <model_path>/collision/.
    """
    mjcf_path = Path(model_path, "model.xml")
    coll_path = Path(model_path, "collision")
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    # ----- helpers -------------------------------------------------- #
    def is_old_coll_mesh(elem):
        return elem.tag == "mesh" and elem.get("file", "").startswith("collision/")

    # bucket handles
    asset_block = root.find("asset")
    object_body = root.find(".//body[@name='object']")
    # fall back to first nested body if name wasn't set
    if object_body is None:
        object_body = root.find("worldbody/body/body")

    # ----- 2-A  purge old collision assets -------------------------- #
    old_quat = "0.7071067811865475 0.7071067811865475 0.0 0.0"
    for mesh in list(asset_block):
        if is_old_coll_mesh(mesh):
            old_quat = mesh.get("refquat", old_quat)
            asset_block.remove(mesh)

    # ----- 2-B  purge old collision geoms --------------------------- #
    for geom in list(object_body):
        if geom.tag == "geom" and geom.get("class") == "collision":
            object_body.remove(geom)

    # ----- 2-C  append new meshes + geoms --------------------------- #
    for idx, fname in enumerate(sorted(coll_path.glob("*.obj"))):
        mesh_name = f"{fname.stem}"  # e.g. straw001_collision_0
        # <mesh …>
        ET.SubElement(
            asset_block,
            "mesh",
            {
                "file": f"collision/{fname.name}",
                "name": mesh_name,
                "scale": "1 1 1",
                "refquat": old_quat,  # preserve old rotation
            },
        )
        # <geom …>
        ET.SubElement(
            object_body,
            "geom",
            {
                "mesh": mesh_name,
                "type": "mesh",
                "class": "collision",
                "group": "0",
                "rgba": "0.8 0.8 0.8 0.0",
            },
        )

    # ----- 2-D  write back ----------------------------------------- #
    tree.write(mjcf_path, encoding="utf-8", xml_declaration=True)
    print(f"✓ updated collision section in {mjcf_path}")


# ------------------------------------------------------------------ #
# 3. driver --------------------------------------------------------- #
def main():
    parser = ParserUtils.get_base_parser()
    parser.add_argument(
        "--parent_path",
        required=True,
        help="folder that contains model.xml, visual/, collision/",
    )
    args = parser.parse_args()

    for obj_inst in os.listdir(args.parent_path):
        path = os.path.join(args.parent_path, obj_inst)
        model_dir = Path(path).resolve()
        raw_obj = next(Path(model_dir, "visual").glob("*.obj"))  # any visual mesh
        coll_dir = Path(model_dir, "collision")

        maybe_decompose(raw_obj, coll_dir, args)  # (re)generate *.obj hulls
        update_xml_with_vhacd(model_dir)  # patch model.xml


if __name__ == "__main__":
    main()
