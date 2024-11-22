import os

from pathlib import Path
import argparse
import tempfile
import shutil
import trimesh
import numpy as np

import lxml.etree as ET

from PIL import Image

from robocasa.utils.model_zoo.utils.mjcf_gen_utils import (
    parse_model_info,
    decompose_convex,
    generate_mjcf,
)

# takes list of xml paths (primary first)
def merge_xmls(xmls, model_name, save_path):

    # Load XML tree
    et = ET.parse(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "../utils/fixture_template.xml"
        )
    )

    root = et.getroot()

    # Relative path from final XML to visuals directory
    visuals = "visuals"

    root.attrib["model"] = model_name
    assets = root.find("asset")

    body = root.find("worldbody/body")
    main = True
    for xml in xmls:
        model_tree = ET.parse(xml)
        model_root = model_tree.getroot()
        # Copy textures, materials, and meshes to main XML, ignoring collision meshes
        for tex in model_root.iterfind("asset/texture"):
            if all(
                [
                    tex.attrib["name"] != existing_tex.attrib["name"]
                    for existing_tex in assets.iterfind("texture")
                ]
            ):
                tex.attrib["file"] = os.path.join(
                    visuals, Path(tex.attrib["file"]).stem + ".png"
                )
                assets.append(tex)

        for mat in model_root.iterfind("asset/material"):
            if all(
                [
                    mat.attrib["name"] != existing_mat.attrib["name"]
                    for existing_mat in assets.iterfind("material")
                ]
            ):
                assets.append(mat)

        for mesh in model_root.iterfind("asset/mesh"):
            if mesh.attrib["name"].split(".")[-1] != "_coll" and all(
                [
                    mesh.attrib["name"] != existing_mesh.attrib["name"]
                    for existing_mesh in assets.iterfind("mesh")
                ]
            ):
                mesh.attrib["file"] = os.path.join(
                    visuals, Path(mesh.attrib["file"]).stem + ".obj"
                )
                assets.append(mesh)

        new_body = ET.Element("body")
        body.append(new_body)

        model_body = model_root.find("worldbody/body/body")

        if main:
            body = body.find("body")
            body.attrib["name"] = "object"
            main = False
        else:
            new_body.attrib["name"] = Path(xml).stem
            new_body.append(ET.Comment("Add Joint Here"))

        # Add geoms (ignoring collisions) to composite XML
        for geom in model_body.iterfind("geom"):
            if geom.attrib["group"] == "1":
                del geom.attrib["group"]
                del geom.attrib["conaffinity"]
                del geom.attrib["contype"]
                geom.attrib["class"] = "visual"
                new_body.append(geom)

        new_body.append(
            ET.Comment(
                f'Add Collision Geom(s) for {Path(xml).stem} here - i.e. <geom class="collision" type="" pos="" size=""/>'
            )
        )

    et.write(save_path)


def generate_xml_from_objs(model_dir, primary, model_name, rot, verbose):
    raw = os.path.join(model_dir, "raw")

    obj_files = [
        os.path.join(model_dir, "raw", name)
        for name in os.listdir(raw)
        if name.endswith(".obj")
    ]

    if len(obj_files) == 0:
        raise Exception(f"Could not find .obj file in {raw}")

    if len(obj_files) > 1 and primary is None:
        raise Exception("Must specify primary obj if there are multiple objs")

    # get model name
    if model_name is not None:
        model_name = model_name
    elif primary is not None:
        model_name = primary[:-4]
    else:
        model_name = Path(obj_files[0]).stem

    visuals = os.path.join(model_dir, "visuals")
    os.makedirs(visuals, exist_ok=True)

    xmls = tempfile.TemporaryDirectory()

    # Load and concatenate all objs in trimesh
    meshes = []
    for obj_file in obj_files:
        meshes.append(trimesh.load(os.path.join(raw, obj_file), force="mesh"))

    conc = meshes[0]
    for m in meshes[1:]:
        conc = trimesh.util.concatenate(conc, m)

    # Compute center and scale factor for composite object
    mat_t = np.eye((4))
    center = (conc.bounds[0] + conc.bounds[1]) / 2
    mat_t[:3, 3] = -center
    transform = mat_t

    mat_s = np.eye((4))
    scale_factor = 1 / np.max(conc.bounds[1] - conc.bounds[0])
    mat_s *= scale_factor

    transform = np.matmul(mat_s, transform)

    # Use mjcf_gen_utils to create a mujoco xml for each component
    for obj_file in obj_files:
        output = tempfile.TemporaryDirectory()
        obj_name = Path(obj_file).stem
        asset_path = output.name
        if not os.path.isdir(asset_path):
            os.mkdir(asset_path)

        model_info = parse_model_info(
            model_path=raw,
            model_name=model_name,
            obj_path=obj_file,
            coll_model_path=None,
            asset_path=asset_path,
            rot=rot,
            transform=transform,
            verbose=verbose,
        )

        # generate mjcf file
        mjcf_path = generate_mjcf(
            asset_path=asset_path,
            model_name=model_name,
            model_info=model_info,
            sc=1,
            verbose=verbose,
            texture_path=os.path.join(visuals, "material_0.png"),
        )

        shutil.copyfile(
            os.path.join(output.name, "model.xml"),
            os.path.join(xmls.name, Path(obj_file).stem + ".xml"),
        )

        if os.path.isfile(os.path.join(output.name, "visual", "material_0.jpeg")):
            im = Image.open(os.path.join(output.name, "visual", "material_0.jpeg"))
            im.save(os.path.join(visuals, "material_0.png"))

        for name in os.listdir(os.path.join(output.name, "visual")):
            if name.endswith(".obj"):
                shutil.copyfile(
                    os.path.join(output.name, "visual", name),
                    os.path.join(visuals, name),
                )

        for name in os.listdir(os.path.join(output.name, "visual")):
            if name.endswith(".png"):
                shutil.copy(
                    os.path.join(output.name, "visual", name),
                    os.path.join(visuals, name),
                )

        output.cleanup()

    # Ensure primary obj is first in the xml list
    prim = primary[:-4] + ".xml"
    xml_paths = os.listdir(xmls.name)
    xml_paths.remove(prim)
    xml_paths.insert(0, prim)

    xml_paths = [os.path.join(xmls.name, path) for path in xml_paths]

    merge_xmls(xml_paths, model_name, os.path.join(model_dir, model_name + ".xml"))

    xmls.cleanup()

    # Write command args to text file so the pipeline is easy to reuse
    with open(os.path.join(model_dir, "command"), "w") as f:
        short_model_path = model_dir[model_dir.index("robosuite_model_zoo") :]
        cmd_string = f"python scripts/objs_to_mjcf.py --path ...{short_model_path}"
        if primary is not None:
            cmd_string += f" --primary {primary}"
        if model_name is not None:
            cmd_string += f" --model_name {model_name}"
        if rot != "x":
            cmd_string += f" --rot {rot}"

        f.write(cmd_string)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True, help="path to model directory"
    )
    parser.add_argument(
        "--rot",
        type=str,
        required=False,
        help="axis to rotate around",
        choices=["x", "y", "z", "none"],
        default="x",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print detailed information for debugging",
    )
    parser.add_argument(
        "--model_name",
        required=False,
        type=str,
        help="(optional) name of saved mjcf model. "
        "Defaults to primary .obj file name",
    )
    parser.add_argument(
        "--primary",
        required=False,
        type=str,
        default="model.obj",
        help="(optional) ONLY if there is one obj file. Otherwise, this is the name of the obj file which is the main body",
    )

    args = parser.parse_args()

    generate_xml_from_objs(
        args.path, args.primary, args.model_name, args.rot, args.verbose
    )
