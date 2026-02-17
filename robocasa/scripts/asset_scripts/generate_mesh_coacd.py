import os
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

import trimesh
import coacd
import robocasa.utils.model_zoo.mjcf_gen_utils as MJCFGenUtils

LIGHTWHEEL_ROOT = "/Users/sepehrnasiriany/robocasa-dev/robocasa/models/assets/objects/lightwheel/saucepan/Saucepan004"
CANONICAL_PREFIX = "collision_mesh"
MAX_HULLS = 32

EXCLUDE_DIRS: set[str] = set()


def find_visual_dirs(root_dir: str) -> list[str]:
    """Return visual directories to process, avoiding nested *_Lid visuals.

    Behavior:
    - If root_dir points to a specific instance folder (contains model.xml),
      only return its own `visual` subfolder.
    - Else, treat root_dir as a category directory and return the immediate
      children `<instance>/visual` folders, skipping any child that ends with
      `_Lid`.
    """
    p = Path(root_dir)
    dirs: list[str] = []

    # Case 1: root is a specific instance folder
    if (p / "model.xml").exists():
        v = p / "visual"
        if v.is_dir():
            dirs.append(str(v))
        return dirs

    # Case 2: root is a category folder; collect only immediate children
    for child in p.iterdir():
        if not child.is_dir():
            continue
        if child.name.endswith("_Lid"):
            continue
        v = child / "visual"
        if v.is_dir():
            # Optional: filter by EXCLUDE_DIRS on object type directory name
            object_type = p.name
            if object_type in EXCLUDE_DIRS:
                print(f"Skipping excluded object type: {object_type}")
                continue
            dirs.append(str(v))
    return dirs


def indent_xml(elem: ET.Element, level: int = 0) -> None:
    """Pretty print XML output."""
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for c in elem:
            indent_xml(c, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def strip_collision_geoms(body_elem: ET.Element) -> None:
    """Remove existing collision geoms (leave group=1 debug/vis alone)."""
    for g in list(body_elem.findall("geom")):
        if (
            g.get("class") == "collision" or "collision" in (g.get("mesh") or "")
        ) and g.get("group") != "1":
            body_elem.remove(g)
    for b in body_elem.findall("body"):
        strip_collision_geoms(b)


def process_directory(input_dir: str) -> None:
    input_path = Path(input_dir)
    obj_dir = input_path.parent
    prefix = obj_dir.name.split("_")[0]
    model_xml_path = obj_dir / "model.xml"
    collision_output_dir = obj_dir / "collision"

    print(f"\n=== Processing {obj_dir.name} ===")

    # fresh collision dir
    if collision_output_dir.exists():
        shutil.rmtree(collision_output_dir)
    os.makedirs(collision_output_dir, exist_ok=True)

    # Load all visual .obj meshes and merge (NO processing → keep original frame!)
    mesh_files = list(input_path.glob("*.obj"))
    if not mesh_files:
        raise RuntimeError("No .obj meshes found in input directory")

    meshes = [trimesh.load(f, force="mesh", process=False) for f in mesh_files]
    merged = trimesh.util.concatenate(meshes)
    merged_path = obj_dir / f"{CANONICAL_PREFIX}.obj"
    merged.export(merged_path, file_type="obj", include_texture=False)
    print(f"Exported merged mesh to {merged_path.name}")

    # Run CoACD decomposition in the same frame
    mesh_tm = trimesh.load(str(merged_path), force="mesh", process=False)
    mesh = coacd.Mesh(mesh_tm.vertices, mesh_tm.faces)
    parts = coacd.run_coacd(
        mesh,
        threshold=0.01,
        preprocess_mode="accurate",
        mcts_max_depth=3,
        mcts_iterations=50,
        max_convex_hull=MAX_HULLS,
        decimate=False,
        max_ch_vertex=512,
    )

    # Parse XML
    tree = ET.parse(model_xml_path)
    root = tree.getroot()
    asset = root.find("asset")
    defaults = root.find("default")
    worldbody = root.find("worldbody")
    obj_body = (
        worldbody.find(".//body[@name='object']") if worldbody is not None else None
    )

    if asset is None or defaults is None or obj_body is None:
        raise RuntimeError("Missing <asset>, <default>, or <body name='object'> in XML")

    # Ensure a 'collision' default exists
    if not any(d.get("class") == "collision" for d in defaults.findall("default")):
        col_def = ET.SubElement(defaults, "default", {"class": "collision"})
        ET.SubElement(
            col_def, "geom", {"group": "0", "rgba": "0.5 0 0 0.5", "density": "1000.0"}
        )

    # Reuse transform parameters from a representative visual mesh (scale, refquat, refpos)
    scale = refquat = refpos = None
    for mesh_tag in asset.findall("mesh"):
        # pick any visual mesh that already carries these attributes
        if any(mesh_tag.get(k) for k in ("scale", "refquat", "refpos")):
            scale = mesh_tag.get("scale")
            refquat = mesh_tag.get("refquat")
            refpos = mesh_tag.get("refpos")
            break

    # Remove any previous collision mesh assets
    for m in list(asset.findall("mesh")):
        if "collision" in (m.get("file") or ""):
            asset.remove(m)

    # Remove old collision geoms in body hierarchy
    strip_collision_geoms(obj_body)

    # Find the body that holds the visual geoms so collisions share the same frame
    # (ElementTree doesn't support nested predicates; search manually)
    visual_body = None
    for b in [obj_body] + list(obj_body.findall(".//body")):
        for g in b.findall("geom"):
            if g.get("class") == "visual":
                visual_body = b
                break
        if visual_body is not None:
            break
    if visual_body is None:
        visual_body = obj_body

    # Write each hull as an asset mesh + geom
    for i, (verts, faces) in enumerate(parts):
        mesh_name = f"{prefix}_collision_mesh_{i}"
        rel_file = f"collision/{CANONICAL_PREFIX}_collision_{i}.obj"
        mesh_out = collision_output_dir / f"{CANONICAL_PREFIX}_collision_{i}.obj"

        trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(mesh_out)

        m = ET.SubElement(asset, "mesh", {"file": rel_file, "name": mesh_name})
        if scale:
            m.set("scale", scale)
        if refquat:
            m.set("refquat", refquat)
        if refpos:
            m.set("refpos", refpos)

        ET.SubElement(
            visual_body,
            "geom",
            {"mesh": mesh_name, "type": "mesh", "class": "collision"},
        )

    merged_path.unlink()

    # Save XML
    indent_xml(root)
    tree.write(str(model_xml_path), encoding="utf-8", xml_declaration=True)
    print("model.xml updated")


if __name__ == "__main__":
    visual_dirs = find_visual_dirs(LIGHTWHEEL_ROOT)
    for d in visual_dirs:
        process_directory(d)
