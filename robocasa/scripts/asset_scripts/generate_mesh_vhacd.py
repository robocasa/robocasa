import os
import shutil
import trimesh
from pathlib import Path
import xml.etree.ElementTree as ET
import robocasa.utils.model_zoo.mjcf_gen_utils as MJCFGenUtils

# === CONFIGURE THESE ===
input_dirs = [
    "/Users/sepehrnasiriany/robocasa-dev/robocasa/models/assets/objects/lightwheel/fruit_bowl/FruitBowl001_edited/visual/",
    "/Users/sepehrnasiriany/robocasa-dev/robocasa/models/assets/objects/lightwheel/jar/Jar005_edited/visual/",
    "/Users/sepehrnasiriany/robocasa-dev/robocasa/models/assets/objects/lightwheel/soap_dispenser/SoapDispenser001_edited/visual/",
    "/Users/sepehrnasiriany/robocasa-dev/robocasa/models/assets/objects/lightwheel/oil_and_vinegar_bottle/OilBottle001_edited/visual/",
    "/Users/sepehrnasiriany/robocasa-dev/robocasa/models/assets/objects/lightwheel/oil_and_vinegar_bottle/OilBottle001_edited/OilBottle001_Auxiliary/visual/",
    "/Users/sepehrnasiriany/robocasa-dev/robocasa/models/assets/objects/lightwheel/salt_and_pepper_shaker/SaltShaker001_edited/visual/",
    "/Users/sepehrnasiriany/robocasa-dev/robocasa/models/assets/objects/lightwheel/salt_and_pepper_shaker/SaltShaker001_edited/SaltShaker001_Auxiliary/visual/",
]


def process_directory(input_dir):
    print(f"\n--- Processing {input_dir} ---")
    input_path = Path(input_dir)
    object_dir = input_path.parent
    prefix = object_dir.name.split("_")[0]
    model_xml_path = object_dir / "model.xml"

    collision_output_dir = object_dir / "collision"

    # --- DELETE OLD COLLISION DIRECTORY ---
    if collision_output_dir.exists():
        shutil.rmtree(collision_output_dir)
    os.makedirs(collision_output_dir, exist_ok=True)

    # Load and merge all visual meshes
    visual_meshes = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".obj"):
            mesh_path = os.path.join(input_dir, filename)
            print(f"Loading {mesh_path}...")
            mesh = trimesh.load(mesh_path, force="mesh")
            visual_meshes.append(mesh)

    if not visual_meshes:
        raise RuntimeError("No visual meshes found!")

    print("Merging meshes...")
    merged_mesh = trimesh.util.concatenate(visual_meshes)

    # Save merged mesh temporarily
    merged_mesh_prefix = "collision_mesh"
    merged_mesh_path = object_dir / f"{merged_mesh_prefix}.obj"
    merged_mesh.export(merged_mesh_path, file_type="obj", include_texture=False)
    print(f"Merged mesh saved to {merged_mesh_path}")

    print("Running VHACD...")
    max_output_hulls = 4
    MJCFGenUtils.decompose_convex(
        merged_mesh_path.resolve(),
        collision_output_dir,
        max_output_convex_hulls=max_output_hulls,
        voxel_resolution=100000,
        volume_error_percent=1.0,
        max_hull_vert_count=64,
    )

    # Modify XML
    print(f"Modifying {model_xml_path}...")
    tree = ET.parse(model_xml_path)
    root = tree.getroot()

    asset_elem = root.find("asset")
    worldbody_elem = root.find("worldbody")
    main_body = worldbody_elem.find(".//body[@name='object']")

    if asset_elem is None or main_body is None:
        raise RuntimeError("Could not find <asset> or <body name='object'> in XML.")

    # Remove old collision meshes
    for mesh in list(asset_elem.findall("mesh")):
        if "collision" in mesh.get("file", ""):
            asset_elem.remove(mesh)

    # Remove old collision geoms recursively
    def remove_all_collision_geoms(body_elem):
        for geom in list(body_elem.findall("geom")):
            if geom.get("class") == "collision":
                body_elem.remove(geom)
        for sub_body in body_elem.findall("body"):
            remove_all_collision_geoms(sub_body)

    remove_all_collision_geoms(main_body)

    # Add new meshes/geoms
    for i in range(max_output_hulls):
        mesh_name = f"{prefix}_collision_mesh_{i}"
        mesh_file = f"collision/{merged_mesh_prefix}_collision_{i}.obj"

        mesh_elem = ET.SubElement(asset_elem, "mesh")
        mesh_elem.set("file", mesh_file)
        mesh_elem.set("name", mesh_name)

        geom_elem = ET.SubElement(main_body, "geom")
        geom_elem.set("mesh", mesh_name)
        geom_elem.set("type", "mesh")
        geom_elem.set("class", "collision")

    # format xml
    def indent_xml(elem, level=0):
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            for child in elem:
                indent_xml(child, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    indent_xml(root)
    tree.write(model_xml_path, encoding="utf-8", xml_declaration=True)
    print(f"Updated model.xml written to: {model_xml_path}")

    if merged_mesh_path.exists():
        os.remove(merged_mesh_path)


for input_dir in input_dirs:
    process_directory(input_dir)
