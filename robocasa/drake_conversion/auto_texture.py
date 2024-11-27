import lxml.etree as le
from robocasa.drake_conversion.just_geom_conversion import convert_geoms_to_obj

"""
PRELIMINARY WORKFLOW

(1) Get the association of material to png file + mesh to obj file
(2) Use worldbody to figure out which material goes to which obj
(3) Write the new mtl/obj files
(4) Update the meshes with the new obj files
"""


def execute(xml_filename):
    ### (0) Read the file
    with open(xml_filename, "r") as f:
        doc = le.parse(f)

    ### (1) Get associations

    material_to_file = {}
    mesh_to_file = {}

    asset = doc.find(".//asset")
    if asset is not None:

        ### ASSOCIATE MATERIAL NAME TO PNG

        # search for all materials
        materials = asset.findall("material")
        for material in materials:
            material_name = material.get("name")
            texture_name = material.get("texture")
            texture = doc.find(f".//texture[@name='{texture_name}']")
            if texture is not None:
                texture_file = texture.get("file")

                # TODO: eventually, can add more details
                material_to_file[material_name] = texture_file
            else:
                # TODO: WHAT TO DO ABOUT THESE GUYS?
                # material_to_file[material_name] = "objs/grey.png"
                print(
                    f"Material {material_name} with texture {texture_name} not found!"
                )
        ### GET MESH NAME TO OBJ
        meshes = asset.findall("mesh")

        for mesh in meshes:
            mesh_to_file[mesh.get("name")] = mesh.get("file")

    # print(material_to_file)
    # print(mesh_to_file)

    obj_to_png = {}

    # (2) Associate obj to mtl

    # (2a) Create an association
    # convert_geoms_to_obj("baseline_model.xml")

    worldbody = doc.find(".//worldbody")
    if worldbody is not None:
        bodies = worldbody.findall("body")
        for body in bodies:
            body_name = body.get("name")
            geoms = body.findall("geom")

            # print(body_name)

            for geom in geoms:
                # print(geom)
                if geom.get("type") == "mesh":
                    material = geom.get("material")
                    mesh = geom.get("mesh")

                    # print(mesh)

                    obj_file = mesh_to_file.get(mesh, None)
                    material_file = material_to_file.get(material, None)

                    # print(obj_file)
                    # assert obj_file != "objs/wall_room_g0_vis.obj"

                    if material_file is not None and obj_file is not None:
                        obj_to_png[obj_file] = {
                            "file": material_file,
                            "name": body_name,
                        }
                    else:
                        print(
                            f"cannot find {material}, {mesh}: getting {material_file}, {obj_file}"
                        )
                        if material_file is not None:
                            obj_to_png[mesh] = {
                                "file": material_file,
                                "name": body_name,
                            }

                        # assert body_name != "stack_4_main_group_2_right_visual"
                else:
                    material = geom.get("material")
                    material_file = material_to_file.get(material, None)

                    if material_file is not None:
                        name = geom.get("name")
                        obj_to_png[f"obj/{name}.obj"] = {
                            "file": material_file,
                            "name": body_name,
                        }

    # print("objs/wall_room_g0_vis.obj" in obj_to_png)
    # print(obj_to_png["objs/wall_room_g0_vis.obj"])
    # assert False
    # for obj, png in obj_to_png.items():
    #     if obj[:3] == "obj":
    #         print(obj, png)
    #
    # assert False

    # (3) Write the new mtl files

    import os
    import shutil

    # Create the destination directory if it doesn't exist
    destination_folder = "meshes"
    os.makedirs(destination_folder, exist_ok=True)

    for obj_path, info_dict in obj_to_png.items():
        png_path = info_dict["file"]
        item_name = info_dict["name"]
        obj_file = item_name + "_" + (obj_path.split("/")[-1])
        png_file = item_name + "_" + (png_path.split("/")[-1])
        png_filename = png_file.split(".")[0]

        # print(obj_file, png_file)
        # print("PREFIX", obj_path[:4])

        try:
            shutil.copy(obj_path, os.path.join(destination_folder, obj_file))
            shutil.copy(png_path, os.path.join(destination_folder, png_file))
            # print(obj_file)
            # print("copy", obj_path, "to", os.path.join(destination_folder, obj_file))
            # print("copy", png_path, "to", os.path.join(destination_folder, png_file))

            # Modify the .obj file header
            obj_file_path = os.path.join(destination_folder, obj_file)

            # Read the contents of the .obj file
            with open(obj_file_path, "r") as file:
                content = file.readlines()

            # Update the specific lines
            new_content = []
            for line in content:
                if line.startswith("mtllib"):
                    line = f"mtllib {png_filename}.mtl\n"
                elif line.startswith("usemtl"):
                    line = (
                        f"usemtl {png_filename}\n"  # Change to the associated PNG name
                    )
                new_content.append(line)

            # Write the modified content back to the .obj file
            with open(obj_file_path, "w") as file:
                file.writelines(new_content)

            mtl_file_path = os.path.join(destination_folder, f"{png_filename}.mtl")

            with open(mtl_file_path, "w") as mtlf:
                mtlf.write(
                    f"""newmtl {png_filename}
    Ns 225.000000
    Ka 1.000000 1.000000 1.000000
    Kd 0.800000 0.800000 0.800000
    Ks 0.500000 0.500000 0.500000
    Ke 0.000000 0.000000 0.000000
    Ni 1.450000
    d 1.000000
    illum 2
    map_Kd {png_file}
    """
                )

            # print(f"Copied {obj_file} to {os.path.join(destination_folder, png_file)}")
        except Exception as e:
            print(f"Failed to copy {obj_file}: {e}")
        # assert obj_path[:4] != "objs"

    # assert False

    # (4) LASTLY, update the file

    obj_to_update = obj_to_png.keys()

    # Find all mesh tags
    mesh_tags = doc.findall(".//mesh")

    # Iterate through each mesh tag
    for mesh in mesh_tags:
        # Get the current file attribute
        current_file = mesh.get("file")

        # print(current_file)
        # print(current_file in obj_to_update)
        # assert current_file != "objs/wall_room_g0_vis.obj"
        # assert current_file != "wall_room_g0_vis_wall_room_g0_vis.obj"

        # Check if the current file is in the list of objects to update
        if current_file in obj_to_update:
            # Update the file attribute to the new value
            new_file_name = (
                obj_to_png[current_file]["name"] + "_" + current_file.split("/")[-1]
            )
            mesh.set("file", new_file_name)
            # print(current_file, new_file_name)
            # print(f"Updated mesh file from {current_file} to {new_file_name}")
        else:
            pass
            # # NOTE: placeholder grey
            # new_file_name = "grey.png"
            # mesh.set('file', new_file_name)
            # obj_to_png[current_file] = new_file_name
            # print("COULD NOT FIND", current_file)

    ###################################################

    # NOW, we want to copy over all of the boxes

    # Write the modified XML back to the file
    with open(xml_filename, "wb") as f:
        doc.write(f, pretty_print=True, xml_declaration=True, encoding="UTF-8")
