import xml.etree.ElementTree as ET
import os
import math
import numpy as np
from xml.dom import minidom
# from auto_texture import execute

# XML_FILE = "example.xml"

def quaternion_to_rotation_matrix(quat):
    """Convert a quaternion to a rotation matrix."""
    w, x, y, z = quat
    return np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
    ])

def euler_to_quaternion(roll, pitch, yaw):
    """Convert a euler to a quaternion."""
    # Convert degrees to radians
    roll_rad = math.radians(roll)
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)

    # Calculate half angles
    half_roll = roll_rad / 2
    half_pitch = pitch_rad / 2
    half_yaw = yaw_rad / 2

    # Calculate sine and cosine of half angles
    c_phi = math.cos(half_roll)
    s_phi = math.sin(half_roll)
    c_theta = math.cos(half_pitch)
    s_theta = math.sin(half_pitch)
    c_psi = math.cos(half_yaw)
    s_psi = math.sin(half_yaw)

    # Calculate quaternion components
    w = c_phi * c_theta * c_psi + s_phi * s_theta * s_psi
    x = s_phi * c_theta * c_psi - c_phi * s_theta * s_psi
    y = c_phi * s_theta * c_psi + s_phi * c_theta * s_psi
    z = c_phi * c_theta * s_psi - s_phi * s_theta * c_psi

    return (w, x, y, z)

def parse_geom(geom):
    # Extract attributes with default values
    pos = geom.attrib.get('pos', '0.0 0.0 0.0')  # Default position
    size = geom.attrib.get('size', '1.0 1.0')  # Default size
    euler = geom.attrib.get('euler', '0.0 0.0 0.0')  # Default to no rotation
    name = geom.attrib.get('name', 'unnamed')  # Default name if not provided
    geom_type = geom.attrib.get('type', 'box')  # Default to box if not specified

    pos = list(map(float, pos.split()))
    size = list(map(float, size.split()))
    euler = list(map(float, euler.split()))  # Convert euler to a list of floats

    return pos, size, euler, name, geom_type

def calculate_box_vertices(pos, size):
    hx = size[0] / 2
    hy = size[1] / 2
    hz = size[2] / 2 if len(size) > 2 else 0  # Default height to 0 for 2D box

    # Define the vertices of the box
    vertices = [
        (pos[0] - hx, pos[1] - hy, pos[2] - hz),
        (pos[0] + hx, pos[1] - hy, pos[2] - hz),
        (pos[0] + hx, pos[1] + hy, pos[2] - hz),
        (pos[0] - hx, pos[1] + hy, pos[2] - hz),
        (pos[0] - hx, pos[1] - hy, pos[2] + hz),
        (pos[0] + hx, pos[1] - hy, pos[2] + hz),
        (pos[0] + hx, pos[1] + hy, pos[2] + hz),
        (pos[0] - hx, pos[1] + hy, pos[2] + hz),
    ]
    
    return vertices

def create_box_obj_from_geometry(body_pos, body_quat, geom_size):
    """Create a box OBJ file from body position, quaternion, and geometry size."""
    # Define the half sizes for the box dimensions
    half_size = np.array(geom_size)
    # print(half_size)

    # Define the vertices of the box centered at the origin
    vertices = np.array([
        [-half_size[0], -half_size[1], -half_size[2]],
        [ half_size[0], -half_size[1], -half_size[2]],
        [ half_size[0],  half_size[1], -half_size[2]],
        [-half_size[0],  half_size[1], -half_size[2]],
        [-half_size[0], -half_size[1],  half_size[2]],
        [ half_size[0], -half_size[1],  half_size[2]],
        [ half_size[0],  half_size[1],  half_size[2]],
        [-half_size[0],  half_size[1],  half_size[2]],
    ])

    # Convert quaternion to rotation matrix
    rotation_matrix = quaternion_to_rotation_matrix(body_quat)

    # Apply rotation and translation to vertices
    transformed_vertices = []
    for vertex in vertices:
        rotated_vertex = rotation_matrix.dot(vertex)
        translated_vertex = rotated_vertex + body_pos
        transformed_vertices.append(translated_vertex)

    return transformed_vertices


def calculate_cylinder_vertices(pos, size, height=1.0, segments=16):
    radius = size[0] / 2
    height = size[1]
    vertices = []

    # Create vertices for the cylinder
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        vertices.append((pos[0] + x, pos[1] + y, pos[2] - height / 2))  # Bottom circle
        vertices.append((pos[0] + x, pos[1] + y, pos[2] + height / 2))  # Top circle

    return vertices

def write_obj_file(vertices, name, geom_type, foldername="objs"):
    filename = f"{foldername}/{name}.obj"
    with open(filename, 'w') as f:
        f.write(f"# OBJ file generated from geom tag: {name}\n")
        f.write(f"mtllib\n")
        f.write(f"usemtl\n")
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")


        if geom_type == 'box':
            # Define the faces of the box
            #             f.write("""
            #  # Texture coordinates
            #  vt 0.0 0.0
            #  vt 1.0 0.0
            #  vt 1.0 1.0
            #  vt 0.0 1.0
            #  
            #  # Normals
            #  vn 0.0 0.0 -1.0
            #  vn 0.0 0.0 1.0
            #  vn -1.0 0.0 0.0
            #  vn 1.0 0.0 0.0
            #  vn 0.0 -1.0 0.0
            #  vn 0.0 1.0 0.0
            #  
            #  # Faces
            #  f 1/1/1 2/2/1 3/3/1 4/4/1
            #  f 5/1/1 6/2/1 7/3/1 6/4/1
            #  f 1/1/1 5/2/1 8/3/1 4/4/1
            #  f 2/1/1 6/2/1 7/3/1 3/4/1
            #  f 1/1/1 2/2/1 6/3/1 5/4/1
            #  f 4/1/1 3/2/1 7/3/1 8/4/1
            # """)

            f.write("""
# Texture coordinates
vt 1.0 1.0
vt 1.0 0.0
vt 0.0 0.0
vt 0.0 1.0

# Normals
vn 0.0 0.0 1.0
vn 0.0 0.0 -1.0
vn 1.0 0.0 0.0
vn -1.0 0.0 0.0
vn 0.0 1.0 0.0
vn 0.0 -1.0 0.0

# Faces
f 1/1/1 2/2/1 3/3/1 4/4/1
f 5/1/1 8/2/1 7/3/1 6/4/1
f 1/1/1 4/2/1 8/3/1 5/4/1
f 2/1/1 6/2/1 7/3/1 3/4/1
f 1/1/1 5/2/1 6/3/1 2/4/1
f 4/1/1 3/2/1 7/3/1 8/4/1
""")



        elif geom_type == 'cylinder':
            # segments = len(vertices) // 2
            # faces = []
            # for i in range(segments):
            #     next_i = (i + 1) % segments
            #     # Bottom face
            #     faces.append((2 * i + 1, 2 * next_i + 1, 2 * next_i + 2, 2 * i + 2))
            #     # Top face
            #     faces.append((2 * i + 2, 2 * next_i + 2, 2 * next_i + 1, 2 * i + 1))

            # for face in faces:
            #     f.write(f"f {' '.join(map(str, face))}\n")

            # print(len(vertices))

            f.write("""
# Texture coordinates
vt 1.0 1.0  # Top right
vt 1.0 0.0  # Bottom right
vt 0.0 0.0  # Bottom left
vt 0.0 1.0  # Top left

# Additional texture coordinates for the sides
# Assuming 32 segments, we will create 32 pairs of coordinates for the sides
vt 0.0 0.0  # Vertex 1 (0 degrees)
vt 0.03125 0.0  # Vertex 2 (11.25 degrees)
vt 0.0625 0.0  # Vertex 3 (22.5 degrees)
vt 0.09375 0.0  # Vertex 4 (33.75 degrees)
vt 0.125 0.0  # Vertex 5 (45 degrees)
vt 0.15625 0.0  # Vertex 6 (56.25 degrees)
vt 0.1875 0.0  # Vertex 7 (67.5 degrees)
vt 0.21875 0.0  # Vertex 8 (78.75 degrees)
vt 0.25 0.0  # Vertex 9 (90 degrees)
vt 0.28125 0.0  # Vertex 10 (101.25 degrees)
vt 0.3125 0.0  # Vertex 11 (112.5 degrees)
vt 0.34375 0.0  # Vertex 12 (123.75 degrees)
vt 0.375 0.0  # Vertex 13 (135 degrees)
vt 0.40625 0.0  # Vertex 14 (146.25 degrees)
vt 0.4375 0.0  # Vertex 15 (157.5 degrees)
vt 0.46875 0.0  # Vertex 16 (168.75 degrees)
vt 0.5 0.0  # Vertex 17 (180 degrees)
vt 0.53125 0.0  # Vertex 18 (191.25 degrees)
vt 0.5625 0.0  # Vertex 19 (202.5 degrees)
vt 0.59375 0.0  # Vertex 20 (213.75 degrees)
vt 0.625 0.0  # Vertex 21 (225 degrees)
vt 0.65625 0.0  # Vertex 22 (236.25 degrees)
vt 0.6875 0.0  # Vertex 23 (247.5 degrees)
vt 0.71875 0.0  # Vertex 24 (258.75 degrees)
vt 0.75 0.0  # Vertex 25 (270 degrees)
vt 0.78125 0.0  # Vertex 26 (281.25 degrees)
vt 0.8125 0.0  # Vertex 27 (292.5 degrees)
vt 0.84375 0.0  # Vertex 28 (303.75 degrees)
vt 0.875 0.0  # Vertex 29 (315 degrees)
vt 0.90625 0.0  # Vertex 30 (326.25 degrees)
vt 0.9375 0.0  # Vertex 31 (337.5 degrees)
vt 0.96875 0.0  # Vertex 32 (348.75 degrees)

# Normals
vn 0.0 0.0 1.0  # Normal pointing outwards for the top face
vn 0.0 0.0 -1.0 # Normal pointing outwards for the bottom face

# Normals for the sides (assuming they point outward)
# For each side vertex, the normal will point outward from the center
# Example for 8 segments (you can repeat for 32)
vn 1.0 0.0 0.0  # Normal for vertex 1
vn 0.92388 0.0 0.38268  # Normal for vertex 2
vn 0.70711 0.0 0.70711  # Normal for vertex 3
vn 0.38268 0.0 0.92388  # Normal for vertex 4
vn 0.0 0.0 1.0  # Normal for vertex 5
vn -0.38268 0.0 0.92388  # Normal for vertex 6
vn -0.70711 0.0 0.70711  # Normal for vertex 7
vn -0.92388 0.0 0.38268  # Normal for vertex 8

# Faces
# Top face (assuming vertices 1 to 32 are the top vertices)
f 1/1/1 2/2/1 3/3/1 4/4/1 5/5/1 6/6/1 7/7/1 8/8/1 9/9/1 10/10/1 11/11/1 12/12/1 13/13/1 14/14/1 15/15/1 16/16/1 17/17/1 18/18/1 19/19/1 20/20/1 21/21/1 22/22/1 23/23/1 24/24/1 25/25/1 26/26/1 27/27/1 28/28/1 29/29/1 30/30/1 31/31/1 32/32/1

# Bottom face (assuming vertices 33 to 64 are the bottom vertices)
f 33/1/2 34/2/2 35/3/2 36/4/2 37/5/2 38/6/2 39/7/2 40/8/2 41/9/2 42/10/2 43/11/2 44/12/2 45/13/2 46/14/2 47/15/2 48/16/2 49/17/2 50/18/2 51/19/2 52/20/2 53/21/2 54/22/2 55/23/2 56/24/2 57/25/2 58/26/2 59/27/2 60/28/2 61/29/2 62/30/2 63/31/2 64/32/2
""")
            # Side faces (assuming vertices 1-32 are the top and 33-64 are the bottom)
            for i in range(1, 33):
                next_i = (i % 32) + 1
                f.write(f"f {i}/{i}/{1} {next_i}/{next_i}/{1} {next_i + 32}/{next_i}/{2} {i + 32}/{i}/{2}\n")
            # print(filename)



def convert_geoms_to_obj(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    geometries_to_remove = []
    geom_to_add = []
    mesh_to_add = []

    all_names = set()

    for body in root.findall('.//body'):

        # print("POS", body.get("pos", "0 0 0"))
        # print("QUAT", body.get("quat", "1 0 0 0"))

        # TODO: ROTATION TYPE DOUBLE-CHECK
        body_pos = np.array(list(map(float, body.get("pos", "0 0 0").split(" "))))
        body_quat = np.array(list(map(float, body.get("quat", "1 0 0 0").split(" "))))

        for geom in body.findall('./geom'):
            pos, size, euler, name, geom_type = parse_geom(geom)
            
            material = (geom.attrib.get("material"))

            if material is None:
                continue

            if geom_type == 'box':
                # vertices = calculate_box_vertices(pos, size)
                # TODO: ROTATION ADDITION WITH DRAKE
                # vertices = create_box_obj_from_geometry(body_pos + pos, body_quat + euler_to_quaternion(*euler), size) 
                vertices = create_box_obj_from_geometry(body_pos + pos, body_quat, size) 
                # vertices = create_box_obj_from_geometry(np.array([0,0,0]), np.array([1,0,0,0]), size) 

            elif geom_type == 'cylinder':
                # TODO: body_pos + pos
                vertices = calculate_cylinder_vertices(pos, size)
            elif geom_type == "mesh":
                # print(f"not processing meshes currently...")
                continue
            else:
                print(f"Unknown geometry type '{geom_type}' for '{name}'. Skipping.")
                continue

            if name in all_names:
                print("NOT ADDING", name)
                continue
        
            write_obj_file(vertices, name, geom_type)

            geom.set("name", name)
            # TODO: SHOULD EVENTUALLY REMOVE!
            geom.set("material", geom.attrib.get("material"))
            geom.set("file", f"objs/{name}.obj")
            geom.set("type", "mesh")
            geom.set("mesh", name)

            # Create a new mesh entry in the XML
            new_geom = ET.Element('geom', attrib={'name': name, 'material': geom.attrib.get("material"), 'file': f"objs/{name}.obj", "type": "mesh", "mesh": name})
            new_mesh = ET.Element('mesh', attrib={'name': name, 'material': geom.attrib.get("material"), 'file': f"objs/{name}.obj", "type": "mesh", "mesh": name})
            # print(new_mesh)
            mesh_to_add.append(new_mesh)
            geom_to_add.append(new_geom)

            geometries_to_remove.append(geom)

            # print(geom)

            # # Optionally, you can remove the original geometry
            # root.remove(geom)

            all_names.add(name)
            # print(f"OBJ file '{name}.obj' created successfully.")
        
    worldbody = root.find('.//worldbody')
    
    for mesh in geom_to_add:
        body = ET.Element('body')
        body.set('name', mesh.get('name', 'default_name'))

        # Append the existing <mesh> element to the new <body>
        body.append(mesh)

        # Append the new <body> to <worldbody>
        worldbody.append(body)

    asset = root.find('.//asset')
    
    for mesh in mesh_to_add:
        # print(mesh.get("file"))
        asset.append(mesh)

    # # List to hold geometries to be removed
    # geoms_to_remove = []

    # # Find all geom elements with type 'body' regardless of depth
    # for geom in root.findall('.//geom'):
    #     # if geom.get('type') in ('box', 'cylinder') and ("cab" not in geom.get('name', '') and "stack" not in geom.get('name', '')):
    #     if geom.get('type') in ('box', 'cylinder'):
    #         # print(geom.get("name"))
    #         geoms_to_remove.append(geom)


    # # Remove the collected geometries
    # for geom in geoms_to_remove:
    #     # Find the parent of the geom
    #     for parent in root.findall('.//'):
    #         if geom in parent:
    #             parent.remove(geom)
    #             break  # Exit after removing the first match

    # tree.write(output_file)

    # Convert the ElementTree to a string
    xml_str = ET.tostring(tree.getroot(), encoding='utf-8', method='xml')
    # Use minidom to pretty-print the XML string
    pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="    ")
    # Write the pretty-printed XML to the output file
    with open(xml_file[:-4] + "_uv.xml", 'w') as f:
        print("writing", xml_file, "...")
        f.write(pretty_xml)

# def convert_geoms_to_obj(xml_file):
#     tree = ET.parse(xml_file)
#     root = tree.getroot()
# 
#     geometries_to_remove = []
#     geom_to_add = []
#     mesh_to_add = []
# 
#     all_names = set()
# 
#     for body in root.findall('.//body'):
#         # Extract body position and quaternion
#         body_pos = np.array(list(map(float, body.get("pos", "0 0 0").split(" "))))
#         body_quat = np.array(list(map(float, body.get("quat", "1 0 0 0").split(" "))))
# 
#         for geom in body.findall('./geom'):
#             # Parse geom attributes
#             pos, size, euler, name, geom_type = parse_geom(geom)
#             material = geom.attrib.get("material")
# 
#             if material is None:
#                 continue
# 
#             # Check for duplicate names
#             if name in all_names:
#                 print(f"Duplicate geom name detected: '{name}'. Skipping.")
#                 continue
#             all_names.add(name)
# 
#             # Generate OBJ file based on geom type
#             if geom_type == 'box':
#                 vertices = create_box_obj_from_geometry(body_pos + pos, body_quat, size)
#             elif geom_type == 'cylinder':
#                 vertices = calculate_cylinder_vertices(pos, size)
#             elif geom_type == "mesh":
#                 print(f"Skipping mesh geometry for '{name}'.")
#                 continue
#             else:
#                 print(f"Unknown geometry type '{geom_type}' for '{name}'. Skipping.")
#                 continue
# 
#             # Write OBJ file
#             foldername = "objs"
#             os.makedirs(foldername, exist_ok=True)
#             write_obj_file(vertices, name, geom_type, foldername)
# 
#             # Prepare updated geom tag
#             new_geom = ET.Element("geom")
#             new_geom.set("type", "mesh")
#             new_geom.set("name", name)
#             new_geom.set("pos", " ".join(map(str, pos)))
#             new_geom.set("size", " ".join(map(str, size)))
#             new_geom.set("material", material)
#             new_geom.set("mesh", f"{foldername}/{name}.obj")
#             new_geom.set("file", f"{foldername}/{name}.obj")
# 
#             # Mark original geom for removal
#             geometries_to_remove.append(geom)
#             geom_to_add.append((body, new_geom))
# 
#     # # Remove old geoms and add new ones
#     # for old_geom in geometries_to_remove:
#     #     parent = old_geom.getparent()
#     #     if parent is not None:
#     #         parent.remove(old_geom)
# 
#     for body, new_geom in geom_to_add:
#         body.append(new_geom)
# 
#     # Write the updated XML back to the file
#     updated_xml_file = os.path.splitext(xml_file)[0] + "_uv.xml"
#     tree.write(updated_xml_file)
#     print(f"Updated XML saved to {updated_xml_file}")

# Example usage
# xml_file = "../robocasa/model_2024-09-03 09:59:59.429691.xml" 
# xml_file = "mve_cabinet.xml"  # Replace with your XML file containing multiple geom tags
# convert_geoms_to_obj(xml_file)

# convert_geoms_to_obj("model_2025-01-13 07:23:14.398406_no_collision.xml")

# convert_geoms_to_obj(XML_FILE)
