import argparse
import os
from tqdm import tqdm
import glob
from lxml import etree

from robosuite.utils.mjcf_utils import (
    find_elements,
    string_to_array as s2a,
    array_to_string as a2s,
)


def remove_unused_assets(xml_path):
    model_path = os.path.abspath(os.path.join(xml_path, os.pardir))
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(xml_path, parser)
    asset = tree.find("asset")

    model_name = tree.getroot().get("model")

    for mat in asset.findall("material"):
        if mat.get("name") == model_name:
            mat.getparent().remove(mat)

    # remove duplicate textures
    scanned_textures = []
    for tex in asset.findall("texture"):
        if tex.get("name") == "tex-" + model_name:
            tex.getparent().remove(tex)
            continue
        
        for tex2 in scanned_textures:
            if tex.tag == tex2.tag and tex.attrib == tex2.attrib and tex.tail == tex2.tail:
                # duplicate texture found, remove
                tex.getparent().remove(tex)
                break
        scanned_textures.append(tex)

    xml = etree.tostring(tree, pretty_print=True, encoding=str)

    with open(xml_path, "w") as file:
        file.write(xml)

    xml_backup_path = os.path.join(model_path, "bak_model.xml")
    if os.path.exists(xml_backup_path):
        os.remove(xml_backup_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        help="path to model(s)",
    )
    args = parser.parse_args()

    paths = []
    for root, dirs, files in os.walk(os.path.expanduser(args.folder)):
        for f in files:
            if f == "model.xml":
                paths.append(os.path.join(root, f))
    
    for p in tqdm(paths):
        remove_unused_assets(p)