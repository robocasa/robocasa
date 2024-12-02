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


def remove_unused_assets(model_path):
    xml_path = os.path.join(model_path, "model.xml")
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(xml_path, parser)
    asset = tree.find("asset")

    model_name = tree.getroot().get("model")

    for mat in asset.findall("material"):
        if mat.get("name") == model_name:
            mat.getparent().remove(mat)

    for tex in asset.findall("texture"):
        if tex.get("name") == "tex-" + model_name:
            tex.getparent().remove(tex)

    xml = etree.tostring(tree, pretty_print=True, encoding=str)

    with open(os.path.join(model_path, "model.xml"), "w") as file:
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

    paths = glob.glob(os.path.join(args.folder, "*/*"))

    for p in tqdm(paths):
        if os.path.isdir(p):
            remove_unused_assets(p)
