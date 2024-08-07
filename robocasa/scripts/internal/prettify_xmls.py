import argparse
import os

from lxml import etree
from tqdm import tqdm


def prettify_xmls(folder=None, filepath=None):
    if filepath is not None:
        filepaths = [filepath]
    else:
        filepaths = []
        for root, dirs, files in os.walk(os.path.expanduser(folder)):
            for f in files:
                if f.endswith(".xml"):
                    filepaths.append(os.path.join(root, f))

    for f in tqdm(filepaths):
        parser = etree.XMLParser(remove_blank_text=True)
        tree = etree.parse(f, parser)
        xml = etree.tostring(tree, pretty_print=True, encoding=str)
        with open(f, "w") as file:
            file.write(xml)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="folder to scan xml files")
    parser.add_argument("--filepath", type=str, help="file path")
    args = parser.parse_args()

    prettify_xmls(folder=args.folder, filepath=args.filepath)
