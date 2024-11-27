import xml.etree.ElementTree as ET
import re


def rm_collision(INPUT_FILE):
    # Define the regex pattern
    pattern = re.compile(r"cab_.*door")

    # Load the XML file
    tree = ET.parse(INPUT_FILE)
    root = tree.getroot()

    # List to hold elements to remove
    elements_to_remove = []

    # Iterate through all elements in the XML tree
    for elem in root.findall(".//*"):
        # Check if the element has a "name" attribute and if it contains "cab" and "door"
        if ("name" in elem.attrib and "coll" in elem.attrib["name"]) or (
            "rgba" in elem.attrib
            and (
                elem.attrib["rgba"] == "0.5 0 0 0.5"
                or elem.attrib["rgba"] == "0.5 0 0 1"
            )
        ):
            elements_to_remove.append(elem)

    # Remove the collected elements from their parents
    for elem in elements_to_remove:
        # Find the parent of the element
        for parent in root.findall(".//*"):
            if elem in list(parent):
                parent.remove(elem)
                break  # Exit the loop once the element is removed

    # Save the modified XML back to a file

    tree.write(INPUT_FILE[:-4] + "_no_collision.xml")
