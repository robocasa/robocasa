import xml.etree.ElementTree as ET
import re

# Define the regex pattern
pattern = re.compile(r'cab_.*door')

# Load the XML file
INPUT_FILE = 'modified_baseline_model.xml'
tree = ET.parse(INPUT_FILE)
root = tree.getroot()

# List to hold elements to remove
elements_to_remove = []

# Iterate through all elements in the XML tree
for elem in root.findall('.//*'):
    # Check if the element has a "name" attribute and if it contains "cab" and "door"
    if "name" in elem.attrib and ("cab" in elem.attrib["name"] or "stack" in elem.attrib["name"]) and "door" in elem.attrib["name"]:
        elements_to_remove.append(elem)

# Remove the collected elements from their parents
for elem in elements_to_remove:
    # Find the parent of the element
    for parent in root.findall('.//*'):
        if elem in list(parent):
            print("removing..")
            parent.remove(elem)
            break  # Exit the loop once the element is removed

# Save the modified XML back to a file
tree.write('modified_baseline_model_no_doors.xml')

