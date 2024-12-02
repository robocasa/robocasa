# RoboCasa Model Zoo Guide
This guide explains how to import and visualize assets in RoboCasa.

## Overview
The structure of generated assets is as follows:
- `model.xml`: mjcf file containing visual geoms, collision geoms, joints, and sites
- `raw`: the original, unmodified model files (includes .glb file, .obj/.mtl files, image texture files)
- `visual`: the processed visual data (.obj files, image texture files)
- `collision`: the collision meshes .obj files (if applicable)

## Importing objects
#### `import_glb_model.py`
Import a `.glb` model:
```
python robocasa/scripts/model_zoo/import_glb_model.py --prescale --center --no_cached_coll --path /Users/soroushnasiriany/tmp/model_zoo_test/model1
```

## Inspecting objects
#### `browse_mjcf_model.py`
Visualize the assets interactively:
```
python robocasa/scripts/browse_mjcf_model.py --show_bbox --show_coll_geoms --mjcf <path-to-model.xml>
```
Arguments:
- `show_bbox`: visualize the bounding box
- `show_coll_geoms`: show collision geoms

<!-- #### `object_play.py`
Play around with generated object model interactively with the robot arm.

Arguments:
- `mjcf_path`: name of the mjcf model
- (all other arguements from [collect_human_demonstrations.py](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/scripts/collect_human_demonstrations.py)) -->

## Importing fixtures
1. Bring object into Blender
- Use objaverse to find an object model and download it in the .glb format. A more manual process can be done with other file types, but none are well tested.
- Create a directory for the model in robosuite_model_zoo/assets/kitchen_fixtures/<fixture_type>
- Put the .glb file in the newly created model directory
- Use the Blender scripting tab to load scripts/glb_to_blender.py and set the string on line 29 to point to the model directory
- Run the script to bring the object into blender and export the textures - it should create a separate 'raw' directory in the model directory

2. Export object from Blender
- Select objects that are part of a single body (not connected by a joint)
    - Be sure to select the actual objects (with the triangle symbol in front), and not just the parent groupings
- File > Export as a Wavefront .obj and ensure that Limit to Selected Only is checked
- Export the files into the raw directory
- This should create both a .obj file and a .mtl file with the same name if done correctly
- Repeat for each connected body, using a different name for the .obj/.mtl files for each export

3. Turn .obj files into Mujoco xml
```
    python objs_to_mjcf.py --path <path_to_dir> --primary <primary_body_obj_in_directory>
```
- The path argument should be the model directory created in section 1, and the primary is the obj file for the main body of the object
- If you would like the model name to be different from the name of the primary body, add a --model_name parameter
- After running the script, there should be a composite .xml file, a visuals directory, and a command file in the model directory.
    - The composite .xml file is a mujoco XML which correctly references all visual assets in visuals, is prescaled, and is centered at the origin
    - The command file specifies the command used 

4. Add joints, collision boxes, and sites to the xml
- The .xml file has comments where to include the collision geoms and joints
    - Make sure to use class="collision" for the collision geoms
- Use the mujoco viewer to check the fit of the joints, collision geoms, and sites while working
```
    python -m mujoco.viewer --mjcf=<xml_path>
```

5. Add object to the kitchen
- The current directory structure for visuals in robosuite-model-zoo does not match the one required for kitchen fixtures, so it is not easy to import
    - Eventually, robosuite will pull objects from robosuite-model-zoo for fixtures
    - This has not been tested, but it may be possible to just copy the contents of the object directory as a way to test temporarily
