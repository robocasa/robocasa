import numpy as np
import yaml
from robosuite.utils.mjcf_utils import array_to_string as a2s
from robosuite.utils.mjcf_utils import string_to_array as s2a

from robocasa.models.scenes.scene_registry import get_layout_path, get_style_path
from robocasa.models.scenes.scene_utils import *
from robocasa.models.fixtures import *

# fixture string to class
FIXTURES = dict(
    hinge_cabinet=HingeCabinet,
    single_cabinet=SingleCabinet,
    open_cabinet=OpenCabinet,
    panel_cabinet=PanelCabinet,
    housing_cabinet=HousingCabinet,
    drawer=Drawer,
    counter=Counter,
    stove=Stove,
    stove_wide=Stove,
    stovetop=Stovetop,
    oven=Oven,
    microwave=Microwave,
    hood=Hood,
    sink=Sink,
    fridge_french_door=FridgeFrenchDoor,
    fridge_side_by_side=FridgeSideBySide,
    fridge_bottom_freezer=FridgeBottomFreezer,
    dishwasher=Dishwasher,
    wall=Wall,
    floor=Floor,
    box=Box,
    accessory=Accessory,
    paper_towel=Accessory,
    plant=Accessory,
    knife_block=Accessory,
    stool=Stool,
    utensil_holder=Accessory,
    coffee_machine=CoffeeMachine,
    toaster=Toaster,
    toaster_oven=ToasterOven,
    blender=Blender,
    stand_mixer=StandMixer,
    utensil_rack=WallAccessory,
    electric_kettle=ElectricKettle,
    wall_accessory=WallAccessory,
    window=Window,
    framed_window=FramedWindow,
    # needs some additional work
    # slide_cabinet=SlideCabinet,
)
# fixtures that are attached to other fixtures, disables positioning system in this script
FIXTURES_INTERIOR = dict(
    sink=Sink, stovetop=Stovetop, accessory=Accessory, wall_accessory=WallAccessory
)

ALL_SIDES = ["left", "right", "front", "back", "bottom", "top"]


def check_syntax(fixture):
    """
    Checks that specifications of a fixture follows syntax rules
    """

    if fixture["type"] != "stack" and fixture["type"] not in FIXTURES:
        raise ValueError(
            'Invalid value for fixture type: "{}".'.format(fixture["type"])
        )

    if "config_name" in fixture and "default_config_name" in fixture:
        raise ValueError('Cannot specify both "config_name" and "default_config_name"')

    if "align_to" in fixture or "side" in fixture or "alignment" in fixture:
        if not ("align_to" in fixture and "side" in fixture):
            raise ValueError(
                'Both or neither of "align_to" and ' '"side" need to be specified.'
            )
        if "pos" in fixture:
            raise ValueError("Cannot specify both relative and absolute positions.")

        # check alignment and side arguments are compatible
        if "alignment" in fixture:
            for keywords in AXES_KEYWORDS.values():
                if fixture["side"] in keywords:
                    # check that neither keyword is used for alignment
                    if (
                        keywords[0] in fixture["alignment"]
                        or keywords[1] in fixture["alignment"]
                    ):
                        raise ValueError(
                            'Cannot set alignment to "{}" when aligning to the "{}" side'.format(
                                fixture["alignment"], fixture["side"]
                            )
                        )

        # check if side is valid
        if fixture["side"] not in ALL_SIDES:
            raise ValueError(
                '"{}" is not a valid side for alignment'.format(fixture["side"])
            )


def create_fixtures(layout_config, style_config, rng=None):
    """
    Initializes fixtures based on the given layout yaml file and style type

    Args:
        layout_config (dict): layout of the kitchen to load

        style_config (dict): style of the kitchen to load

        rng (np.random.Generator): random number generator used for initializing fixture state
    """

    # contains all fixtures with updated configs
    arena = list()

    # Update each fixture config. First iterate through groups: subparts of the arena that can be
    # rotated and displaced together. example: island group, right group, room group, etc
    for group_name, group_config in layout_config.items():
        group_fixtures = list()
        # each group is further divded into similar subcollections of fixtures
        # ex: main group counter accessories, main group top cabinets, etc
        for k, fixture_list in group_config.items():
            # these values are rotations/displacements that are applied to all fixtures in the group
            if k in ["group_origin", "group_z_rot", "group_pos"]:
                continue
            elif type(fixture_list) != list:
                raise ValueError('"{}" is not a valid argument for groups'.format(k))

            # add suffix to support different groups
            for fxtr_config in fixture_list:
                fxtr_config["name"] += "_" + group_name
                # update fixture names for alignment, interior objects, etc.
                for k in ATTACH_ARGS + ["align_to", "stack_fixtures", "size"]:
                    if k in fxtr_config:
                        if isinstance(fxtr_config[k], list):
                            for i in range(len(fxtr_config[k])):
                                if isinstance(fxtr_config[k][i], str):
                                    fxtr_config[k][i] += "_" + group_name
                        else:
                            if isinstance(fxtr_config[k], str):
                                fxtr_config[k] += "_" + group_name

            group_fixtures.extend(fixture_list)

        # update group rotation/displacement if necessary
        if "group_origin" in group_config:
            for fxtr_config in group_fixtures:
                # do not update the rotation of the walls/floor
                if fxtr_config["type"] in ["wall", "floor"]:
                    continue
                fxtr_config["group_origin"] = group_config["group_origin"]
                fxtr_config["group_pos"] = group_config["group_pos"]
                fxtr_config["group_z_rot"] = group_config["group_z_rot"]

        # addto overall fixture list
        arena.extend(group_fixtures)

    # maps each fixture name to its object class
    fixtures = dict()
    # maps each fixture name to its configuration
    configs = dict()
    # names of composites, delete from fixtures before returning
    composites = list()

    # initialize each fixture in the arena by processing config
    for fixture_config in arena:
        check_syntax(fixture_config)

        enable = fixture_config.get("enable", True)
        if not enable:
            # skip including the fixture in the scene if enable=False
            continue

        fixture_name = fixture_config["name"]

        # stack of fixtures, handled separately
        if fixture_config["type"] == "stack":
            stack = FixtureStack(
                fixture_config,
                fixtures,
                configs,
                style_config,
                default_texture=None,
                rng=rng,
            )
            fixtures[fixture_name] = stack
            configs[fixture_name] = fixture_config
            composites.append(fixture_name)
            continue

        # load style information and update config to include it
        default_config = load_style_config(style_config, fixture_config)
        if default_config is not None:
            for k, v in fixture_config.items():
                default_config[k] = v
            fixture_config = default_config

        # set fixture type
        fixture_config["type"] = FIXTURES[fixture_config["type"]]

        # pre-processing for fixture size
        size = fixture_config.get("size", None)
        if isinstance(size, list):
            for i in range(len(size)):
                elem = size[i]
                if isinstance(elem, str):
                    ref_fxtr = fixtures[elem]
                    size[i] = ref_fxtr.size[i]

        # initialize fixture
        fixture = initialize_fixture(fixture_config, fixtures, rng=rng)
        fixtures[fixture_name] = fixture
        configs[fixture_name] = fixture_config

        # update fixture position
        if fixture_config["type"] not in FIXTURES_INTERIOR.values():
            # relative positioning
            if "align_to" in fixture_config:
                pos = get_relative_position(
                    fixture,
                    fixture_config,
                    fixtures[fixture_config["align_to"]],
                    configs[fixture_config["align_to"]],
                )

            elif "stack_on" in fixture_config:
                stack_on = fixtures[fixture_config["stack_on"]]

                # account for off-centered objects
                stack_on_center = stack_on.center

                # infer unspecified axes of position
                pos = fixture_config["pos"]
                if pos[0] is None:
                    pos[0] = stack_on.pos[0] + stack_on_center[0]
                if pos[1] is None:
                    pos[1] = stack_on.pos[1] + stack_on_center[1]

                # calculate height of fixture
                pos[2] = stack_on.pos[2] + stack_on.size[2] / 2 + fixture.size[2] / 2
                pos[2] += stack_on_center[2]
            else:
                # absolute position
                pos = fixture_config.get("pos", None)
            if pos is not None and type(fixture) not in [Wall, Floor]:
                fixture.set_pos(pos)

    # composites are non-MujocoObjects, must remove
    for composite in composites:
        del fixtures[composite]

    # update the rotation and postion of each fixture based on their group
    for name, fixture in fixtures.items():
        # check if updates are necessary
        config = configs[name]
        if "group_origin" not in config:
            continue

        # TODO: add default for group origin?
        # rotate about this coordinate (around the z-axis)
        origin = config["group_origin"]
        pos = config["group_pos"]
        z_rot = config["group_z_rot"]
        displacement = [pos[0] - origin[0], pos[1] - origin[1]]

        if type(fixture) not in [Wall, Floor]:
            dx = fixture.pos[0] - origin[0]
            dy = fixture.pos[1] - origin[1]
            dx_rot = dx * np.cos(z_rot) - dy * np.sin(z_rot)
            dy_rot = dx * np.sin(z_rot) + dy * np.cos(z_rot)

            x_rot = origin[0] + dx_rot
            y_rot = origin[1] + dy_rot
            z = fixture.pos[2]
            pos_new = [x_rot + displacement[0], y_rot + displacement[1], z]

            # account for previous z-axis rotation
            rot_prev = fixture._obj.get("euler")
            if rot_prev is not None:
                # TODO: switch to quaternion since euler rotations are ambiguous
                rot_new = s2a(rot_prev)
                rot_new[2] += z_rot
            else:
                rot_new = [0, 0, z_rot]

            fixture._obj.set("pos", a2s(pos_new))
            fixture._obj.set("euler", a2s(rot_new))

    return fixtures
