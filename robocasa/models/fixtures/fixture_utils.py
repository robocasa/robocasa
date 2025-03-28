from robocasa.models.fixtures import *


def fixture_is_type(fixture, fixture_type):
    """
    Check if a fixture is of a certain type

    Args:
        fixture (Fixture): The fixture to check

        fixture_type (FixtureType): The type to check against
    """
    if fixture_type == FixtureType.COUNTER:
        return isinstance(fixture, Counter)
    elif fixture_type == FixtureType.DINING_COUNTER:
        cls_check = any([isinstance(fixture, cls) for cls in [Counter]])
        if not cls_check:
            return False
        # a hack to identify counters that start with name island
        starts_with_island = fixture.name.startswith("island")
        return starts_with_island or sum(fixture.base_opening) > 0
    elif fixture_type == FixtureType.CABINET:
        return isinstance(fixture, Cabinet)
    elif fixture_type == FixtureType.DRAWER:
        return isinstance(fixture, Drawer)
    elif fixture_type == FixtureType.SINK:
        return isinstance(fixture, Sink)
    elif fixture_type == FixtureType.STOVE:
        return isinstance(fixture, Stove)
    elif fixture_type == FixtureType.OVEN:
        return isinstance(fixture, Oven)
    elif fixture_type == FixtureType.FRIDGE:
        return isinstance(fixture, Fridge)
    elif fixture_type == FixtureType.DISHWASHER:
        return isinstance(fixture, Dishwasher)
    elif fixture_type == FixtureType.COFFEE_MACHINE:
        return isinstance(fixture, CoffeeMachine)
    elif fixture_type == FixtureType.CABINET_TOP:
        cls_check = any(
            [
                isinstance(fixture, cls)
                for cls in [SingleCabinet, HingeCabinet, OpenCabinet]
            ]
        )
        if not cls_check:
            return False
        if "stack" in fixture.name:  # wall stack cabinets not valid
            return False
        # check the height of the cabinet to see if it is a top cabinet
        fxtr_bottom_z = fixture.pos[2] + fixture.bottom_offset[2]
        height_check = 1.0 <= fxtr_bottom_z <= 1.60
        return height_check
    elif fixture_type == FixtureType.MICROWAVE:
        return isinstance(fixture, Microwave)
    elif fixture_type in [FixtureType.DOOR_HINGE, FixtureType.DOOR_TOP_HINGE]:
        cls_check = any(
            [
                isinstance(fixture, cls)
                for cls in [SingleCabinet, HingeCabinet, Microwave]
            ]
        )
        if not cls_check:
            return False
        if fixture_type == FixtureType.DOOR_TOP_HINGE:
            if "stack" in fixture.name:  # wall stack cabinets not valid
                return False
            fxtr_bottom_z = fixture.pos[2] + fixture.bottom_offset[2]
            height_check = 1.0 <= fxtr_bottom_z <= 1.60
            if not height_check:
                return False
        return True
    elif fixture_type in [
        FixtureType.DOOR_HINGE_SINGLE,
        FixtureType.DOOR_TOP_HINGE_SINGLE,
    ]:
        cls_check = any(
            [isinstance(fixture, cls) for cls in [SingleCabinet, Microwave]]
        )
        if not cls_check:
            return False
        if fixture_type == FixtureType.DOOR_TOP_HINGE_SINGLE:
            if "stack" in fixture.name:  # wall stack cabinets not valid
                return False
            fxtr_bottom_z = fixture.pos[2] + fixture.bottom_offset[2]
            height_check = 1.0 <= fxtr_bottom_z <= 1.60
            if not height_check:
                return False
        return True
    elif fixture_type in [
        FixtureType.DOOR_HINGE_DOUBLE,
        FixtureType.DOOR_TOP_HINGE_DOUBLE,
    ]:
        cls_check = any([isinstance(fixture, cls) for cls in [HingeCabinet]])
        if not cls_check:
            return False
        if fixture_type == FixtureType.DOOR_TOP_HINGE_DOUBLE:
            if "stack" in fixture.name:  # wall stack cabinets not valid
                return False
            fxtr_bottom_z = fixture.pos[2] + fixture.bottom_offset[2]
            height_check = 1.0 <= fxtr_bottom_z <= 1.60
            if not height_check:
                return False
        return True
    elif fixture_type == FixtureType.TOASTER:
        return isinstance(fixture, Toaster)
    elif fixture_type == FixtureType.TOASTER_OVEN:
        return isinstance(fixture, ToasterOven)
    elif fixture_type == FixtureType.BLENDER:
        return isinstance(fixture, Blender)
    elif fixture_type == FixtureType.TOP_DRAWER:
        height_check = 0.7 <= fixture.pos[2] <= 0.9
        return height_check and isinstance(fixture, Drawer)
    elif fixture_type == FixtureType.STOOL:
        return isinstance(fixture, Stool)
    elif fixture_type == FixtureType.ISLAND:
        return isinstance(fixture, Counter) and "island" in fixture.name
    elif fixture_type == FixtureType.COUNTER_NON_CORNER:
        return isinstance(fixture, Counter) and "corner" not in fixture.name
    else:
        raise ValueError


def is_fxtr_valid(env, fxtr, size):
    """
    checks if counter is valid for object placement by making sure it is large enough

    Args:
        fxtr (Fixture): fixture to check
        size (tuple): minimum size (x,y) that the counter region must be to be valid

    Returns:
        bool: True if fixture is valid, False otherwise
    """
    for region in fxtr.get_reset_regions(env).values():
        if region["size"][0] >= size[0] and region["size"][1] >= size[1]:
            return True
    return False
