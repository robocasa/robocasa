from robocasa.models.fixtures import *


def fixture_is_type(fixture, fixture_type):
    """
    Check if a fixture is of a certain type

    Args:
        fixture (Fixture): The fixture to check

        fixture_type (FixtureType): The type to check against
    """
    if fixture_type == FixtureType.SINK:
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
    elif fixture_type == FixtureType.TOASTER:
        return isinstance(fixture, Toaster)
    elif fixture_type == FixtureType.TOASTER_OVEN:
        return isinstance(fixture, ToasterOven)
    elif fixture_type == FixtureType.BLENDER:
        return isinstance(fixture, Blender)
    elif fixture_type == FixtureType.MICROWAVE:
        return isinstance(fixture, Microwave)
    elif fixture_type == FixtureType.STOOL:
        return isinstance(fixture, Stool)
    elif fixture_type == FixtureType.ISLAND:
        return isinstance(fixture, Counter) and "island" in fixture.name
    elif fixture_type == FixtureType.COUNTER_NON_CORNER:
        return isinstance(fixture, Counter) and "corner" not in fixture.name
    elif fixture_type == FixtureType.COUNTER:
        return isinstance(fixture, Counter)
    elif fixture_type == FixtureType.DINING_COUNTER:
        cls_check = any([isinstance(fixture, cls) for cls in [Counter]])
        if not cls_check:
            return False
        # a hack to identify counters that start with name island
        starts_with_island = fixture.name.startswith("island")
        return starts_with_island or sum(fixture.base_opening) > 0
    elif fixture_type in [
        FixtureType.CABINET,
        FixtureType.CABINET_WITH_DOOR,
        FixtureType.CABINET_SINGLE_DOOR,
        FixtureType.CABINET_DOUBLE_DOOR,
    ]:
        if fixture_type == FixtureType.CABINET:
            valid_classes = [SingleCabinet, HingeCabinet, OpenCabinet]
        elif fixture_type == FixtureType.CABINET_WITH_DOOR:
            valid_classes = [SingleCabinet, HingeCabinet]
        elif fixture_type == FixtureType.CABINET_SINGLE_DOOR:
            valid_classes = [SingleCabinet]
        elif fixture_type == FixtureType.CABINET_DOUBLE_DOOR:
            valid_classes = [HingeCabinet]
        cls_check = any([isinstance(fixture, cls) for cls in valid_classes])
        if not cls_check:
            return False

        if "stack" in fixture.name:  # wall stack cabinets not valid
            return False
        if fixture.is_corner_cab is True:  # ignore corner cabinets
            return False

        # check that there are valid reset regions
        reset_regions = fixture.get_reset_regions(
            z_range=(1.0, 1.50)
        )  # find reset regions within bounds
        if len(reset_regions) > 0:
            return True
        else:
            return False
    elif fixture_type == FixtureType.TOP_DRAWER:
        height_check = 0.7 <= fixture.pos[2] <= 0.9
        return height_check and isinstance(fixture, Drawer)
    elif fixture_type == FixtureType.DRAWER:
        return isinstance(fixture, Drawer)
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
