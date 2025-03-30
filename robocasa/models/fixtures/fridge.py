from robocasa.models.fixtures import Fixture


class Fridge(Fixture):
    """
    Fridge fixture class
    """

    def __init__(
        self, xml="fixtures/fridges/Refrigerator033", name="fridge", *args, **kwargs
    ):
        super().__init__(
            xml=xml, name=name, duplicate_collision_geoms=False, *args, **kwargs
        )

    @property
    def nat_lang(self):
        return "fridge"


class FridgeFrenchDoor(Fridge):
    RESET_REGION_NAMES = [
        "fridge_left_shelf0",
        "fridge_left_shelf1",
        "fridge_right_shelf0",
        "fridge_right_shelf1",
    ]


class FridgeSideBySide(Fridge):
    pass


class FridgeBottomFreezer(Fridge):
    pass
