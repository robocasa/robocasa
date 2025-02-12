from robocasa.models.fixtures import Fixture


class Fridge(Fixture):
    """
    Fridge fixture class
    """

    def __init__(
        self,
        xml="fixtures/fridges/Refrigerator033/model.xml",
        name="fridge",
        *args,
        **kwargs
    ):
        super().__init__(
            xml=xml, name=name, duplicate_collision_geoms=False, *args, **kwargs
        )

    @property
    def nat_lang(self):
        return "fridge"


class FridgeFrenchDoor(Fridge):
    pass


class FridgeSideBySide(Fridge):
    pass


class FridgeTopFreezer(Fridge):
    pass
