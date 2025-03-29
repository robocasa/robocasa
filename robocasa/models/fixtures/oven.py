from robocasa.models.fixtures import Fixture


class Oven(Fixture):
    """
    Oven fixture class

    Args:
        xml (str): path to mjcf xml file

        name (str): name of the object
    """

    RESET_REGION_NAMES = ["rack_bottom", "rack_top"]

    def __init__(self, xml="fixtures/ovens/Oven001", name="oven", *args, **kwargs):
        super().__init__(
            xml=xml, name=name, duplicate_collision_geoms=False, *args, **kwargs
        )

    @property
    def nat_lang(self):
        return "oven"
