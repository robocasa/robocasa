from robocasa.models.fixtures import Fixture


class Oven(Fixture):
    """
    Oven fixture class

    Args:
        xml (str): path to mjcf xml file

        name (str): name of the object
    """

    def __init__(self, xml="fixtures/ovens/Oven001", name="oven", *args, **kwargs):
        super().__init__(
            xml=xml, name=name, duplicate_collision_geoms=False, *args, **kwargs
        )

    def get_reset_region_names(self):
        return ("rack_bottom", "rack_top")

    @property
    def nat_lang(self):
        return "oven"
