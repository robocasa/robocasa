from robocasa.models.fixtures import Fixture


class Oven(Fixture):
    """
    Oven fixture class

    Args:
        xml (str): path to mjcf xml file

        name (str): name of the object
    """

    def __init__(self, xml="fixtures/ovens/Oven048", name="oven", *args, **kwargs):
        super().__init__(
            xml=xml, name=name, duplicate_collision_geoms=False, *args, **kwargs
        )

    @property
    def door_joint_names(self):
        return [f"{self.name}_door_joint"]

    def get_reset_region_names(self):
        return ("rack0", "rack1")

    @property
    def nat_lang(self):
        return "oven"
