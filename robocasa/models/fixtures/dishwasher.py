from robocasa.models.fixtures import Fixture
from robosuite.utils.mjcf_utils import (
    string_to_array,
    find_elements,
)


class Dishwasher(Fixture):
    """
    Dishwasher fixture class
    """

    def __init__(
        self,
        xml="fixtures/appliances/dishwashers/Dishwasher031",
        name="dishwasher",
        *args,
        **kwargs,
    ):
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
        return "dishwasher"
