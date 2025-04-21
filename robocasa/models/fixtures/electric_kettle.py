from robocasa.models.fixtures import Fixture
import numpy as np


class ElectricKettle(Fixture):
    """
    Electric Kettle fixture

    Args:
        xml (str): Path to the MJCF xml file.
        name (str): Name of the object.
    """

    def __init__(
        self,
        xml="fixtures/electric_kettles/ElectricKettle003",
        name="electric_kettle",
        *args,
        **kwargs,
    ):
        super().__init__(
            xml=xml, name=name, duplicate_collision_geoms=False, *args, **kwargs
        )

    @property
    def nat_lang(self):
        return "electric kettle"
