from robocasa.models.objects.fixtures.fixture import Fixture

import numpy as np
from copy import deepcopy

class Hood(Fixture):
    def __init__(
        self,
        xml="fixtures/hoods/basic_silver.xml",
        name="hood",
        *args,
        **kwargs
    ):
        super().__init__(
            xml=xml,
            name=name,
            duplicate_collision_geoms=False,
            *args,
            **kwargs
        )
    
    @property
    def nat_lang(self):
        return "range hood"