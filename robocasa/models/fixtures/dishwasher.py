from robocasa.models.fixtures import Fixture


class Dishwasher(Fixture):
    """
    Dishwasher fixture class
    """

    def __init__(
        self,
        xml="fixtures/appliances/dishwashers/Dishwasher031",
        name="dishwasher",
        *args,
        **kwargs
    ):
        super().__init__(
            xml=xml, name=name, duplicate_collision_geoms=False, *args, **kwargs
        )

    def get_reset_region_names(self):
        return ("rack_bottom", "rack_top")

    @property
    def nat_lang(self):
        return "dishwasher"
