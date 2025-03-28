from robocasa.models.fixtures import Fixture


class Dishwasher(Fixture):
    """
    Dishwasher fixture class
    """

    RESET_REGION_NAMES = ["rack_top", "rack_bottom"]

    def __init__(
        self,
        xml="fixtures/appliances/dishwashers/pack_1/model.xml",
        name="dishwasher",
        *args,
        **kwargs
    ):
        super().__init__(
            xml=xml, name=name, duplicate_collision_geoms=False, *args, **kwargs
        )

    @property
    def nat_lang(self):
        return "dishwasher"
