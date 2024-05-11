from robocasa.models.objects.fixtures.fixture import Fixture

class Dishwasher(Fixture):
    def __init__(
        self,
        xml="fixtures/appliances/dishwashers/pack_1/model.xml",
        name="dishwasher",
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
        return "dishwasher"