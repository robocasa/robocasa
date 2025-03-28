from robocasa.models.fixtures import Fixture


class ToasterOven(Fixture):
    """
    Toaster oven fixture class
    """

    def __init__(self, xml, name="toaster_oven", *args, **kwargs):
        super().__init__(
            xml=xml, name=name, duplicate_collision_geoms=False, *args, **kwargs
        )

    @property
    def nat_lang(self):
        return "toaster oven"
