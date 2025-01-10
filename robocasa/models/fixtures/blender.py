from robocasa.models.fixtures import Fixture


class Blender(Fixture):
    """
    Blender fixture class
    """

    def __init__(self, xml, name="blender", *args, **kwargs):
        super().__init__(
            xml=xml, name=name, duplicate_collision_geoms=False, *args, **kwargs
        )

    @property
    def nat_lang(self):
        return "blender"
