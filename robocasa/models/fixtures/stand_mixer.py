from robocasa.models.fixtures import Fixture


class StandMixer(Fixture):
    """
    Stand mixer fixture class
    """

    def __init__(
        self, xml, name="stand_mixer", mirror_placement=False, *args, **kwargs
    ):
        super().__init__(
            xml=xml, name=name, duplicate_collision_geoms=False, *args, **kwargs
        )
        self.mirror_placement = mirror_placement

    @property
    def nat_lang(self):
        return "stand mixer"
