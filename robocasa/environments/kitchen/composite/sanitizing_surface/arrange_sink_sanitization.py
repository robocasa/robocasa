from robocasa.environments.kitchen.kitchen import *


class ArrangeSinkSanitization(Kitchen):
    """
    Arrange Sink Sanitization: task for preparing cleaning supplies near the sink.

    Simulates the preparation of cleaning supplies for sink sanitization by
    grabbing a sponge and dishbrush from the cabinet and placing them next to the sink.

    Steps:
        1. Pick up the sponge and dishbrush from the cabinet
        2. Place them on the counter next to the sink

    Args:
        cab_id (int): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which the
            cleaning supplies are picked.
    """

    def __init__(self, cab_id=FixtureType.CABINET_WITH_DOOR, *args, **kwargs):
        self.cab_id = cab_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Place the sponge and dishbrush next to the sink on the counter."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.cab.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="sponge",
                obj_groups="sponge",
                placement=dict(
                    fixture=self.cab,
                    size=(0.80, 0.20),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="dishbrush",
                obj_groups="dish_brush",
                placement=dict(
                    fixture=self.cab,
                    size=(0.8, 0.20),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_cab",
                obj_groups="all",
                exclude_obj_groups=("sponge", "dish_brush"),
                placement=dict(
                    fixture=self.cab,
                    size=(1.0, 0.5),
                    pos=(0.0, -0.5),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        sponge_on_counter = OU.check_obj_any_counter_contact(self, "sponge")
        brush_on_counter = OU.check_obj_any_counter_contact(self, "dishbrush")

        gripper_far = all(
            OU.gripper_obj_far(self, obj) for obj in ["sponge", "dishbrush"]
        )

        sponge_near_sink = (
            OU.obj_fixture_bbox_min_dist(self, "sponge", self.sink) <= 0.25
        )
        brush_near_sink = (
            OU.obj_fixture_bbox_min_dist(self, "dishbrush", self.sink) <= 0.25
        )

        return (
            sponge_on_counter
            and brush_on_counter
            and gripper_far
            and sponge_near_sink
            and brush_near_sink
        )
