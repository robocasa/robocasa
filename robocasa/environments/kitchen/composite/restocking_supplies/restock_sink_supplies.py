from robocasa.environments.kitchen.kitchen import *
from robocasa.environments.kitchen.atomic.kitchen_drawer import ManipulateDrawer


class RestockSinkSupplies(ManipulateDrawer):
    """
    Restock Sink Supplies: composite task for Restocking Supplies activity.
    Simulates the task of restocking the sink area with a sponge and a soap dispenser.
    Steps:
        1. Place the sponge and soap dispenser on the counter and next to the sink.
        2. Ensure the soap dispenser is upright."""

    _MIN_SINK_DIST = 0.25

    def __init__(self, *args, **kwargs):
        super().__init__(behavior="close", *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        ep_meta[
            "lang"
        ] = f"Restock the sink supplies by placing the sponge and soap dispenser on the counter and next to the sink. Make sure the soap dispenser is upright."

        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="sponge",
                obj_groups="sponge",
                placement=dict(
                    fixture=self.drawer,
                    size=(0.5, 0.35),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="soap_dispenser",
                obj_groups="soap_dispenser",
                placement=dict(
                    fixture=self.drawer,
                    size=(0.5, 0.35),
                    pos=(0, -1.0),
                    rotation=np.pi / 2,
                    rotation_axis="x",
                ),
            )
        )

        return cfgs

    def _check_success(self):

        soap_on_any_counter = OU.check_obj_any_counter_contact(
            self, obj_name="soap_dispenser"
        )
        sponge_on_any_counter = OU.check_obj_any_counter_contact(
            self, obj_name="sponge"
        )
        soap_near_sink = (
            OU.obj_fixture_bbox_min_dist(
                self, obj_name="soap_dispenser", fixture=self.sink
            )
            < self._MIN_SINK_DIST
        )
        sponge_near_sink = (
            OU.obj_fixture_bbox_min_dist(self, obj_name="sponge", fixture=self.sink)
            < self._MIN_SINK_DIST
        )

        soap_upright = OU.check_obj_upright(self, obj_name="soap_dispenser")

        objs_far = OU.gripper_obj_far(
            self, obj_name="soap_dispenser"
        ) and OU.gripper_obj_far(self, obj_name="sponge")

        return (
            soap_on_any_counter
            and sponge_on_any_counter
            and soap_near_sink
            and sponge_near_sink
            and soap_upright
            and objs_far
        )
