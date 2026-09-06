from robocasa.environments.kitchen.kitchen import *
from robocasa.environments.kitchen.atomic.kitchen_drawer import ManipulateDrawer


class DrawerCakeSort(ManipulateDrawer):
    """Open a drawer, place two cakes inside it, and close the drawer.

    Both cakes are sampled from a shallow band along the counter's outer edge
    so they do not start deep inside the work surface.
    """

    OUTER_EDGE_SAMPLE_DEPTH = 0.15

    def __init__(self, drawer_id=FixtureType.TOP_DRAWER, *args, **kwargs):
        super().__init__(behavior="open", drawer_id=drawer_id, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.drawer, size=(0.4, 0.2))
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Open the drawer, place both cakes inside it, then close the drawer."
        )
        return ep_meta

    def _get_obj_cfgs(self):
        return [
            dict(
                name="cake1",
                obj_groups="cake",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.drawer),
                    size=(0.3, self.OUTER_EDGE_SAMPLE_DEPTH),
                    pos=("ref", -1.0),
                ),
            ),
            dict(
                name="cake2",
                obj_groups="cake",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.drawer),
                    size=(0.5, self.OUTER_EDGE_SAMPLE_DEPTH),
                    pos=("ref", -1.0),
                ),
            ),
        ]

    def _check_success(self):
        cakes_inside_drawer = all(
            OU.obj_inside_of(self, cake_name, self.drawer)
            and not OU.check_obj_any_counter_contact(self, cake_name)
            for cake_name in ("cake1", "cake2")
        )
        gripper_far = all(
            OU.gripper_obj_far(self, obj_name=cake_name)
            for cake_name in ("cake1", "cake2")
        )
        drawer_closed = self.drawer.is_closed(env=self)
        return cakes_inside_drawer and gripper_far and drawer_closed
