from robocasa.environments.kitchen.kitchen import *


class NewSponge(Kitchen):
    """
    New Sponge: composite task for Sanitize Surface activity.

    Simulates the task of cleaning the countertop.

    Steps:
        Get the sponge from the drawer and place it on the counter next to the sink.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):

        super()._setup_kitchen_references()
        self.drawer = self.register_fixture_ref(
            "drawer", dict(id=FixtureType.TOP_DRAWER, ref=self.cab)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.init_robot_base_pos = self.drawer

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Take the sponge from the drawer and place it on the counter next to the sink."
        )
        return ep_meta

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """

        super()._reset_internal()
        self.cab.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="sponge",
                obj_groups="sponge",
                graspable=True,
                placement=dict(
                    fixture=self.drawer,
                    size=(0.3, 0.3),
                    pos=(1.0, -0.5),
                ),
            )
        ),

        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.sink, loc="left_right"),
                    size=(1.0, 0.30),
                    pos=(0.0, 0.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_sink",
                obj_groups="all",
                washable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(0.25, 0.25),
                    pos=(0.0, 1.0),
                ),
            )
        )

        return cfgs
    
    def _obj_sink_dist(self, obj_name):
        """
        Returns the distance of the object from the sink
        """
        sink_points = self.sink.get_ext_sites(all_points=True, relative=False)
        obj_point = self.sim.data.body_xpos[self.obj_body_id[obj_name]]

        all_dists = [np.linalg.norm(p1 - obj_point) for p1 in sink_points]
        return np.min(all_dists)

    def _check_success(self):
        gripper_obj_far = OU.gripper_obj_far(self, "sponge")
        obj_on_counter = OU.check_obj_fixture_contact(self, "sponge", self.counter)

        obj_name = self.objects['sponge']
        obj_sink_close = self._obj_sink_dist(obj_name) < 0.35

        return gripper_obj_far and obj_sink_close and obj_on_counter
