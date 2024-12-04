from robocasa.environments.kitchen.kitchen import *


class OrganizeCleaningSupplies(Kitchen):
    """
    Organize Cleaning Supplies: composite task for Tidying Cabinets And Drawers activity.

    Simulates the task of preparing to clean the sink.

    Steps:
        Open the cabinet. Pick the cleaner and place it next to the sink.
        Then close the cabinet.

    Args:
        cab_id (str): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which the cleaner is
            picked.
    """

    def __init__(self, cab_id=FixtureType.CABINET_TOP, *args, **kwargs):
        self.cab_id = cab_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id, ref=self.sink))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )
        self.init_robot_base_pos = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        cleaner_name = self.get_obj_lang("cleaner")

        ep_meta["lang"] = (
            "Open the cabinet. "
            f"Pick the {cleaner_name} and place it next to the sink. "
            "Then close the cabinet."
        )
        return ep_meta

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.cab.set_door_state(min=0.0, max=0.0, env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="cleaner",
                obj_groups="cleaner",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.20),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_counter_1",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                    offset=(0.0, 0.30),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_counter_2",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
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

        # must make sure the cleaner is on the counter and close to the sink
        gripper_obj_far = OU.gripper_obj_far(self, obj_name="cleaner")
        obj_on_counter = OU.check_obj_fixture_contact(self, "cleaner", self.counter)

        obj_name = self.objects["cleaner"].name
        obj_sink_close = self._obj_sink_dist(obj_name) < 0.35

        door_state = self.cab.get_door_state(env=self)

        door_closed = True
        for joint_p in door_state.values():
            if joint_p > 0.05:
                door_closed = False
                break

        return gripper_obj_far and obj_on_counter and door_closed and obj_sink_close
