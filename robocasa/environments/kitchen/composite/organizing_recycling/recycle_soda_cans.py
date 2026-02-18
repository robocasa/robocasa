from robocasa.environments.kitchen.kitchen import *


class RecycleSodaCans(Kitchen):
    """
    Soda Can Recycling Setup: composite task for Organizing Recycling.

    Simulates the task of gathering scattered empty soda cans around the kitchen and
    placing them in a neatly clustered group, ready for recycling.

    Steps:
        Pick up all empty soda cans.
        Place them in a tight cluster on the counter near the stove.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.cab = self.register_fixture_ref("cabinet", dict(id=FixtureType.CABINET))
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.microwave = self.register_fixture_ref(
            "microwave", dict(id=FixtureType.MICROWAVE)
        )

        self.counter_microwave = self.register_fixture_ref(
            "counter_microwave", dict(id=FixtureType.COUNTER, ref=self.microwave)
        )
        self.counter_cab = self.register_fixture_ref(
            "counter_cab", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.counter_stove = self.register_fixture_ref(
            "counter_stove", dict(id=FixtureType.COUNTER, ref=self.stove)
        )
        self.dining_counter = self.register_fixture_ref(
            "dining_counter", dict(id=FixtureType.DINING_COUNTER)
        )

        # Randomly choose location for robot initialization
        possible_counters = [
            self.counter_cab,
            self.dining_counter,
            self.counter_microwave,
        ]
        chosen_counter = self.rng.choice(possible_counters)
        self.init_robot_base_ref = chosen_counter.name

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Four empty soda cans are scattered around the kitchen. "
            "Gather all the cans and cluster them next to the stove for recycling."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name=f"can1",
                obj_groups="can",
                placement=dict(
                    fixture=self.counter_microwave,
                    sample_region_kwargs=dict(
                        ref=self.microwave,
                    ),
                    size=(0.3, 0.3),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name=f"can2",
                obj_groups="can",
                placement=dict(
                    fixture=self.counter_cab,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.3, 0.3),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name=f"can3",
                obj_groups="can",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(0.3, 0.3),
                    pos=(1.0, "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name=f"can4",
                obj_groups="can",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(0.3, 0.3),
                    pos=(-1.0, "ref"),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        """
        Success criteria for Soda Can Recycling Setup:
            1. Every can must be within stove_threshold of the stove.
            2. Every can must be within cluster_threshold of at least one other can.
            3. Every can must be in contact with the stove‐counter.
            4. The robot gripper is far from every can.
        """

        cluster_threshold = 0.25
        stove_threshold = 0.25

        can_names = [f"can{i+1}" for i in range(4)]
        can_positions = {}

        # check if all cans are close to the stove
        close_to_stove = True
        for name in can_names:
            pos = np.array(self.sim.data.body_xpos[self.obj_body_id[name]])
            can_positions[name] = pos

            d_stove = OU.obj_fixture_bbox_min_dist(self, name, self.stove)
            if d_stove > stove_threshold:
                close_to_stove = False

        # check if all cans are clustered together
        cluster_ok = True
        for i, name_i in enumerate(can_names):
            pos_i = can_positions[name_i][:2]
            neighbor_ds = [
                np.linalg.norm(pos_i - can_positions[name_j][:2])
                for j, name_j in enumerate(can_names)
                if j != i
            ]
            min_d = min(neighbor_ds) if neighbor_ds else float("inf")
            if min_d > cluster_threshold:
                cluster_ok = False

        on_counter = True
        for name in can_names:
            on_ctr = OU.check_obj_any_counter_contact(self, name)
            if not on_ctr:
                on_counter = False

        all_far = all(
            OU.gripper_obj_far(self, obj_name=name, th=0.15) for name in can_names
        )

        return close_to_stove and cluster_ok and on_counter and all_far
