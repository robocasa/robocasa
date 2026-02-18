from robocasa.environments.kitchen.kitchen import *


class PlaceBeveragesTogether(Kitchen):
    """
    Place Beverages Together: composite task for Arranging Buffet.

    Simulates placing 3 types of drinks in a cluster on the dining counter.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER_NON_DINING)
        )
        self.dining_counter = self.register_fixture_ref(
            "dining_counter", dict(id=FixtureType.DINING_COUNTER)
        )
        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        ep_meta["lang"] = f"Place the drinks on the dining counter in a tight cluster."

        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="alcohol",
                obj_groups="alcohol",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    size=(1.0, 0.30),
                    pos=(-0.5, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="juice",
                obj_groups="juice",
                graspable=True,
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="alcohol",
                    size=(1.0, 0.30),
                    pos=(0.0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bottled_water",
                obj_groups="bottled_water",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="alcohol",
                    size=(1.0, 0.30),
                    pos=(0.5, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distractor_tong",
                obj_groups=("tongs"),
                placement=dict(
                    fixture=self.dining_counter,
                    size=(1.0, 1.0),
                    rotation=(np.pi / 2, np.pi / 2),
                    try_to_place_in="tray",
                ),
            )
        )

        cfgs.append(
            dict(
                name="distractor_cup",
                obj_groups="cup",
                placement=dict(
                    fixture=self.dining_counter,
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distractor_cup_2",
                obj_groups="cup",
                placement=dict(
                    fixture=self.dining_counter,
                    reuse_region_from="distractor_cup",
                    size=(1.0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        drink_names = ["alcohol", "juice", "bottled_water"]
        cluster_threshold = 0.25

        all_on_counter = all(
            OU.check_obj_fixture_contact(self, name, self.dining_counter)
            for name in drink_names
        )

        drink_positions = {
            name: self.sim.data.body_xpos[self.obj_body_id[name]][:2]
            for name in drink_names
        }
        all_clustered = True
        for i, name_i in enumerate(drink_names):
            pi = drink_positions[name_i]
            dists = [
                np.linalg.norm(pi - drink_positions[name_j])
                for j, name_j in enumerate(drink_names)
                if j != i
            ]
            if min(dists) > cluster_threshold:
                all_clustered = False

        all_far = all(
            OU.gripper_obj_far(self, obj_name=name, th=0.15) for name in drink_names
        )

        return all_on_counter and all_clustered and all_far
