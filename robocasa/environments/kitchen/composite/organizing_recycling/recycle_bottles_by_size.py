from robocasa.environments.kitchen.kitchen import *


class RecycleBottlesBySize(Kitchen):
    """
    Recycle Bottles By Size: composite task for Organizing Recycling.

    Simulates the task of sorting empty bottles by size for recycling.

    Steps:
        Gather all empty bottled water and beer bottles from the dining counter.
        Place large bottles in a cluster next the sink, but not inside it.
        Place small bottles inside the sink.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )

        if "stool" in self.fixture_refs:
            self.stool = self.fixture_refs["stool"]
        else:
            registered_stool_ids = set()
            self.stool = None
            while len(registered_stool_ids) < 2:
                for fixture in self.fixtures.values():
                    if isinstance(fixture, robocasa.models.fixtures.accessories.Stool):
                        fixture_id = id(fixture)
                        if fixture_id not in registered_stool_ids:
                            registered_stool_ids.add(fixture_id)
                            if self.stool is None:
                                self.stool = fixture
            self.fixture_refs["stool"] = self.stool

        self.dining_counter = self.register_fixture_ref(
            "dining_counter", dict(id=FixtureType.DINING_COUNTER, ref=self.stool)
        )

        self.init_robot_base_ref = self.dining_counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Place the large bottle from the dining counter next to the other large bottle outside the sink, "
            "and place the small bottle from the dining counter next to the other small bottle inside the sink."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name=f"bottle_large1",
                obj_groups="bottled_drink",
                object_scale=1.40,
                init_robot_here=True,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(0.2, 0.2),
                    pos=(0, -0.9),
                ),
            )
        )

        cfgs.append(
            dict(
                name=f"bottle_small1",
                obj_groups="bottled_drink",
                object_scale=0.90,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(0.2, 0.2),
                    pos=(0.2, -0.9),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bottle_large_sample",
                obj_groups="bottled_drink",
                object_scale=1.40,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.2, 0.4),
                    pos=("ref", -0.9),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bottle_small_sample",
                obj_groups="bottled_drink",
                object_scale=0.90,
                placement=dict(
                    fixture=self.sink,
                    size=(0.2, 0.2),
                    pos=(1.0, 1.0),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        """
        Success criteria for Bottle Sorting task:
        1. The small bottle and its sample must be inside the sink.
        2. The large bottle from the dining counter must be within distance_threshold of the large sample,
            and touching the designated counter.
        3. Both large bottles must form a cluster.
        4. The robot gripper must be far from every bottle.
        """
        distance_threshold = 0.25
        cluster_threshold = 0.40

        small_bottles_in_sink = True
        gripper_far_from_all = True

        for name in ["bottle_small1", "bottle_small_sample"]:
            if not OU.obj_inside_of(self, name, self.sink):
                small_bottles_in_sink = False
            if not OU.gripper_obj_far(self, obj_name=name):
                gripper_far_from_all = False

        large_names = ["bottle_large1", "bottle_large_sample"]
        large_bottles_near_sink = True
        large_bottle_on_counter = True

        for name in large_names:
            d_sink = OU.obj_fixture_bbox_min_dist(self, name, self.sink)
            if d_sink > distance_threshold:
                large_bottles_near_sink = False

            if not OU.check_obj_fixture_contact(self, name, self.counter):
                large_bottle_on_counter = False

            if not OU.gripper_obj_far(self, obj_name=name):
                gripper_far_from_all = False

        positions = [
            self.sim.data.body_xpos[self.obj_body_id[name]] for name in large_names
        ]

        clustered = True
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if np.linalg.norm(positions[i] - positions[j]) > cluster_threshold:
                    clustered = False
                    break
            if not clustered:
                break

        return (
            small_bottles_in_sink
            and large_bottles_near_sink
            and large_bottle_on_counter
            and clustered
            and gripper_far_from_all
        )
