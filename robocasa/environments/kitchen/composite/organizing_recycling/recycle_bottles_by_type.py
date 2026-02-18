from robocasa.environments.kitchen.kitchen import *


class RecycleBottlesByType(Kitchen):
    """
    Recycle Bottles By Type: composite task for Organizing Recycling.

    Simulates the task of sorting bottles by type for recycling.

    Steps:
        Idenitfy the type of each bottle (plastic or glass) in the middle.
        Place plastic bottles in a cluster of plastic bottles on one side.
        Place glass bottles in a cluster of glass bottles on the other side.
    """

    # exclude other layouts since they don't have exactly 3 stools
    other_excluded_layouts = [
        8,
        11,
        12,
        13,
        19,
        21,
        22,
        23,
        25,
        28,
        29,
        30,
        33,
        34,
        38,
        41,
        42,
        46,
        49,
        54,
        57,
        58,
        60,
    ]
    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS + other_excluded_layouts

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.dining_counter = self.register_fixture_ref(
            "dining_counter", dict(id=FixtureType.DINING_COUNTER)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )

        if "stool2" in self.fixture_refs:
            self.stool1 = self.fixture_refs["stool1"]
            self.stool2 = self.fixture_refs["stool2"]
            self.stool3 = self.fixture_refs["stool3"]
            self.end_ref_plastic = self.fixture_refs["end_ref_plastic"]
            self.end_ref_glass = self.fixture_refs["end_ref_glass"]
        else:
            stools = [
                fixture
                for fixture in self.fixtures.values()
                if isinstance(fixture, robocasa.models.fixtures.accessories.Stool)
            ]
            if len(stools) < 3:
                raise RuntimeError(f"Expected at least 3 stools, found {len(stools)}")

            # find the two stools farthest apart
            max_d = -1.0
            ends = (None, None)
            for i in range(len(stools)):
                pi = stools[i].pos
                for j in range(i + 1, len(stools)):
                    pj = stools[j].pos
                    d = np.linalg.norm(pi - pj)
                    if d > max_d:
                        max_d = d
                        ends = (stools[i], stools[j])

            stool_A_candidate, stool_C_candidate = ends

            if stool_A_candidate.pos[0] > stool_C_candidate.pos[0]:
                stool_A_candidate, stool_C_candidate = (
                    stool_C_candidate,
                    stool_A_candidate,
                )

            # pick B as the stool closest to midpoint of A and C
            midpt = 0.5 * (stool_A_candidate.pos + stool_C_candidate.pos)
            best_B = None
            best_dist = float("inf")
            for s in stools:
                if s is stool_A_candidate or s is stool_C_candidate:
                    continue
                d = np.linalg.norm(s.pos - midpt)
                if d < best_dist:
                    best_dist = d
                    best_B = s

            self.stool1 = stool_A_candidate
            self.stool2 = best_B
            self.stool3 = stool_C_candidate

            # decide randomly which end gets plastic vs. glass
            if self.rng.choice([True, False]):
                self.end_ref_plastic = self.stool1
                self.end_ref_glass = self.stool3
            else:
                self.end_ref_plastic = self.stool3
                self.end_ref_glass = self.stool1

            self.fixture_refs["stool1"] = self.stool1
            self.fixture_refs["stool2"] = self.stool2
            self.fixture_refs["stool3"] = self.stool3
            self.fixture_refs["end_ref_plastic"] = self.end_ref_plastic
            self.fixture_refs["end_ref_glass"] = self.end_ref_glass

        self.init_robot_base_ref = self.stool2

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions)
        else:
            ep_meta["lang"] = (
                "Move the plastic bottles in the middle to the plastics group, "
                "and the glass bottles in the middle to the glass group."
            )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        # plastic bottles at one end
        for i in range(2):
            cfgs.append(
                dict(
                    name=f"bottle_plastic{i+1}",
                    obj_groups="bottled_water",
                    placement=dict(
                        fixture=self.dining_counter,
                        sample_region_kwargs=dict(ref=self.end_ref_plastic),
                        size=(0.3, 0.3),
                        pos=(0.0 + i * 0.3, -0.95),
                    ),
                )
            )

        # middle plastics & glass
        cfgs.append(
            dict(
                name="bottle_plastic_middle",
                obj_groups="bottled_water",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(ref=self.stool2),
                    size=(0.3, 0.3),
                    pos=("ref", -0.95),
                ),
            )
        )
        cfgs.append(
            dict(
                name="bottle_glass_middle",
                obj_groups="alcohol",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(ref=self.stool2),
                    size=(0.3, 0.3),
                    pos=("ref", -0.95),
                ),
            )
        )

        # mystery bottle in the middle
        self.choice = self.rng.choice(["alcohol", "bottled_water"])
        cfgs.append(
            dict(
                name="mystery_middle",
                obj_groups=self.choice,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(ref=self.stool2),
                    size=(0.3, 0.3),
                    pos=("ref", -0.95),
                ),
            )
        )

        # glass bottles at the other end
        for i in range(2):
            cfgs.append(
                dict(
                    name=f"bottle_glass{i+1}",
                    obj_groups="alcohol",
                    placement=dict(
                        fixture=self.dining_counter,
                        sample_region_kwargs=dict(ref=self.end_ref_glass),
                        size=(0.3, 0.3),
                        pos=(0.0 + i * 0.3, -0.95),
                    ),
                )
            )

        return cfgs

    def _check_success(self):
        """
        Checks success for RecycleBottlesByType:
        - In each cluster, *every* “middle” bottle (including the mystery)
            must be within cluster_threshold of at least one of the two sample bottles.
        - All bottles must be touching the dining counter.
        - The gripper must be far from every bottle.
        """

        plastic = ["bottle_plastic1", "bottle_plastic2", "bottle_plastic_middle"]
        glass = ["bottle_glass1", "bottle_glass2", "bottle_glass_middle"]
        mystery = "mystery_middle"

        if self.choice == "alcohol":
            glass.append(mystery)
        else:
            plastic.append(mystery)

        cluster_threshold = 0.30

        def cluster_okay(names, thresh):
            sample_names = [n for n in names if not n.endswith("_middle")]
            middle_names = [n for n in names if n.endswith("_middle")]

            pos2d = {
                n: np.array(self.sim.data.body_xpos[self.obj_body_id[n]])[:2]
                for n in names
            }

            clustered = True
            for m in middle_names:

                dists = [np.linalg.norm(pos2d[m] - pos2d[s]) for s in sample_names]
                if not any(d <= thresh for d in dists):
                    clustered = False
            return clustered

        plastic_ok = cluster_okay(plastic, cluster_threshold)
        glass_ok = cluster_okay(glass, cluster_threshold)

        all_names = plastic + glass
        on_table = all(
            OU.check_obj_fixture_contact(self, name, self.dining_counter)
            for name in all_names
        )
        all_far = all(OU.gripper_obj_far(self, obj_name=name) for name in all_names)

        return plastic_ok and glass_ok and on_table and all_far
