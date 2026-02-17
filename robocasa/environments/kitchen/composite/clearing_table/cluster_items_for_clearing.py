from robocasa.environments.kitchen.kitchen import *


class ClusterItemsForClearing(Kitchen):
    """
    Cluster Items For Clearing: composite task for Clearing Table activity.

    Simulates the process of clustering items on one side of the dining counter
    for efficient clearing and organization.

    Steps:
        Move the two items from the dining counter to cluster with the third item
        that is already positioned on one side of the dining counter.

    Restricted to layouts with a dining table.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS + [22]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.DINING_COUNTER, full_depth_region=True),
        )

        if "refs" in self._ep_meta:
            self.anchor_side = self._ep_meta["refs"]["anchor_side"]
        else:
            self.anchor_side = self.rng.choice([-1.0, 1.0])

        self.init_robot_base_ref = self.dining_table

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        item1_lang = self.get_obj_lang("item1")
        item2_lang = self.get_obj_lang("item2")

        if item1_lang == item2_lang:
            items_text = f"{item1_lang}s"
        else:
            items_text = f"{item1_lang} and {item2_lang}"

        ep_meta["lang"] = (
            f"Move the {items_text} to "
            f"cluster with the {self.get_obj_lang('anchor_item')} on the dining counter for efficient clearing."
        )
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["anchor_side"] = self.anchor_side
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="anchor_item",
                obj_groups=["plate", "pitcher", "glass_cup"],
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    sample_region_kwargs=dict(
                        full_depth_region=True,
                    ),
                    size=(0.25, 0.25),
                    pos=(self.anchor_side, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="item1",
                obj_groups=["mug", "bowl", "cup"],
                graspable=True,
                init_robot_here=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.30, 0.30),
                    pos=(-0.2, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="item2",
                obj_groups=["mug", "bowl", "cup"],
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.30, 0.30),
                    pos=(0.0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        cluster_threshold = 0.3

        all_on_counter = all(
            OU.check_obj_fixture_contact(self, name, self.dining_table)
            for name in ["anchor_item", "item1", "item2"]
        )

        item_positions = {
            name: self.sim.data.body_xpos[self.obj_body_id[name]][:2]
            for name in ["anchor_item", "item1", "item2"]
        }

        all_clustered = True
        item_names = ["anchor_item", "item1", "item2"]
        for i, name_i in enumerate(item_names):
            pi = item_positions[name_i]
            dists = [
                np.linalg.norm(pi - item_positions[name_j])
                for j, name_j in enumerate(item_names)
                if j != i
            ]
            if min(dists) > cluster_threshold:
                all_clustered = False
                break

        all_far = all(
            OU.gripper_obj_far(self, obj_name=name, th=0.15)
            for name in ["anchor_item", "item1", "item2"]
        )

        return all_on_counter and all_clustered and all_far
