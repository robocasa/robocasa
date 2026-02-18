from robocasa.environments.kitchen.kitchen import *
from robocasa.environments.kitchen.atomic.kitchen_drawer import *


class CuttingToolSelection(ManipulateDrawer):
    """Tool Selection: composite task for the chopping vegetables activity.
    Simulates the task of selecting the appropriate cutting tool based on the food item on the cutting board.
    Steps:
        Place the appropriate cutting tool (peeler or knife) on the cutting board based on the food item.
    """

    _CUTTING_MAP = {
        "garlic": "knife",
        "eggplant": "peeler",
        "broccoli": "knife",
        "carrot": "peeler",
        "cucumber": "peeler",
        "potato": "peeler",
        "avocado": "knife",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(behavior="close", *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.drawer, size=(0.5, 0.5))
        )
        if "refs" in self._ep_meta:
            self.food = self._ep_meta["refs"]["food"]
        else:
            self.food = self.rng.choice(list(self._CUTTING_MAP.keys()))

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        cat = self.get_obj_lang("food")
        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions).format(cat=cat)
        else:
            ep_meta[
                "lang"
            ] = f"Place the appropriate cutting tool for cutting the {cat} skin on the cutting board."
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["food"] = self.food
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="peeler",
                obj_groups="peeler",
                placement=dict(fixture=self.drawer, pos=(None, -0.5), size=(0.3, 0.3)),
                object_scale=[1, 1, 1.75],
            )
        )

        cfgs.append(
            dict(
                name="knife",
                obj_groups="knife",
                placement=dict(fixture=self.drawer, pos=(None, -0.5), size=(0.3, 0.3)),
                object_scale=[1, 1, 1.75],
            )
        )

        cfgs.append(
            dict(
                name="distr",
                exclude_obj_groups=["food"],
                placement=dict(
                    fixture=self.counter,
                    size=(0.6, 0.2),
                    sample_region_kwargs=dict(ref=self.drawer),
                    pos=(0, 1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="food",
                obj_groups=self.food,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.drawer, top_size=(0.5, 0.5)),
                    size=(0.5, 0.5),
                    pos=("ref", -1.0),
                    try_to_place_in="cutting_board",
                ),
            )
        )
        return cfgs

    def _check_success(self):
        peeler_on_cutting_board = OU.check_obj_in_receptacle(
            self, "peeler", "food_container"
        )
        knife_on_cutting_board = OU.check_obj_in_receptacle(
            self, "knife", "food_container"
        )

        correct_tool = self._CUTTING_MAP[self.food]
        correct_tool_chosen = (
            knife_on_cutting_board and not peeler_on_cutting_board
            if correct_tool == "knife"
            else not knife_on_cutting_board and peeler_on_cutting_board
        )
        gripper_obj_far = OU.gripper_obj_far(self, "food_container")

        return gripper_obj_far and correct_tool_chosen
