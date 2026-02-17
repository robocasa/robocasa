from robocasa.environments.kitchen.kitchen import *


class MoveToCounter(Kitchen):
    """
    Move To Counter: composite task for Defrosting Food activity.

    Simulates the task of moving meat from the fridge to the counter.

    Steps:
        Pick place all of the fruits in the running sink and all of the
        vegetables in a bowl on the counter.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.fridge)
        )
        self.init_robot_base_ref = self.fridge

    def _setup_scene(self):
        self.fridge.open_door(env=self)
        super()._setup_scene()

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        food_lang = self.get_obj_lang("frosted_food")
        plate_lang = self.get_obj_lang("plate")
        ep_meta[
            "lang"
        ] = f"Pick the {food_lang} from the fridge and place it on the {plate_lang} on the counter."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="frosted_food",
                obj_groups=("meat", "vegetable"),
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.3, 0.25),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distractor_obj",
                obj_groups="all",
                exclude_obj_groups=("meat", "vegetable"),
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.3, 0.25),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                placement=dict(
                    fixture=self.counter,
                    size=(0.40, 0.40),
                    pos=(0, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        return OU.check_obj_in_receptacle(
            self, "frosted_food", "plate"
        ) and OU.gripper_obj_far(self, "frosted_food")
