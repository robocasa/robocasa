from robocasa.environments.kitchen.kitchen import *


class PrepareCoffee(Kitchen):
    """
    Prepare Coffee: composite task for Brewing activity.

    Simulates the process of preparing coffee.

    Steps:
        Pick the mug from the cabinet, place it under the coffee machine dispenser,
        and press the start button.

    Args:
        cab_id (str): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which the mug is
            picked.
    """

    def __init__(self, cab_id=FixtureType.CABINET, *args, **kwargs):
        self.cab_id = cab_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.coffee_machine = self.register_fixture_ref(
            "coffee_machine", dict(id=FixtureType.COFFEE_MACHINE)
        )
        self.cab = self.register_fixture_ref(
            "cab", dict(id=self.cab_id, ref=self.coffee_machine)
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_name = self.get_obj_lang()
        ep_meta[
            "lang"
        ] = f"Pick the {obj_name} from the cabinet, place it under the coffee machine dispenser, and press the start button."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups="mug",
                placement=dict(
                    fixture=self.cab,
                    size=(0.30, 0.20),
                    pos=(0, -1.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="distr_cab",
                obj_groups="all",
                placement=dict(
                    fixture=self.cab,
                    size=(1.0, 0.20),
                    pos=(0.0, 1.0),
                    offset=(0.0, 0.0),
                ),
            )
        )

        return cfgs

    def _setup_scene(self):
        super()._setup_scene()
        self.cab.open_door(env=self)

    def _check_success(self):
        gripper_obj_far = OU.gripper_obj_far(self)
        contact_check = self.coffee_machine.check_receptacle_placement_for_pouring(
            self, "obj"
        )
        gripper_button_far = self.coffee_machine.gripper_button_far(self)
        return (
            contact_check
            and gripper_obj_far
            and self.coffee_machine._turned_on
            and gripper_button_far
        )
