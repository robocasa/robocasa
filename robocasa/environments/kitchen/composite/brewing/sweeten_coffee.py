from robocasa.environments.kitchen.kitchen import *


class SweetenCoffee(Kitchen):
    """
    Sweeten Coffee: composite task for Brewing activity.

    Simulates the task of sweetening coffee by adding milk and sugar.

    Steps:
        1. Take a sugar cube from the saucer and place it inside the coffee mug.
        2. Get milk from the fridge and place it next to the coffee on the counter next to the coffee machine.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.coffee_machine = self.register_fixture_ref(
            "coffee_machine", dict(id=FixtureType.COFFEE_MACHINE)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.coffee_machine)
        )
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Take a sugar cube from the saucer and place it inside the coffee. "
            "Then, grab milk from the fridge and place it next to the coffee on the counter to sweeten the coffee. "
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(self)
        OU.add_obj_liquid_site(self, "coffee", [0.65, 0.45, 0.25, 0.8])

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="coffee",
                obj_groups="mug",
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.coffee_machine,
                    ),
                    size=(0.50, 0.25),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="milk",
                obj_groups="milk",
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        z_range=(1.0, 1.5),
                    ),
                    size=(0.30, 0.15),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="saucer_plate",
                obj_groups="plate",
                object_scale=[0.6, 0.6, 1],
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.coffee_machine,
                    ),
                    size=(0.70, 0.35),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="sugar_cube",
                obj_groups="sugar_cube",
                object_scale=1.1,
                placement=dict(
                    object="saucer_plate",
                    size=(0.5, 0.5),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        milk_on_counter = OU.check_obj_any_counter_contact(self, "milk")
        coffee_on_counter = OU.check_obj_any_counter_contact(self, "coffee")

        if milk_on_counter and coffee_on_counter:
            milk_pos = self.sim.data.body_xpos[self.obj_body_id["milk"]][:2]
            coffee_pos = self.sim.data.body_xpos[self.obj_body_id["coffee"]][:2]
            distance = np.linalg.norm(milk_pos - coffee_pos)
            milk_close_to_coffee = distance <= 0.25
        else:
            milk_close_to_coffee = False

        sugar_in_coffee = OU.check_obj_in_receptacle(self, "sugar_cube", "coffee")

        gripper_far = OU.gripper_obj_far(self, obj_name="milk") and OU.gripper_obj_far(
            self, obj_name="sugar_cube"
        )

        return (
            milk_on_counter
            and coffee_on_counter
            and milk_close_to_coffee
            and sugar_in_coffee
            and gripper_far
        )
