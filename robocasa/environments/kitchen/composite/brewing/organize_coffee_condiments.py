from robocasa.environments.kitchen.kitchen import *


class OrganizeCoffeeCondiments(Kitchen):
    """
    Organize Condiments: composite task for Brewing activity.
    Simulates the task of organizing condiments.

    Steps:
        Open the cabinet, pick the condiments, and place them near the mug.
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
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.coffee_machine_counter = self.register_fixture_ref(
            "coffee_machine_counter",
            dict(id=FixtureType.COUNTER, ref=self.coffee_machine),
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj1_name = self.get_obj_lang("obj1")
        obj2_name = self.get_obj_lang("obj2")
        ep_meta[
            "lang"
        ] = f"Pick the {obj1_name} and {obj2_name} from the {self.cab.nat_lang} and place them next to the mug."
        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()
        self.cab.open_door(self)
        OU.add_obj_liquid_site(self, "mug", [0.4, 0.25, 0.15, 0.8])

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj1",
                obj_groups=("syrup_bottle", "cinnamon"),
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
                name="obj2",
                obj_groups="honey_bottle",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.20),
                    pos=(0, -1.0),
                ),
            )
        )

        # distractors
        cfgs.append(
            dict(
                name="mug",
                obj_groups="mug",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(1.0, 0.30),
                    pos=(0.0, -1.0),
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

    def _close_to_mug(self, obj_name, dist=0.3):
        """
        Check if the object is close to the coffee machine.

        Args:
            obj_name (str): name of the object

            dist (float): distance threshold

        Returns:
            bool: True if the object is close to the coffee machine
        """
        obj = self.objects[obj_name]
        obj_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[obj.name]])
        mug_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["mug"]])
        val = np.linalg.norm(obj_pos - mug_pos)
        return val < dist

    def _check_success(self):
        """
        Check if the cabinet to counter pick and place task is successful.
        Checks if the object is on the counter and the gripper is far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        obj1_close = self._close_to_mug("obj1")
        obj2_close = self._close_to_mug("obj2")
        obj1_on_counter = self.check_contact(
            self.objects["obj1"], self.coffee_machine_counter
        )
        obj2_on_counter = self.check_contact(
            self.objects["obj2"], self.coffee_machine_counter
        )
        gripper_obj_far = OU.gripper_obj_far(
            self, "obj1", th=0.15
        ) and OU.gripper_obj_far(self, "obj2", th=0.15)
        return (
            obj1_close
            and obj2_close
            and obj1_on_counter
            and obj2_on_counter
            and gripper_obj_far
        )
