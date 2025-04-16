from robocasa.environments.kitchen.kitchen import *


class PrepMarinatingMeat(Kitchen):
    """
    Prep Marinating Meat: composite task for Meat Preparation activity.

    Simulates the task of preparing meat for marinating.

    Steps:
        Take the meat from its container and place it on the cutting board. Then,
        take the condiment from the cabinet and place it next to the cutting board.

    Args:
        cab_id (str): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which the condiment
            is picked.
    """

    def __init__(self, cab_id=FixtureType.CABINET_DOUBLE_DOOR, *args, **kwargs):
        # fixture type hingle double bc there will be a large counter space under the cabinet
        self.cab_id = cab_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab, size=(0.6, 0.6))
        )
        self.init_robot_base_pos = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        cond_name = self.get_obj_lang("condiment")
        meat_name = self.get_obj_lang("meat")
        cont_name = self.get_obj_lang("meat_container")
        ep_meta["lang"] = (
            f"Pick the {meat_name} from the {cont_name} and place it on the cutting board. "
            f"Then pick the {cond_name} from the cabinet and place it next to the cutting board."
        )

        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()
        self.cab.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="cutting_board",
                obj_groups="cutting_board",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(1.0, 0.5),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="meat",
                obj_groups="meat",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                    try_to_place_in="container",
                ),
            )
        )

        cfgs.append(
            dict(
                name="condiment",
                obj_groups="condiment_bottle",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.1),
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

    def _check_success(self):
        gripper_obj_far = OU.gripper_obj_far(self, "condiment") and OU.gripper_obj_far(
            self, "meat"
        )
        condiment_on_counter = self.check_contact(
            self.objects["condiment"], self.counter
        )
        meat_on_cutting_board = OU.check_obj_in_receptacle(
            self, "meat", "cutting_board"
        )
        cutting_board_on_counter = OU.check_obj_fixture_contact(
            self, "cutting_board", self.counter
        )
        return (
            gripper_obj_far
            and meat_on_cutting_board
            and cutting_board_on_counter
            and condiment_on_counter
        )
