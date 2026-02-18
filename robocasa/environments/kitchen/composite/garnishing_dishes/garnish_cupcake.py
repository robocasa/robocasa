from robocasa.environments.kitchen.kitchen import *
import numpy as np


class GarnishCupcake(Kitchen):
    """
    Garnish Cupcake: composite task for Garnishing Dishes activity.

    Simulates the task of garnishing a cupcake with chocolate and cinnamon.

    Steps:
        Take cinnamon from the cabinet and place it next to the plate (0.3 distance).
        Take chocolate from the counter and place it on the plate with the cupcake.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cabinet = self.register_fixture_ref(
            "cabinet", dict(id=FixtureType.CABINET)
        )
        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.dining_counter = self.register_fixture_ref(
            "dining_counter",
            dict(id=FixtureType.DINING_COUNTER, ref=self.stool),
        )
        self.init_robot_base_ref = self.cabinet

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Take the cinnamon bottle from the cabinet and place it on the dining table, next to the plate with the cupcake. "
            "Then take the chocolate from the dining counter and place it on the cupcake plate."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.cabinet.open_door(self)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="cinnamon",
                obj_groups="cinnamon",
                graspable=True,
                placement=dict(
                    fixture=self.cabinet,
                    size=(1.0, 0.2),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_cabinet",
                obj_groups=["cup", "mug", "bowl"],
                graspable=True,
                placement=dict(
                    fixture=self.cabinet,
                    size=(1.0, 0.3),
                    pos=(0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="cupcake",
                obj_groups="cupcake",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(ref=self.stool),
                    size=(0.35, 0.35),
                    pos=("ref", "ref"),
                    try_to_place_in="plate",
                ),
            )
        )

        cfgs.append(
            dict(
                name="chocolate",
                obj_groups="chocolate",
                object_scale=[0.8, 0.8, 1.8],
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(ref=self.stool),
                    size=(1.0, 0.3),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        cupcake_container_name = "cupcake_container"
        cinnamon_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["cinnamon"]])
        plate_pos = np.array(
            self.sim.data.body_xpos[self.obj_body_id[cupcake_container_name]]
        )
        cinnamon_distance = np.linalg.norm(cinnamon_pos[:2] - plate_pos[:2])
        cinnamon_next_to_plate = cinnamon_distance <= 0.3

        chocolate_on_plate = OU.check_obj_in_receptacle(
            self, "chocolate", cupcake_container_name
        )
        cupcake_on_plate = OU.check_obj_in_receptacle(
            self, "cupcake", cupcake_container_name
        )
        plate_on_table = OU.check_obj_fixture_contact(
            self, cupcake_container_name, self.dining_counter
        )

        gripper_far_cinnamon = OU.gripper_obj_far(self, "cinnamon")
        gripper_far_chocolate = OU.gripper_obj_far(self, "chocolate")
        return (
            cinnamon_next_to_plate
            and chocolate_on_plate
            and cupcake_on_plate
            and plate_on_table
            and gripper_far_cinnamon
            and gripper_far_chocolate
        )
