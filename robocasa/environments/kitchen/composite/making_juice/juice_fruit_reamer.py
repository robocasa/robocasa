from robocasa.environments.kitchen.kitchen import *
from robocasa.models.objects.kitchen_objects import BASE_ASSET_ZOO_PATH
from scipy.spatial.transform import Rotation as R


class JuiceFruitReamer(Kitchen):
    """
    JuiceFruitReamer: composite task for the making juice activity.
    Simulates the task of juicing a sliced fruit using a fruit reamer.
    Steps:
        1. Pick up the sliced fruit from the counter.
        2. Press the sliced fruit against the reamer to juice it.
    """

    _SLICED_OBJECTS = [
        {"instance": "orange_1", "split": "pretrain"},
        {"instance": "kiwi_1", "split": "pretrain"},
        {"instance": "kiwi_2", "split": "pretrain"},
        {"instance": "kiwi_6", "split": "target"},
    ]

    _CONTACT_TIMESTEPS = 10

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER)
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = OU.get_obj_lang(self, "fruit")
        ep_meta["lang"] = f"Juice the {obj_lang} by pressing it against the reamer."

        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.contact_timesteps = 0

    def _get_obj_cfgs(self):
        cfgs = []

        valid_insts = [
            info["instance"]
            for info in self._SLICED_OBJECTS
            if self.obj_instance_split is None
            or info["split"] in self.obj_instance_split
        ]
        fruit_inst = self.rng.choice(valid_insts)
        fruit_cat = fruit_inst.split("_")[0]
        obj_path = os.path.join(
            BASE_ASSET_ZOO_PATH, "objaverse", fruit_cat, f"{fruit_inst}", "model.xml"
        )

        cfgs.append(
            dict(
                name="fruit",
                obj_groups=obj_path,
                placement=dict(
                    fixture=self.counter,
                    pos=(0, -1.0),
                    # make sure that the cut side is facing downwards so that
                    # we can pick it up and juice it on the reamer without having
                    # to re-orient it.
                    rotation=(np.pi),
                    rotation_axis="y",
                    size=(0.30, 0.30),
                ),
                init_robot_here=True,
            )
        )

        cfgs.append(
            dict(
                name="reamer",
                obj_groups="reamer",
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="fruit",
                    pos=(0, -1.0),
                    size=(0.50, 0.30),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        fruit_reamer_contact = self.check_contact(
            self.objects["fruit"], self.objects["reamer"]
        )
        fruit_body_id = self.obj_body_id["fruit"]
        fruit_ori = self.sim.data.body_xquat[fruit_body_id]
        fruit_euler = R.from_quat(
            [fruit_ori[1], fruit_ori[2], fruit_ori[3], fruit_ori[0]]
        ).as_euler("xyz", degrees=True)
        fruit_zpos = self.sim.data.body_xpos[fruit_body_id][2]
        reamer_zpos = self.sim.data.body_xpos[self.obj_body_id["reamer"]][2]
        max_reamer_zpos = reamer_zpos + self.objects["reamer"].size[2] / 2
        roll = fruit_euler[0]

        # make sure that the cut part of the fruit is in contact with the reamer top part.
        if (
            fruit_reamer_contact
            and abs(180 - roll) < 20
            and OU.check_obj_grasped(self, "fruit", threshold=0.99)
            and fruit_zpos > max_reamer_zpos
        ):
            self.contact_timesteps += 1

        return self.contact_timesteps > self._CONTACT_TIMESTEPS
