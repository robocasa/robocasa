from copy import deepcopy

import numpy as np

from robocasa.models.objects.fixtures.fixture import Fixture

STOVE_LOCATIONS = [
    "rear_left",
    "rear_center",
    "rear_right",
    "front_left",
    "front_center",
    "front_right",
    "center",
]


class Stove(Fixture):
    def __init__(self, xml="fixtures/stoves/stove_orig", name="stove", *args, **kwargs):
        self._knob_joints = None
        self._burner_sites = None

        super().__init__(
            xml=xml, name=name, duplicate_collision_geoms=False, *args, **kwargs
        )

    def get_reset_regions(self, env, locs=None):
        regions = dict()

        if locs is None:
            locs = STOVE_LOCATIONS
        for location in locs:
            site = self.worldbody.find(
                "./body/body/site[@name='{}burner_{}_place_site']".format(
                    self.naming_prefix, location
                )
            )
            if site is None:
                site = self.worldbody.find(
                    "./body/body/site[@name='{}burner_on_{}']".format(
                        self.naming_prefix, location
                    )
                )
            if site is None:
                continue
            burner_pos = [float(x) for x in site.get("pos").split()]
            regions[location] = {
                "offset": burner_pos,
                "size": [0.10, 0.10],
            }

        return regions

    def update_state(self, env):
        for location in STOVE_LOCATIONS:
            site = self.burner_sites[location]
            if site is None:
                continue
            site_id = env.sim.model.site_name2id(
                "{}burner_on_{}".format(self.naming_prefix, location)
            )

            joint = self.knob_joints[location]
            if joint is None:
                env.sim.model.site_rgba[site_id][3] = 0.0
                continue
            joint_id = env.sim.model.joint_name2id(
                "{}knob_{}_joint".format(self.naming_prefix, location)
            )

            joint_qpos = deepcopy(env.sim.data.qpos[joint_id])
            joint_qpos = joint_qpos % (2 * np.pi)
            if joint_qpos < 0:
                joint_qpos += 2 * np.pi

            if 0.35 <= np.abs(joint_qpos) <= 2 * np.pi - 0.35:
                env.sim.model.site_rgba[site_id][3] = 0.5
            else:
                env.sim.model.site_rgba[site_id][3] = 0.0

    def set_knob_state(self, env, rng, knob, mode="on"):
        assert mode in ["on", "off"]
        if mode == "off":
            joint_val = 0.0
        else:
            if np.random.uniform() < 0.5:
                joint_val = rng.uniform(0.50, np.pi / 2)
            else:
                joint_val = rng.uniform(2 * np.pi - np.pi / 2, 2 * np.pi - 0.50)

        env.sim.data.set_joint_qpos(
            "{}knob_{}_joint".format(self.naming_prefix, knob), joint_val
        )

    def get_knobs_state(self, env):
        knobs_state = {}
        for location in STOVE_LOCATIONS:
            joint = self.knob_joints[location]
            if joint is None:
                continue
            site = self.burner_sites[location]
            if site is None:
                continue

            joint_id = env.sim.model.joint_name2id(
                "{}knob_{}_joint".format(self.naming_prefix, location)
            )

            joint_qpos = deepcopy(env.sim.data.qpos[joint_id])
            joint_qpos = joint_qpos % (2 * np.pi)
            if joint_qpos < 0:
                joint_qpos += 2 * np.pi

            knobs_state[location] = joint_qpos
        return knobs_state

    @property
    def knob_joints(self):
        if self._knob_joints is None:
            self._knob_joints = {}
            for location in STOVE_LOCATIONS:
                joint = self.worldbody.find(
                    "./body/body/body/joint[@name='{}knob_{}_joint']".format(
                        self.naming_prefix, location
                    )
                )
                self._knob_joints[location] = joint

        return self._knob_joints

    @property
    def burner_sites(self):
        if self._burner_sites is None:
            self._burner_sites = {}
            for location in STOVE_LOCATIONS:
                site = self.worldbody.find(
                    "./body/body/site[@name='{}burner_on_{}']".format(
                        self.naming_prefix, location
                    )
                )
                self._burner_sites[location] = site

        return self._burner_sites

    @property
    def nat_lang(self):
        return "stove"


class Stovetop(Stove):
    def __init__(self, xml="fixtures/stoves/stove_orig", name="stove", *args, **kwargs):
        super().__init__(xml=xml, name=name, *args, **kwargs)


class Oven(Fixture):
    def __init__(self, xml="fixtures/ovens/samsung", name="oven", *args, **kwargs):
        super().__init__(
            xml=xml, name=name, duplicate_collision_geoms=False, *args, **kwargs
        )

    @property
    def nat_lang(self):
        return "oven"
