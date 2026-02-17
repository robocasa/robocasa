from robocasa.models.fixtures import Fixture
import numpy as np


class StandMixer(Fixture):
    """
    Stand mixer fixture class
    """

    def __init__(
        self,
        xml="fixtures/stand_mixers/StandMixer003",
        name="stand_mixer",
        mirror_placement=False,
        *args,
        **kwargs,
    ):
        super().__init__(
            xml=xml, name=name, duplicate_collision_geoms=False, *args, **kwargs
        )
        self.mirror_placement = mirror_placement

        self._button_head_lock = False
        self._speed_dial_knob_value = 0.0
        self._head_value = 0.0

        # Joint name mapping
        prefix = self._get_joint_prefix()
        self._joint_names = {
            "button_head_lock": f"{prefix}button_head_lock_joint",
            "bowl": f"{prefix}bowl_joint",
            "knob_speed": f"{prefix}knob_speed_joint",
            "head": f"{prefix}head_joint",
        }

    def _get_joint_prefix(self):
        return f"{self.naming_prefix}"

    def get_reset_region_names(self):
        return ("bowl",)

    def set_speed_dial_knob(self, env, knob_val):
        """
        Sets the speed of the stand mixer

        Args:
            knob_val (float): normalized value between 0 and 1 (max speed)
        """
        self._speed_dial_knob_value = np.clip(knob_val, 0.0, 1.0)
        jn = self._joint_names["knob_speed"]
        self.set_joint_state(
            env=env,
            min=self._speed_dial_knob_value,
            max=self._speed_dial_knob_value,
            joint_names=[jn],
        )

    def set_head_pos(self, env, head_val=1.0):
        """
        Sets the position of the head

        Args:
            head_val (float): normalized value between 0 (closed) and 1 (open)
        """
        self._head_value = np.clip(head_val, 0.0, 1.0)
        jn = self._joint_names["head"]

        self.set_joint_state(
            env=env,
            min=self._head_value,
            max=self._head_value,
            joint_names=[jn],
        )

    def update_state(self, env):
        """
        Update the state of the stand mixer
        """
        # read power button
        btn = self._joint_names["button_head_lock"]
        if btn in env.sim.model.joint_names:
            pos = self.get_joint_state(env, [btn])[btn]
            if pos > 0.75:
                self._button_head_lock = True

        # sync positions back into internal values
        mapping = {
            "_speed_dial_knob_value": "knob_speed",
            "_head_value": "head",
        }
        for attr, key in mapping.items():
            jn = self._joint_names[key]
            if jn in env.sim.model.joint_names:
                setattr(self, attr, self.get_joint_state(env, [jn])[jn])

    def check_item_in_bowl(self, env, obj_name, partial_check=False):
        """
        Check if an object is in the bowl of the stand mixer.
        """
        obj = env.objects[obj_name]
        sites = self.get_int_sites(relative=False)
        if partial_check:
            pts = [env.sim.data.body_xpos[env.obj_body_id[obj.name]].copy()]
            tol = 0.0
        else:
            pos = env.sim.data.body_xpos[env.obj_body_id[obj.name]]
            quat = env.sim.data.body_xquat[env.obj_body_id[obj.name]]
            pts = obj.get_bbox_points(trans=pos, rot=quat)
            tol = 1e-2

        for (_, (p0, px, py, pz)) in sites.items():
            u, v, w = px - p0, py - p0, pz - p0
            mid = p0 + 0.5 * (pz - p0)
            all_in = True
            for q in pts:
                cu, cv, cw = np.dot(u, q), np.dot(v, q), np.dot(w, q)
                if not (
                    (np.dot(u, p0) - tol <= cu <= np.dot(u, px) + tol)
                    and (np.dot(v, p0) - tol <= cv <= np.dot(v, py) + tol)
                    and (np.dot(w, p0) - tol <= cw <= np.dot(w, mid) + tol)
                ):
                    all_in = False
                    break
            if all_in:
                return True
        return False

    def get_state(self, env):
        """
        Returns a dictionary representing the state of the stand mixer.
        """
        st = {}
        for name, jn in self._joint_names.items():
            if jn in env.sim.model.joint_names:
                st[name] = getattr(self, f"_{name}_value", None)
        return st

    @property
    def nat_lang(self):
        return "stand mixer"
