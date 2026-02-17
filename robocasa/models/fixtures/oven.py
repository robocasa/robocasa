from robocasa.models.fixtures import Fixture
import numpy as np
import re


class Oven(Fixture):
    """
    Oven fixture class

    Args:
        xml (str): path to mjcf xml file
        name (str): name of the object
    """

    def __init__(self, xml="fixtures/ovens/Oven048", name="oven", *args, **kwargs):
        super().__init__(
            xml=xml, name=name, duplicate_collision_geoms=False, *args, **kwargs
        )
        self._door = 0.0
        self._timer = 0.0
        self._temperature = 0.0
        self._rack = {}

        self._joint_names = {
            "door": f"{self.naming_prefix}door_joint",
            "timer": f"{self.naming_prefix}knob_time_joint",
            "temperature": f"{self.naming_prefix}knob_temp_joint",
            "rack": f"{self.naming_prefix}rack0_joint",
        }

    def _get_joint_prefix(self):
        return f"{self.naming_prefix}"

    def get_reset_region_names(self):
        return ("rack0", "rack1")

    def get_reset_regions(self, env=None, rack_level=0, z_range=None):
        rack_regions = []

        for key, reg in self._regions.items():
            m = re.fullmatch(r"rack(\d+)", key)
            if not m:
                continue
            idx = int(m.group(1))
            p0, px, py, pz = reg["p0"], reg["px"], reg["py"], reg["pz"]
            hx, hy, hz = (px[0] - p0[0]) / 2, (py[1] - p0[1]) / 2, (pz[2] - p0[2]) / 2
            center = p0 + np.array([hx, hy, hz])
            entry = (idx, key, p0, px, py, pz)
            rack_regions.append(entry)

        rack_regions.sort(key=lambda x: x[0])
        region = (
            rack_regions[1]
            if rack_level == 1 and len(rack_regions) > 1
            else rack_regions[0]
            if rack_regions
            else None
        )

        if region:
            level = region[0]
            self._joint_names["rack"] = f"{self._get_joint_prefix()}rack{level}_joint"
            self._rack[f"rack{level}"] = 0.0
        else:
            raise ValueError(f"No rack reset regions found for rack_level {rack_level}")

        idx, key, p0, px, py, pz = region
        offset = (
            float(np.mean((p0[0], px[0]))),
            float(np.mean((p0[1], py[1]))),
            float(p0[2]),
        )
        size = (float(px[0] - p0[0]), float(py[1] - p0[1]))
        height = float(pz[2] - p0[2])
        return {key: {"offset": offset, "size": size, "height": height}}

    def open_door(self, env):
        self._door = 1.0
        self.set_joint_state(
            min=1.0, max=1.0, env=env, joint_names=[self._joint_names["door"]]
        )

    def close_door(self, env):
        self._door = 0.0
        self.set_joint_state(
            min=0.0, max=0.0, env=env, joint_names=[self._joint_names["door"]]
        )

    def set_timer(self, env, value):
        self._timer = np.clip(value, 0.0, 1.0)
        self.set_joint_state(
            min=self._timer,
            max=self._timer,
            env=env,
            joint_names=[self._joint_names["timer"]],
        )

    def set_temperature(self, env, value):
        self._temperature = np.clip(value, 0.0, 1.0)
        self.set_joint_state(
            min=self._temperature,
            max=self._temperature,
            env=env,
            joint_names=[self._joint_names["temperature"]],
        )

    def has_multiple_rack_levels(self):
        """
        Returns True if the oven has multiple rack levels, False if only one.
        """
        rack_levels = set()
        for key in self._regions:
            m = re.fullmatch(r"rack(\d+)", key)
            if m:
                idx = int(m.group(1))
                rack_levels.add(idx)
        return len(rack_levels) > 1

    def slide_rack(self, env, value=1.0, rack_level=0):
        joint_prefix = self._get_joint_prefix()

        for level in [rack_level, 0]:
            joint = f"{joint_prefix}rack{level}_joint"
            if joint in env.sim.model.joint_names:
                name = f"rack{level}"
                self._rack[name] = value
                self.set_joint_state(min=value, max=value, env=env, joint_names=[joint])
                return

        raise ValueError(
            f"No rack found for level {rack_level}, and fallback to 0 also failed."
        )

    def update_state(self, env):
        for name in ["door", "timer", "temperature"]:
            jn = self._joint_names[name]
            if jn in env.sim.model.joint_names:
                val = self.get_joint_state(env, [jn])[jn]
                setattr(self, f"_{name}", val)

        self._rack.clear()
        for joint_name in env.sim.model.joint_names:
            if "rack" in joint_name and joint_name.startswith(self.naming_prefix):
                val = self.get_joint_state(env, [joint_name])[joint_name]
                key = joint_name.replace("_joint", "")
                self._rack[key] = val
                if key.endswith("rack0"):
                    self._rack["rack0"] = val
                elif key.endswith("rack1"):
                    self._rack["rack1"] = val

    def check_rack_contact(self, env, object_name, rack_level=0):
        """
        Checks whether the specified object is in contact with the oven rack at the given level.

        Args:
            env: The simulation environment.
            object_name (str): Name of the object to check for contact.
            rack_level (int): Which rack level to check (default is 0).

        Returns:
            bool: True if contact exists between object and rack, False otherwise.
        """
        joint_prefix = self._get_joint_prefix()
        for level in [rack_level, 0]:
            joint = f"{joint_prefix}rack{level}_joint"
            if joint in env.sim.model.joint_names:
                contact_name = joint.replace("_joint", "")
                break
        else:
            raise RuntimeError(f"No rack found for level {rack_level}")

        body_id = env.sim.model.body_name2id(contact_name)
        shelf_geoms = [
            env.sim.model.geom_id2name(i)
            for i in range(env.sim.model.ngeom)
            if env.sim.model.geom_bodyid[i] == body_id
        ]

        obj_body = env.obj_body_id[object_name]
        obj_geoms = [
            env.sim.model.geom_id2name(gid)
            for gid in range(env.sim.model.ngeom)
            if env.sim.model.geom_bodyid[gid] == obj_body
        ]

        return env.check_contact(shelf_geoms, obj_geoms)

    def get_state(self, rack_level=0):
        state = {}
        for key in [f"rack{rack_level}", "rack0"]:
            if key in self._rack:
                state[key] = self._rack[key]
                break
        else:
            raise ValueError(f"No rack state available for level {rack_level}")

        state["door"] = self._door
        state["timer"] = self._timer
        state["temperature"] = self._temperature
        return state

    @property
    def nat_lang(self):
        return "oven"
