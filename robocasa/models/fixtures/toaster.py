from robocasa.models.fixtures import Fixture
import numpy as np
from robosuite.utils.mjcf_utils import find_elements


class Toaster(Fixture):
    """
    Toaster fixture class
    """

    def __init__(self, xml, name="toaster", *args, **kwargs):
        super().__init__(
            xml=xml, name=name, duplicate_collision_geoms=False, *args, **kwargs
        )

        floor_names = [
            g.get("name")
            for g in self.worldbody.iter("geom")
            if g.get("name", "").endswith("_floor")
        ]

        prefix = self._get_joint_prefix()
        self._slots = []
        for full_name in floor_names:
            slot = full_name
            if slot.startswith(prefix):
                slot = slot[len(prefix) :]
            if slot.endswith("_floor"):
                slot = slot[: -len("_floor")]
            self._slots.append(slot)

        self._slot_pairs = list(range(int(len(self._slots) / 2)))

        self._floor_geoms = {}
        for slot_name in self._slots:
            geom = find_elements(
                self.worldbody,
                "geom",
                {"name": f"{self.naming_prefix}{slot_name}_floor"},
            )
            if geom is not None:
                self._floor_geoms[slot_name] = geom

        self._controls = {
            "button": "button_cancel",
            "knob": "knob_doneness",
            "lever": "lever",
        }

        self._joint_names = {}
        for ctrl, tag in self._controls.items():
            names = sorted(
                j.get("name")
                for j in self.worldbody.iter("joint")
                if tag in j.get("name", "")
            )
            for pair, jn in zip(self._slot_pairs, names):
                self._joint_names[f"{ctrl}_{pair}"] = jn

        self._state = {s: {c: 0.0 for c in self._controls} for s in self._slot_pairs}
        self._turned_on = {s: False for s in self._slot_pairs}
        self._num_steps_on = {s: 0 for s in self._slot_pairs}
        self._cooldown = {s: 0 for s in self._slot_pairs}

    def _get_joint_prefix(self):
        return f"{self.naming_prefix}"

    def get_reset_region_names(self):
        return (
            "slotL",
            "slotR",
            "sideL_slotL",
            "sideL_slotR",
            "sideR_slotL",
            "sideR_slotR",
        )

    def get_reset_regions(self, env=None, slot_pair=None, side=None, **kwargs):
        """
        Returns the reset regions for the toaster slots.

        Args:
            env: environment instance
            slot_pair (int or None): 0 to N-1 slot pairs, or None for all pairs
            side (str or None): "left", "right", or None for both sides

        Returns:
            dict: reset regions for the specified slots
        """
        if slot_pair is not None and (
            not isinstance(slot_pair, int)
            or not (0 <= slot_pair < len(self._slot_pairs))
        ):
            raise ValueError(
                f"Invalid slot_pair {slot_pair!r}; must be None or an int in 0-{len(self._slot_pairs)-1}"
            )

        if side is not None and side not in ("left", "right"):
            raise ValueError(f"Invalid side {side!r}; must be None, 'left' or 'right'")

        all_regions = list(self.get_reset_region_names())

        if slot_pair is None:
            filtered_regions = all_regions
        elif slot_pair == 0:
            # Return left side regions
            filtered_regions = [
                r
                for r in all_regions
                if "sideL" in r or (r in ["slotL", "slotR"] and "side" not in r)
            ]
        elif slot_pair == 1:
            # Return right side regions
            filtered_regions = [r for r in all_regions if "sideR" in r]
        else:
            filtered_regions = []

        # Filter by side
        if side is not None:
            if side == "left":
                filtered_regions = [r for r in filtered_regions if "slotL" in r]
            elif side == "right":
                filtered_regions = [r for r in filtered_regions if "slotR" in r]

        return super().get_reset_regions(
            env=env, reset_region_names=filtered_regions, **kwargs
        )

    def set_doneness_knob(self, env, slot_pair, value):
        """
        Sets the toasting doneness knob

        Args:
            slot_pair (int): the slot pair to set the knob for (0 to N-1 slot pairs)
            value (float): normalized value between 0 (min) and 1 (max)
        """
        if slot_pair not in self._slot_pairs:
            raise ValueError(f"Unknown slot_pair '{slot_pair}'")
        val = np.clip(value, 0.0, 1.0)
        self._state[slot_pair]["knob"] = val

        jn = self._joint_names.get(f"knob_{slot_pair}")
        if jn and jn in env.sim.model.joint_names:
            self.set_joint_state(
                env=env,
                min=val,
                max=val,
                joint_names=[jn],
            )

    def set_lever(self, env, slot_pair, value):
        """
        Sets the power lever

        Args:
            slot_pair (int): the slot pair to set the lever for (0 to N-1 slot pairs)
            value (float): normalized value between 0 (off) and 1 (on)
        """
        if slot_pair not in self._slot_pairs:
            raise ValueError(f"Unknown slot_pair '{slot_pair}'")
        val = np.clip(value, 0.0, 1.0)
        self._state[slot_pair]["lever"] = val

        jn = self._joint_names.get(f"lever_{slot_pair}")
        if jn and jn in env.sim.model.joint_names:
            self.set_joint_state(
                env=env,
                min=val,
                max=val,
                joint_names=[jn],
            )

    def update_state(self, env):
        """
        Update the state of the toaster
        """
        for sp in self._slot_pairs:
            for control in self._controls:
                key = f"{control}_{sp}"
                jn = self._joint_names.get(key)
                if jn and jn in env.sim.model.joint_names:
                    q = self.get_joint_state(env, [jn])[jn]
                    self._state[sp][control] = np.clip(q, 0.0, 1.0)

        for sp in self._slot_pairs:
            lev_val = self._state[sp]["lever"]

            if lev_val <= 0.70:
                self._cooldown[sp] = 0

            if lev_val >= 0.90 and not self._turned_on[sp] and self._cooldown[sp] == 0:
                self._turned_on[sp] = True

            if self._turned_on[sp]:
                if self._num_steps_on[sp] < 500:
                    self._num_steps_on[sp] += 1
                    self.set_lever(env, sp, 1.0)
                else:
                    self._turned_on[sp] = False
                    self._num_steps_on[sp] = 0
                    self._cooldown[sp] = 1

            if 0 < self._cooldown[sp] < 1000:
                self._cooldown[sp] += 1
            elif self._cooldown[sp] >= 1000:
                self._cooldown[sp] = 0

    def check_slot_contact(self, env, obj_name, slot_pair=None, side=None):
        """
        Returns True if the specified object is in contact with any of the toaster slot-floor geom(s).

        Args:
            env (MujocoEnv): the environment
            obj_name (str): name of the object to check
            slot_pair (int or None): 0 to N-1 slot pairs
            side (str or None): None = both sides; otherwise “left” or “right”

        Returns:
            bool: whether any of the object’s geoms are touching any selected slot-floor geom
        """
        if slot_pair is not None and (
            not isinstance(slot_pair, int)
            or not (0 <= slot_pair < len(self._slot_pairs))
        ):
            raise ValueError(
                f"Invalid slot_pair {slot_pair!r}; must be None or an int in 0-{len(self._slot_pairs)-1}"
            )

        # pick slot side
        if side is None:
            sides = ["left", "right"]
        elif side in ("left", "right"):
            sides = [side]
        else:
            raise ValueError(f"Invalid side {side!r}; must be None, 'left' or 'right'")

        slot_floor_names = []
        for slot in self._slots:
            slot_floor_names.append(f"{self.naming_prefix}{slot}_floor")

        slot_tokens = []
        if "left" in sides:
            slot_tokens.append("slotL")
        if "right" in sides:
            slot_tokens.append("slotR")

        # pick slot pair side
        if set(slot_tokens) == {"slotL", "slotR"}:
            side_filtered = slot_floor_names.copy()
        else:
            side_filtered = [
                n for n in slot_floor_names if any(tok in n for tok in slot_tokens)
            ]

        n_pairs = len(self._slot_pairs)
        if n_pairs == 1:
            if slot_pair in (None, 0):
                final_floor_names = side_filtered
            else:
                raise ValueError("Only slot_pair=0 is valid for a single-pair toaster")
        elif n_pairs == 2:
            if slot_pair is None:
                final_floor_names = side_filtered
            elif slot_pair == 0:
                final_floor_names = [n for n in side_filtered if "sideL" in n]
            elif slot_pair == 1:
                final_floor_names = [n for n in side_filtered if "sideR" in n]
            else:
                raise ValueError(
                    "slot_pair must be None, 0, or 1 for a two-pair toaster"
                )

        # get item geom names
        item_body_id = env.obj_body_id[obj_name]
        item_geom_names = [
            env.sim.model.geom_id2name(gid)
            for gid in range(env.sim.model.ngeom)
            if env.sim.model.geom_bodyid[gid] == item_body_id
        ]

        # check contact
        for slot_floor_name in final_floor_names:
            if env.check_contact(slot_floor_name, item_geom_names):
                return True

        return False

    def get_state(self, env, slot_pair=None):
        """
        Returns the current state of the toaster as a dictionary.

        Args:
            slot_pair (int or None): 0 to N-1 slot pairs

        Returns:
            dict: the current state of the toaster
        """
        full = {
            s: {**self._state[s], "turned_on": self._turned_on[s]}
            for s in self._slot_pairs
        }

        if slot_pair is None:
            return full

        if isinstance(slot_pair, int):
            try:
                slot_pair = self._slot_pairs[slot_pair]
            except (IndexError, TypeError):
                raise ValueError(
                    f"Invalid slot index {slot_pair!r}, must be between "
                    f"0 and {len(self._slots)-1}"
                )

        if slot_pair not in full:
            raise ValueError(
                f"Invalid slot_pair {slot_pair!r}, must be one of {list(full)}"
            )

        return full[slot_pair]

    @property
    def nat_lang(self):
        return "toaster"
