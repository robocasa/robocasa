import numpy as np
from robosuite.utils import transform_utils as T

import robocasa.utils.object_utils as OU
from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    STAGE_TILT,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# A bowl holding two leftover items sits on the counter. Pick up the bowl, tilt it to dump
# the leftovers out (onto the counter), then place the empty bowl in the sink to be washed.
# The bowl is held throughout -> one continuous grasp split geometrically into Pick (grab +
# lift) -> Tilt (tilt-pour the leftovers out) -> Place (carry to the sink + set down). No
# bank atomic fits a bare pick / tilt-pour / counter->sink place, so Pick/Tilt/Place are
# bank-extension skills (like PanTransfer). Doc-canonical count = 3.
PICK_SUBTASK = "Pick"
TILT_SUBTASK = "Tilt"
PLACE_SUBTASK = "Place"

PICK_LANG = "pick up the bowl from the counter"
PLACE_LANG = "place the bowl in the sink"
DONE_LANG = "task complete"

LEFTOVERS = ("leftover1", "leftover2")


@register_labeler
class DumpLeftoversLabeler(BaseSubtaskLabeler):
    """Labeler for the DumpLeftovers composite task.

    A bowl with two leftovers is on the counter; dump the leftovers out of the bowl, then put
    the bowl in the sink. The bowl is grasped the whole time: Pick (grab + lift) -> Tilt
    (rotate the bowl far from rest to pour the leftovers out) -> Place (carry to the sink +
    set down + release). Pick->Tilt = first sustained lift-off; Tilt->Place = when BOTH
    leftovers have left the bowl; Place ends at the bowl's release.
    """

    ENV_NAME = "DumpLeftovers"

    BOWL = "bowl"
    LIFT_TH = 0.05
    POUR_MIN_TILT = 25.0      # the bowl must be tilted at least this much when it empties
    TILT_ONSET_TH = 15.0      # Pick->Tilt = where the final tilt ramp crosses up through this
    ONSET_RUN = 3             # consecutive low-tilt frames that mark the ramp's foot
    EMPTY_W = 18              # window for the "stays empty" test
    EMPTY_MAX_NOISE = 4       # tolerated in-bowl re-reads inside that window
    MOTION_TH = 0.004
    RELEASE_PAD = 8
    REL_FRAC = 0.5
    REST_W = 10

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.bowl_bid = env.obj_body_id[self.BOWL]
        self.leftovers = [o for o in LEFTOVERS if o in env.obj_body_id]
        self.lo_bid = {o: env.obj_body_id[o] for o in self.leftovers}
        self.bowl_geoms = env.objects[self.BOWL].contact_geoms
        self.gripper = env.robots[0].gripper["right"]
        self.fadr = [env.sim.model.get_joint_qpos_addr(j) for j in self.gripper.joints]
        langs = []
        for o in self.leftovers:
            try:
                langs.append(env.get_obj_lang(o))
            except Exception:
                langs.append("leftover")
        if len(langs) == 2 and langs[0] == langs[1]:
            self.leftovers_text = f"{langs[0]}s"
        else:
            self.leftovers_text = " and ".join(langs) if langs else "leftovers"
        self.extra_subtasks = {PICK_SUBTASK, TILT_SUBTASK, PLACE_SUBTASK}
        self.min_segment_frames = 12
        self.expected_done_idx = None

    def per_step(self, env, t: int) -> dict:
        q = np.array(env.sim.data.body_xquat[self.bowl_bid])  # wxyz
        bowl_up = T.quat2mat(q[[1, 2, 3, 0]])[:, 2].copy()
        in_bowl = 0
        for o in self.leftovers:
            try:
                in_bowl += int(OU.check_obj_in_receptacle(env, o, self.BOWL))
            except Exception:
                pass
        return {
            "bowl_pos": env.sim.data.body_xpos[self.bowl_bid].copy(),
            "bowl_up": bowl_up,
            "in_bowl": in_bowl,  # how many leftovers are still in the bowl
            "fw": float(sum(abs(env.sim.data.qpos[a]) for a in self.fadr)),
            "grasp": int(env._check_grasp(gripper=self.gripper, object_geoms=self.bowl_geoms)),
        }

    def _fw_open(self, fw, fw_open, grasp_t, hi):
        seg = fw[grasp_t:hi]
        if seg.size < 4:
            return None
        held = float(np.min(seg[: min(25, seg.size)]))
        level = held + self.REL_FRAC * (fw_open - held)
        above = seg > level
        for t in range(1, seg.size - 2):
            if above[t] and above[t + 1] and above[t + 2] and not above[t - 1]:
                return grasp_t + t
        return None

    def _release(self, grasp_arr, obj, fw, fw_open, grasp_t, hi):
        spd = np.linalg.norm(np.diff(obj, axis=0), axis=1)
        mv = np.where(spd[grasp_t:hi] > self.MOTION_TH)[0]
        setdown = int(grasp_t + int(mv[-1]) + 2) if mv.size else int(grasp_t + 1)
        fwo = self._fw_open(fw, fw_open, grasp_t, hi)
        gc = np.where(grasp_arr[grasp_t:hi] > 0)[0]
        if gc.size:
            rel = grasp_t + int(gc[-1]) + self.RELEASE_PAD
            if fwo is not None:
                rel = max(rel, fwo)
        elif fwo is not None:
            rel = fwo
        else:
            rel = setdown
        return int(min(max(rel, grasp_t + 1), hi))

    def _pour_complete(self, in_bowl, tilt):
        """First frame the bowl stays (mostly) empty WHILE actually tilted.

        in_bowl is noisy (transient false-empties as items shift, and false
        re-reads near the end once the bowl is set down), so a frame only counts
        when it reads empty, the bowl is tilted past POUR_MIN_TILT (so a
        carry-phase dropout can't qualify), and the next EMPTY_W frames hold at
        most EMPTY_MAX_NOISE in-bowl re-reads.
        """
        n = len(in_bowl)
        for t in range(n):
            if in_bowl[t] != 0 or tilt[t] <= self.POUR_MIN_TILT:
                continue
            hi = min(t + self.EMPTY_W, n)
            if int(np.sum(in_bowl[t:hi] > 0)) <= self.EMPTY_MAX_NOISE:
                return t
        return None

    def _tilt_onset(self, tilt, pour_complete, floor=3):
        """Pick -> Tilt = foot of the final tilt ramp before the pour.

        Walk back from pour_complete to the nearest run of ONSET_RUN consecutive
        low-tilt frames; its top (closest to the pour) is where the ramp begins.
        This skips an earlier lift-time tilt bump (demo_44) because the bowl goes
        back below TILT_ONSET_TH between that bump and the real pour.
        """
        run = 0
        for t in range(int(pour_complete), floor - 1, -1):
            if tilt[t] < self.TILT_ONSET_TH:
                run += 1
                if run >= self.ONSET_RUN:
                    return t + self.ONSET_RUN - 1
            else:
                run = 0
        return floor

    def resolve(self, history):
        T_ = len(history)
        bowl_pos = np.stack([h["bowl_pos"] for h in history])
        bowl_up = np.stack([h["bowl_up"] for h in history])
        grasp = np.array([h["grasp"] for h in history])
        fw = np.array([h["fw"] for h in history])
        fw_open = float(fw.max())
        in_bowl = np.array([h["in_bowl"] for h in history])

        z0 = float(np.median(bowl_pos[: self.REST_W, 2]))
        tilt = np.degrees(np.arccos(np.clip(bowl_up @ bowl_up[0], -1.0, 1.0)))
        lifted = (bowl_pos[:, 2] - z0) > self.LIFT_TH
        lift_off = next((t for t in range(T_ - 2)
                         if lifted[t] and lifted[t + 1] and lifted[t + 2]), 0)

        # Tilt -> Place = the pour is complete (leftovers have left the bowl and
        # stay out) -- robust to the noisy in-bowl signal; may precede OR follow
        # the lift, since the robot sometimes pours low over the counter first.
        pc = self._pour_complete(in_bowl, tilt)
        pour_detected = pc is not None
        pour_end = int(min(max(pc, 1), T_ - 1)) if pour_detected else min(lift_off + 1, T_ - 1)

        # Pick -> Tilt = foot of the tilt ramp leading into that pour.
        tilt_onset = int(min(self._tilt_onset(tilt, pour_end), max(pour_end - 1, 1)))

        release = min(self._release(grasp, bowl_pos, fw, fw_open, max(pour_end, 1), T_), T_ - 1)

        segments = [
            (tilt_onset, PICK_SUBTASK, STAGE_PICK, PICK_LANG),
            (pour_end, TILT_SUBTASK, STAGE_TILT, f"tilt the bowl to dump out the {self.leftovers_text}"),
            (release, PLACE_SUBTASK, STAGE_PLACE, PLACE_LANG),
            (T_, DONE, STAGE_DONE, DONE_LANG),
        ]
        self.expected_done_idx = 3

        review = []
        if not pour_detected:
            review.append("leftovers never registered out of the bowl (dump not detected)")
        result = self.build_result(T_, segments, debug={
            "lift_off": lift_off, "tilt_onset": tilt_onset, "pour_end": pour_end,
            "release": release, "max_tilt": round(float(tilt.max()), 1),
            "pour_detected": pour_detected, "n_segments": len(segments)})
        result.review = review
        return result

    def validate(self, result):
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        names = set(str(v) for v in result.subtask_name[firsts])
        for req in (PICK_SUBTASK, TILT_SUBTASK, PLACE_SUBTASK):
            if req not in names:
                issues.append(f"missing {req} subtask")
        dbg = getattr(result, "debug", None) or {}
        if not dbg.get("pour_detected", True):
            issues.append("leftovers never registered out of the bowl (dump not detected)")
        return issues
