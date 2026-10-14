import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_NAVIGATE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Take the blender jug off the counter and set it down in the sink. A single
# counter->sink pick-place, INTERRUPTED by a navigation from the counter to the
# sink. PickPlaceCounterToSink + NavigateKitchen are bank atomics -- no extras.
SUBTASK_MOVE = "PickPlaceCounterToSink"
SUBTASK_NAV = "NavigateKitchen"

NAV_LANG = "carry the blender jug to the sink"
DONE_LANG = "task complete"


@register_labeler
class CleanBlenderJugLabeler(BaseSubtaskLabeler):
    """Labeler for the CleanBlenderJug composite task.

    Task: a blender jug sits on the counter (where the robot starts). Carry it to
    the sink and set it down inside the sink (to prep it for washing).

    Doc-canonical count = 3 (`mobile: Yes`): the jug's pick-place, INTERRUPTED by a
    navigation -- PickPlaceCounterToSink[pick] -> NavigateKitchen -> [place]. 3
    subtasks + done (expected_done_idx = 3). The nav is ADAPTIVE (gated on
    MIN_NAV_DIST): counter and sink can be adjacent, so a few demos transfer the jug
    mostly with the arm and barely drive (2 segments).

    Boundaries (DeliverBrewedCoffee / base-stop convention):
      - t_pick = the jug first displaces MOVE_TOL from its rest (on-counter) pose.
      - the nav is the carry, ending at the base stop (last base-moving frame); the
        place is the set-down + gripper RELEASE after the base parks.
    """

    ENV_NAME = "CleanBlenderJug"

    MOVE_TOL = 0.10  # jug displaced this far from rest == grasped & being carried
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08  # grip->jug rising this far above carry level == let go
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    BASE_MOVE_TH = 0.003  # per-frame base speed above this == the base is driving
    MIN_NAV_DIST = 0.20  # peak base displacement over the carry to emit a nav
    REST_W = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.jug_body_id = env.obj_body_id["blender_jug"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.jug_lang = env.get_obj_lang("blender_jug")
        except Exception:
            self.jug_lang = "blender jug"

        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])

        self.extra_subtasks = set()
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        return {
            "jug": env.sim.data.body_xpos[self.jug_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array(
                [float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]
            ),
        }

    @staticmethod
    def _sustained_start(mask, start, sustain):
        run = 0
        for t in range(int(start), len(mask)):
            if mask[t]:
                run += 1
                if run >= sustain:
                    return t - sustain + 1
            else:
                run = 0
        return None

    def resolve(self, history):
        T = len(history)
        jug = np.array([h["jug"] for h in history])
        grip = np.array([h["grip"] for h in history])
        base_xy = (
            np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        )
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        rest = np.median(jug[: self.REST_W], axis=0)
        disp = np.linalg.norm(jug - rest, axis=1)
        t_pick = self._sustained_start(disp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
        if t_pick is None:
            mv = np.where(disp >= self.MOVE_TOL)[0]
            t_pick = int(mv[0]) if mv.size else max(T - 1, 0)

        gd = np.linalg.norm(grip - jug, axis=1)
        hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
        t_placed = self._sustained_start(
            gd > hold_gd + self.RELEASE_MARGIN, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN
        )
        if t_placed is None:
            t_placed = T - 1

        # nav/place split: the base drives (carry) then the arm sets the jug down
        moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
        t_arrive = min(t_pick + int(moving[-1]) + 1, t_placed) if moving.size else t_pick
        if base_xy is not None and t_arrive > t_pick:
            off = base_xy[t_pick: t_arrive] - base_xy[t_pick]
            nav_disp = float(np.max(np.linalg.norm(off, axis=1)))
        else:
            nav_disp = 0.0

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up the {self.jug_lang} from the counter")
        if t_arrive > cursor and nav_disp >= self.MIN_NAV_DIST:
            emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, NAV_LANG)
        emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place the {self.jug_lang} in the sink")
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        debug = {
            "t_pick": t_pick, "t_arrive": t_arrive, "t_placed": t_placed,
            "nav_disp": round(nav_disp, 2), "n_segments": len(segments),
        }
        return self.build_result(T, segments, debug=debug)

    def validate(self, result):
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        names = result.subtask_name[firsts]
        stages = result.subtask_stage[firsts]

        def count(name, stage=None):
            m = names == name
            if stage is not None:
                m = m & (stages == stage)
            return int(np.sum(m))

        if count(SUBTASK_MOVE, STAGE_PICK) != 1:
            issues.append(f"expected 1 jug pick, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 1:
            issues.append(f"expected 1 jug place, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        if count(SUBTASK_NAV) not in (0, 1):
            issues.append(f"expected 0-1 NavigateKitchen, found {count(SUBTASK_NAV)}")
        return issues
