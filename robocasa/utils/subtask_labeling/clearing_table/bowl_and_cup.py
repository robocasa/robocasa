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


# First drop the cup into the bowl on the dining table (a local pick-place), then
# carry the bowl (now holding the cup) to a counter (a nav-interrupted pick-place).
# Neither move has an exact bank name (cup->bowl-object, dining-counter->counter), so
# both are extra_subtasks; NavigateKitchen is a bank atomic.
SUBTASK_CUP = "PickPlaceCounterToBowl"     # cup: dining table -> into the bowl
SUBTASK_BOWL = "PickPlaceCounterToCounter"  # bowl (with cup): dining table -> counter
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"


@register_labeler
class BowlAndCupLabeler(BaseSubtaskLabeler):
    """Labeler for the BowlAndCup composite task.

    Task: a cup and a bowl sit on the dining table (robot starts there). Put the cup
    inside the bowl, then carry the bowl (holding the cup) to any counter.

    Doc-canonical count = 4 (`mobile: Yes`): the cup's local pick-place into the bowl
    (no nav -- both on the dining table) + the bowl's pick-place, INTERRUPTED by a
    navigation to the counter. The ORDER is fixed (the cup must go in the bowl before
    the bowl moves): `cup[pick] → [place in bowl] → bowl[pick] → NavigateKitchen →
    [place on counter]`. 5 segments without the nav (expected_done_idx = 4), 6 with.
    The nav is ADAPTIVE (gated on MIN_NAV_DIST).

    Boundaries:
      - cup (first-move split): t_move = cup displaces from its dining-table rest;
        t_placed = the gripper RELEASE (dropped in the bowl).
      - bowl: rest re-anchored to its pose right after the cup is dropped (the cup
        drop can jostle it). t_move = bowl displaces; the nav is the carry, split at
        the base stop; the place ends at the gripper RELEASE.
    """

    ENV_NAME = "BowlAndCup"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    BASE_MOVE_TH = 0.003
    MIN_NAV_DIST = 0.20
    REST_W = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.cup_body_id = env.obj_body_id["cup"]
        self.bowl_body_id = env.obj_body_id["bowl"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.cup_lang = env.get_obj_lang("cup")
        except Exception:
            self.cup_lang = "cup"
        try:
            self.bowl_lang = env.get_obj_lang("bowl")
        except Exception:
            self.bowl_lang = "bowl"

        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])

        self.extra_subtasks = {SUBTASK_CUP, SUBTASK_BOWL}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        return {
            "cup": env.sim.data.body_xpos[self.cup_body_id].copy(),
            "bowl": env.sim.data.body_xpos[self.bowl_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
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

    def _release(self, opos, grip, t_move, T):
        gd = np.linalg.norm(grip - opos, axis=1)
        hold_gd = float(np.median(gd[t_move: min(t_move + self.HOLD_W, T)]))
        t_rel = self._sustained_start(
            gd > hold_gd + self.RELEASE_MARGIN, t_move + self.MIN_CARRY, self.RELEASE_SUSTAIN
        )
        return T - 1 if t_rel is None else t_rel

    def resolve(self, history):
        T = len(history)
        cup = np.array([h["cup"] for h in history])
        bowl = np.array([h["bowl"] for h in history])
        grip = np.array([h["grip"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        # cup: dining table -> into the bowl
        crest = np.median(cup[: self.REST_W], axis=0)
        cdisp = np.linalg.norm(cup - crest, axis=1)
        t_cmove = self._sustained_start(cdisp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
        if t_cmove is None:
            mv = np.where(cdisp >= self.MOVE_TOL)[0]
            t_cmove = int(mv[0]) if mv.size else max(T - 1, 0)
        t_cplaced = self._release(cup, grip, t_cmove, T)

        # bowl: re-anchor its rest to just after the cup is dropped (the drop can
        # jostle it), then it is grasped and carried to the counter.
        w0 = min(t_cplaced, max(T - self.REST_W, 0))
        brest = np.median(bowl[w0: w0 + self.REST_W], axis=0)
        bdisp = np.linalg.norm(bowl - brest, axis=1)
        t_bmove = self._sustained_start(bdisp >= self.MOVE_TOL, t_cplaced, self.MOVE_SUSTAIN)
        if t_bmove is None:
            mv = np.where(bdisp[t_cplaced:] >= self.MOVE_TOL)[0]
            t_bmove = t_cplaced + int(mv[0]) if mv.size else max(t_cplaced, T - 1)
        t_bmove = max(t_bmove, t_cplaced)
        t_bplaced = self._release(bowl, grip, t_bmove, T)

        moving = np.where(binst[t_bmove: t_bplaced] > self.BASE_MOVE_TH)[0]
        t_arrive = min(t_bmove + int(moving[-1]) + 1, t_bplaced) if moving.size else t_bmove
        if base_xy is not None and t_arrive > t_bmove:
            nav_disp = float(np.max(np.linalg.norm(base_xy[t_bmove: t_arrive] - base_xy[t_bmove], axis=1)))
        else:
            nav_disp = 0.0

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        emit(t_cmove, SUBTASK_CUP, STAGE_PICK, f"pick up the {self.cup_lang} from the dining table")
        emit(t_cplaced, SUBTASK_CUP, STAGE_PLACE, f"place the {self.cup_lang} in the {self.bowl_lang}")
        emit(t_bmove, SUBTASK_BOWL, STAGE_PICK, f"pick up the {self.bowl_lang} from the dining table")
        if t_arrive > cursor and nav_disp >= self.MIN_NAV_DIST:
            emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE,
                 f"carry the {self.bowl_lang} to the counter next to the sink")
        emit(t_bplaced, SUBTASK_BOWL, STAGE_PLACE, f"place the {self.bowl_lang} on the counter")
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        debug = {
            "t_cmove": t_cmove, "t_cplaced": t_cplaced, "t_bmove": t_bmove,
            "t_arrive": t_arrive, "t_bplaced": t_bplaced,
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

        for nm, label in ((SUBTASK_CUP, "cup"), (SUBTASK_BOWL, "bowl")):
            if count(nm, STAGE_PICK) != 1:
                issues.append(f"expected 1 {label} pick, found {count(nm, STAGE_PICK)}")
            if count(nm, STAGE_PLACE) != 1:
                issues.append(f"expected 1 {label} place, found {count(nm, STAGE_PLACE)}")
        if count(SUBTASK_NAV) not in (0, 1):
            issues.append(f"expected 0-1 NavigateKitchen, found {count(SUBTASK_NAV)}")
        return issues
