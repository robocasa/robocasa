import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_EXECUTE,
    STAGE_NAVIGATE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Pick the vegetable out of the fridge, carry it to the running sink, hold it under the
# water to rinse it, then set it on the tray to prepare for roasting. Fridge -> tray and the
# rinse have no bank names (extra_subtasks); NavigateKitchen is a bank atomic. == WashFish.
SUBTASK_MOVE = "PickPlaceFridgeToTray"
SUBTASK_RINSE = "RinseVegetable"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"


@register_labeler
class PrepareVegetableRoastingLabeler(BaseSubtaskLabeler):
    """Labeler for the PrepareVegetableRoasting composite task.

    Task (mobile): a vegetable is in the fridge and the sink faucet is running. Pick it
    up, carry it to the sink, hold it under the running water to rinse it, then set it on the
    tray. Doc-canonical count = 4: PickPlaceFridgeToTray[pick] -> NavigateKitchen ->
    RinseVegetable -> [place] (the short sink->dish-rack hop folds into the place; the nav is
    adaptive).

    Boundaries (same as WashFish):
      - t_carry: the teapot first displaces MOVE_TOL from its counter rest -> pick ends.
      - t_sink:  the base stops driving before the rinse -> nav ends, rinse begins.
      - t_rinse_end: the teapot is lifted back out after the last under-water frame.
      - t_placed: the gripper releases it into the dish rack.
    """

    ENV_NAME = "PrepareVegetableRoasting"

    MOVE_TOL = 0.12
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    BASE_MOVE_TH = 0.003
    MIN_NAV_DIST = 0.20
    LIFT_TOL = 0.12
    REST_W = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_id = env.obj_body_id["vegetable"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.sink = getattr(env, "sink", None)
        try:
            self.obj_lang = env.get_obj_lang("vegetable")
        except Exception:
            self.obj_lang = "vegetable"
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[env.sim.model.joint_name2id(nm)])
            except Exception:
                pass
        self.extra_subtasks = {SUBTASK_MOVE, SUBTASK_RINSE}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        try:
            uw = 1.0 if self.sink.check_obj_under_water(env, "vegetable") else 0.0
        except Exception:
            uw = 0.0
        return {
            "obj": env.sim.data.body_xpos[self.obj_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "uw": uw,
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

    def resolve(self, history):
        T = len(history)
        obj = np.array([h["obj"] for h in history])
        grip = np.array([h["grip"] for h in history])
        uw = np.array([h["uw"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        rest = np.median(obj[: self.REST_W], axis=0)
        disp = np.linalg.norm(obj - rest, axis=1)
        t_carry = self._sustained_start(disp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
        if t_carry is None:
            mv = np.where(disp >= self.MOVE_TOL)[0]
            t_carry = int(mv[0]) if mv.size else 0

        gd = np.linalg.norm(grip - obj, axis=1)
        hold_gd = float(np.median(gd[t_carry: min(t_carry + self.HOLD_W, T)]))
        t_placed = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN,
                                         t_carry + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_placed is None:
            t_placed = T - 1

        uw_idx = np.where(uw > 0.5)[0]
        uw_idx = uw_idx[(uw_idx >= t_carry) & (uw_idx < t_placed)] if uw_idx.size else uw_idx
        has_rinse = uw_idx.size > 0
        if has_rinse:
            first_uw, last_uw = int(uw_idx[0]), int(uw_idx[-1])
            moving = np.where(binst[t_carry: first_uw] > self.BASE_MOVE_TH)[0]
            t_sink = t_carry + int(moving[-1]) + 1 if moving.size else t_carry
            t_sink = int(min(max(t_sink, t_carry), first_uw))
            fz = obj[:, 2]
            dip = float(np.min(fz[t_sink: last_uw + 1])) if last_uw >= t_sink else float(fz[last_uw])
            t_rinse_end = self._sustained_start(fz > dip + self.LIFT_TOL, last_uw, 3)
            if t_rinse_end is None:
                t_rinse_end = last_uw
            t_rinse_end = int(min(max(t_rinse_end, t_sink), t_placed))
        else:
            moving = np.where(binst[t_carry: t_placed] > self.BASE_MOVE_TH)[0]
            t_sink = min(t_carry + int(moving[-1]) + 1, t_placed) if moving.size else t_carry
            t_rinse_end = t_sink

        if base_xy is not None and t_sink > t_carry:
            off = base_xy[t_carry: t_sink] - base_xy[t_carry]
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

        w = self.obj_lang
        emit(t_carry, SUBTASK_MOVE, STAGE_PICK, f"pick up the {w} from the fridge")
        if t_sink > cursor and nav_disp >= self.MIN_NAV_DIST:
            emit(t_sink, SUBTASK_NAV, STAGE_NAVIGATE, f"carry the {w} to the sink")
        if has_rinse:
            emit(t_rinse_end, SUBTASK_RINSE, STAGE_EXECUTE, f"rinse the {w} under the running water")
        emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place the {w} on the tray")
        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        review = []
        if not has_rinse:
            review.append("teapot never registered under running water -- check the rinse")
        result = self.build_result(T, segments, debug={
            "t_carry": t_carry, "t_sink": t_sink, "t_rinse_end": t_rinse_end,
            "t_placed": t_placed, "nav_disp": round(nav_disp, 2),
            "has_rinse": has_rinse, "n_segments": len(segments)})
        result.review = review
        return result

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
            issues.append("expected 1 teapot pick")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 1:
            issues.append("expected 1 teapot place")
        if count(SUBTASK_RINSE) != 1:
            issues.append(f"expected 1 RinseVegetable, found {count(SUBTASK_RINSE)}")
        if count(SUBTASK_NAV) not in (0, 1):
            issues.append("expected 0-1 NavigateKitchen")
        return issues
