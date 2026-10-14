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


# Take the fish out of the (already-open) fridge, carry it to the running sink, hold
# it under the water to rinse it, then set it on the tray next to the sink. The
# fridge->tray move and the rinse have no bank names (extra_subtasks); NavigateKitchen
# is a bank atomic.
SUBTASK_MOVE = "PickPlaceFridgeToTray"
SUBTASK_RINSE = "RinseFish"
SUBTASK_NAV = "NavigateKitchen"

NAV_LANG = "carry the fish to the sink"
RINSE_LANG = "rinse the fish in the sink"
DONE_LANG = "task complete"


@register_labeler
class WashFishLabeler(BaseSubtaskLabeler):
    """Labeler for the WashFish composite task.

    Task: a fish sits in an ALREADY-OPEN fridge; the sink faucet is running. Take the
    fish out, carry it to the sink, hold it under the running water to rinse it, then
    set it on the tray on the counter next to the sink.

    Doc-canonical count = 4 (`mobile: Yes`): the fish's pick-place (fridge -> tray) is
    INTERRUPTED by a navigation to the sink and a rinse ->
    PickPlaceFridgeToTray[pick] -> NavigateKitchen -> RinseFish -> [place]. 5 segments
    (expected_done_idx = 4). The nav is adaptive; the short sink->tray hop is folded
    into the place.

    Boundaries:
      - t_carry:  the fish first displaces MOVE_TOL from its in-fridge rest -> the
                  pick ends.
      - t_sink:   the base stops driving (the fridge->sink drive is done) before the
                  rinse -> the nav ends, the rinse begins.
      - t_rinse_end: the fish is lifted back out of the sink after the last under-water
                  frame (its Z rises above the rinse dip) -> the rinse ends.
      - t_placed: the gripper releases the fish onto the tray.
    """

    ENV_NAME = "WashFish"

    MOVE_TOL = 0.12
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    BASE_MOVE_TH = 0.003
    MIN_NAV_DIST = 0.20
    LIFT_TOL = 0.12  # fish Z rising this far above the rinse dip == lifted out
    REST_W = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.env = env
        self.fish_body_id = env.obj_body_id["obj"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.fish_lang = env.get_obj_lang("obj")
        except Exception:
            self.fish_lang = "fish"

        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])

        self.extra_subtasks = {SUBTASK_MOVE, SUBTASK_RINSE}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        try:
            uw = 1.0 if env.sink.check_obj_under_water(env, "obj") else 0.0
        except Exception:
            uw = 0.0
        return {
            "fish": env.sim.data.body_xpos[self.fish_body_id].copy(),
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
        fish = np.array([h["fish"] for h in history])
        grip = np.array([h["grip"] for h in history])
        uw = np.array([h["uw"] for h in history])
        base_xy = (
            np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        )
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        # pick -> carry: the fish is grasped and pulled out of the fridge
        rest = np.median(fish[: self.REST_W], axis=0)
        disp = np.linalg.norm(fish - rest, axis=1)
        t_carry = self._sustained_start(disp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
        if t_carry is None:
            mv = np.where(disp >= self.MOVE_TOL)[0]
            t_carry = int(mv[0]) if mv.size else 0

        # place ends at the gripper release (onto the tray)
        gd = np.linalg.norm(grip - fish, axis=1)
        hold_gd = float(np.median(gd[t_carry: min(t_carry + self.HOLD_W, T)]))
        t_placed = self._sustained_start(
            gd > hold_gd + self.RELEASE_MARGIN, t_carry + self.MIN_CARRY, self.RELEASE_SUSTAIN
        )
        if t_placed is None:
            t_placed = T - 1

        # rinse: the fish is held under the running water. Bracket by the under-water
        # frames; the rinse begins when the base parks at the sink (nav done) and ends
        # when the fish is lifted back out.
        uw_idx = np.where(uw > 0.5)[0]
        uw_idx = uw_idx[(uw_idx >= t_carry) & (uw_idx < t_placed)] if uw_idx.size else uw_idx
        has_rinse = uw_idx.size > 0
        if has_rinse:
            first_uw, last_uw = int(uw_idx[0]), int(uw_idx[-1])
            # nav ends: the last base-moving frame before the fish goes under water
            moving = np.where(binst[t_carry: first_uw] > self.BASE_MOVE_TH)[0]
            t_sink = t_carry + int(moving[-1]) + 1 if moving.size else t_carry
            t_sink = int(min(max(t_sink, t_carry), first_uw))
            # rinse ends: the fish is lifted back out of the sink after the last
            # under-water frame (its Z rises above the rinse dip)
            fz = fish[:, 2]
            dip = float(np.min(fz[t_sink: last_uw + 1])) if last_uw >= t_sink else float(fz[last_uw])
            t_rinse_end = self._sustained_start(fz > dip + self.LIFT_TOL, last_uw, 3)
            if t_rinse_end is None:
                t_rinse_end = last_uw
            t_rinse_end = int(min(max(t_rinse_end, t_sink), t_placed))
        else:
            # no under-water frames -- fall back to a base-stop split (no rinse)
            moving = np.where(binst[t_carry: t_placed] > self.BASE_MOVE_TH)[0]
            t_sink = min(t_carry + int(moving[-1]) + 1, t_placed) if moving.size else t_carry
            t_rinse_end = t_sink

        # nav (fridge -> sink): peak base displacement over [t_carry, t_sink]
        if base_xy is not None and t_sink > t_carry:
            off = base_xy[t_carry: t_sink] - base_xy[t_carry]
            nav_disp = float(np.max(np.linalg.norm(off, axis=1)))
        else:
            nav_disp = 0.0

        # -- assemble --
        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        emit(t_carry, SUBTASK_MOVE, STAGE_PICK, f"pick up the {self.fish_lang} from the fridge")
        if t_sink > cursor and nav_disp >= self.MIN_NAV_DIST:
            emit(t_sink, SUBTASK_NAV, STAGE_NAVIGATE, NAV_LANG)
        if has_rinse:
            emit(t_rinse_end, SUBTASK_RINSE, STAGE_EXECUTE, RINSE_LANG)
        emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place the {self.fish_lang} on the tray")
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        review = []
        if not has_rinse:
            review.append("fish never registered under running water -- check the rinse")

        debug = {
            "t_carry": t_carry, "t_sink": t_sink, "t_rinse_end": t_rinse_end,
            "t_placed": t_placed, "nav_disp": round(nav_disp, 2),
            "has_rinse": has_rinse, "n_segments": len(segments),
        }
        result = self.build_result(T, segments, debug=debug)
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
            issues.append(f"expected 1 fish pick, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 1:
            issues.append(f"expected 1 fish place, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        if count(SUBTASK_RINSE) != 1:
            issues.append(f"expected 1 RinseFish, found {count(SUBTASK_RINSE)}")
        n_nav = count(SUBTASK_NAV)
        if n_nav not in (0, 1):
            issues.append(f"expected 0-1 NavigateKitchen, found {n_nav}")
        return issues
