import re

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


# The toaster oven starts CLOSED; a bread item (on a plate) and a meat item (on a plate) are
# on the counter. Open the door, then for EACH item: slide its rack level out, put the item
# on it, and slide the rack back in. The two items go to different levels (top vs bottom).
# OpenToasterOvenDoor / SlideToasterOvenRack / PickPlaceCounterToToasterOven are bank atomics;
# NavigateKitchen is bank. rack_level 1 == "top", 0 == "bottom".
SUBTASK_OPEN = "OpenToasterOvenDoor"
SUBTASK_RACK = "SlideToasterOvenRack"
SUBTASK_MOVE = "PickPlaceCounterToToasterOven"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"

ITEMS = ("bread", "meat")


@register_labeler
class ToastOnCorrectRackLabeler(BaseSubtaskLabeler):
    """Labeler for the ToastOnCorrectRack composite task.

    Task (mobile): the toaster oven starts closed; a bread item and a meat item sit on plates
    on the counter. Open the door, then put each item on its designated rack level (one on the
    top rack, the other on the bottom), sliding that rack out before and in after.

    Doc-canonical count = 7 (`mobile: Yes`), action-level: OpenToasterOvenDoor + per item
    (slide its rack out + counter->oven pick/place + slide it back in), ordered by first
    grasped move.

    Boundaries: t_open = door reaches its open plateau; per item its RACK slides out (reaches
    the out plateau) before the pick, the item is grasped/carried/RELEASEd, then its rack
    slides back in (returns toward closed). A leading NavigateKitchen is emitted only when the
    base actually drives before a pick.
    """

    ENV_NAME = "ToastOnCorrectRack"

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.06
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    REST_W = 15
    GRASP_TH = 0.08
    DOOR_OPEN_FRAC = 0.80
    OPEN_SUSTAIN = 5
    RACK_OUT_FRAC = 0.85   # rack past this fraction of its out peak == slid out
    RACK_IN_FRAC = 0.30    # rack back below this fraction of its out peak == slid in
    RACK_MIN = 0.05        # min out travel to count as a real slide
    BASE_MOVE_TH = 0.008
    MIN_NAV_DIST = 0.30
    MIN_PICK_GAP = 12

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = [o for o in ITEMS if o in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.lang = {}
        for o in self.names:
            try:
                self.lang[o] = env.get_obj_lang(o)
            except Exception:
                self.lang[o] = o
        bl = int(ep_meta.get("bread_rack_level", getattr(env, "bread_rack_level", 1)))
        ml = int(getattr(env, "meat_rack_level", 1 - bl))
        self.level = {"bread": bl, "meat": ml}
        self.direction = {o: ("top" if self.level[o] == 1 else "bottom") for o in self.names}
        to = env.toaster_oven
        try:
            self.receptacle = to.rack_or_tray(env)
        except Exception:
            self.receptacle = "rack"
        # map each rack/tray level -> its joint qpos address ('...rack0_joint' -> level 0)
        prefix = to.naming_prefix
        self.rec_level_adr = {}
        self.rec_is_tray = False
        for nm in env.sim.model.joint_names:
            if not nm or not nm.startswith(prefix):
                continue
            m = re.search(r"(rack|tray)(\d+)", nm)
            if m:
                self.rec_level_adr[int(m.group(2))] = env.sim.model.get_joint_qpos_addr(nm)
                self.rec_is_tray = self.rec_is_tray or (m.group(1) == "tray")
        self.rec_levels = sorted(self.rec_level_adr)
        n = to._joint_names.get("door")
        self.door_adr = (env.sim.model.get_joint_qpos_addr(n)
                         if n and n in env.sim.model.joint_names else None)
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[env.sim.model.joint_name2id(nm)])
            except Exception:
                pass
        self.extra_subtasks = set()
        self.expected_done_idx = None
        self.min_segment_frames = 6

    def per_step(self, env, t: int) -> dict:
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "door": float(env.sim.data.qpos[self.door_adr]) if self.door_adr is not None else 0.0,
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
            "recs": np.array([float(env.sim.data.qpos[self.rec_level_adr[l]]) for l in self.rec_levels]),
        }
        for o in self.names:
            d[o] = env.sim.data.body_xpos[self.obj_body_ids[o]].copy()
        return d

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

    @staticmethod
    def _last_sustained_end(mask, sustain):
        last_end, run = None, 0
        for i, v in enumerate(mask):
            if v:
                run += 1
            else:
                if run >= sustain:
                    last_end = i
                run = 0
        if run >= sustain:
            last_end = len(mask)
        return last_end

    def resolve(self, history):
        T = len(history)
        opos = {o: np.array([h[o] for h in history]) for o in self.names}
        grip = np.array([h["grip"] for h in history])
        door = np.array([h["door"] for h in history])
        recs = np.array([h["recs"] for h in history]) if T else np.zeros((0, len(self.rec_levels)))
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in self.names}
        review = []

        def base_disp(a, b):
            return (float(np.max(np.linalg.norm(base_xy[a: b] - base_xy[a], axis=1)))
                    if base_xy is not None and b > a else 0.0)

        def rack_of(o):
            lvl = self.level[o]
            if lvl in self.rec_levels:
                return recs[:, self.rec_levels.index(lvl)]
            return np.zeros(T)

        # door open
        peak_door = float(door.max()) if T else 0.0
        if peak_door > 0.2:
            t_open = self._sustained_start(door >= self.DOOR_OPEN_FRAC * peak_door, 0, self.OPEN_SUSTAIN)
        else:
            t_open = None
            review.append("door never opened -- verify")
        if t_open is None:
            t_open = max(1, T // 10)

        # order items by first grasped move (after the door opens)
        first_move = {}
        for o in self.names:
            rest = np.median(opos[o][: self.REST_W], axis=0)
            dsp = np.linalg.norm(opos[o] - rest, axis=1)
            fm = self._sustained_start((dsp >= self.MOVE_TOL) & (gd[o] < self.GRASP_TH), t_open, self.MOVE_SUSTAIN)
            if fm is None:
                fm = self._sustained_start(dsp >= self.MOVE_TOL, t_open, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted([o for o in self.names if first_move[o] < T], key=lambda o: first_move[o])

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, "open the toaster oven door")

        n_moved = 0
        for o in order:
            c0 = cursor  # this item's start, before any nav/slide
            pos = opos[o]
            rack = rack_of(o)
            rrest = float(np.median(rack[: self.REST_W]))
            rpeak = float(rack.max() - rrest)
            direction = self.direction.get(o, "")

            # pick / place (item on a counter plate -> onto the slid-out rack)
            r = np.median(pos[max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(pos - r, axis=1)
            t_pick = self._sustained_start((dsp >= self.MOVE_TOL) & (gd[o] < self.GRASP_TH), cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                t_pick = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick = max(t_pick, cursor)
            hold = float(np.median(gd[o][t_pick: min(t_pick + self.HOLD_W, T)]))
            t_place = self._sustained_start(gd[o] > hold + self.RELEASE_MARGIN, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_place is None:
                t_place = T - 1

            # this item's rack SLIDE-OUT window, before the pick: rises from rest to the out
            # plateau. Computed BEFORE the nav so a slide-out that overlaps a repositioning
            # drive still gets its own subtask (the nav is capped to end when the rack begins
            # rising). ##demo_15/50 2nd rack
            # skip the slide-out if this item's rack is already (mostly) out at its start --
            # it slid out earlier (e.g. overlapping the previous item's slide-in); emitting it
            # would be a collapsed 1-frame segment.
            rise_start = t_so_end = None
            if rpeak > self.RACK_MIN and (float(rack[min(c0, T - 1)]) - rrest) < 0.5 * rpeak:
                rs = self._sustained_start(rack - rrest > 0.30 * rpeak, cursor, self.MOVE_SUSTAIN)
                te = self._sustained_start(rack - rrest >= self.RACK_OUT_FRAC * rpeak, cursor, self.MOVE_SUSTAIN)
                if rs is not None and te is not None and rs < t_pick:
                    rise_start = int(min(max(rs, cursor), t_pick))
                    t_so_end = int(min(max(te, rise_start + 1), max(t_pick - self.MIN_PICK_GAP, rise_start + 1)))

            # leading nav (only when the base actually drove), ending before the rack rises
            nav_limit = rise_start if rise_start is not None else t_pick
            end = self._last_sustained_end(binst[cursor: nav_limit] > self.BASE_MOVE_TH, self.MOVE_SUSTAIN)
            t_nav = cursor + end if end is not None else cursor
            t_nav = min(t_nav, max(nav_limit - 2, cursor))
            if t_nav > cursor and base_disp(cursor, t_nav) >= self.MIN_NAV_DIST:
                emit(t_nav, SUBTASK_NAV, STAGE_NAVIGATE, "navigate to the toaster oven")

            # slide-out
            if t_so_end is not None and t_so_end > cursor:
                emit(t_so_end, SUBTASK_RACK, STAGE_EXECUTE,
                     f"slide the {direction} toaster oven {self.receptacle} out")

            emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up the {self.lang[o]} from the plate")
            emit(t_place, SUBTASK_MOVE, STAGE_PLACE,
                 f"place the {self.lang[o]} on the {direction} {self.receptacle} in the toaster oven")

            # slide this item's rack IN (after the place) -- only when the rack is still
            # substantially OUT at release; if it was pushed in DURING the place (rack already
            # in by release), there is no distinct post-place slide and it folds into the
            # place. ##demo_20/23/68
            if rpeak > self.RACK_MIN and (float(rack[min(t_place, T - 1)]) - rrest) > self.RACK_IN_FRAC * rpeak:
                t_si = self._sustained_start(rack - rrest <= self.RACK_IN_FRAC * rpeak, t_place, self.MOVE_SUSTAIN)
                if t_si is not None and t_si - cursor >= self.MOVE_SUSTAIN:
                    emit(t_si, SUBTASK_RACK, STAGE_EXECUTE,
                         f"slide the {direction} toaster oven {self.receptacle} in")
            n_moved += 1

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        if n_moved != 2:
            review.append(f"only {n_moved}/2 items moved")
        result = self.build_result(T, segments, debug={
            "order": order, "t_open": t_open, "direction": self.direction,
            "receptacle": self.receptacle, "n_moved": n_moved, "n_segments": len(segments),
        })
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

        if count(SUBTASK_OPEN) != 1:
            issues.append(f"expected 1 OpenToasterOvenDoor, found {count(SUBTASK_OPEN)}")
        if count(SUBTASK_MOVE, STAGE_PICK) != 2:
            issues.append(f"expected 2 item picks, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 2:
            issues.append(f"expected 2 item places, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        if count(SUBTASK_RACK) not in (0, 1, 2, 3, 4):
            issues.append(f"unexpected SlideToasterOvenRack count {count(SUBTASK_RACK)}")
        return issues
