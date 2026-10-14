import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_EXECUTE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Open the (closed, double-door) cabinet; take the bowl and condiment out onto the counter,
# then place the lime and garlic (already on the counter) on the cutting board. OpenCabinet +
# PickPlaceCabinetToCounter are bank atomics; counter -> cutting-board has no bank name ->
# PickPlaceCounterToCounter (extension). Fixed base; no nav.
SUBTASK_OPEN = "OpenCabinet"
SUBTASK_CAB = "PickPlaceCabinetToCounter"
SUBTASK_CB = "PickPlaceCounterToCounter"

DONE_LANG = "task complete"


@register_labeler
class SpicyMarinadeLabeler(BaseSubtaskLabeler):
    """Labeler for the SpicyMarinade composite task.

    Task (fixed-base): a bowl and a condiment sit in a CLOSED double-door cabinet; a lime and a
    garlic sit on the counter; a cutting board sits on the counter. Open the cabinet, set the
    bowl and condiment on the counter, then set the lime and garlic on the cutting board.

    Doc-canonical decomposition (5 high-level: open + 4 moves). Emitted: OpenCabinet, then the
    four objects in first-motion order, each named by its own source/destination:
      OpenCabinet [execute]
      PickPlaceCabinetToCounter  bowl/condiment  cabinet -> counter
      PickPlaceCounterToCounter  lime/garlic     counter -> cutting board
    plus a terminal done.

    Detection: OpenCabinet ends when the LAST opened leaf reaches min(OPEN_ABS, 0.9*peak)
    sustained (double door). Object ORDER and windows are GRASP-based (gd < GRASP_TH), not
    displacement: the lime/garlic start close to the board so their net displacement can be
    < MOVE_TOL (demo_96: 0.10), and the gripper lingers by a just-placed veg (which sits near
    the next one), so a release-only place over-extends and swallows the next grab (demo_18).
    Each object's place is therefore BOUNDED by the next object's grasp. Per object: t_pick =
    the object's motion onset within its grasp window; t_place = the gripper RELEASE, capped at
    the next grasp.
    """

    ENV_NAME = "SpicyMarinade"

    # object -> (natural word, source phrase, dest phrase, subtask name)
    ITEMS = {
        "bowl": ("bowl", "the cabinet", "on the counter", SUBTASK_CAB),
        "condiment": (None, "the cabinet", "on the counter", SUBTASK_CAB),
        "lime": ("lime", "the counter", "on the cutting board", SUBTASK_CB),
        "garlic": ("garlic", "the counter", "on the cutting board", SUBTASK_CB),
    }

    MOVE_TOL = 0.05       # motion-onset tol WITHIN a window (the lift off the counter);
                          # smaller than MOVE_TOL_STD because lime/garlic barely translate.
    MOVE_SUSTAIN = 3
    MOVE_TOL_STD = 0.10   # displacement that reliably marks a big cabinet->counter move
    MOVE_SUSTAIN_STD = 5
    GRASP_TH = 0.06       # gd below this == a small object is in hand (lime/garlic gd_min
                          # ~0.01-0.02); a big bowl grabbed by the rim never gets this close to
                          # its centre, so grasp is a FALLBACK, not the primary signal.
    GRASP_SUSTAIN = 4
    HOLD_W = 30
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 8
    MIN_PICK_GAP = 8      # a place ends at least this far before the next grasp (visible pick)
    REST_W = 15
    OPEN_ABS = 0.85
    OPEN_MIN = 0.25
    OPEN_SUSTAIN = 5

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = list(self.ITEMS.keys())
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.word = {}
        for o in self.names:
            w = self.ITEMS[o][0]
            if w is None:
                try:
                    w = env.get_obj_lang(o)
                except Exception:
                    w = o
            self.word[o] = w
        cab = env.cab
        joint_names = list(getattr(cab, "door_joint_names", []) or [])
        if not joint_names:
            pref = cab.name
            joint_names = [n for n in env.sim.model.joint_names
                           if n and n.startswith(pref) and ("door" in n.lower() or "hinge" in n.lower())]
        self.door_adrs, self.door_scales = [], []
        for j in joint_names:
            try:
                adr = env.sim.model.get_joint_qpos_addr(j)
                jid = env.sim.model.joint_name2id(j)
                lo, hi = env.sim.model.jnt_range[jid]
            except Exception:
                continue
            self.door_scales.append(max(abs(float(hi)), abs(float(lo)), 1e-6))
            self.door_adrs.append(adr)
        self.extra_subtasks = {SUBTASK_CB}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        doors = np.array([abs(float(env.sim.data.qpos[a])) / s
                          for a, s in zip(self.door_adrs, self.door_scales)])
        d = {"grip": env.sim.data.site_xpos[self.eef_site_id].copy(), "doors": doors}
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
    def _longest_run(mask, min_len):
        best, best_len, rs = None, 0, None
        for t in range(len(mask)):
            if mask[t]:
                if rs is None:
                    rs = t
            else:
                if rs is not None:
                    L = t - rs
                    if L >= min_len and L > best_len:
                        best_len, best = L, (rs, t)
                    rs = None
        if rs is not None:
            L = len(mask) - rs
            if L >= min_len and L > best_len:
                best = (rs, len(mask))
        return best

    def resolve(self, history):
        T = len(history)
        opos = {o: np.array([h[o] for h in history]) for o in self.names}
        grip = np.array([h["grip"] for h in history])
        doorv = (np.array([h["doors"] for h in history])
                 if T and len(history[0]["doors"]) else np.zeros((T, 0)))
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in self.names}

        # HYBRID manipulation-start per object: the earliest of a big displacement (works for
        # the bowl/condiment, which travel cabinet->counter but never let the gripper near
        # their centre) OR a tight grasp (works for the lime/garlic, which barely translate).
        manip = {}
        for o in self.names:
            rest = np.median(opos[o][: self.REST_W], axis=0)
            dfm = self._sustained_start(
                np.linalg.norm(opos[o] - rest, axis=1) >= self.MOVE_TOL_STD, 0, self.MOVE_SUSTAIN_STD)
            grun = self._longest_run(gd[o] < self.GRASP_TH, self.GRASP_SUSTAIN)
            gst = grun[0] if grun is not None else None
            cands = [x for x in (dfm, gst) if x is not None]
            manip[o] = min(cands) if cands else None
        order = sorted([o for o in self.names if manip[o] is not None], key=lambda o: manip[o])

        # OpenCabinet: end when the LAST opened leaf reaches its adaptive open level.
        open_frames, n_open = [], 0
        for i in range(doorv.shape[1]):
            col = doorv[:, i]
            peak = float(col.max())
            if peak < self.OPEN_MIN:
                continue
            lvl = min(self.OPEN_ABS, 0.9 * peak)
            fr = self._sustained_start(col >= lvl, 0, self.OPEN_SUSTAIN)
            if fr is not None:
                open_frames.append(fr)
                n_open += 1
        earliest_manip = min([manip[o] for o in order], default=T)
        t_open = max(open_frames) if open_frames else 0
        t_open = int(min(t_open, max(earliest_manip - 1, 0)))
        door_word = "doors" if n_open >= 2 else "door"

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        review = []
        if n_open > 0 and t_open > 0:
            emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, f"open the cabinet {door_word} above")
        elif len(self.door_adrs) > 0:
            review.append("cabinet has a door but it never opened -- verify")

        n_moved = 0
        for i, o in enumerate(order):
            word, src, dst, sub = self.word[o], self.ITEMS[o][1], self.ITEMS[o][2], self.ITEMS[o][3]
            g0 = max(int(manip[o]), cursor)
            win_end = int(manip[order[i + 1]]) if i + 1 < len(order) else T
            # t_pick = the object's motion onset (lift off its rest) within the grasp window,
            # so the pick covers the reach+grasp and the place the lift+carry+set-down.
            r = np.median(opos[o][max(g0 - self.REST_W, 0): g0 + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_pick = self._sustained_start(dsp[:win_end] >= self.MOVE_TOL, g0, self.MOVE_SUSTAIN)
            if t_pick is None:
                t_pick = g0
            t_pick = int(min(max(t_pick, cursor), win_end - 1))
            emit(t_pick, sub, STAGE_PICK, f"pick up the {word} from {src}")
            # t_place = the gripper RELEASE, but never past the next object's grasp (a lingering
            # gripper by an adjacent placed veg must not swallow the next grab).
            hold_gd = float(np.median(gd[o][t_pick: min(t_pick + self.HOLD_W, T)]))
            released = gd[o] > hold_gd + self.RELEASE_MARGIN
            t_placed = self._sustained_start(released, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = win_end
            if i + 1 < len(order):
                t_placed = min(t_placed, win_end - self.MIN_PICK_GAP)
            t_placed = int(min(max(t_placed, t_pick + 1), T))
            emit(t_placed, sub, STAGE_PLACE, f"place the {word} {dst}")
            n_moved += 1

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        if n_moved != 4:
            review.append(f"only {n_moved}/4 items moved")

        result = self.build_result(T, segments, debug={
            "t_open": t_open, "n_open": n_open, "order": order,
            "n_moved": n_moved, "n_segments": len(segments),
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

        if count(SUBTASK_OPEN) not in (0, 1):
            issues.append(f"expected 0-1 OpenCabinet, found {count(SUBTASK_OPEN)}")
        if int(np.sum(stages == STAGE_PICK)) != 4:
            issues.append(f"expected 4 pick stages, found {int(np.sum(stages == STAGE_PICK))}")
        if int(np.sum(stages == STAGE_PLACE)) != 4:
            issues.append(f"expected 4 place stages, found {int(np.sum(stages == STAGE_PLACE))}")
        return issues
