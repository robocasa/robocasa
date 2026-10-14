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


# Open the toaster oven and move the heatable items (sausage, bread) from the plate onto its rack,
# leaving the cold items (tomato, onion) on the plate. OpenToasterOvenDoor, SlideToasterOvenRack,
# and PickPlaceCounterToToasterOven are all bank atomics. Fixed base / local.
SUBTASK_OPEN = "OpenToasterOvenDoor"
SUBTASK_SLIDE = "SlideToasterOvenRack"
SUBTASK_MOVE = "PickPlaceCounterToToasterOven"

DONE_LANG = "task complete"

HEATABLE = ("sausage", "bread")


@register_labeler
class ToastHeatableIngredientsLabeler(BaseSubtaskLabeler):
    """Labeler for the ToastHeatableIngredients composite task.

    Task (fixed-base): a plate on the counter holds a sausage, bread, tomato, and onion beside a
    (closed) toaster oven. Open the toaster oven and put the two heatable items (sausage, bread) on
    its rack; leave the tomato and onion on the plate. The robot starts at the counter.

    Doc-canonical count = 4 (action-level): open the door, an optional rack slide, and the two
    heatable moves. Emitted:
      OpenToasterOven [execute]                 -> open the toaster oven door
      (SlideToasterOvenRack [execute])          -> slide the toaster oven rack out (if pulled)
      per heatable item (grasp order):
        PickPlaceCounterToToasterOven [pick]    -> pick up the {sausage|bread} from the plate
        PickPlaceCounterToToasterOven [place]   -> place the {sausage|bread} {on the rack}
    plus a terminal done. Only the sausage and bread move; the tomato and onion stay on the plate.

    Boundaries: OpenToasterOven ends when the door joint rises to its adaptive open level (before
    the first item is grabbed); item grasp = first contact / lift; place = the gripper release
    (finger-width open).
    """

    ENV_NAME = "ToastHeatableIngredients"

    MOVE_TH = 0.10
    Z_LIFT = 0.03
    MOTION_TH = 0.004
    RELEASE_PAD = 8
    REL_FRAC = 0.5
    REST_W = 10
    OPEN_FRAC = 0.85     # door reaches this fraction of its open peak -> counted open
    OPEN_MIN = 0.30      # door peak must exceed this for an open to register
    OPEN_SUSTAIN = 5
    SLIDE_TH = 0.15
    SLIDE_FRAC = 0.85

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.movers = [o for o in HEATABLE if o in env.obj_body_id]
        self.body_ids = {n: env.obj_body_id[n] for n in self.movers}
        self.geoms = {n: env.objects[n].contact_geoms for n in self.movers}
        self.gripper = env.robots[0].gripper["right"]
        self.fadr = [env.sim.model.get_joint_qpos_addr(j) for j in self.gripper.joints]
        self.obj_lang = {}
        for n in self.movers:
            try:
                self.obj_lang[n] = env.get_obj_lang(n)
            except Exception:
                self.obj_lang[n] = n
        tov = env.toaster_oven
        self.door_adr = env.sim.model.get_joint_qpos_addr(tov._joint_names["door"])
        self.multi_rack = tov.has_multiple_rack_levels()
        prefix = tov._joint_names["door"][: -len("door_joint")]
        jointset = set(env.sim.model.joint_names)
        self.rack_adrs = {}
        for kind in ("rack", "tray"):
            lvls = {}
            for lvl in (0, 1):
                jn = f"{prefix}{kind}{lvl}_joint"
                if jn in jointset:
                    lvls[lvl] = env.sim.model.get_joint_qpos_addr(jn)
            if lvls:
                self.rack_adrs = lvls
                break
        self.min_segment_frames = 12
        self.expected_done_idx = None

    def per_step(self, env, t: int) -> dict:
        return {
            "pos": {n: env.sim.data.body_xpos[b].copy() for n, b in self.body_ids.items()},
            "fw": float(sum(abs(env.sim.data.qpos[a]) for a in self.fadr)),
            "grasp": {n: int(env._check_grasp(gripper=self.gripper, object_geoms=g))
                      for n, g in self.geoms.items()},
            "door": float(env.sim.data.qpos[self.door_adr]),
            "rack": {lvl: float(env.sim.data.qpos[a]) for lvl, a in self.rack_adrs.items()},
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

    def _grasp_onset(self, grasp_arr, obj):
        gi = np.where(grasp_arr > 0)[0]
        if gi.size:
            return int(gi[0])
        z0 = float(np.median(obj[: self.REST_W, 2]))
        lifted = np.where(obj[:, 2] > z0 + self.Z_LIFT)[0]
        if lifted.size:
            return int(lifted[0])
        disp = np.linalg.norm(obj[:, :2] - obj[0, :2], axis=1)
        w = np.where(disp > self.MOVE_TH)[0]
        return int(w[0]) if w.size else 0

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

    def _release_for(self, grasp_arr, obj, fw, fw_open, grasp_t, hi):
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

    def resolve(self, history):
        T = len(history)
        pos = {n: np.array([h["pos"][n] for h in history]) for n in self.movers}
        grasp = {n: np.array([h["grasp"][n] for h in history]) for n in self.movers}
        fw = np.array([h["fw"] for h in history])
        fw_open = float(fw.max())
        door = np.array([h["door"] for h in history])
        rack = {lvl: np.array([h["rack"][lvl] for h in history]) for lvl in self.rack_adrs}

        info = {}
        for n in self.movers:
            p = pos[n]
            disp = float(np.linalg.norm(p[:, :2] - p[0, :2], axis=1).max())
            if disp > self.MOVE_TH:
                info[n] = dict(pos=p, grasp0=self._grasp_onset(grasp[n], p))
        order = sorted(info, key=lambda n: info[n]["grasp0"])
        first_grasp = min([info[n]["grasp0"] for n in order], default=T)

        segments = []
        cursor = 0
        review = []

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        # --- OpenToasterOven: the door rises from shut to its open peak, before the first grasp ---
        seg = door[:first_grasp]
        peak = float(seg.max()) if seg.size else 0.0
        if peak >= self.OPEN_MIN and peak - float(door[0]) >= 0.10:
            t_open = self._sustained_start(door[:first_grasp] >= self.OPEN_FRAC * peak, 0, self.OPEN_SUSTAIN)
            if t_open is not None and t_open > cursor:
                emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, "open the toaster oven door")
            else:
                review.append("toaster oven door open not cleanly detected -- verify")
        else:
            review.append("toaster oven door never opened -- verify")

        # --- optional rack slide (rack joint rises when pulled out) ---
        pulled_level, slide_end = None, None
        for lvl, rv in rack.items():
            if rv.max() > self.SLIDE_TH and (pulled_level is None or rv.max() > rack[pulled_level].max()):
                pulled_level = lvl
        if pulled_level is not None:
            rv = rack[pulled_level]
            se = np.where(rv >= self.SLIDE_FRAC * float(rv.max()))[0]
            slide_end = int(se[0]) if se.size else 0
        rack_name = (("bottom rack" if (pulled_level or 0) == 0 else "top rack")
                     if self.multi_rack else "rack")
        place_dest = f"on the {rack_name} of the toaster oven" if self.multi_rack else "on the toaster oven rack"

        slide_done = pulled_level is None
        n_moved = 0
        for i, n in enumerate(order):
            b = info[n]
            g = max(int(b["grasp0"]), cursor)
            hi = info[order[i + 1]]["grasp0"] if i + 1 < len(order) else T
            hi = max(hi, g + 1)
            rel = self._release_for(grasp[n], b["pos"], fw, fw_open, g, hi)
            # emit the rack slide only if it forms a real segment; a sub-min-length slide (the rack
            # is pulled in a few frames) folds into the adjacent actions instead of collapsing.
            if (not slide_done and slide_end is not None and slide_end <= g
                    and slide_end - cursor >= self.min_segment_frames):
                emit(slide_end, SUBTASK_SLIDE, STAGE_EXECUTE, f"slide the {rack_name} of the toaster oven out"
                     if self.multi_rack else "slide the toaster oven rack out")
                slide_done = True
            elif not slide_done and slide_end is not None and slide_end <= g:
                slide_done = True  # negligible slide -- skip the tiny segment
            lang = self.obj_lang.get(n, n)
            emit(g, SUBTASK_MOVE, STAGE_PICK, f"pick up the {lang} from the plate")
            emit(rel, SUBTASK_MOVE, STAGE_PLACE, f"place the {lang} {place_dest}")
            n_moved += 1

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        if n_moved != 2:
            review.append(f"only {n_moved}/2 heatable items moved")

        result = self.build_result(T, segments, debug={
            "order": ",".join(order), "n_moved": n_moved, "door_peak": round(peak, 2),
            "pulled_level": pulled_level, "n_segments": len(segments),
        })
        result.review = review
        return result

    def validate(self, result):
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        stages = result.subtask_stage[firsts]
        names = result.subtask_name[firsts]
        if int(np.sum(stages == STAGE_PICK)) != 2:
            issues.append(f"expected 2 picks, found {int(np.sum(stages == STAGE_PICK))}")
        if int(np.sum(stages == STAGE_PLACE)) != 2:
            issues.append(f"expected 2 places, found {int(np.sum(stages == STAGE_PLACE))}")
        if int(np.sum(names == SUBTASK_OPEN)) not in (0, 1):
            issues.append(f"expected 0-1 OpenToasterOven, found {int(np.sum(names == SUBTASK_OPEN))}")
        return issues
