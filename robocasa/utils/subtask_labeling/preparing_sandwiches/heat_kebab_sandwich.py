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

# All four are bank atomic tasks (no declared extension needed).
SUBTASK_MOVE = "PickPlaceCounterToToasterOven"   # kebab / baguette -> oven
SUBTASK_SLIDE = "SlideToasterOvenRack"           # pull the rack out (optional)
SUBTASK_CLOSE = "CloseToasterOvenDoor"           # close the door
SUBTASK_TIMER = "TurnOnToasterOven"              # set the timer dial
SUBTASK_NAV = "NavigateKitchen"

NAV_TO_OVEN_LANG = "navigate to the toaster oven"
NAV_TO_PLATE_LANG = "navigate back to the plate"
CLOSE_LANG = "close the toaster oven door"
TIMER_LANG = "set the toaster oven timer"
DONE_LANG = "task complete"


@register_labeler
class HeatKebabSandwichLabeler(BaseSubtaskLabeler):
    """Labeler for the HeatKebabSandwich composite task.

    A kebab skewer and a baguette start on a plate on the counter; the toaster
    oven sits next to it (spawned open). The robot moves both items into the oven,
    closes the door, and sets the timer -- the six subtasks are the two moves
    (pick + place each), the door close, and the timer.

    The two items are handled in grasp order: PickPlaceCounterToToasterOven
    [pick] -> (NavigateKitchen carry, only if the base drives) -> [place]. Grasp
    onset is the item's first contact (falling back to its lift off the plate);
    the place ends fully-ungrasped (later of the contact-grasp end and the
    finger-width gripper-open). The door close and timer are read straight off the
    toaster-oven joints: the door joint falls from ~1.4 (open) to 0 (closed), and
    the time knob rises from 0 to its set value. The close subtask runs from the
    last item's release to the door-closed frame; the timer subtask from there to
    when the knob stops turning.
    """

    ENV_NAME = "HeatKebabSandwich"

    OBJS = ["kebab", "baguette"]
    OBJ_LANG = {"kebab": "kebab skewer", "baguette": "baguette"}

    MOVE_TH = 0.10       # planar displacement above which an item counts as moved
    Z_LIFT = 0.03        # height an item rises off the plate to count as grasped
    MOTION_TH = 0.004
    RELEASE_PAD = 8
    REL_FRAC = 0.5
    REST_W = 10
    DOOR_CLOSED = 0.10   # door joint below this counts as closed (open is ~1.4)
    DOOR_CLOSE_START_FRAC = 0.9  # the door swing begins once the door drops below
                          # this fraction of its open value
    SETTLE_MARGIN = 0.015  # the rack has "settled" (stopped moving in) once it is
                          # within this of the plateau it holds before the door swings
    TIME_FRAC = 0.9      # fraction of the final time value at which the turn is done
    TIME_MIN = 0.1       # a set timer reaches at least this (success threshold)
    SLIDE_TH = 0.15      # rack joint above this counts as slid out (a clear pull
                         # reaches ~0.18-0.21; a small nudge ~0.13 does not count)
    SLIDE_FRAC = 0.85    # fraction of the rack's out-peak at which the slide is done
    SLIDE_BACK_TH = 0.05 # rack joint below this (after the out-peak) counts as slid
                         # back in; some demos leave the rack out and never return

    NAV_MOVING_TH = 0.004
    NAV_SMOOTH_W = 7
    MIN_NAV_DIST = 0.65

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.body_ids = {n: env.obj_body_id[n] for n in self.OBJS}
        self.geoms = {n: env.objects[n].contact_geoms for n in self.OBJS}
        self.gripper = env.robots[0].gripper["right"]
        self.fadr = [env.sim.model.get_joint_qpos_addr(j) for j in self.gripper.joints]
        try:
            self.obj_lang = {n: env.get_obj_lang(n) for n in self.OBJS}
        except Exception:
            self.obj_lang = dict(self.OBJ_LANG)
        tov = env.toaster_oven
        self.door_adr = env.sim.model.get_joint_qpos_addr(tov._joint_names["door"])
        self.time_adr = env.sim.model.get_joint_qpos_addr(tov._joint_names["time"])
        # Sliding-shelf joints (level 0 = bottom, 1 = top). A pulled shelf adds a
        # SlideToasterOvenRack subtask, and a multi-level oven names the rack.
        # The pull-out shelf is articulated as either rack{lvl}_joint OR
        # tray{lvl}_joint depending on the oven model -- some ovens' declared rack
        # joint doesn't exist in the sim and their shelves are trays. Mirror the
        # fixture's slide_rack(): prefer rack joints, fall back to tray, matched by
        # the oven's exact joint prefix (a substring match grabs another fixture's
        # rack joints, e.g. the stove's or dishwasher's).
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
        try:
            self.base_body_id = env.sim.model.body_name2id("mobilebase0_base")
        except Exception:
            self.base_body_id = None
        self.min_segment_frames = 15
        self.expected_done_idx = None

    def per_step(self, env, t: int) -> dict:
        base_xy = (
            env.sim.data.body_xpos[self.base_body_id][:2].copy()
            if self.base_body_id is not None
            else np.array([np.nan, np.nan])
        )
        return {
            "pos": {n: env.sim.data.body_xpos[b].copy() for n, b in self.body_ids.items()},
            "base_xy": base_xy,
            "fw": float(sum(abs(env.sim.data.qpos[a]) for a in self.fadr)),
            "grasp": {
                n: int(env._check_grasp(gripper=self.gripper, object_geoms=g))
                for n, g in self.geoms.items()
            },
            "door": float(env.sim.data.qpos[self.door_adr]),
            "time": float(env.sim.data.qpos[self.time_adr]),
            "rack": {lvl: float(env.sim.data.qpos[a]) for lvl, a in self.rack_adrs.items()},
            "rack_contact": (
                {lvl: int(any(env.toaster_oven.check_rack_contact(env, o, rack_level=lvl)
                              for o in self.OBJS))
                 for lvl in self.rack_adrs}
                if self.multi_rack else {}
            ),
        }

    # ---- event helpers -------------------------------------------------

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

    def _nav_end(self, base_speed, base_xy, a, b):
        if b - a < 2 or base_xy is None:
            return a
        seg = base_speed[a:b]
        moving = np.where(seg > self.NAV_MOVING_TH)[0]
        if not moving.size:
            return a
        nav_end = min(a + int(moving[-1]) + 1, b)
        disp = float(np.linalg.norm(base_xy[min(nav_end, b - 1)] - base_xy[a]))
        return nav_end if disp >= self.MIN_NAV_DIST else a

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        pos = {n: np.array([h["pos"][n] for h in history]) for n in self.OBJS}
        base_xy = np.array([h["base_xy"] for h in history])
        grasp = {n: np.array([h["grasp"][n] for h in history]) for n in self.OBJS}
        fw = np.array([h["fw"] for h in history])
        fw_open = float(fw.max())
        door = np.array([h["door"] for h in history])
        time = np.array([h["time"] for h in history])
        rack = {lvl: np.array([h["rack"][lvl] for h in history]) for lvl in self.rack_adrs}

        has_base = np.isfinite(base_xy).all()
        if has_base and T >= 2:
            inst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            W = self.NAV_SMOOTH_W
            base_speed = np.convolve(inst, np.ones(W) / W, mode="same")
            base_speed = np.concatenate([base_speed, base_speed[-1:]])
        else:
            base_speed = np.zeros(T)
            base_xy = None

        info = {}
        for n in self.OBJS:
            p = pos[n]
            disp = float(np.linalg.norm(p[:, :2] - p[0, :2], axis=1).max())
            if disp > self.MOVE_TH:
                info[n] = dict(pos=p, grasp0=self._grasp_onset(grasp[n], p))
        order = sorted(info, key=lambda n: info[n]["grasp0"])

        # Optional rack slide: a rack joint rises from 0 to ~0.2 when pulled out.
        pulled_level = None
        for lvl, rv in rack.items():
            if rv.max() > self.SLIDE_TH and (
                pulled_level is None or rv.max() > rack[pulled_level].max()):
                pulled_level = lvl
        slide_end = None
        if pulled_level is not None:
            rv = rack[pulled_level]
            se = np.where(rv >= self.SLIDE_FRAC * float(rv.max()))[0]
            slide_end = int(se[0]) if se.size else 0
        # Rack level used (for naming a multi-level oven): the pulled rack, else
        # the rack the items end up on.
        if pulled_level is not None:
            used_level = pulled_level
        elif self.multi_rack and history:
            last_rc = history[-1].get("rack_contact", {})
            used_level = next((l for l, v in sorted(last_rc.items()) if v), 0)
        else:
            used_level = 0
        rack_name = (("bottom rack" if used_level == 0 else "top rack")
                     if self.multi_rack else "rack")

        prev_release = 0
        for i, n in enumerate(order):
            b = info[n]
            g = max(int(b["grasp0"]), prev_release)
            # release bounded by the next item's grasp, else the door-close.
            if i + 1 < len(order):
                hi = info[order[i + 1]]["grasp0"]
            else:
                dc = np.where(door[g:] < self.DOOR_CLOSED)[0]
                hi = int(g + dc[0]) if dc.size else T
            hi = max(hi, g + 1)
            b["grasp"] = g
            b["release"] = self._release_for(grasp[n], b["pos"], fw, fw_open, g, hi)
            prev_release = b["release"]

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        def maybe_nav(end, lang):
            nav_end = self._nav_end(base_speed, base_xy, cursor, end)
            if nav_end > cursor:
                emit(nav_end, SUBTASK_NAV, STAGE_NAVIGATE, lang)

        if self.multi_rack:
            slide_lang = f"slide the {rack_name} of the toaster oven out"
            slide_in_lang = f"slide the {rack_name} of the toaster oven back in"
            place_dest = f"on the {rack_name} of the toaster oven"
        else:
            slide_lang = "slide the toaster oven rack out"
            slide_in_lang = "slide the toaster oven rack back in"
            place_dest = "in the toaster oven"

        slide_done = pulled_level is None
        last_release = 0
        for i, n in enumerate(order):
            b = info[n]
            lang = self.obj_lang.get(n, n)
            # Insert the rack slide before the item it precedes (usually first).
            if not slide_done and slide_end is not None and slide_end <= b["grasp"]:
                emit(slide_end, SUBTASK_SLIDE, STAGE_EXECUTE, slide_lang)
                slide_done = True
            if i > 0:
                maybe_nav(b["grasp"], NAV_TO_PLATE_LANG)
            emit(b["grasp"], SUBTASK_MOVE, STAGE_PICK,
                 f"pick up the {lang} from the plate")
            maybe_nav(b["release"], NAV_TO_OVEN_LANG)
            emit(b["release"], SUBTASK_MOVE, STAGE_PLACE,
                 f"place the {lang} {place_dest}")
            last_release = b["release"]
        if not slide_done and slide_end is not None:
            emit(slide_end, SUBTASK_SLIDE, STAGE_EXECUTE, slide_lang)

        # Slide the rack back in: a pulled-out rack is always pushed back before
        # the door can close. The slide-in subtask spans the window where the rack
        # is actually moving in, and ends when the rack SETTLES -- it reaches the
        # plateau it then holds while the robot repositions to shut the door. The
        # idle repositioning and the door swing both belong to the close subtask,
        # not the slide-in. (Distinguishing signal: rack pos moves -> slide-in;
        # rack pos still, door pos moves -> close.)
        slide_in_end = None
        if pulled_level is not None:
            rv = rack[pulled_level]
            door_open = float(np.median(door[: self.REST_W]))
            dcl = np.where(door[last_release:] < self.DOOR_CLOSED)[0]
            door_closed_f = int(last_release + int(dcl[0])) if dcl.size else T
            # The final swing begins at the last frame the door is still ~fully
            # open before it reaches closed.
            openish = np.where(
                door[last_release:door_closed_f]
                > self.DOOR_CLOSE_START_FRAC * door_open
            )[0]
            door_close_start = (
                int(last_release + int(openish[-1])) if openish.size else last_release
            )
            # held = the plateau the rack holds just before the door swings; the
            # slide-in ends the moment the rack first reaches that plateau. Search
            # only after the last place -- if the rack is already settled by then
            # (it was pushed in during the placement), there is no separate
            # slide-in subtask.
            held = float(rv[door_close_start])
            reach = np.where(
                rv[last_release : door_close_start + 1] <= held + self.SETTLE_MARGIN
            )[0]
            slide_in_end = (
                int(last_release + int(reach[0])) if reach.size else door_close_start
            )
            if slide_in_end <= last_release:
                slide_in_end = None
        close_from = last_release
        if slide_in_end is not None:
            emit(slide_in_end, SUBTASK_SLIDE, STAGE_EXECUTE, slide_in_lang)
            close_from = slide_in_end

        # Close the door: the joint falls from ~1.4 (open) to 0 (closed).
        dc = np.where(door[close_from:] < self.DOOR_CLOSED)[0]
        door_closed = int(close_from + int(dc[0])) if dc.size else T
        emit(door_closed, SUBTASK_CLOSE, STAGE_EXECUTE, CLOSE_LANG)

        # Set the timer: the time knob rises to its final value.
        time_final = float(time[-1])
        if time_final > self.TIME_MIN:
            level = self.TIME_FRAC * time_final
            tt = np.where(time[door_closed:] >= level)[0]
            timer_done = int(door_closed + int(tt[0])) if tt.size else T
        else:
            timer_done = T
        emit(timer_done, SUBTASK_TIMER, STAGE_EXECUTE, TIMER_LANG)

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        debug = {
            "order": ",".join(order),
            "slide": "" if pulled_level is None else f"L{pulled_level}@{slide_end}",
            "slide_in": "" if slide_in_end is None else f"@{slide_in_end}",
            "rack_name": rack_name,
            "door_closed": door_closed,
            "timer_done": timer_done,
            "n_moved": len(order),
            "n_segments": len(segments),
        }
        return self.build_result(T, segments, debug=debug)

    # ---- structural validation ----------------------------------------

    def validate(self, result):
        issues = super().validate(result)
        if getattr(result, "debug", None) and result.debug.get("manual"):
            return issues
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        names = result.subtask_name[firsts]
        stages = result.subtask_stage[firsts]
        n_pick = int(np.sum(stages == STAGE_PICK))
        n_place = int(np.sum(stages == STAGE_PLACE))
        if n_pick != n_place:
            issues.append(f"pick/place mismatch: {n_pick} picks, {n_place} places")
        if n_pick != 2:
            issues.append(f"expected 2 items moved to the oven, found {n_pick}")
        if SUBTASK_CLOSE not in names:
            issues.append("missing CloseToasterOvenDoor subtask")
        if SUBTASK_TIMER not in names:
            issues.append("missing TurnOnToasterOven (timer) subtask")
        return issues
