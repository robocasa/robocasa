import numpy as np

import robocasa.utils.object_utils as OU
from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_NAVIGATE,
    STAGE_PICK,
    STAGE_PLACE,
    STAGE_WAIT,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Weigh meat on a digital scale, then plate it: take the meat from the fridge, set it on the scale
# on the counter, wait for the reading, then carry it to the plate on the dining counter. Two moves
# of one meat with a wait between. Fridge -> scale (counter) -> PickPlaceFridgeToCounter (ext);
# WeighItem is the wait; scale -> plate (counter -> counter) -> PickPlaceCounterToCounter (ext).
SUBTASK_FRIDGE = "PickPlaceFridgeToCounter"
SUBTASK_WEIGH = "WeighItem"
SUBTASK_PLATE = "PickPlaceCounterToCounter"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"


@register_labeler
class ScalePortioningLabeler(BaseSubtaskLabeler):
    """Labeler for the ScalePortioning composite task.

    Task (mobile): a meat sits in the (open) fridge; a digital scale sits on the counter by the
    fridge; a plate sits on the dining counter. Take the meat from the fridge, set it on the
    scale, wait for the reading, then carry it to the plate. The robot starts at the fridge.

    Doc-canonical count = 6. Emitted:
      PickPlaceFridgeToCounter [pick]   -> pick up the {meat} from the fridge
      (NavigateKitchen [navigate])      -> carry the {meat} to the scale  (adaptive; usually local)
      PickPlaceFridgeToCounter [place]  -> place the {meat} on the digital scale
      WeighItem [wait]                  -> wait for the scale to read the weight
      PickPlaceCounterToCounter [pick]  -> pick up the {meat} from the digital scale
      NavigateKitchen [navigate]        -> carry the {meat} to the plate on the dining counter
      PickPlaceCounterToCounter [place] -> place the {meat} on the plate
    plus a terminal done.

    Boundaries: the meat has two grasp cycles (fridge->scale, scale->plate). t_pick = first
    sustained displacement from the rest re-anchored at the segment start; place = the gripper
    RELEASE; the WAIT spans the release on the scale until the gripper re-grasps the meat.
    """

    ENV_NAME = "ScalePortioning"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    REST_W = 15
    GRASP_TH = 0.05
    GRASP_SUSTAIN = 5
    WAIT_GAP = 8
    MIN_PICK_GAP = 15  # the scale pick keeps at least this many frames (grab + lift)
    BASE_MOVE_TH = 0.008
    MIN_NAV_DIST = 0.25

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_id = env.obj_body_id["meat"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        # finger-width joints -> a clean release/re-grasp signal (the gripper opens to let go and
        # closes to grab, even when it hovers close to the meat and the eef-to-meat distance barely
        # changes -- which happens for bulky meats).
        self.fadr = [env.sim.model.get_joint_qpos_addr(j) for j in env.robots[0].gripper["right"].joints]
        try:
            self.meat = env.get_obj_lang("meat")
        except Exception:
            self.meat = "meat"
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])
        self.extra_subtasks = {SUBTASK_FRIDGE, SUBTASK_WEIGH, SUBTASK_PLATE}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        try:
            on_scale = int(OU.check_obj_in_receptacle(env, "meat", "digital_scale"))
        except Exception:
            on_scale = 0
        return {
            "obj": env.sim.data.body_xpos[self.obj_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "fw": float(sum(abs(env.sim.data.qpos[a]) for a in self.fadr)),
            "on_scale": on_scale,
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
        gd = np.linalg.norm(grip - obj, axis=1)
        fw = np.array([h["fw"] for h in history])
        # finger-open / finger-closed thresholds from this demo's own width distribution
        fw_th = 0.5 * (float(np.percentile(fw, 15)) + float(np.percentile(fw, 90)))
        f_open = fw > fw_th   # fingers open (released)
        f_closed = fw < fw_th  # fingers closed (grasping)
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        segments = []
        cursor = 0
        review = []

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        on_scale = np.array([h["on_scale"] for h in history])

        def first_move(cur):
            r = np.median(obj[max(cur - self.REST_W, 0): cur + 1], axis=0)
            dsp = np.linalg.norm(obj - r, axis=1)
            tm = self._sustained_start(dsp >= self.MOVE_TOL, cur, self.MOVE_SUSTAIN)
            if tm is None:
                mv = np.where(dsp[cur:] >= self.MOVE_TOL)[0]
                tm = cur + int(mv[0]) if mv.size else min(cur + 1, T - 1)
            return max(tm, cur)

        def release_after(start):
            # released when the fingers open OR the eef pulls away from the meat -- whichever first
            hold = float(np.median(gd[start: min(start + self.HOLD_W, T)]))
            rel = self._sustained_start(f_open | (gd > hold + self.RELEASE_MARGIN),
                                        start + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            return rel if rel is not None else T - 1

        def carry_nav(t_move, t_placed, lang):
            if base_xy is None:
                return
            moving = np.where(binst[t_move: t_placed] > self.BASE_MOVE_TH)[0]
            t_arrive = min(t_move + int(moving[-1]) + 1, t_placed) if moving.size else t_move
            nav_disp = (float(np.max(np.linalg.norm(base_xy[t_move: t_arrive] - base_xy[t_move], axis=1)))
                        if t_arrive > t_move else 0.0)
            if t_arrive > cursor and nav_disp >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, lang)

        # the meat sits on the scale over [t_on, t_off); the wait is bounded by this, so the two
        # moves and the reading are cleanly separated regardless of the grasp signature.
        t_on = self._sustained_start(on_scale > 0, 0, 3)
        onoff = np.where(on_scale > 0)[0]
        t_off = int(onoff[-1]) + 1 if onoff.size else None

        # --- move 1: fridge -> scale ---
        t_pick1 = first_move(0)
        t_place1 = t_on if (t_on is not None and t_on > t_pick1) else release_after(t_pick1)
        emit(t_pick1, SUBTASK_FRIDGE, STAGE_PICK, f"pick up the {self.meat} from the fridge")
        carry_nav(t_pick1, t_place1, f"carry the {self.meat} to the scale")
        emit(t_place1, SUBTASK_FRIDGE, STAGE_PLACE, f"place the {self.meat} on the digital scale")

        # --- wait: on the scale from the farthest-eef frame (the gripper starts its re-approach)
        # toward the re-grasp, split at the MIDPOINT of the two -- so the reading wait covers the
        # idle plus roughly half the reach, and the scale pick keeps the rest of the reach + lift. ---
        if t_on is not None and t_off is not None and t_off > cursor + 2:
            seg = gd[cursor: t_off]
            gd_peak = cursor + int(np.argmax(seg)) if seg.size else cursor + 1
            regrab = self._sustained_start(f_closed, gd_peak, self.GRASP_SUSTAIN)
            hi = regrab if (regrab is not None and regrab < t_off) else t_off
            t_grab2 = gd_peak + (hi - gd_peak) // 2
            t_grab2 = int(min(max(t_grab2, gd_peak, cursor + 1), t_off - self.MIN_PICK_GAP))
            emit(t_grab2, SUBTASK_WEIGH, STAGE_WAIT, "wait for the scale to read the weight")
            t_lift2 = t_off
        else:
            review.append("scale wait not detected (meat never registered on the scale) -- verify")
            t_lift2 = first_move(cursor)

        # --- move 2: scale -> plate ---
        t_lift2 = max(t_lift2, cursor + 1)
        t_place2 = release_after(t_lift2)
        emit(t_lift2, SUBTASK_PLATE, STAGE_PICK, f"pick up the {self.meat} from the digital scale")
        carry_nav(t_lift2, t_place2, f"carry the {self.meat} to the plate on the dining counter")
        emit(t_place2, SUBTASK_PLATE, STAGE_PLACE, f"place the {self.meat} on the plate")

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        result = self.build_result(T, segments, debug={
            "t_pick1": int(t_pick1), "t_place1": int(t_place1),
            "t_on": (int(t_on) if t_on is not None else None),
            "t_off": (int(t_off) if t_off is not None else None),
            "t_lift2": int(t_lift2), "t_place2": int(t_place2), "n_segments": len(segments),
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

        if count(SUBTASK_FRIDGE, STAGE_PICK) != 1:
            issues.append(f"expected 1 fridge pick, found {count(SUBTASK_FRIDGE, STAGE_PICK)}")
        if count(SUBTASK_FRIDGE, STAGE_PLACE) != 1:
            issues.append(f"expected 1 scale place, found {count(SUBTASK_FRIDGE, STAGE_PLACE)}")
        if count(SUBTASK_WEIGH) not in (0, 1):
            issues.append(f"expected 0-1 WeighItem, found {count(SUBTASK_WEIGH)}")
        if count(SUBTASK_PLATE, STAGE_PICK) != 1:
            issues.append(f"expected 1 scale pick, found {count(SUBTASK_PLATE, STAGE_PICK)}")
        if count(SUBTASK_PLATE, STAGE_PLACE) != 1:
            issues.append(f"expected 1 plate place, found {count(SUBTASK_PLATE, STAGE_PLACE)}")
        return issues
