import numpy as np
from robosuite.utils import transform_utils as T

import robocasa.utils.object_utils as OU
from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_NAVIGATE,
    STAGE_PICK,
    STAGE_PLACE,
    STAGE_TILT,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Plate a balanced meal: put the meat (from a pan on the stove) and the vegetable (from a bowl on
# the counter) onto the same plate. Each item is delivered one of two ways, detected per demo:
#   - DIRECT GRASP: the item is grasped and set on the plate. Meat -> plate is the bank atomic
#     PickPlaceStoveToCounter; vegetable -> plate (counter -> counter) -> PickPlaceCounterToCounter.
#   - TILT-POUR: the robot lifts the pan/bowl and tilts it to dump the item onto the plate (the
#     container's up-axis rotates far while it is lifted). Modeled with the Pick/Tilt/Place
#     bank-extension skills (as in PanTransfer).
SUBTASK_STOVE = "PickPlaceStoveToCounter"
SUBTASK_COUNTER = "PickPlaceCounterToCounter"
SUBTASK_NAV = "NavigateKitchen"
SUBTASK_PICK = "Pick"
SUBTASK_TILT = "Tilt"
SUBTASK_PLACE = "Place"

DONE_LANG = "task complete"
PLATE_LM = "the plate next to the stove"


@register_labeler
class BalancedMealPrepLabeler(BaseSubtaskLabeler):
    """Labeler for the BalancedMealPrep composite task.

    Task (mobile, usually local): a meat sits in a pan on the stove, a vegetable in a bowl on the
    counter, and a plate on the counter. Put both onto the plate. The robot starts at the stove.

    Doc-canonical decomposition (2 moves), emitted in first-motion order. Each item is either a
    direct grasp:
      PickPlaceStoveToCounter/CounterToCounter [pick]  -> pick up the {item} from the {pan|bowl}
      (NavigateKitchen [navigate])                     -> carry the {item} to the plate ... (adaptive)
      PickPlaceStoveToCounter/CounterToCounter [place] -> place the {item} on the plate next to the stove
    or a tilt-pour of its container (detected when the container's up-axis rotates far while lifted):
      Pick [pick]   -> pick up the {pan|bowl} from the {stove|counter}
      Tilt [tilt]   -> tilt the {pan|bowl} to pour the {item} onto the plate next to the stove
      (Place [place] -> place the {pan|bowl} back ...  only if the container is returned to rest)
    plus a terminal done. `item` names come from get_obj_lang.
    """

    ENV_NAME = "BalancedMealPrep"

    # item -> (container body, container word, direct pick_from, container pick_from, direct subtask)
    ITEMS = {
        "meat": ("meat_container", "pan", "the pan", "the stove", SUBTASK_STOVE),
        "vegetable": ("vegetable_container", "bowl", "the bowl", "the counter", SUBTASK_COUNTER),
    }

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 12
    REST_W = 15
    BASE_MOVE_TH = 0.008
    MIN_NAV_DIST = 0.25
    # tilt-pour detection (mirrors PanTransfer)
    TILT_TH = 45.0       # container up-axis rotation (deg) to count as a pour
    CONT_DISP_TH = 0.15  # ...and the container must also be carried this far
    LIFT_TH = 0.05
    CONT_RETURN_TH = 0.12  # container ends within this of its start -> it was returned -> Place

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.items = [o for o in self.ITEMS if o in env.obj_body_id]
        self.conts = {o: self.ITEMS[o][0] for o in self.items if self.ITEMS[o][0] in env.obj_body_id}
        bodies = set(self.items) | set(self.conts.values())
        self.obj_body_ids = {o: env.obj_body_id[o] for o in bodies}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.word = {o: env.get_obj_lang(o) for o in self.items}
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])
        self.extra_subtasks = {SUBTASK_COUNTER, SUBTASK_PICK, SUBTASK_TILT, SUBTASK_PLACE}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
        }
        for o in self.obj_body_ids:
            d[o] = env.sim.data.body_xpos[self.obj_body_ids[o]].copy()
        for o, c in self.conts.items():
            q = np.array(env.sim.data.body_xquat[self.obj_body_ids[c]])
            d[c + "_up"] = T.quat2mat(q[[1, 2, 3, 0]])[:, 2].copy()
            try:
                d[o + "_onplate"] = int(OU.check_obj_in_receptacle(env, o, "plate"))
            except Exception:
                d[o + "_onplate"] = 0
            try:
                d[o + "_incont"] = int(OU.check_obj_in_receptacle(env, o, c))
            except Exception:
                d[o + "_incont"] = 0
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

    def resolve(self, history):
        T_ = len(history)
        opos = {o: np.array([h[o] for h in history]) for o in self.obj_body_ids}
        grip = np.array([h["grip"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T_ >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T_)

        first_move = {}
        for o in self.items:
            rest = np.median(opos[o][: self.REST_W], axis=0)
            dsp = np.linalg.norm(opos[o] - rest, axis=1)
            fm = self._sustained_start(dsp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T_
        order = sorted([o for o in self.items if first_move[o] < T_], key=lambda o: first_move[o])

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T_))
            segments.append((end, name, stage, lang))
            cursor = end

        review = []
        n_moved = 0
        for o in order:
            cont_body, cont_word, direct_from, cont_from, sub = self.ITEMS[o]
            word = self.word[o]
            has_cont = o in self.conts

            # tilt-pour test: container up-axis rotates far AND it is carried
            tilt_pour = False
            cup = til = None
            if has_cont:
                cup = opos.get(cont_body + "_up")
                if cup is None:
                    cup = np.array([h[cont_body + "_up"] for h in history])
                til = np.degrees(np.arccos(np.clip(cup @ cup[0], -1.0, 1.0)))
                cdisp = np.linalg.norm(opos[cont_body] - opos[cont_body][0], axis=1)
                tilt_pour = (float(til.max()) > self.TILT_TH) and (float(cdisp.max()) > self.CONT_DISP_TH)

            if tilt_pour:
                self._emit_tilt_pour(o, cont_body, cont_word, cont_from, word, history,
                                     opos, grip, til, cursor, emit)
            else:
                self._emit_direct(o, direct_from, sub, word, opos, grip, base_xy, binst, cursor, emit)
            n_moved += 1
            cursor = segments[-1][0]

        emit(T_, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        if n_moved != 2:
            review.append(f"only {n_moved}/2 items moved")

        result = self.build_result(T_, segments, debug={
            "order": order, "n_moved": n_moved, "n_segments": len(segments),
        })
        result.review = review
        return result

    def _emit_direct(self, o, pick_from, sub, word, opos, grip, base_xy, binst, cursor, emit):
        T_ = len(grip)
        r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
        dsp = np.linalg.norm(opos[o] - r, axis=1)
        t_pick = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
        if t_pick is None:
            mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
            t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T_ - 1)
        t_pick = max(t_pick, cursor)
        gd = np.linalg.norm(grip - opos[o], axis=1)
        hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T_)]))
        t_placed = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_placed is None:
            t_placed = T_ - 1
        emit(t_pick, sub, STAGE_PICK, f"pick up the {word} from {pick_from}")
        moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
        t_arrive = min(t_pick + int(moving[-1]) + 1, t_placed) if moving.size else t_pick
        nav_disp = (float(np.max(np.linalg.norm(base_xy[t_pick: t_arrive] - base_xy[t_pick], axis=1)))
                    if base_xy is not None and t_arrive > t_pick else 0.0)
        if t_arrive > cursor and nav_disp >= self.MIN_NAV_DIST:
            emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry the {word} to {PLATE_LM}")
        emit(t_placed, sub, STAGE_PLACE, f"place the {word} on {PLATE_LM}")

    def _emit_tilt_pour(self, o, cont_body, cont_word, cont_from, word, history, opos, grip, til, cursor, emit):
        T_ = len(grip)
        cpos = opos[cont_body]
        z0 = float(np.median(cpos[max(cursor - self.REST_W, 0): cursor + self.REST_W, 2])) if T_ else 0.0
        lifted = (cpos[:, 2] - z0) > self.LIFT_TH
        # Pick -> Tilt: the container's first sustained lift-off after the segment start
        lift_off = self._sustained_start(lifted, cursor, 3)
        if lift_off is None:
            lift_off = min(cursor + 1, T_ - 1)
        # Tilt -> (done/Place): the item is delivered onto the plate (on plate, out of the container)
        onp = np.array([h[o + "_onplate"] for h in history])
        inc = np.array([h[o + "_incont"] for h in history])
        deliv = np.where((onp[lift_off:] > 0) & (inc[lift_off:] == 0))[0]
        if deliv.size:
            pour_end = lift_off + int(deliv[0])
        else:
            # fallback: end of the sustained high-tilt stretch
            hi = np.where(til[lift_off:] > self.TILT_TH)[0]
            pour_end = lift_off + int(hi[-1]) + 1 if hi.size else min(lift_off + 1, T_ - 1)
        pour_end = int(min(max(pour_end, lift_off + 1), T_ - 1))

        emit(lift_off, SUBTASK_PICK, STAGE_PICK, f"pick up the {cont_word} from {cont_from}")
        emit(pour_end, SUBTASK_TILT, STAGE_TILT,
             f"tilt the {cont_word} to pour the {word} onto {PLATE_LM}")
        # Place only if the container is returned close to where it started
        returned = float(np.linalg.norm(cpos[-1] - cpos[0])) < self.CONT_RETURN_TH
        if returned:
            gdc = np.linalg.norm(grip - cpos, axis=1)
            hold = float(np.median(gdc[lift_off: min(lift_off + self.HOLD_W, T_)]))
            rel = self._sustained_start(gdc > hold + self.RELEASE_MARGIN, pour_end + 1, self.RELEASE_SUSTAIN)
            rel = rel if rel is not None else T_ - 1
            if rel > pour_end:
                emit(rel, SUBTASK_PLACE, STAGE_PLACE, f"place the {cont_word} back {cont_from.replace('the ', 'on the ')}")

    def validate(self, result):
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        stages = result.subtask_stage[firsts]
        # each item is either a direct grasp (pick+place) or a tilt-pour (pick+tilt[+place]).
        npick = int(np.sum(stages == STAGE_PICK))
        if npick != 2:
            issues.append(f"expected 2 item picks, found {npick}")
        n_tilt = int(np.sum(stages == STAGE_TILT))
        n_place = int(np.sum(stages == STAGE_PLACE))
        # a direct item ends in a place; a tilt item ends in a tilt (+ optional place)
        if n_tilt + n_place < 2:
            issues.append(f"expected >=2 terminal (place/tilt) stages, found {n_tilt + n_place}")
        return issues
