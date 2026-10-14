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

# The croissant is moved counter -> cutting board (both on the counter); the jam
# is retrieved from the open cabinet -> cutting board. PickPlaceCabinetToCounter
# is a bank atomic; PickPlaceCounterToCounter is a declared extension.
SUBTASK_CROISSANT = "PickPlaceCounterToCounter"
SUBTASK_JAM = "PickPlaceCabinetToCounter"
SUBTASK_NAV = "NavigateKitchen"

NAV_TO_BOARD_LANG = "navigate to the cutting board"
NAV_BACK_LANG = "navigate back to the cutting board"
DONE_LANG = "task complete"


@register_labeler
class BreadSelectionLabeler(BaseSubtaskLabeler):
    """Labeler for the BreadSelection composite task.

    Task (mobile): select the croissant from among the pastries on the counter
    and place it on the cutting board, then retrieve the jar of jam from the open
    cabinet and place it alongside the croissant. Two pastry distractors and a
    counter distractor are ignored. The robot starts at the cutting board.

    Doc-canonical decomposition (2 subtasks: croissant, jam). Emitted fine-grained:
      1. PickPlaceCounterToCounter [pick/place]  -> croissant onto the cutting board
      2. PickPlaceCabinetToCounter [pick/place]  -> jam onto the cutting board
    plus a terminal done, with adaptive NavigateKitchen segments inserted where the
    base actually drives. The croissant is on the counter by the robot, so it has
    no navigation. The cabinet is reachable in most layouts (no nav), but in some
    it is across the kitchen -- those demos drive out to the cabinet for the jam
    and back to the cutting board to place it.

    Per object, the pick ends at the grasp (contact-grasp onset) and the place
    ends at the release (contact-grasp end); the dishes-style finger-width signal
    is unreliable here but the contact check is clean for the croissant and jam.
    """

    ENV_NAME = "BreadSelection"

    DEPART_TOL = 0.10
    PLACE_TOL = 0.12
    REST_W = 10

    NAV_MOVING_TH = 0.004
    NAV_SMOOTH_W = 7
    MIN_NAV_DIST = 0.65

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.croissant_body_id = env.obj_body_id["croissant"]
        self.jam_body_id = env.obj_body_id["jam"]
        try:
            self.croissant_lang = env.get_obj_lang("croissant")
        except Exception:
            self.croissant_lang = "croissant"
        try:
            self.jam_lang = env.get_obj_lang("jam")
        except Exception:
            self.jam_lang = "jam"
        self.gripper = env.robots[0].gripper["right"]
        self.croissant_geoms = env.objects["croissant"].contact_geoms
        self.jam_geoms = env.objects["jam"].contact_geoms
        try:
            self.base_body_id = env.sim.model.body_name2id("mobilebase0_base")
        except Exception:
            self.base_body_id = None
        # The place can be a fast set-down on the adjacent cutting board (~20
        # frames), so the short-segment guard is below the usual 30.
        self.min_segment_frames = 15
        self.extra_subtasks = {SUBTASK_CROISSANT}
        self.expected_done_idx = None

    def per_step(self, env, t: int) -> dict:
        base_xy = (
            env.sim.data.body_xpos[self.base_body_id][:2].copy()
            if self.base_body_id is not None
            else np.array([np.nan, np.nan])
        )
        return {
            "croissant": env.sim.data.body_xpos[self.croissant_body_id].copy(),
            "jam": env.sim.data.body_xpos[self.jam_body_id].copy(),
            "base_xy": base_xy,
            "grasp_c": int(
                env._check_grasp(gripper=self.gripper, object_geoms=self.croissant_geoms)
            ),
            "grasp_j": int(
                env._check_grasp(gripper=self.gripper, object_geoms=self.jam_geoms)
            ),
        }

    # ---- event helpers -------------------------------------------------

    def _lift_time(self, obj, T):
        start = np.median(obj[: self.REST_W], axis=0)
        moved = np.linalg.norm(obj - start, axis=1) > self.DEPART_TOL
        cand = np.where(moved[:T])[0]
        return int(cand[0]) if cand.size else 0

    def _grasp_time(self, grasp, lift, lo, hi):
        # Pick -> place at the grasp onset (first contact within (lo, hi]); the
        # lift is the fallback when the contact check never fires.
        seg = np.where(grasp[lo:hi] > 0)[0]
        return int(lo + int(seg[0])) if seg.size else int(lift)

    def _release_time(self, grasp, obj, lo, hi, T):
        # Release = the ungrasp, taken as the LATER of the contact-grasp end and
        # the object's arrival on the cutting board. The contact check is reliable
        # for a firm grasp but for some demos it registers only a 1-2 frame run on
        # the (soft) croissant, which would collapse the place -- the arrival is
        # the robust fallback in that case.
        final = np.median(obj[-self.REST_W:], axis=0)
        far = np.linalg.norm(obj - final, axis=1) >= self.PLACE_TOL
        idxs = np.where(far[:T])[0]
        idxs = idxs[(idxs >= lo) & (idxs < hi)]
        t_arrival = int(idxs[-1]) + 1 if idxs.size else lo + 1
        seg = np.where(grasp[lo:hi] > 0)[0]
        t_grasp_end = int(lo + int(seg[-1]) + 1) if seg.size else lo
        return int(min(max(t_arrival, t_grasp_end), T))

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
        crois = np.array([h["croissant"] for h in history])
        jam = np.array([h["jam"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history])
        grasp_c = np.array([h["grasp_c"] for h in history])
        grasp_j = np.array([h["grasp_j"] for h in history])

        has_base = np.isfinite(base_xy).all()
        if has_base and T >= 2:
            inst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            W = self.NAV_SMOOTH_W
            base_speed = np.convolve(inst, np.ones(W) / W, mode="same")
            base_speed = np.concatenate([base_speed, base_speed[-1:]])
        else:
            base_speed = np.zeros(T)
            base_xy = None

        objs = {
            "croissant": dict(
                pos=crois, grasp=grasp_c, lift=self._lift_time(crois, T),
                name=SUBTASK_CROISSANT, lang=self.croissant_lang,
                src="counter", is_jam=False,
            ),
            "jam": dict(
                pos=jam, grasp=grasp_j, lift=self._lift_time(jam, T),
                name=SUBTASK_JAM, lang=self.jam_lang,
                src="cabinet", is_jam=True,
            ),
        }
        # The object lifted first is handled first (usually the croissant, but
        # some demos retrieve the jam first).
        order = sorted(objs, key=lambda o: objs[o]["lift"])
        first, second = objs[order[0]], objs[order[1]]

        hi1 = max(second["lift"], first["lift"] + 1)
        first["t_grasp"] = self._grasp_time(first["grasp"], first["lift"], 0, hi1)
        first["t_release"] = self._release_time(first["grasp"], first["pos"], 0, hi1, T)
        second["t_grasp"] = self._grasp_time(
            second["grasp"], second["lift"], first["t_release"], T
        )
        second["t_release"] = self._release_time(
            second["grasp"], second["pos"],
            max(first["t_release"], second["t_grasp"]), T, T,
        )

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

        def emit_object(o):
            # The jam comes from the cabinet, so it can require a drive out to the
            # cabinet (before the grasp) and back to the cutting board (during the
            # carry). The croissant is on the counter by the robot.
            if o["is_jam"]:
                maybe_nav(
                    o["t_grasp"],
                    f"navigate to the open cabinet with the {o['lang']}",
                )
                emit(o["t_grasp"], o["name"], STAGE_PICK,
                     f"pick up the {o['lang']} from the cabinet")
                maybe_nav(o["t_release"], NAV_BACK_LANG)
                emit(o["t_release"], o["name"], STAGE_PLACE,
                     f"place the {o['lang']} on the cutting board")
            else:
                emit(o["t_grasp"], o["name"], STAGE_PICK,
                     f"pick up the {o['lang']} from the counter")
                maybe_nav(o["t_release"], NAV_TO_BOARD_LANG)
                emit(o["t_release"], o["name"], STAGE_PLACE,
                     f"place the {o['lang']} on the cutting board")

        emit_object(first)
        emit_object(second)
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        debug = {
            "order": ",".join(order),
            "first_grasp": int(first["t_grasp"]),
            "first_release": int(first["t_release"]),
            "second_grasp": int(second["t_grasp"]),
            "second_release": int(second["t_release"]),
            "base_max": float(np.linalg.norm(base_xy - base_xy[0], axis=1).max())
            if base_xy is not None else 0.0,
            "n_segments": len(segments),
        }
        return self.build_result(T, segments, debug=debug)

    # ---- structural validation ----------------------------------------

    def validate(self, result):
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        stages = result.subtask_stage[firsts]
        n_pick = int(np.sum(stages == STAGE_PICK))
        n_place = int(np.sum(stages == STAGE_PLACE))
        if n_pick != 2:
            issues.append(f"expected 2 pick stages, found {n_pick}")
        if n_place != 2:
            issues.append(f"expected 2 place stages, found {n_place}")
        return issues
