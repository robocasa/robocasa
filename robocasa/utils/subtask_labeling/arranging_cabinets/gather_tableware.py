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


# The lone mug (glass3) is carried from the second cabinet into the main cabinet,
# split across a navigation: pick -> NavigateKitchen -> place. No exact bank name
# covers cabinet -> cabinet, so the PickPlace name is a declared extension; the
# name is unchanged across the pick/place stages.
SUBTASK_MUG = "PickPlaceCabinetToCabinet"

# Optional final steps (only in some demos): nudge the bowl aside so it sits on
# the opposite side from the mugs, and/or tidy the two resident mugs to one side
# of the cabinet. Pushed, not grasped; single stage each. Declared extensions.
SUBTASK_BOWL = "MoveBowlAside"
SUBTASK_MUGS = "MoveMugsAside"

# NavigateKitchen (bank atomic): the initial drive to the second cabinet, and the
# carry back to the main cabinet.
SUBTASK_NAV = "NavigateKitchen"
NAV_TO_OTHER_LANG = "navigate to the other open cabinet holding the lone mug"

DONE_LANG = "task complete"


@register_labeler
class GatherTablewareLabeler(BaseSubtaskLabeler):
    """Labeler for the GatherTableware composite task.

    Task: gather all the tableware into one cabinet, glasses on one side and the
    bowl on the other. The two cabinets start open. Three mugs (glass1, glass2)
    and a bowl already sit in the main cabinet; one lone mug (glass3) sits in the
    second cabinet. The robot drives to the second cabinet, takes that mug, and
    brings it back to the main cabinet with the others (the bowl and the two mugs
    there only get minor positioning nudges).

    Doc-canonical decomposition (4 subtasks):
      1. NavigateKitchen                       -> drive to the second cabinet
      2. PickPlaceCabinetToCabinet [pick]      -> take the lone mug
      3. NavigateKitchen                       -> carry it to the main cabinet
      4. PickPlaceCabinetToCabinet [place]     -> set it with the other mugs
    plus a terminal done. The single transported object is glass3 (verified the
    consistent mover across the dataset; the other objects move < ~0.2 m).

    Boundaries:
      - t_pick:  the mug first moves DEPART_TOL from its rest pose in the second
                 cabinet while held (transport begun). Direction-agnostic.
      - t_place: the mug settles within PLACE_TOL of its final pose with the
                 gripper released, sustained.
      - NavigateKitchen segments are carved where the base drives at least
        MIN_NAV_DIST (the drive to the second cabinet; the carry back). The
        leading nav is capped so the reach+grasp keeps >= MIN_PICK_FRAMES.
        Validation is structural (one pick + one place; the mug ends in the main
        cabinet).
    """

    ENV_NAME = "GatherTableware"

    DEPART_TOL = 0.12
    GRASP_TH = 0.10
    PLACE_TOL = 0.12
    RELEASE_TH = 0.12
    RELEASE_SUSTAIN = 5
    REST_W = 10

    NAV_MOVING_TH = 0.004
    NAV_SMOOTH_W = 7
    MIN_NAV_DIST = 0.65
    MIN_PICK_FRAMES = 20

    # Bowl-move detection: the bowl is repositioned only in some demos. It counts
    # as a real move when it displaces past BOWL_MOVE_TOL from its rest pose AFTER
    # the mug is placed (so an incidental bump during the mug placement doesn't
    # trigger it). BOWL_SETTLE_TOL bounds the end of the push.
    BOWL_MOVE_TOL = 0.12
    BOWL_SETTLE_TOL = 0.04
    # The two resident mugs are tidied to one side in some demos (pushed, not
    # grasped). Counts as a move past MUG_MOVE_TOL after the lone mug is placed.
    MUG_MOVE_TOL = 0.10

    # The lone mug usually travels 0.6-7.7 m. Below this it was never gathered
    # (sort-only demo: only the tidy-up steps are labelled).
    GATHER_MIN_DISP = 0.20

    MUG = "glass3"  # the lone mug in the second cabinet
    OTHER_MUGS = ["glass1", "glass2"]

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.mug_body_id = env.obj_body_id[self.MUG]
        self.other_mug_body_ids = [env.obj_body_id[n] for n in self.OTHER_MUGS]
        self.bowl_body_id = env.obj_body_id["bowl"]
        try:
            self.mug_lang = env.get_obj_lang(self.MUG)
        except Exception:
            self.mug_lang = "mug"
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.base_body_id = env.sim.model.body_name2id("mobilebase0_base")
        except Exception:
            self.base_body_id = None
        # Agentview camera (matches the rendered view) for the bowl's left/right
        # push direction.
        try:
            self.cam_id = env.sim.model.camera_name2id("robot0_agentview_center")
        except Exception:
            self.cam_id = None
        # Main-cabinet reference (where the mug ends up). Used only for the
        # structural check; the lang is identity-independent.
        self.main_cab_xy = None
        try:
            self.main_cab_xy = np.array(env.cab.pos)[:2]
        except Exception:
            self.main_cab_xy = None
        self.expected_done_idx = None
        self.extra_subtasks = {SUBTASK_MUG, SUBTASK_BOWL, SUBTASK_MUGS}

    def per_step(self, env, t: int) -> dict:
        base_xy = (
            env.sim.data.body_xpos[self.base_body_id][:2].copy()
            if self.base_body_id is not None
            else np.array([np.nan, np.nan])
        )
        cam_right = (
            env.sim.data.cam_xmat[self.cam_id].reshape(3, 3)[:, 0].copy()
            if self.cam_id is not None
            else np.array([np.nan, np.nan, np.nan])
        )
        return {
            "mug": env.sim.data.body_xpos[self.mug_body_id].copy(),
            "other_mugs": np.stack(
                [env.sim.data.body_xpos[b].copy() for b in self.other_mug_body_ids]
            ),
            "bowl": env.sim.data.body_xpos[self.bowl_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": base_xy,
            "cam_right": cam_right,
        }

    # ---- event helpers -------------------------------------------------

    def _release_time(self, mug, grip, T):
        final = np.median(mug[-self.REST_W:], axis=0)
        settled = np.linalg.norm(mug - final, axis=1) < self.PLACE_TOL
        released = np.linalg.norm(mug - grip, axis=1) > self.RELEASE_TH
        ok = settled & released
        run = 0
        for t in range(T):
            if ok[t]:
                run += 1
                if run >= self.RELEASE_SUSTAIN:
                    return t - self.RELEASE_SUSTAIN + 1
            else:
                run = 0
        w = np.where(settled)[0]
        return int(w[0]) if w.size else T - 1

    def _depart_time(self, mug, grip, t_release):
        start = np.median(mug[: self.REST_W], axis=0)
        moved = np.linalg.norm(mug - start, axis=1) > self.DEPART_TOL
        held = np.linalg.norm(mug - grip, axis=1) < self.GRASP_TH
        end = max(t_release, 1)
        cand = np.where((moved & held)[:end])[0]
        if cand.size:
            return int(cand[0])
        cand = np.where(moved[:end])[0]
        return int(cand[0]) if cand.size else int(t_release)

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
        mug = np.array([h["mug"] for h in history])
        other_mugs = np.array([h["other_mugs"] for h in history])  # (T, 2, 3)
        bowl = np.array([h["bowl"] for h in history])
        grip = np.array([h["grip"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history])
        cam_right = np.array([h["cam_right"] for h in history])
        has_base = np.isfinite(base_xy).all()

        if has_base and T >= 2:
            inst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            W = self.NAV_SMOOTH_W
            base_speed = np.convolve(inst, np.ones(W) / W, mode="same")
            base_speed = np.concatenate([base_speed, base_speed[-1:]])
        else:
            base_speed = np.zeros(T)
            base_xy = None

        t_release = self._release_time(mug, grip, T)
        t_depart = self._depart_time(mug, grip, t_release)

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        def maybe_nav(end, lang, cap=None):
            nav_end = self._nav_end(base_speed, base_xy, cursor, end)
            if cap is not None:
                nav_end = min(nav_end, max(cursor, cap))
            if nav_end > cursor:
                emit(nav_end, SUBTASK_NAV, STAGE_NAVIGATE, lang)

        pick_lang = f"pick up the {self.mug_lang} from the cabinet"
        place_lang = f"place the {self.mug_lang} in the cabinet with the other {self.mug_lang}s"
        carry_lang = f"navigate back to the other cabinet where the other {self.mug_lang}s are"

        # Did the lone mug actually get gathered? In a few demos it never moves
        # (it starts and ends in the second cabinet) and the robot only sorts the
        # already-gathered tableware -- those demos have ONLY the tidy-up steps.
        gathered = (
            np.linalg.norm(
                np.median(mug[-self.REST_W:], axis=0) - np.median(mug[: self.REST_W], axis=0)
            )
            >= self.GATHER_MIN_DISP
        )

        if gathered:
            # Drive to the second cabinet -> reach + grasp + lift the mug out.
            maybe_nav(t_depart, NAV_TO_OTHER_LANG, cap=t_depart - self.MIN_PICK_FRAMES)
            emit(t_depart, SUBTASK_MUG, STAGE_PICK, pick_lang)
            # Carry back to the main cabinet -> place with the others.
            maybe_nav(t_release, carry_lang)
            emit(t_release, SUBTASK_MUG, STAGE_PLACE, place_lang)

        # Tidy-up steps (only in some demos): push the bowl to the opposite side,
        # and/or tidy the two resident mugs to one side. Each is a push (not
        # grasped), detected from the object's displacement past its threshold.
        # When the mug was gathered, only moves AFTER the placement count (so an
        # incidental bump during placement isn't a tidy step); in a sort-only demo
        # the whole episode is scanned. Direction (right/left) is the displacement
        # projected onto the agentview camera's right axis.
        tidy_ref = t_release if gathered else 0

        def _side(disp_vec, at_frame):
            cr = cam_right[min(max(at_frame, 0), T - 1)]
            sign = float(np.dot(disp_vec, cr)) if np.isfinite(cr).all() else float(disp_vec[1])
            return "right" if sign > 0 else "left"

        def _push(obj_xyz, tol):
            # obj_xyz: (T,3). Returns (start, end, disp_vec) if it moves past tol
            # after tidy_ref, else None.
            start = np.median(obj_xyz[: self.REST_W], axis=0)
            final = np.median(obj_xyz[-self.REST_W:], axis=0)
            after = np.where((np.linalg.norm(obj_xyz - start, axis=1) > tol)[tidy_ref:])[0]
            if np.linalg.norm(final - start) <= tol or not after.size:
                return None
            s = tidy_ref + int(after[0])
            away = np.linalg.norm(obj_xyz - final, axis=1) > self.BOWL_SETTLE_TOL
            w = np.where(away[tidy_ref:])[0]
            e = tidy_ref + int(w[-1]) + 1 if w.size else T
            return s, e, (final - start)

        end_subs = []  # (start, end, name, lang)

        bowl_push = _push(bowl, self.BOWL_MOVE_TOL)
        if bowl_push is not None:
            s, e, dv = bowl_push
            lang = (
                f"move the bowl to the {_side(dv, s)} to place it on the opposite "
                f"side of the {self.mug_lang}s"
            )
            end_subs.append((s, e, SUBTASK_BOWL, lang))

        # The most-displaced resident mug stands in for the pair.
        om_start = np.median(other_mugs[: self.REST_W], axis=0)   # (2,3)
        om_final = np.median(other_mugs[-self.REST_W:], axis=0)   # (2,3)
        mi = int(np.argmax(np.linalg.norm(om_final - om_start, axis=1)))
        mug_push = _push(other_mugs[:, mi, :], self.MUG_MOVE_TOL)
        if mug_push is not None:
            s, e, dv = mug_push
            lang = f"move the {self.mug_lang}s to the {_side(dv, s)} side of the cabinet"
            end_subs.append((s, e, SUBTASK_MUGS, lang))

        # Emit the tidy-up steps in the order they occur, then done.
        end_subs.sort(key=lambda x: x[0])
        for s, e, name, lang in end_subs:
            e = int(min(max(e, cursor + 1), T - 1))
            if e > cursor:
                emit(e, name, STAGE_EXECUTE, lang)

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        mug_end_xy = np.median(mug[-self.REST_W:], axis=0)[:2]
        ends_in_main = (
            bool(np.linalg.norm(mug_end_xy - self.main_cab_xy) < 0.6)
            if self.main_cab_xy is not None
            else True
        )
        debug = {
            "t_depart": int(t_depart),
            "t_release": int(t_release),
            "gathered": int(gathered),
            "ends_in_main": int(ends_in_main),
            "bowl_moved": int(bowl_push is not None),
            "mugs_moved": int(mug_push is not None),
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
        names = result.subtask_name[firsts]
        n_pick = int(np.sum(stages == STAGE_PICK))
        n_place = int(np.sum(stages == STAGE_PLACE))
        if result.debug.get("gathered") == 1:
            # Standard demo: the lone mug is carried into the main cabinet.
            if n_pick != 1:
                issues.append(f"expected 1 pick stage, found {n_pick}")
            if n_place != 1:
                issues.append(f"expected 1 place stage, found {n_place}")
            if result.debug.get("ends_in_main") == 0:
                issues.append("mug did not end in the main cabinet")
        else:
            # Sort-only demo: the lone mug never moves; only tidy-up steps. Expect
            # no pick/place and at least one tidy step.
            n_tidy = int(np.sum((names == SUBTASK_BOWL) | (names == SUBTASK_MUGS)))
            if n_pick or n_place:
                issues.append(f"sort-only demo but found pick/place ({n_pick}/{n_place})")
            if n_tidy == 0:
                issues.append("sort-only demo with no tidy-up step")
        return issues
