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


# The kettle is moved counter -> tray (both on the counter); the mug is moved
# cabinet -> tray. Each splits into pick/place. PickPlaceCabinetToCounter is a
# bank atomic; PickPlaceCounterToCounter is a declared extension.
SUBTASK_KETTLE = "PickPlaceCounterToCounter"
SUBTASK_MUG = "PickPlaceCabinetToCounter"

# Close the (double-door) cabinet at the end. The bank has OpenCabinet but no
# close variant, so CloseCabinet is a declared extension. Single stage.
SUBTASK_CLOSE = "CloseCabinet"
CLOSE_LANG = "close the cabinet doors"

DONE_LANG = "task complete"


@register_labeler
class ArrangeTeaLabeler(BaseSubtaskLabeler):
    """Labeler for the ArrangeTea composite task.

    Task (non-mobile): set up a tea tray. Put the kettle (from the counter) and
    the mug (from the open cabinet) onto the tray, then close the cabinet doors.

    Doc-canonical decomposition (3 subtasks):
      1. PickPlaceCounterToCounter [pick/place]  -> kettle onto the tray
      2. PickPlaceCabinetToCounter [pick/place]  -> mug onto the tray
      3. CloseCabinet                            -> close the cabinet doors
    plus a terminal done. The two pick-places are auto-ordered by which object
    settles on the tray first (teleop order varies -- often the mug first), and
    the close is always last.

    Per object: t_pick = it departs its rest while held (displacement fallback for
    the kettle, which is grasped loosely by its handle); t_place = it settles on
    the tray with the gripper released. t_close = the cabinet's last open frame,
    +1. Validation is structural (two picks + two places, one CloseCabinet; the
    cabinet ends closed).
    """

    ENV_NAME = "ArrangeTea"

    DEPART_TOL = 0.10
    GRASP_TH = 0.13
    PLACE_TOL = 0.10
    RELEASE_TH = 0.12
    # The kettle is grasped by its handle, so the gripper-to-body distance is
    # large (~0.18) even while held -- an absolute release threshold fires as soon
    # as it settles, before the robot lets go. Detect release RELATIVE to the
    # grasp distance instead: the gripper has retracted RETRACT_MARGIN past the
    # tightest grasp it achieved on this object.
    RETRACT_MARGIN = 0.10
    RELEASE_SUSTAIN = 5
    REST_W = 10

    CLOSE_TH = 0.05

    # name -> (object key, lang word, subtask name, source phrase)
    OBJECTS = [
        ("obj2", "kettle", SUBTASK_KETTLE, "the counter"),
        ("obj", "mug", SUBTASK_MUG, "the cabinet"),
    ]

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_keys = [o[0] for o in self.OBJECTS]
        self.obj_lang = []
        for key, fallback, _, _ in self.OBJECTS:
            try:
                self.obj_lang.append(env.get_obj_lang(key))
            except Exception:
                self.obj_lang.append(fallback)
        self.obj_body_ids = [env.obj_body_id[k] for k in self.obj_keys]
        # Cabinet door hinges (model joints under the cabinet fixture's name).
        cab = getattr(env, "cab", None)
        nm = getattr(cab, "name", None)
        self.door_addrs = []
        if nm is not None:
            for j in range(env.sim.model.njnt):
                try:
                    jn = env.sim.model.joint_id2name(j)
                except Exception:
                    jn = None
                if jn and jn.startswith(nm) and "doorhinge" in jn:
                    self.door_addrs.append(env.sim.model.jnt_qposadr[j])
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.expected_done_idx = None
        self.extra_subtasks = {SUBTASK_KETTLE, SUBTASK_CLOSE}

    def per_step(self, env, t: int) -> dict:
        door = (
            sum(abs(env.sim.data.qpos[a]) for a in self.door_addrs)
            if self.door_addrs
            else 0.0
        )
        return {
            "objs": np.stack(
                [env.sim.data.body_xpos[b].copy() for b in self.obj_body_ids]
            ),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "door": float(door),
        }

    # ---- event helpers -------------------------------------------------

    def _release_time(self, obj, grip, T):
        final = np.median(obj[-self.REST_W:], axis=0)
        settled = np.linalg.norm(obj - final, axis=1) < self.PLACE_TOL
        gdist = np.linalg.norm(obj - grip, axis=1)
        # Release threshold = the gripper has pulled away from its tightest grasp
        # (handle-robust), with an absolute floor.
        grasp_dist = float(np.min(gdist))
        thr = max(self.RELEASE_TH, grasp_dist + self.RETRACT_MARGIN)
        released = gdist > thr
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

    def _depart_time(self, obj, grip, t_release):
        start = np.median(obj[: self.REST_W], axis=0)
        moved = np.linalg.norm(obj - start, axis=1) > self.DEPART_TOL
        held = np.linalg.norm(obj - grip, axis=1) < self.GRASP_TH
        end = max(t_release, 1)
        cand = np.where((moved & held)[:end])[0]
        if cand.size:
            return int(cand[0])
        cand = np.where(moved[:end])[0]
        return int(cand[0]) if cand.size else int(t_release)

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        objs = np.array([h["objs"] for h in history])  # (T, 2, 3)
        grip = np.array([h["grip"] for h in history])
        door = np.array([h["door"] for h in history])

        ops = []
        for i, (key, _, name, src) in enumerate(self.OBJECTS):
            o = objs[:, i, :]
            t_rel = self._release_time(o, grip, T)
            t_dep = self._depart_time(o, grip, t_rel)
            ops.append(dict(idx=i, lang=self.obj_lang[i], name=name, src=src,
                            dep=t_dep, rel=t_rel))
        order = sorted(ops, key=lambda o: o["rel"])

        # Close = last open frame, +1.
        above = np.where(door > self.CLOSE_TH)[0]
        t_close = min(int(above[-1]) + 1, T) if above.size else T

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        for op in order:
            t_dep = max(op["dep"], cursor)
            t_rel = max(op["rel"], t_dep)
            emit(t_dep, op["name"], STAGE_PICK,
                 f"pick up the {op['lang']} from {op['src']}")
            emit(t_rel, op["name"], STAGE_PLACE,
                 f"place the {op['lang']} on the tray")
        emit(t_close, SUBTASK_CLOSE, STAGE_EXECUTE, CLOSE_LANG)

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        closed = bool(door[-1] < self.CLOSE_TH) if T else True
        debug = {
            "order": np.array([o["idx"] for o in order]),
            "t_close": int(t_close),
            "closed_at_end": int(closed),
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
        n_close = int(np.sum(names == SUBTASK_CLOSE))
        if n_pick != 2:
            issues.append(f"expected 2 pick stages, found {n_pick}")
        if n_place != 2:
            issues.append(f"expected 2 place stages, found {n_place}")
        if n_close != 1:
            issues.append(f"expected 1 CloseCabinet, found {n_close}")
        if result.debug.get("closed_at_end") == 0:
            issues.append("cabinet did not end closed")
        return issues
