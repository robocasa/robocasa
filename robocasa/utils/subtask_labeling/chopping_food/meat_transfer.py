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


# Open the (double-door) cabinet, take the container out and set it on the counter,
# then pick up the raw meat from the counter and drop it into the container.
# OpenCabinet + PickPlaceCabinetToCounter + NavigateKitchen are bank atomics; the
# meat's counter->container move has no exact bank name (the container is a small
# object, not a fixture) so it is an extra_subtask.
SUBTASK_OPEN = "OpenCabinet"
SUBTASK_CONT = "PickPlaceCabinetToCounter"      # container: cabinet -> counter
SUBTASK_MEAT = "PickPlaceCounterToContainer"    # meat: counter -> the container
SUBTASK_NAV = "NavigateKitchen"

OPEN_LANG = "open the cabinet doors"
DONE_LANG = "task complete"


@register_labeler
class MeatTransferLabeler(BaseSubtaskLabeler):
    """Labeler for the MeatTransfer composite task.

    Task: a container (a pan, bowl, or tupperware) sits inside a closed double-door
    cabinet; a piece of raw meat sits on the counter next to the cabinet (where the
    robot starts). Open the cabinet, take the container out and set it on the
    counter, then place the meat into the container.

    Doc-canonical count = 3 (`mobile: Yes`): OpenCabinet + the container's
    cabinet->counter pick-place + the meat's counter->container pick-place. The
    container move is INTERRUPTED by a short navigation from the cabinet to the
    counter (adaptive -- cabinet and counter are adjacent, so many demos transfer
    the container mostly with the arm). Segments: open + container[pick,(nav),place]
    + meat[pick,place] + done; expected_done_idx is set per demo (6 without a nav,
    7 with).

    Boundaries (each from its OWN signal):
      - t_open:   the cabinet door (max fraction over the two hinge leaves) reaches
                  OPEN_ABS of fully-open and stays -> OpenCabinet ends. Emitted only
                  if a door actually swings (a few demos may spawn it ajar).
      - container: t_cpick = the container first displaces MOVE_TOL from its rest
                  (in-cabinet) pose (grasped & extracted -- horizontal, so
                  displacement not a Z-lift). The nav is the drive to the counter,
                  split at the base stop (t_carrive = last base-moving frame). The
                  place runs to the gripper RELEASE.
      - meat:     t_mpick = the meat first displaces MOVE_TOL from its rest (counter)
                  pose, after the container is placed. t_mplaced = the gripper
                  releases the meat into the container.
    """

    ENV_NAME = "MeatTransfer"

    OPEN_MIN = 0.25  # peak door fraction (of RANGE) below this == the leaf never opened
    OPEN_FRAC = 0.95  # a leaf is "open" at this fraction of its OWN peak (a leaf may
    # only swing part-way, e.g. to 0.75 of its range, so gate on its own peak)
    OPEN_SUSTAIN = 5
    MOVE_TOL = 0.12  # object displaced this far from rest == grasped & being carried
    MOVE_SUSTAIN = 5
    HOLD_W = 40  # window after a pick used to estimate the carry (hold) grip dist
    RELEASE_MARGIN = 0.06  # grip->obj rising this far above carry level == let go
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    BASE_MOVE_TH = 0.003  # per-frame base speed above this == the base is driving
    MIN_NAV_DIST = 0.20  # peak base displacement over the container carry to emit a nav
    REST_W = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.cont_body_id = env.obj_body_id["container"]
        self.meat_body_id = env.obj_body_id["meat"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.cont_lang = env.get_obj_lang("container")
        except Exception:
            self.cont_lang = "container"
        try:
            self.meat_lang = env.get_obj_lang("meat")
        except Exception:
            self.meat_lang = "meat"

        # cabinet door hinge joints (double-door -> up to 2 leaves), each tracked as
        # a fraction of its own fully-open range (see FillKettle).
        cab = env.cab
        joint_names = list(getattr(cab, "door_joint_names", []) or [])
        if not joint_names:
            pref = cab.name
            joint_names = [
                n for n in env.sim.model.joint_names
                if n and n.startswith(pref) and ("door" in n.lower() or "hinge" in n.lower())
            ]
        self.door_adrs, self.door_scales = [], []
        for j in joint_names:
            try:
                adr = env.sim.model.get_joint_qpos_addr(j)
                jid = env.sim.model.joint_name2id(j)
                lo, hi = env.sim.model.jnt_range[jid]
            except Exception:
                continue
            scale = max(abs(float(hi)), abs(float(lo)), 1e-6)
            self.door_adrs.append(adr)
            self.door_scales.append(scale)

        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])

        self.extra_subtasks = {SUBTASK_MEAT}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        doors = np.array(
            [abs(float(env.sim.data.qpos[a])) / s
             for a, s in zip(self.door_adrs, self.door_scales)]
        )
        return {
            "doors": doors,
            "container": env.sim.data.body_xpos[self.cont_body_id].copy(),
            "meat": env.sim.data.body_xpos[self.meat_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array(
                [float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]
            ),
        }

    # ---- helpers -------------------------------------------------------

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

    def _release(self, opos, grip, t_pick, T):
        gd = np.linalg.norm(grip - opos, axis=1)
        hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
        t_rel = self._sustained_start(
            gd > hold_gd + self.RELEASE_MARGIN, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN
        )
        return T - 1 if t_rel is None else t_rel

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        doorv = np.array([h["doors"] for h in history]) if self.door_adrs else np.zeros((T, 0))
        cont = np.array([h["container"] for h in history])
        meat = np.array([h["meat"] for h in history])
        grip = np.array([h["grip"] for h in history])
        base_xy = (
            np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        )
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        # container: grasp & extract from the cabinet (horizontal -> displacement).
        # The container is stationary in the closed cabinet until grasped, so its
        # pick can be found independently of the door-opening (search from t=0).
        crest = np.median(cont[: self.REST_W], axis=0)
        cdisp = np.linalg.norm(cont - crest, axis=1)
        t_cpick = self._sustained_start(cdisp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
        if t_cpick is None:
            mv = np.where(cdisp >= self.MOVE_TOL)[0]
            t_cpick = int(mv[0]) if mv.size else T - 1
        t_cplaced = self._release(cont, grip, t_cpick, T)

        # open the cabinet: the double-door is opened one leaf at a time, and a leaf
        # may only swing PART-way (e.g. to ~0.75 of its range) -- so a fixed
        # fraction-of-RANGE gate misses the second leaf entirely and ends OpenCabinet
        # at the first. Gate each leaf on its OWN peak instead: a leaf is open once it
        # reaches OPEN_FRAC of the maximum openness it ever attains, and OpenCabinet
        # ends when the LAST leaf that opens gets there. Only leaves that actually
        # swing (peak fraction of range >= OPEN_MIN) count.
        door_peak = float(doorv.max()) if doorv.size else 0.0
        has_open = door_peak >= self.OPEN_MIN
        if has_open and doorv.shape[1]:
            open_frames = []
            for i in range(doorv.shape[1]):
                peak_i = float(doorv[:, i].max())
                if peak_i < self.OPEN_MIN:
                    continue  # this leaf stayed (essentially) shut
                fr = self._sustained_start(
                    doorv[:, i] >= self.OPEN_FRAC * peak_i, 0, self.OPEN_SUSTAIN
                )
                if fr is not None:
                    open_frames.append(fr)
            t_open = max(open_frames) if open_frames else 0
            t_open = int(min(t_open, t_cpick))
        else:
            t_open = 0

        # nav (container carry): base drive between extraction and set-down, split
        # at the base stop.
        moving = np.where(binst[t_cpick: t_cplaced] > self.BASE_MOVE_TH)[0]
        t_carrive = min(t_cpick + int(moving[-1]) + 1, t_cplaced) if moving.size else t_cpick
        if base_xy is not None and t_carrive > t_cpick:
            off = base_xy[t_cpick: t_carrive] - base_xy[t_cpick]
            nav_disp = float(np.max(np.linalg.norm(off, axis=1)))
        else:
            nav_disp = 0.0

        # meat: grasp from the counter (after the container is set down) and drop
        # into the container.
        mrest = np.median(meat[: self.REST_W], axis=0)
        mdisp = np.linalg.norm(meat - mrest, axis=1)
        t_mpick = self._sustained_start(mdisp >= self.MOVE_TOL, t_cplaced, self.MOVE_SUSTAIN)
        if t_mpick is None:
            mv = np.where(mdisp[t_cplaced:] >= self.MOVE_TOL)[0]
            t_mpick = t_cplaced + int(mv[0]) if mv.size else max(t_cplaced, T - 1)
        t_mpick = max(t_mpick, t_cplaced)
        t_mplaced = self._release(meat, grip, t_mpick, T)

        # -- assemble --
        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        if has_open:
            emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, OPEN_LANG)
        emit(t_cpick, SUBTASK_CONT, STAGE_PICK,
             f"pick up the {self.cont_lang} from the cabinet")
        if t_carrive > cursor and nav_disp >= self.MIN_NAV_DIST:
            emit(t_carrive, SUBTASK_NAV, STAGE_NAVIGATE,
                 f"carry the {self.cont_lang} to the counter next to the {self.meat_lang}")
        emit(t_cplaced, SUBTASK_CONT, STAGE_PLACE,
             f"place the {self.cont_lang} on the counter")
        emit(t_mpick, SUBTASK_MEAT, STAGE_PICK,
             f"pick up the {self.meat_lang} from the counter")
        emit(t_mplaced, SUBTASK_MEAT, STAGE_PLACE,
             f"place the {self.meat_lang} in the {self.cont_lang}")
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        review = []
        if not has_open:
            review.append(
                f"cabinet door never opened (peak {door_peak:.2f}) -- no OpenCabinet; verify"
            )

        debug = {
            "t_open": t_open, "t_cpick": t_cpick, "t_carrive": t_carrive,
            "t_cplaced": t_cplaced, "t_mpick": t_mpick, "t_mplaced": t_mplaced,
            "nav_disp": round(nav_disp, 2), "door_peak": round(door_peak, 2),
            "n_segments": len(segments),
        }
        result = self.build_result(T, segments, debug=debug)
        result.review = review
        return result

    # ---- structural validation ----------------------------------------

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
        if count(SUBTASK_CONT, STAGE_PICK) != 1:
            issues.append(f"expected 1 container pick, found {count(SUBTASK_CONT, STAGE_PICK)}")
        if count(SUBTASK_CONT, STAGE_PLACE) != 1:
            issues.append(f"expected 1 container place, found {count(SUBTASK_CONT, STAGE_PLACE)}")
        if count(SUBTASK_MEAT, STAGE_PICK) != 1:
            issues.append(f"expected 1 meat pick, found {count(SUBTASK_MEAT, STAGE_PICK)}")
        if count(SUBTASK_MEAT, STAGE_PLACE) != 1:
            issues.append(f"expected 1 meat place, found {count(SUBTASK_MEAT, STAGE_PLACE)}")
        if count(SUBTASK_NAV) not in (0, 1):
            issues.append(f"expected 0-1 NavigateKitchen, found {count(SUBTASK_NAV)}")
        return issues
