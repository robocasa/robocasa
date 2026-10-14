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


# Subtasks 1-3: add butter + two sugar cubes to the stand-mixer bowl. Each is a
# PickPlaceCounterToStandMixer (bank atomic; splits into pick / place stages).
# The order the three are added VARIES per demo, so it is resolved dynamically.
SUBTASK_ADD = "PickPlaceCounterToStandMixer"

# Subtask 4: close the stand-mixer head (bank atomic, single EXECUTE stage). The
# head spawns OPEN (set_head_pos default 1.0) so the robot can drop items into
# the bowl, then closes it before turning the mixer on.
SUBTASK_CLOSE = "CloseStandMixerHead"
CLOSE_LANG = "close the stand mixer head"

# Subtask 5: turn the speed dial to start the mixer. No bank atomic covers this
# (the bank has TurnOnBlender/Toaster/... but no stand mixer), so it is a declared
# extension.
SUBTASK_TURNON = "TurnOnStandMixer"
TURNON_LANG = "turn on the stand mixer"

DONE_LANG = "task complete"


@register_labeler
class MakeCheesecakeFillingLabeler(BaseSubtaskLabeler):
    """Labeler for the MakeCheesecakeFilling composite task.

    Task: add butter and two sugar cubes to the stand-mixer bowl, close the mixer
    head, then turn the speed dial to start mixing the frosting. Everything is
    within arm reach of the mixer (the ingredients spawn on the same counter), so
    the base never drives -- there is no NavigateKitchen subtask.

    Doc-canonical count = 5 (`mobile: Yes`). Decomposition, in physical order:
      1-3. PickPlaceCounterToStandMixer [pick]/[place]  -> add each ingredient to
           the bowl. There are three (butter, sugar_0, sugar_1); the ORDER they
           are added varies per demo and is resolved from when each is released
           into the bowl.
      4.   CloseStandMixerHead          [execute]  -> swing the head down/closed
      5.   TurnOnStandMixer             [execute]  -> turn the speed dial on
    plus a terminal 'done'. Three PickPlace subtasks (2 stages each) + close +
    turn-on + done = 9 segments; expected_done_idx = 8 (fixed -- no nav).

    Each ingredient's pick/place is resolved from its OWN grasp trajectory:
      - The gripper->object distance identifies the contiguous "carry" grasp run
        (the held run during which the object is lifted the highest). Because all
        three end up clustered in the same bowl, keying off each object's own
        grasp run -- rather than a global "gripper near object" test -- keeps a
        later placement from contaminating an earlier object's boundaries.
      - grasp   = start of that run; release = end of that run (the deposit).
      - t_lift  = first sustained frame the object rises LIFT_TOL above its rest
                  (counter) height -> the pick->place boundary.
    Ordering the three objects by release gives the add order.

    - t_head_closed: first frame after the last release that the head joint drops
                  below HEAD_CLOSED_ABS (spawns open ~1.57 rad, closes toward 0).
    - t_knob_on:  first frame after the head closes that the speed-dial joint
                  reaches KNOB_ON_FRAC of its peak -> the mixer is running.
    """

    ENV_NAME = "MakeCheesecakeFilling"

    OBJECTS = ("butter", "cream_cheese", "sugar")

    # --- grasp / pick / place ---
    # Grasp is detected per object with an ADAPTIVE threshold: max(GRASP_TH,
    # min_grip + GRASP_MARGIN). A fixed threshold fails on the elongated butter
    # stick -- it is grabbed at one end, so the eef-to-body-center distance never
    # drops below ~0.05 (vs ~0.005 for a sugar cube). Keying off each object's own
    # closest approach recovers the butter while keeping the sugars tight.
    GRASP_TH = 0.04  # floor for the grasp threshold
    GRASP_MARGIN = 0.015  # how far above its closest approach still counts as held
    HOLD_MARGIN = 0.02  # release extends while within this of the grasp threshold
    MIN_RUN = 5  # min held-run length (frames) to count as a real grasp (not a brush)
    LIFT_TOL = 0.06  # object Z rise above rest to count as lifted off the counter
    LIFT_SUSTAIN = 15  # frames the lift must persist (rejects a transient bump)
    REST_W = 20  # window for rest-height medians
    # --- stand mixer head (open ~1.57 rad, closed ~0) ---
    HEAD_CLOSED_ABS = 0.15  # head joint below this counts as closed
    # --- speed dial (0 -> peak ~2-3 rad when on) ---
    # Take the FINAL rise to (near) peak as the turn-on, not the first crossing: in
    # some demos the dial is nudged partway early and only turned to full speed at
    # the end. A high fraction anchors the boundary to that final turn.
    KNOB_ON_FRAC = 0.85  # dial past this fraction of its peak == turned to full

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.OBJECTS}
        self.eef_site_id = env.robots[0].eef_site_id["right"]

        sm = env.stand_mixer
        self.head_qpos_adr = env.sim.model.get_joint_qpos_addr(
            sm._joint_names["head"]
        )
        self.knob_qpos_adr = env.sim.model.get_joint_qpos_addr(
            sm._joint_names["knob_speed"]
        )

        # Natural-language object names for the instructions.
        self.obj_lang = {}
        for o in self.OBJECTS:
            try:
                self.obj_lang[o] = env.get_obj_lang(o)
            except Exception:
                self.obj_lang[o] = {"butter": "butter stick", "cream_cheese": "cream cheese stick", "sugar": "sugar cube"}[o]

        self.extra_subtasks = {SUBTASK_TURNON}
        self.expected_done_idx = None  # set per demo in resolve()
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {
            "head": float(env.sim.data.qpos[self.head_qpos_adr]),
            "knob": float(env.sim.data.qpos[self.knob_qpos_adr]),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
        }
        for o in self.OBJECTS:
            d[o] = env.sim.data.body_xpos[self.obj_body_ids[o]].copy()
        return d

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

    def _carry(self, grip, opos):
        """Resolve (grasp, release, t_lift, rest_z) for one ingredient from its
        gripper-distance + height trajectory, or None if never grasped."""
        T = len(opos)
        gd = np.linalg.norm(grip - opos, axis=1)
        # adaptive grasp threshold: floor, or a margin above this object's closest
        # approach (handles objects grabbed off-center, e.g. the butter stick)
        grasp_th = max(self.GRASP_TH, float(gd.min()) + self.GRASP_MARGIN)
        held = gd < grasp_th
        # contiguous held runs of at least MIN_RUN frames
        runs = []
        i = 0
        while i < T:
            if held[i]:
                j = i
                while j < T and held[j]:
                    j += 1
                if j - i >= self.MIN_RUN:
                    runs.append((i, j - 1))
                i = j
            else:
                i += 1
        if not runs:
            return None
        # the carry run lifts the object highest (isolates the real add from any
        # incidental brush while the gripper is over the shared bowl)
        rest0 = float(np.median(opos[: self.REST_W, 2]))
        grasp, last = max(runs, key=lambda r: opos[r[0]: r[1] + 1, 2].max() - rest0)
        # extend the release to the last still-holding frame of that run's grasp
        hold_th = grasp_th + self.HOLD_MARGIN
        rel = last
        while rel + 1 < T and gd[rel + 1] < hold_th:
            rel += 1
        release = rel + 1
        rest_z = float(np.median(opos[: max(grasp, self.REST_W), 2]))
        lifted = opos[:, 2] > rest_z + self.LIFT_TOL
        t_lift = self._sustained_start(lifted, grasp, self.LIFT_SUSTAIN)
        if t_lift is None:
            lf = np.where(lifted[grasp:])[0]
            t_lift = grasp + int(lf[0]) if lf.size else grasp
        return grasp, release, t_lift, rest_z

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        head = np.array([h["head"] for h in history])
        knob = np.array([h["knob"] for h in history])
        grip = np.array([h["grip"] for h in history])
        opos = {o: np.array([h[o] for h in history]) for o in self.OBJECTS}

        # per-object carry (grasp/release/lift); order by release == add order
        carries = {}
        for o in self.OBJECTS:
            c = self._carry(grip, opos[o])
            if c is not None:
                carries[o] = c
        order = sorted(carries, key=lambda o: carries[o][1])  # by release
        last_release = max((carries[o][1] for o in order), default=0)

        # head close: first frame at/after the last add where the head is closed
        hc = np.where(head[last_release:] < self.HEAD_CLOSED_ABS)[0]
        t_head_closed = last_release + int(hc[0]) if hc.size else T - 1

        # knob on: first frame at/after the head close reaching a fraction of peak
        peak = float(knob.max())
        if peak > 0.2:
            kf = np.where(knob[t_head_closed:] >= self.KNOB_ON_FRAC * peak)[0]
            t_knob_on = t_head_closed + int(kf[0]) if kf.size else T - 1
        else:
            t_knob_on = T - 1

        # -- assemble segments --
        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        for o in order:
            grasp, release, t_lift, _ = carries[o]
            lang = self.obj_lang[o]
            emit(t_lift, SUBTASK_ADD, STAGE_PICK, f"pick up the {lang} from the counter")
            emit(release, SUBTASK_ADD, STAGE_PLACE, f"place the {lang} in the stand mixer bowl")
        emit(t_head_closed, SUBTASK_CLOSE, STAGE_EXECUTE, CLOSE_LANG)
        emit(t_knob_on, SUBTASK_TURNON, STAGE_EXECUTE, TURNON_LANG)
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        review = []
        if len(order) != len(self.OBJECTS):
            review.append(
                f"only {len(order)}/{len(self.OBJECTS)} ingredients had a clean "
                f"grasp -- check the adds"
            )

        debug = {
            "add_order": ",".join(order),
            "releases": [int(carries[o][1]) for o in order],
            "t_head_closed": t_head_closed,
            "t_knob_on": t_knob_on,
            "knob_peak": round(peak, 3),
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

        n_pick = count(SUBTASK_ADD, STAGE_PICK)
        n_place = count(SUBTASK_ADD, STAGE_PLACE)
        if n_pick != 3:
            issues.append(f"expected 3 ingredient picks, found {n_pick}")
        if n_place != 3:
            issues.append(f"expected 3 ingredient places, found {n_place}")
        if count(SUBTASK_CLOSE) != 1:
            issues.append(f"expected 1 CloseStandMixerHead, found {count(SUBTASK_CLOSE)}")
        if count(SUBTASK_TURNON) != 1:
            issues.append(f"expected 1 TurnOnStandMixer, found {count(SUBTASK_TURNON)}")
        return issues
