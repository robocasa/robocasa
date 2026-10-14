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

# All bank atomics. The colander is moved counter -> sink and each fruit is
# moved counter -> colander (which sits in the sink), so both reuse the
# CounterToSink atomic; the language differentiates "in the sink" vs "in the
# colander". Then the water is turned on to rinse.
SUBTASK_MOVE = "PickPlaceCounterToSink"
SUBTASK_ON = "TurnOnSinkFaucet"
SUBTASK_SPOUT = "TurnSinkSpout"   # optional: some operators swivel the spout over
                                  # the colander before turning the water on

COLANDER = "colander"
ON_LANG = "turn on the sink faucet"
DONE_LANG = "task complete"


@register_labeler
class WashFruitColanderLabeler(BaseSubtaskLabeler):
    """Labeler for the WashFruitColander composite task.

    Fixed-base task: a colander and 1-3 fruit (`fruit0..fruitN-1`) sit on the
    counter by the sink. The robot puts the colander in the sink, puts each fruit
    into the colander, then turns on the faucet to pour water over it. The subtask
    count therefore scales with num_fruit (3/4/5 for 1/2/3 fruit); all bank
    atomics.

    Decomposition, in grasp order (colander first, then the fruit as grasped --
    the fruit INDEX is not the placement order):

        PickPlaceCounterToSink[pick/place] (colander -> sink)
        PickPlaceCounterToSink[pick/place] (each fruit -> colander)
        TurnOnSinkFaucet
        -> done

    Detection off privileged sim state (the wide rim-grasped colander makes
    contact-grasp unreliable, so grasp/release use the finger-width and the object
    motion, not contact): the grasp onset is the finger-width close just before
    the object's SUSTAINED lift (bump-proof) or, for the slid colander, its
    displacement; the place ends at the motion set-down maxed with the contact
    let-go. The faucet turn-on ends at the first frame the sink water is on.
    """

    ENV_NAME = "WashFruitColander"

    MOVE_TH = 0.10
    Z_LIFT = 0.03
    MOTION_TH = 0.004
    RELEASE_PAD = 8
    REL_FRAC = 0.5
    REST_W = 10
    GRASP_WINDOW = 120  # search this many frames before the object's movement for
                        # the finger-width close (the grasp)
    CONTACT_LINGER = 60  # a contact-grasp end this far past the motion set-down
                         # still counts as the release (gripper lingering); beyond
                         # it is a spurious later re-contact, ignored
    PLACE_TOL = 0.06       # object within this of its final resting pose = settled
    SPOUT_TH = 0.15        # spout swivel (rad, wrapped) beyond this = a real rotation
    SPOUT_MOTION_TH = 0.01 # per-frame spout angular speed marking the rotation's end
    SPOUT_PAD = 5

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.gripper = env.robots[0].gripper["right"]
        self.fadr = [env.sim.model.get_joint_qpos_addr(j) for j in self.gripper.joints]
        self.num_fruit = int(ep_meta.get("refs", {}).get("num_fruit", env.num_fruit))
        self.fruits = [f"fruit{i}" for i in range(self.num_fruit)]
        self.movables = [COLANDER] + self.fruits
        self.bids = {n: env.obj_body_id[n] for n in self.movables}
        self.geoms = {n: env.objects[n].contact_geoms for n in self.movables}
        self.lang = {}
        for n in self.movables:
            try:
                self.lang[n] = env.get_obj_lang(n)
            except Exception:
                self.lang[n] = COLANDER if n == COLANDER else "fruit"
        self.sink = env.sink
        self.min_segment_frames = 10
        self.expected_done_idx = None

    def per_step(self, env, t: int) -> dict:
        hs = self.sink.get_handle_state(env=env)
        d = {
            "fw": float(sum(abs(env.sim.data.qpos[a]) for a in self.fadr)),
            "water_on": int(bool(hs.get("water_on", False))),
            "spout_joint": float(hs.get("spout_joint", 0.0)),
            "spout_ori": str(hs.get("spout_ori", "center")),
        }
        for n in self.movables:
            d[f"pos_{n}"] = env.sim.data.body_xpos[self.bids[n]].copy()
            d[f"grasp_{n}"] = int(env._check_grasp(gripper=self.gripper, object_geoms=self.geoms[n]))
        return d

    # ---- event helpers -------------------------------------------------

    def _movement_onset(self, obj):
        # First frame the object leaves its resting pose for its carry. Prefer the
        # SUSTAINED (>=3 frame) lift -- bump-proof, since a neighbouring object
        # being placed can NUDGE a still-resting fruit horizontally (a false
        # displacement) but cannot lift it off the counter. The colander is slid
        # into the sink and never rises, so fall to the sustained displacement --
        # safe because the colander is grasped first, so nothing nudges it.
        z0 = float(np.median(obj[: self.REST_W, 2]))
        lifted = obj[:, 2] > z0 + self.Z_LIFT
        for t in range(len(lifted) - 2):
            if lifted[t] and lifted[t + 1] and lifted[t + 2]:
                return t
        disp = np.linalg.norm(obj[:, :2] - obj[0, :2], axis=1) > self.MOVE_TH
        for t in range(len(disp) - 2):
            if disp[t] and disp[t + 1] and disp[t + 2]:
                return t
        return None

    def _fw_grasp_before(self, fw, fw_open, move):
        # The grasp = the finger-width CLOSING just before the object starts
        # moving: the last frame the gripper is still open (fw high) before it
        # closes into the carry. Robust where contact-grasp fires late or drops
        # out mid-carry (the wide rim-grasped colander). None if no clean close.
        lo = max(0, move - self.GRASP_WINDOW)
        if move - lo < 3:
            return None
        closed = float(np.min(fw[max(0, move - 6): move + 2]))
        level = closed + self.REL_FRAC * (fw_open - closed)
        last_open = None
        for t in range(lo, min(move + 1, len(fw))):
            if fw[t] >= level:
                last_open = t
        if last_open is None or last_open + 1 >= move:
            return None
        return last_open + 1

    def _grasp_onset(self, obj, fw, fw_open):
        move = self._movement_onset(obj)
        if move is None:
            return 0
        fwc = self._fw_grasp_before(fw, fw_open, move)
        return fwc if fwc is not None else move

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

    def _release_for(self, obj, grasp_arr, grasp_t, hi):
        # Release = when the object is set down and the gripper lets go. The
        # finger-width is unreliable here: a LOOSE colander grasp barely moves the
        # fingers, so a fw-open false-triggers mid-carry (colander still airborne).
        # Instead take the object's motion SET-DOWN (last real motion, incl. the
        # gripper-retraction jerk) and MAX it with the last contact-grasp frame --
        # motion floors it when contact drops out early (the release comes after
        # the object stops), contact extends it when the gripper lingers after the
        # object has settled. A contact end far past the set-down is a spurious
        # re-contact (reaching past the object later) and is ignored.
        # Arrival = the last frame the object is still far from its final resting
        # pose (it stays put after placement). This is where it has SETTLED, and
        # excludes the trailing micro-jitter of the gripper nudging the already-
        # placed object as it retracts (which a raw last-motion set-down includes).
        lo = max(grasp_t, hi - 20)
        final = np.median(obj[lo:hi], axis=0) if hi > lo else obj[hi - 1]
        far = np.where(np.linalg.norm(obj[grasp_t:hi] - final, axis=1) > self.PLACE_TOL)[0]
        arrival = int(grasp_t + int(far[-1]) + 2) if far.size else int(grasp_t + 1)
        rel = arrival
        # Extend to the gripper let-go if it lingers on the settled object (the
        # colander is often held a beat after it settles). A contact end far past
        # the arrival is a spurious later re-contact and is ignored; contact that
        # drops out early (loose grasp) simply leaves the arrival as the release.
        ci = np.where(grasp_arr[grasp_t:hi] > 0)[0]
        if ci.size:
            contact_end = int(grasp_t + int(ci[-1]) + 2)
            if contact_end <= arrival + self.CONTACT_LINGER:
                rel = max(rel, contact_end)
        return int(min(max(rel, grasp_t + 1), hi))

    @staticmethod
    def _wrap(a):
        # signed wrap of an angle (array) to [-pi, pi]
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _spout_rotation(self, spout, spout_ori):
        """Detect an optional spout swivel (whenever it occurs in the demo).

        Returns (rotated, t_done, direction). The spout starts centered; some
        operators swivel it over the colander -- usually right before turning the
        water on, but sometimes mid-sequence (between two fruit). If the wrapped
        displacement from the initial angle ever exceeds SPOUT_TH the rotation is
        real; t_done is where the swivel is essentially complete (first reaches
        90% of its peak displacement, + pad) and direction is the sink's own
        left/right classification of the settled spout.
        """
        sp0 = float(np.median(spout[: self.REST_W]))
        dwrap = np.abs(self._wrap(spout - sp0))
        mx = float(dwrap.max())
        if mx <= self.SPOUT_TH:
            return False, None, None
        reach = np.where(dwrap >= 0.9 * mx)[0]
        t_done = int(reach[0]) + self.SPOUT_PAD if reach.size else int(np.argmax(dwrap))
        t_done = int(min(t_done, len(spout) - 1))
        direction = str(spout_ori[t_done])
        if direction == "center":
            later = [o for o in spout_ori[t_done:] if o != "center"]
            direction = str(later[0]) if later else "left"
        return True, t_done, direction

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        fw = np.array([h["fw"] for h in history])
        fw_open = float(fw.max())
        water_on = np.array([h["water_on"] for h in history])
        spout = np.array([h["spout_joint"] for h in history])
        spout_ori = [h["spout_ori"] for h in history]
        pos = {n: np.array([h[f"pos_{n}"] for h in history]) for n in self.movables}
        grasp = {n: np.array([h[f"grasp_{n}"] for h in history]) for n in self.movables}

        g = {n: self._grasp_onset(pos[n], fw, fw_open) for n in self.movables}
        order = sorted(self.movables, key=lambda n: g[n])

        # Each item's place ends before the next item is grasped; the last item
        # runs to the end (bounded below by the faucet turn-on).
        rel = {}
        for i, n in enumerate(order):
            hi = g[order[i + 1]] if i + 1 < len(order) else T
            rel[n] = self._release_for(pos[n], grasp[n], g[n], max(hi, g[n] + 1))
        last_rel = rel[order[-1]]

        won = np.where(water_on[last_rel:] > 0)[0]
        turn_on_done = int(last_rel + int(won[0])) if won.size else T - 1

        rotated, t_spout, spout_dir = self._spout_rotation(spout, spout_ori)

        # Build every segment boundary as a (end, name, stage, lang) event and
        # emit them in time order. The optional spout swivel is inserted at its
        # actual moment -- sometimes between two fruit, sometimes just before the
        # water turn-on -- rather than pinned to the end.
        events = []
        for n in order:
            target = "sink" if n == COLANDER else "colander"
            events.append((int(g[n]), SUBTASK_MOVE, STAGE_PICK, f"pick up the {self.lang[n]} from the counter"))
            events.append((int(rel[n]), SUBTASK_MOVE, STAGE_PLACE, f"place the {self.lang[n]} in the {target}"))
        if rotated:
            events.append((int(t_spout), SUBTASK_SPOUT, STAGE_EXECUTE, f"turn the sink spout to the {spout_dir}"))
        events.append((int(turn_on_done), SUBTASK_ON, STAGE_EXECUTE, ON_LANG))
        events.sort(key=lambda e: e[0])

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        for end, name, stage, lang in events:
            emit(end, name, stage, lang)
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        debug = {
            "order": ",".join(order),
            "num_fruit": self.num_fruit,
            "grasp": {n: int(g[n]) for n in order},
            "release": {n: int(rel[n]) for n in order},
            "spout_rotated": bool(rotated),
            "spout_done": int(t_spout) if rotated else None,
            "spout_dir": spout_dir,
            "turn_on_done": turn_on_done,
            "water_on_frames": int(water_on.sum()),
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
        move = names == SUBTASK_MOVE
        npi = int(np.sum(move & (stages == STAGE_PICK)))
        npl = int(np.sum(move & (stages == STAGE_PLACE)))
        exp = self.num_fruit + 1  # colander + each fruit
        if npi != exp or npl != exp:
            issues.append(f"expected {exp} picks + {exp} places, found {npi} + {npl}")
        nameset = set(str(v) for v in names)
        if SUBTASK_ON not in nameset:
            issues.append(f"missing {SUBTASK_ON} subtask")
        dbg = getattr(result, "debug", None) or {}
        if dbg.get("water_on_frames", 1) < 1:
            issues.append("water never turned on (faucet not detected)")
        if dbg.get("spout_rotated") and SUBTASK_SPOUT not in nameset:
            issues.append(f"spout rotated but {SUBTASK_SPOUT} subtask missing")
        return issues
