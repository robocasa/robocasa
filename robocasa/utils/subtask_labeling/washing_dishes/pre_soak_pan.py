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

# All bank atomic tasks.
SUBTASK_MOVE = "PickPlaceCounterToSink"   # pan / sponge counter -> sink
SUBTASK_ON = "TurnOnSinkFaucet"

ON_LANG = "turn on the sink faucet"
DONE_LANG = "task complete"


@register_labeler
class PreSoakPanLabeler(BaseSubtaskLabeler):
    """Labeler for the PreSoakPan composite task.

    A pan (`obj1`) and a sponge (`obj2`) sit on the counter next to the sink; the
    robot places both into the sink and turns on the water. Fixed base. Three
    subtasks (all bank atomics): PickPlaceCounterToSink for the pan and the sponge
    (in grasp order) and TurnOnSinkFaucet.

    Detection off privileged sim state: each item's grasp onset is its first
    SUSTAINED contact-grasp (fallback lift/displace on a loose 1-2 frame blip) and
    its place ends fully-ungrasped. The faucet turn-on ends when the sink handle
    joint enters the water-on range (0.40 < qpos mod 2*pi < pi), mirroring the
    env's own success check.
    """

    ENV_NAME = "PreSoakPan"

    PAN = "obj1"
    SPONGE = "obj2"

    MOVE_TH = 0.10
    Z_LIFT = 0.03
    MOTION_TH = 0.004
    RELEASE_PAD = 8
    REL_FRAC = 0.5
    REST_W = 10
    FW_NEAR = 30      # only fold the fw-open into a clean contact-release if within
                      # this many frames (else it's a spurious later gripper opening)
    WATER_LO = 0.40   # handle qpos (mod 2pi) above this (and below pi) = water on

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.gripper = env.robots[0].gripper["right"]
        self.fadr = [env.sim.model.get_joint_qpos_addr(j) for j in self.gripper.joints]
        self.items = (self.PAN, self.SPONGE)
        self.bids = {n: env.obj_body_id[n] for n in self.items}
        self.geoms = {n: env.objects[n].contact_geoms for n in self.items}
        self.lang = {}
        for n in self.items:
            try:
                self.lang[n] = env.get_obj_lang(n)
            except Exception:
                self.lang[n] = "item"
        self.handle_adr = env.sim.model.get_joint_qpos_addr(
            "{}handle_joint".format(env.sink.naming_prefix)
        )
        self.min_segment_frames = 10
        self.expected_done_idx = None

    def per_step(self, env, t: int) -> dict:
        hq = float(env.sim.data.qpos[self.handle_adr]) % (2 * np.pi)
        d = {
            "fw": float(sum(abs(env.sim.data.qpos[a]) for a in self.fadr)),
            "water_on": int(self.WATER_LO < hq < np.pi),
        }
        for n in self.items:
            d[f"pos_{n}"] = env.sim.data.body_xpos[self.bids[n]].copy()
            d[f"grasp_{n}"] = int(env._check_grasp(gripper=self.gripper, object_geoms=self.geoms[n]))
        return d

    # ---- event helpers -------------------------------------------------

    def _grasp_onset(self, grasp_arr, obj):
        # First SUSTAINED (>=3 frame) contact grasp -- a 1-2 frame contact blip
        # (loose/no-contact grasp) is not the grasp, so fall to the lift/displace.
        g = grasp_arr > 0
        for t in range(len(g) - 2):
            if g[t] and g[t + 1] and g[t + 2]:
                return t
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
        # End of the last SUSTAINED (>=3 frame) contact run -- a firm grasp's
        # release. A 1-2 frame contact blip (loose/no-contact grasp) is ignored.
        gseg = grasp_arr[grasp_t:hi]
        sustained_end = None
        run = 0
        for i in range(len(gseg)):
            if gseg[i] > 0:
                run += 1
                if run >= 3:
                    sustained_end = grasp_t + i
            else:
                run = 0
        if sustained_end is not None:
            rel = sustained_end + self.RELEASE_PAD
            # Fold in the fw-open only if near the contact end -- a far-later full
            # gripper opening (e.g. while reaching for the faucet) is spurious.
            if fwo is not None and fwo <= sustained_end + self.FW_NEAR:
                rel = max(rel, fwo)
        elif fwo is not None:
            # loose/blip grasp: the gripper-open is the real release.
            rel = fwo
        else:
            rel = setdown
        return int(min(max(rel, grasp_t + 1), hi))

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        fw = np.array([h["fw"] for h in history])
        fw_open = float(fw.max())
        water_on = np.array([h["water_on"] for h in history])
        pos = {n: np.array([h[f"pos_{n}"] for h in history]) for n in self.items}
        grasp = {n: np.array([h[f"grasp_{n}"] for h in history]) for n in self.items}

        g = {n: self._grasp_onset(grasp[n], pos[n]) for n in self.items}
        order = sorted(self.items, key=lambda n: g[n])
        g0, g1 = g[order[0]], g[order[1]]
        r0 = self._release_for(grasp[order[0]], pos[order[0]], fw, fw_open, g0, max(g1, g0 + 1))
        r1 = self._release_for(grasp[order[1]], pos[order[1]], fw, fw_open, g1, T)

        # Turn-on ends when the water first comes on after the second item is in.
        won = np.where(water_on[r1:] > 0)[0]
        turn_on_done = int(r1 + int(won[0])) if won.size else T - 1
        turn_on_done = int(min(max(turn_on_done, r1 + 1), T - 1))

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        rels = {order[0]: r0, order[1]: r1}
        for n in order:
            emit(g[n], SUBTASK_MOVE, STAGE_PICK, f"pick up the {self.lang[n]} from the counter")
            emit(rels[n], SUBTASK_MOVE, STAGE_PLACE, f"place the {self.lang[n]} in the sink")
        emit(turn_on_done, SUBTASK_ON, STAGE_EXECUTE, ON_LANG)
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        debug = {
            "order": ",".join(order),
            "grasp": [g0, g1],
            "release": [r0, r1],
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
        if npi != 2 or npl != 2:
            issues.append(f"expected 2 picks + 2 places into the sink, found {npi} + {npl}")
        if SUBTASK_ON not in set(str(v) for v in names):
            issues.append(f"missing {SUBTASK_ON} subtask")
        dbg = getattr(result, "debug", None) or {}
        if dbg.get("water_on_frames", 1) < 1:
            issues.append("water never turned on (faucet not detected)")
        return issues
