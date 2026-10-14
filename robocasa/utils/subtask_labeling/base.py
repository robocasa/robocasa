from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from robocasa.utils.subtask_labeling.subtask_bank import DONE, is_known, is_pick_place


# A segment is (end_frame_exclusive, subtask_name, stage, lang). Segments are
# contiguous: each starts where the previous ended.
Segment = Tuple[int, str, str, str]

# Stage names. Pick-and-place subtasks split into PICK then PLACE. TurnOn*
# (lever-press) subtasks split into PRESS then WAIT (the agent actively pushes
# the lever down, then idles while the toaster heats / lever springs back).
# TILT is the pour stage of the bowl-carry strategy (grab the container, then
# tilt it to empty its contents). EXECUTE is the catch-all single-stage marker
# for atomic skills that don't split. DONE is the terminal stage.
STAGE_PICK = "pick"
STAGE_PLACE = "place"
STAGE_EXECUTE = "execute"
STAGE_PRESS = "press"
STAGE_WAIT = "wait"
STAGE_TILT = "tilt"
STAGE_NAVIGATE = "navigate"
STAGE_DONE = "done"


@dataclass
class LabelResult:
    """Per-demo labeling output. All per-frame arrays are aligned 1:1 with the
    demo's states/actions."""

    # Running finest-segment counter. Range [0, expected_done_idx].
    subtask_idx: np.ndarray  # (T,) int32

    # Bank atomic-task name for the current subtask (e.g. "PickPlaceCounterTo
    # Blender", "TurnOnToaster"), or "done" for the terminal segment.
    subtask_name: np.ndarray  # (T,) str

    # Stage within the subtask: "pick"/"place" for pick-and-place skills,
    # "execute" for other skills, "done" for the terminal segment.
    subtask_stage: np.ndarray  # (T,) str

    # Natural-language instruction for the current stage.
    subtask_lang: np.ndarray  # (T,) str

    # Soft review notes: a labeler attaches these when it auto-detects something
    # that is correctly labeled but worth a human glance (e.g. a tilt-pour
    # strategy variant). Unlike validate()'s issues, these are NOT errors -- the
    # demo passes the sanity checks; the note just routes it to a light "confirm"
    # review queue rather than the "investigate" queue.
    review: List[str] = field(default_factory=list)

    # Optional debug fields the labeler may attach.
    debug: Dict[str, np.ndarray] = field(default_factory=dict)


class BaseSubtaskLabeler:
    """Subclass per composite task.

    Subclasses must define:
      - ENV_NAME: str matching env_args.env_name in the hdf5.

    And must implement:
      - per_step(env, t) -> dict: state values for timestep t, called once per
              frame after the sim state has been replayed.
      - resolve(history) -> LabelResult: given the list of per_step dicts for
              the whole demo, return per-frame labels. Use build_result() to
              turn a list of segment boundaries into a LabelResult.

    setup() should set self.expected_done_idx so validate() can check the demo
    decomposed into the expected number of segments. Subtask names should come
    from the atomic-task bank (see subtask_bank.py); any intentional non-bank
    names must be listed in self.extra_subtasks.
    """

    ENV_NAME: str = ""

    def __init__(self):
        # Expected terminal value of subtask_idx (== number of segments - 1).
        self.expected_done_idx: Optional[int] = None
        # Non-bank subtask names the labeler intentionally uses.
        self.extra_subtasks = set()
        # If set, validate() flags any non-terminal segment shorter than this
        # many frames. Catches near-degenerate subtasks (a boundary that nearly
        # collapsed) that the exact zero-length check misses. Opt-in per labeler;
        # None disables it.
        self.min_segment_frames: Optional[int] = None

    def setup(self, env, ep_meta: dict) -> None:
        """Called once per demo after the env is reset to the initial state.
        Override to cache per-demo constants (e.g. object count)."""
        self.ep_meta = ep_meta

    def per_step(self, env, t: int) -> dict:
        raise NotImplementedError

    def resolve(self, history: List[dict]) -> LabelResult:
        raise NotImplementedError

    def build_result(
        self, T: int, segments: List[Segment], debug: Optional[dict] = None
    ) -> LabelResult:
        """Expand an ordered list of contiguous segments into a LabelResult.

        Each segment is (end_exclusive, subtask_name, stage, lang). Segment i
        covers [cursor, end); subtask_idx is the segment ordinal. Boundaries are
        clamped monotonically, so out-of-order inputs collapse to zero-length
        segments rather than corrupting the timeline (validate() then flags it).
        """
        subtask_idx = np.zeros(T, dtype=np.int32)
        subtask_name = np.empty(T, dtype=object)
        subtask_stage = np.empty(T, dtype=object)
        subtask_lang = np.empty(T, dtype=object)

        cursor = 0
        for i, (end, name, stage, lang) in enumerate(segments):
            end = min(max(int(end), cursor), T)
            subtask_idx[cursor:end] = i
            subtask_name[cursor:end] = name
            subtask_stage[cursor:end] = stage
            subtask_lang[cursor:end] = lang
            cursor = end
        # Any leftover frames belong to the terminal segment.
        if cursor < T:
            last = len(segments) - 1
            subtask_idx[cursor:T] = last
            subtask_name[cursor:T] = segments[last][1]
            subtask_stage[cursor:T] = segments[last][2]
            subtask_lang[cursor:T] = segments[last][3]

        return LabelResult(
            subtask_idx=subtask_idx,
            subtask_name=subtask_name,
            subtask_stage=subtask_stage,
            subtask_lang=subtask_lang,
            debug=debug or {},
        )

    def validate(self, result: LabelResult) -> List[str]:
        """Run sanity checks on a LabelResult. Returns a list of issue strings;
        an empty list means the demo passed and needs no manual review."""
        issues = []
        idx = result.subtask_idx
        if idx.size == 0:
            return ["empty trajectory"]

        if np.any(np.diff(idx) < 0):
            issues.append("subtask_idx is not monotonic non-decreasing")
        if int(idx[0]) != 0:
            issues.append(f"subtask_idx starts at {int(idx[0])}, expected 0")

        present = {int(v) for v in np.unique(idx)}
        max_idx = max(present)
        missing = sorted(set(range(max_idx + 1)) - present)
        if missing:
            issues.append(f"zero-length segment(s): subtask_idx skips {missing}")

        if self.expected_done_idx is not None and max_idx != self.expected_done_idx:
            issues.append(
                f"ends at subtask_idx {max_idx}, expected done index "
                f"{self.expected_done_idx} (wrong segment count / never finished)"
            )

        if result.subtask_stage.size and result.subtask_stage[-1] != STAGE_DONE:
            issues.append("trajectory does not end in the 'done' stage")

        # Every subtask name should be a bank atomic task (or a declared extra).
        for name in {str(v) for v in np.unique(result.subtask_name)}:
            if not is_known(name) and name not in self.extra_subtasks:
                issues.append(f"subtask '{name}' is not in the atomic-task bank")

        # Flag suspiciously short (but non-empty) subtasks -- a boundary that
        # nearly collapsed. The terminal done segment is exempt (it is legitimately
        # a single frame when a demo is cut the instant the task succeeds).
        if self.min_segment_frames is not None and idx.size:
            boundaries = np.concatenate(
                [[0], np.where(np.diff(idx) != 0)[0] + 1, [idx.size]]
            )
            for j in range(len(boundaries) - 1):
                start, end = int(boundaries[j]), int(boundaries[j + 1])
                seg_idx = int(idx[start])
                if seg_idx == max_idx:  # terminal done segment
                    continue
                length = end - start
                if length < self.min_segment_frames:
                    stage = str(result.subtask_stage[start])
                    name = str(result.subtask_name[start])
                    issues.append(
                        f"subtask {seg_idx} ({name} [{stage}]) is only {length} "
                        f"frames (< {self.min_segment_frames}); boundary may have "
                        f"collapsed"
                    )

        return issues


LABELERS: Dict[str, Callable[[], BaseSubtaskLabeler]] = {}


def register_labeler(cls):
    """Decorator: register a SubtaskLabeler subclass under its ENV_NAME."""
    assert cls.ENV_NAME, f"{cls.__name__} must set ENV_NAME"
    assert cls.ENV_NAME not in LABELERS, f"duplicate labeler for {cls.ENV_NAME}"
    LABELERS[cls.ENV_NAME] = cls
    return cls


def get_labeler(env_name: str) -> Optional[BaseSubtaskLabeler]:
    cls = LABELERS.get(env_name)
    return cls() if cls is not None else None
