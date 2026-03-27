"""Pipeline state machine: enforces stage ordering and transition legality."""

import logging

from aml_toolkit.core.enums import AbstentionReason, PipelineStage

logger = logging.getLogger("aml_toolkit")

# Legal forward transitions (happy path + abstention from any stage)
_TRANSITIONS: dict[PipelineStage, list[PipelineStage]] = {
    PipelineStage.INIT: [PipelineStage.DATA_VALIDATED, PipelineStage.ABSTAINED],
    PipelineStage.DATA_VALIDATED: [PipelineStage.PROFILED, PipelineStage.ABSTAINED],
    PipelineStage.PROFILED: [PipelineStage.PROBED, PipelineStage.ABSTAINED],
    PipelineStage.PROBED: [PipelineStage.INTERVENTION_SELECTED, PipelineStage.ABSTAINED],
    PipelineStage.INTERVENTION_SELECTED: [PipelineStage.TRAINING_ACTIVE, PipelineStage.ABSTAINED],
    PipelineStage.TRAINING_ACTIVE: [PipelineStage.MODEL_SELECTED, PipelineStage.ABSTAINED],
    PipelineStage.MODEL_SELECTED: [PipelineStage.CALIBRATED, PipelineStage.ABSTAINED],
    PipelineStage.CALIBRATED: [PipelineStage.ENSEMBLED, PipelineStage.ABSTAINED],
    PipelineStage.ENSEMBLED: [PipelineStage.EXPLAINED, PipelineStage.ABSTAINED],
    PipelineStage.EXPLAINED: [PipelineStage.COMPLETED, PipelineStage.ABSTAINED],
    PipelineStage.COMPLETED: [],
    PipelineStage.ABSTAINED: [],
}


class PipelineStateMachine:
    """Tracks and enforces pipeline stage transitions.

    The state machine starts at INIT and only allows forward transitions
    along the happy path, or transitions to ABSTAINED from any stage.
    """

    def __init__(self) -> None:
        self._current = PipelineStage.INIT
        self._history: list[PipelineStage] = [PipelineStage.INIT]
        self._abstention_reason: AbstentionReason | None = None

    @property
    def current(self) -> PipelineStage:
        return self._current

    @property
    def history(self) -> list[PipelineStage]:
        return list(self._history)

    @property
    def abstention_reason(self) -> AbstentionReason | None:
        return self._abstention_reason

    @property
    def is_terminal(self) -> bool:
        return self._current in (PipelineStage.COMPLETED, PipelineStage.ABSTAINED)

    def can_transition(self, target: PipelineStage) -> bool:
        return target in _TRANSITIONS.get(self._current, [])

    def transition(self, target: PipelineStage) -> None:
        """Transition to the target stage.

        Raises:
            ValueError: If the transition is illegal.
        """
        if not self.can_transition(target):
            raise ValueError(
                f"Illegal transition: {self._current.value} -> {target.value}. "
                f"Allowed: {[s.value for s in _TRANSITIONS.get(self._current, [])]}"
            )
        logger.info(f"Pipeline: {self._current.value} -> {target.value}")
        self._current = target
        self._history.append(target)

    def abstain(self, reason: AbstentionReason) -> None:
        """Transition to ABSTAINED with a typed reason."""
        self._abstention_reason = reason
        self.transition(PipelineStage.ABSTAINED)
