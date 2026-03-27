"""Resource guard: catches OOM and resource failures, converts to structured abstention (FR-129)."""

import logging
import signal
import time
from contextlib import contextmanager
from typing import Any, Generator

from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.core.exceptions import ResourceAbstentionError

logger = logging.getLogger("aml_toolkit")


class ResourceGuard:
    """Monitors resource usage and enforces compute budgets.

    Catches OOM errors, time limit violations, and other resource failures.
    Converts them into structured ResourceAbstentionError events.
    """

    def __init__(self, config: ToolkitConfig) -> None:
        self._max_time = config.compute.max_training_time_seconds
        self._abstain_on_oom = config.compute.resource_abstention_on_oom
        self._start_time: float | None = None

    def start_timer(self) -> None:
        """Start the execution timer."""
        self._start_time = time.time()

    def check_time_budget(self, candidate_id: str = "") -> None:
        """Check if the time budget has been exceeded.

        Args:
            candidate_id: Candidate being trained (for logging).

        Raises:
            ResourceAbstentionError: If time budget exceeded.
        """
        if self._start_time is None:
            return
        elapsed = time.time() - self._start_time
        if elapsed > self._max_time:
            raise ResourceAbstentionError(
                f"Training time budget exceeded for {candidate_id}: "
                f"{elapsed:.1f}s > {self._max_time}s limit.",
                resource_type="time",
            )

    def elapsed_seconds(self) -> float:
        """Return elapsed time since timer start."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @contextmanager
    def guarded_execution(self, candidate_id: str = "") -> Generator[None, None, None]:
        """Context manager that catches resource failures and converts to abstention.

        Catches MemoryError and RuntimeError (common for CUDA OOM) and wraps
        them in ResourceAbstentionError.

        Args:
            candidate_id: Candidate being trained (for error context).

        Yields:
            Control to the caller.

        Raises:
            ResourceAbstentionError: On OOM or resource failure.
        """
        try:
            yield
        except MemoryError as e:
            if self._abstain_on_oom:
                logger.error(f"OOM during training of {candidate_id}: {e}")
                raise ResourceAbstentionError(
                    f"Out of memory during training of {candidate_id}.",
                    resource_type="memory",
                ) from e
            raise
        except RuntimeError as e:
            # PyTorch CUDA OOM raises RuntimeError
            err_str = str(e).lower()
            if "out of memory" in err_str or "cuda" in err_str:
                if self._abstain_on_oom:
                    logger.error(f"GPU OOM during training of {candidate_id}: {e}")
                    raise ResourceAbstentionError(
                        f"GPU out of memory during training of {candidate_id}.",
                        resource_type="gpu_memory",
                    ) from e
            raise
