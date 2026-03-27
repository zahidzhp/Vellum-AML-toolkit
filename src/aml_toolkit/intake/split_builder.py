"""Split building: stratified, grouped, temporal, and provided splits."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

from aml_toolkit.core.config import ToolkitConfig
from aml_toolkit.core.enums import SplitStrategy
from aml_toolkit.core.exceptions import SchemaValidationError


class SplitResult:
    """Container for train/val/test split indices."""

    def __init__(
        self,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        test_indices: np.ndarray,
        strategy: SplitStrategy,
        warnings: list[str] | None = None,
    ) -> None:
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
        self.strategy = strategy
        self.warnings = warnings or []


def build_splits(
    n_samples: int,
    labels: np.ndarray,
    config: ToolkitConfig,
    groups: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
) -> SplitResult:
    """Build train/val/test splits according to the configured strategy.

    Args:
        n_samples: Total number of samples.
        labels: Label array (1D for single-label, 2D for multilabel).
        config: Toolkit configuration.
        groups: Group IDs for grouped splitting (optional).
        timestamps: Timestamps for temporal splitting (optional).

    Returns:
        SplitResult with train/val/test index arrays.
    """
    strategy_str = config.splitting.strategy.upper()
    try:
        strategy = SplitStrategy(strategy_str)
    except ValueError:
        raise SchemaValidationError(
            f"Unknown split strategy: '{strategy_str}'. "
            f"Supported: {[s.value for s in SplitStrategy]}"
        )

    if strategy == SplitStrategy.STRATIFIED:
        return _stratified_split(n_samples, labels, config)
    elif strategy == SplitStrategy.GROUPED:
        return _grouped_split(n_samples, labels, config, groups)
    elif strategy == SplitStrategy.TEMPORAL:
        return _temporal_split(n_samples, config, timestamps)
    elif strategy == SplitStrategy.PROVIDED:
        raise SchemaValidationError(
            "Strategy 'PROVIDED' requires pre-defined split indices. "
            "Use build_provided_splits() instead."
        )
    else:
        raise SchemaValidationError(f"Unhandled split strategy: {strategy}")


def _stratified_split(
    n_samples: int,
    labels: np.ndarray,
    config: ToolkitConfig,
) -> SplitResult:
    """Stratified split preserving class distribution."""
    test_ratio = config.splitting.test_ratio
    val_ratio = config.splitting.val_ratio
    seed = config.splitting.random_seed
    warnings: list[str] = []

    # For multilabel, fall back to random split (sklearn stratification needs 1D)
    if labels.ndim == 2:
        warnings.append(
            "Multilabel detected: using random split instead of stratified."
        )
        indices = np.arange(n_samples)
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
        test_size = int(n_samples * test_ratio)
        val_size = int(n_samples * val_ratio)
        test_idx = indices[:test_size]
        val_idx = indices[test_size : test_size + val_size]
        train_idx = indices[test_size + val_size :]
        return SplitResult(train_idx, val_idx, test_idx, SplitStrategy.STRATIFIED, warnings)

    # Check for class absence risk
    unique, counts = np.unique(labels, return_counts=True)
    min_count = counts.min()
    if min_count < 3:
        warnings.append(
            f"Class '{unique[counts.argmin()]}' has only {min_count} samples. "
            "May be absent from some splits."
        )

    # First split: train+val vs test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    all_indices = np.arange(n_samples)
    trainval_idx, test_idx = next(sss1.split(all_indices, labels))

    # Second split: train vs val (from trainval)
    val_fraction_of_trainval = val_ratio / (1.0 - test_ratio)
    val_fraction_of_trainval = min(val_fraction_of_trainval, 0.5)

    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction_of_trainval, random_state=seed
    )
    trainval_labels = labels[trainval_idx]
    train_local, val_local = next(sss2.split(trainval_idx, trainval_labels))
    train_idx = trainval_idx[train_local]
    val_idx = trainval_idx[val_local]

    return SplitResult(train_idx, val_idx, test_idx, SplitStrategy.STRATIFIED, warnings)


def _grouped_split(
    n_samples: int,
    labels: np.ndarray,
    config: ToolkitConfig,
    groups: np.ndarray | None,
) -> SplitResult:
    """Group-aware split ensuring no group appears in multiple splits."""
    if groups is None:
        raise SchemaValidationError(
            "Grouped split strategy requires a group column, but none was provided. "
            "Set dataset.group_column in config."
        )

    test_ratio = config.splitting.test_ratio
    val_ratio = config.splitting.val_ratio
    seed = config.splitting.random_seed

    # First split: trainval vs test by group
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    all_indices = np.arange(n_samples)
    trainval_idx, test_idx = next(gss1.split(all_indices, labels, groups))

    # Second split: train vs val by group
    val_fraction = val_ratio / (1.0 - test_ratio)
    val_fraction = min(val_fraction, 0.5)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    trainval_labels = labels[trainval_idx]
    trainval_groups = groups[trainval_idx]
    train_local, val_local = next(
        gss2.split(trainval_idx, trainval_labels, trainval_groups)
    )
    train_idx = trainval_idx[train_local]
    val_idx = trainval_idx[val_local]

    return SplitResult(train_idx, val_idx, test_idx, SplitStrategy.GROUPED)


def _temporal_split(
    n_samples: int,
    config: ToolkitConfig,
    timestamps: np.ndarray | None,
) -> SplitResult:
    """Temporal split: oldest → train, middle → val, newest → test."""
    if timestamps is None:
        raise SchemaValidationError(
            "Temporal split strategy requires a time column, but none was provided. "
            "Set dataset.time_column in config."
        )

    test_ratio = config.splitting.test_ratio
    val_ratio = config.splitting.val_ratio

    sorted_indices = np.argsort(timestamps)
    n_test = int(n_samples * test_ratio)
    n_val = int(n_samples * val_ratio)
    n_train = n_samples - n_test - n_val

    train_idx = sorted_indices[:n_train]
    val_idx = sorted_indices[n_train : n_train + n_val]
    test_idx = sorted_indices[n_train + n_val :]

    return SplitResult(train_idx, val_idx, test_idx, SplitStrategy.TEMPORAL)


def build_provided_splits(
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
) -> SplitResult:
    """Wrap user-provided split indices into a SplitResult.

    Args:
        train_indices: Pre-defined training indices.
        val_indices: Pre-defined validation indices.
        test_indices: Pre-defined test indices.

    Returns:
        SplitResult with PROVIDED strategy.
    """
    return SplitResult(
        train_indices, val_indices, test_indices, SplitStrategy.PROVIDED
    )
