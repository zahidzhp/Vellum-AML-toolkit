"""Tests for Phase 2.2 — Run history store."""

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from aml_toolkit.artifacts.run_history import DatasetSignature, RunHistoryRecord
from aml_toolkit.history.dataset_signature_builder import build_dataset_signature
from aml_toolkit.history.run_history_store import RunHistoryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sig(**kwargs) -> DatasetSignature:
    defaults = dict(
        modality="TABULAR",
        task_type="BINARY",
        n_classes=2,
        log_n_samples=4.0,
        log_n_features=2.0,
        imbalance_ratio=1.5,
        missingness_pct=0.02,
        duplicate_pct=0.01,
        ood_shift_score=0.1,
        label_noise_score=0.05,
    )
    defaults.update(kwargs)
    return DatasetSignature(**defaults)


def _make_record(sig: DatasetSignature | None = None, **kwargs) -> RunHistoryRecord:
    if sig is None:
        sig = _make_sig()
    defaults = dict(
        run_id="run_test",
        dataset_signature=sig,
        best_candidate_id="rf_001",
        best_candidate_family="rf",
        best_macro_f1=0.85,
        config_mode="BALANCED",
    )
    defaults.update(kwargs)
    return RunHistoryRecord(**defaults)


# ---------------------------------------------------------------------------
# DatasetSignature.to_vector
# ---------------------------------------------------------------------------

class TestDatasetSignatureVector:
    def test_vector_length(self):
        sig = _make_sig()
        vec = sig.to_vector()
        assert len(vec) == 10

    def test_vector_all_in_unit_interval(self):
        sig = _make_sig()
        vec = sig.to_vector()
        assert np.all(vec >= 0.0), f"Below 0: {vec}"
        assert np.all(vec <= 1.0), f"Above 1: {vec}"

    def test_vector_dtype_float32(self):
        sig = _make_sig()
        vec = sig.to_vector()
        assert vec.dtype == np.float32

    def test_vector_identical_signatures_equal(self):
        sig1 = _make_sig()
        sig2 = _make_sig()
        assert np.allclose(sig1.to_vector(), sig2.to_vector())

    def test_vector_different_signatures_differ(self):
        sig1 = _make_sig(log_n_samples=2.0, imbalance_ratio=1.0)
        sig2 = _make_sig(log_n_samples=5.0, imbalance_ratio=50.0)
        assert not np.allclose(sig1.to_vector(), sig2.to_vector())

    def test_imbalance_capped_at_100(self):
        sig = _make_sig(imbalance_ratio=500.0)
        vec = sig.to_vector()
        assert vec[2] == pytest.approx(1.0)

    def test_log_n_samples_capped_at_6(self):
        sig = _make_sig(log_n_samples=10.0)  # > 6
        vec = sig.to_vector()
        assert vec[0] == pytest.approx(1.0)

    def test_flags_reflected_in_vector(self):
        sig = _make_sig(
            has_label_noise=True, has_ood_shift=True, has_severe_imbalance=True
        )
        vec = sig.to_vector()
        assert vec[7] == 1.0  # has_label_noise
        assert vec[8] == 1.0  # has_ood_shift
        assert vec[9] == 1.0  # has_severe_imbalance


# ---------------------------------------------------------------------------
# RunHistoryRecord auto-vector
# ---------------------------------------------------------------------------

class TestRunHistoryRecord:
    def test_vector_auto_populated(self):
        record = _make_record()
        assert len(record.dataset_signature_vector) == 10

    def test_vector_matches_signature(self):
        sig = _make_sig(log_n_samples=3.0, imbalance_ratio=10.0)
        record = _make_record(sig=sig)
        expected = sig.to_vector().tolist()
        assert record.dataset_signature_vector == pytest.approx(expected, abs=1e-5)

    def test_roundtrip_json(self):
        record = _make_record()
        json_str = record.model_dump_json()
        restored = RunHistoryRecord.model_validate_json(json_str)
        assert restored.run_id == record.run_id
        assert len(restored.dataset_signature_vector) == 10


# ---------------------------------------------------------------------------
# RunHistoryStore save/load
# ---------------------------------------------------------------------------

class TestRunHistoryStoreSaveLoad:
    def test_save_creates_file(self, tmp_path):
        store = RunHistoryStore(tmp_path / "history.jsonl")
        record = _make_record()
        store.save(record)
        assert (tmp_path / "history.jsonl").exists()

    def test_save_load_roundtrip(self, tmp_path):
        store = RunHistoryStore(tmp_path / "history.jsonl")
        records = [_make_record(run_id=f"run_{i}", best_macro_f1=float(i) / 10) for i in range(5)]
        for r in records:
            store.save(r)
        loaded = store.load_all()
        assert len(loaded) == 5
        run_ids = {r.run_id for r in loaded}
        assert run_ids == {f"run_{i}" for i in range(5)}

    def test_load_empty_file_returns_empty(self, tmp_path):
        store = RunHistoryStore(tmp_path / "empty.jsonl")
        assert store.load_all() == []

    def test_load_nonexistent_path_returns_empty(self, tmp_path):
        store = RunHistoryStore(tmp_path / "nonexistent" / "history.jsonl")
        assert store.load_all() == []

    def test_save_bad_path_does_not_raise(self):
        # Path with null byte — guaranteed to fail on all OS
        store = RunHistoryStore("/nonexistent_path_xyz_abc_/history.jsonl")
        record = _make_record()
        # Should not raise
        store.save(record)

    def test_load_respects_max_records(self, tmp_path):
        store = RunHistoryStore(tmp_path / "history.jsonl")
        for i in range(20):
            store.save(_make_record(run_id=f"run_{i}"))
        loaded = store.load_all(max_records=5)
        assert len(loaded) <= 5

    def test_load_skips_malformed_lines(self, tmp_path):
        path = tmp_path / "history.jsonl"
        # Write one valid and one invalid line
        valid = _make_record(run_id="valid")
        path.write_text(valid.model_dump_json() + "\n{not valid json\n")
        store = RunHistoryStore(path)
        loaded = store.load_all()
        assert len(loaded) == 1
        assert loaded[0].run_id == "valid"


# ---------------------------------------------------------------------------
# RunHistoryStore similarity search
# ---------------------------------------------------------------------------

class TestRunHistoryStoreSimilarity:
    def test_find_similar_identical_signature(self, tmp_path):
        store = RunHistoryStore(tmp_path / "history.jsonl")
        sig = _make_sig(log_n_samples=4.0, imbalance_ratio=5.0)
        store.save(_make_record(sig=sig, run_id="exact_match"))
        results = store.find_similar(sig, top_k=5)
        assert len(results) == 1
        assert results[0].run_id == "exact_match"

    def test_find_similar_returns_top_k(self, tmp_path):
        store = RunHistoryStore(tmp_path / "history.jsonl")
        sig = _make_sig()
        for i in range(10):
            store.save(_make_record(
                sig=_make_sig(log_n_samples=float(i), imbalance_ratio=float(i + 1)),
                run_id=f"run_{i}",
            ))
        results = store.find_similar(sig, top_k=3)
        assert len(results) <= 3

    def test_find_similar_same_modality_only(self, tmp_path):
        store = RunHistoryStore(tmp_path / "history.jsonl")
        tabular_sig = _make_sig(modality="TABULAR")
        image_sig = _make_sig(modality="IMAGE")
        store.save(_make_record(sig=tabular_sig, run_id="tabular_run"))
        store.save(_make_record(sig=image_sig, run_id="image_run"))

        results = store.find_similar(tabular_sig, same_modality_only=True)
        run_ids = {r.run_id for r in results}
        assert "image_run" not in run_ids
        assert "tabular_run" in run_ids

    def test_find_similar_cross_modality_when_flag_false(self, tmp_path):
        store = RunHistoryStore(tmp_path / "history.jsonl")
        tabular_sig = _make_sig(modality="TABULAR")
        image_sig = _make_sig(modality="IMAGE")
        store.save(_make_record(sig=image_sig, run_id="image_run"))

        results = store.find_similar(tabular_sig, same_modality_only=False)
        assert len(results) == 1

    def test_recency_decay_weights_recent(self, tmp_path):
        store = RunHistoryStore(tmp_path / "history.jsonl")
        sig = _make_sig(log_n_samples=4.0, imbalance_ratio=2.0)

        old_ts = datetime.now(timezone.utc) - timedelta(days=100)
        recent_ts = datetime.now(timezone.utc) - timedelta(minutes=1)

        old_record = _make_record(sig=sig, run_id="old")
        old_record = old_record.model_copy(update={"timestamp": old_ts})

        recent_record = _make_record(sig=sig, run_id="recent")
        recent_record = recent_record.model_copy(update={"timestamp": recent_ts})

        # Write manually to preserve timestamps
        path = tmp_path / "history.jsonl"
        with open(path, "w") as f:
            f.write(old_record.model_dump_json() + "\n")
            f.write(recent_record.model_dump_json() + "\n")

        results = store.find_similar(sig, top_k=2, recency_decay=0.9)
        # Recent should rank first
        assert results[0].run_id == "recent"

    def test_find_similar_empty_store_returns_empty(self, tmp_path):
        store = RunHistoryStore(tmp_path / "history.jsonl")
        sig = _make_sig()
        assert store.find_similar(sig) == []


# ---------------------------------------------------------------------------
# DatasetSignatureBuilder
# ---------------------------------------------------------------------------

class TestDatasetSignatureBuilder:
    def test_build_from_dicts(self):
        manifest = {
            "modality": "TABULAR",
            "task_type": "BINARY",
            "n_classes": 2,
            "n_samples": 1000,
            "n_features": 20,
            "class_counts": {"0": 800, "1": 200},
        }
        profile = {
            "mean_missingness": 0.05,
            "duplicate_pct": 0.01,
            "ood_shift_score": 0.0,
            "label_noise_score": 0.0,
        }
        sig = build_dataset_signature(manifest, profile)
        assert sig.modality == "TABULAR"
        assert sig.n_classes == 2
        assert sig.imbalance_ratio == pytest.approx(4.0)

    def test_build_sets_severe_imbalance_flag(self):
        manifest = {
            "modality": "TABULAR",
            "task_type": "BINARY",
            "n_classes": 2,
            "n_samples": 1000,
            "n_features": 10,
            "class_counts": {"0": 990, "1": 10},
        }
        profile = {}
        sig = build_dataset_signature(manifest, profile)
        assert sig.has_severe_imbalance is True
        assert sig.imbalance_ratio == pytest.approx(99.0)

    def test_build_image_modality(self):
        manifest = {
            "modality": "IMAGE",
            "task_type": "MULTICLASS",
            "n_classes": 5,
            "n_samples": 5000,
            "n_features": 0,
            "backbone": "resnet18",
            "image_size_bucket": "small",
            "n_channels": 3,
        }
        profile = {}
        sig = build_dataset_signature(manifest, profile)
        assert sig.modality == "IMAGE"
        assert sig.log_n_features == 0.0  # image
        assert sig.backbone == "resnet18"

    def test_build_graceful_failure(self):
        # Even with garbage input, should not raise
        sig = build_dataset_signature(None, None)
        assert sig.modality == "TABULAR"
        vec = sig.to_vector()
        assert len(vec) == 10

    def test_build_vector_all_in_unit_interval(self):
        manifest = {
            "modality": "TABULAR",
            "task_type": "BINARY",
            "n_classes": 2,
            "n_samples": 500000,
            "n_features": 500,
            "class_counts": {"0": 490000, "1": 10000},
        }
        profile = {
            "mean_missingness": 0.3,
            "duplicate_pct": 0.05,
            "ood_shift_score": 0.7,
            "label_noise_score": 0.2,
        }
        sig = build_dataset_signature(manifest, profile)
        vec = sig.to_vector()
        assert np.all(vec >= 0.0) and np.all(vec <= 1.0)
