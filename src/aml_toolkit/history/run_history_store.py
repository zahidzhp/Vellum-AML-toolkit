"""Run history store — persists and retrieves RunHistoryRecord objects."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from aml_toolkit.artifacts.run_history import DatasetSignature, RunHistoryRecord

logger = logging.getLogger("aml_toolkit")


class RunHistoryStore:
    """JSON-lines file-backed store for pipeline run history.

    Each line in the JSONL file is one serialized RunHistoryRecord.
    Thread-safety: append-only writes are atomic on most OS/FS.
    """

    def __init__(self, store_path: str | Path):
        self._path = Path(store_path).expanduser()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(self, record: RunHistoryRecord) -> None:
        """Append a record to the store. Never raises — failures are logged."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a") as f:
                f.write(record.model_dump_json() + "\n")
        except Exception as e:
            logger.warning(f"RunHistoryStore.save failed (non-fatal): {e}")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load_all(self, max_records: int = 1000) -> list[RunHistoryRecord]:
        """Load all records from the store (most recent first, up to max_records)."""
        if not self._path.exists():
            return []
        records: list[RunHistoryRecord] = []
        try:
            lines = self._path.read_text().splitlines()
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(RunHistoryRecord.model_validate_json(line))
                except Exception as e:
                    logger.warning(f"Skipping malformed history line: {e}")
                if len(records) >= max_records:
                    break
        except Exception as e:
            logger.warning(f"RunHistoryStore.load_all failed: {e}")
        return records

    # ------------------------------------------------------------------
    # Similarity search
    # ------------------------------------------------------------------

    def find_similar(
        self,
        sig: DatasetSignature,
        top_k: int = 5,
        recency_decay: float = 0.9,
        same_modality_only: bool = True,
        max_records: int = 1000,
    ) -> list[RunHistoryRecord]:
        """Cosine similarity search with recency weighting.

        Score = cosine_similarity(sig_vector, record_vector) * (recency_decay ** days_ago)

        Args:
            sig: Query DatasetSignature.
            top_k: Return at most this many records.
            recency_decay: Per-day decay factor (0.9 = 10% penalty per day).
            same_modality_only: Filter to records with the same modality.
            max_records: Max records to load from store.

        Returns:
            List of RunHistoryRecord sorted by descending similarity score.
        """
        records = self.load_all(max_records=max_records)
        if not records:
            return []

        query_vec = sig.to_vector()
        query_norm = float(np.linalg.norm(query_vec))
        if query_norm < 1e-8:
            return records[:top_k]

        now = datetime.now(timezone.utc)
        scored: list[tuple[float, RunHistoryRecord]] = []

        for record in records:
            if same_modality_only and record.dataset_signature.modality != sig.modality:
                continue

            rec_vec = np.array(record.dataset_signature_vector, dtype=np.float32)
            rec_norm = float(np.linalg.norm(rec_vec))
            if rec_norm < 1e-8:
                continue

            cosine_sim = float(np.dot(query_vec, rec_vec) / (query_norm * rec_norm))

            ts = record.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            days_ago = max(0.0, (now - ts).total_seconds() / 86400.0)
            recency_weight = recency_decay ** days_ago

            scored.append((cosine_sim * recency_weight, record))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k]]
