"""SQLite-backed experiment store + blob directory.

Experiments are keyed by `(domain, id)`. `values` and `artifacts` are stored
as JSON blobs on the row; the spreadsheet UI reads/writes them through this
interface. Binary artifacts (CIFs, spectra, images) live under `blob_dir/`.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Iterable

from .schema import Experiment, InputValue


_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT NOT NULL,
    domain TEXT NOT NULL,
    values_json TEXT NOT NULL,
    artifacts_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (domain, id)
);
CREATE INDEX IF NOT EXISTS ix_experiments_domain ON experiments(domain);
"""


class ExperimentStore:
    def __init__(self, db_path: str | Path, blob_dir: str | Path | None = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.blob_dir = Path(blob_dir) if blob_dir else self.db_path.parent / "blobs"
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def upsert(self, exp: Experiment) -> None:
        values_json = json.dumps(
            {k: v.model_dump(mode="json") for k, v in exp.values.items()},
            default=_json_default,
        )
        artifacts_json = json.dumps(
            {k: str(v) for k, v in exp.artifacts.items()}
        )
        metadata_json = json.dumps(exp.metadata, default=_json_default)
        self._conn.execute(
            "INSERT INTO experiments(id, domain, values_json, artifacts_json, metadata_json) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(domain, id) DO UPDATE SET "
            "values_json=excluded.values_json, "
            "artifacts_json=excluded.artifacts_json, "
            "metadata_json=excluded.metadata_json",
            (exp.id, exp.domain, values_json, artifacts_json, metadata_json),
        )
        self._conn.commit()

    def bulk_upsert(self, exps: Iterable[Experiment]) -> int:
        n = 0
        for e in exps:
            self.upsert(e)
            n += 1
        return n

    def list_domains(self) -> list[str]:
        cur = self._conn.execute("SELECT DISTINCT domain FROM experiments ORDER BY domain")
        return [r[0] for r in cur.fetchall()]

    def count(self, domain: str) -> int:
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE domain = ?", (domain,)
        )
        return int(cur.fetchone()[0])

    def iter_experiments(self, domain: str) -> Iterable[Experiment]:
        cur = self._conn.execute(
            "SELECT id, domain, values_json, artifacts_json, metadata_json "
            "FROM experiments WHERE domain = ? ORDER BY created_at",
            (domain,),
        )
        for row in cur.fetchall():
            yield _row_to_experiment(row)

    def get(self, domain: str, exp_id: str) -> Experiment:
        cur = self._conn.execute(
            "SELECT id, domain, values_json, artifacts_json, metadata_json "
            "FROM experiments WHERE domain = ? AND id = ?",
            (domain, exp_id),
        )
        row = cur.fetchone()
        if row is None:
            raise KeyError(f"{domain}/{exp_id} not found")
        return _row_to_experiment(row)

    def delete(self, domain: str, exp_id: str) -> None:
        self._conn.execute(
            "DELETE FROM experiments WHERE domain = ? AND id = ?", (domain, exp_id)
        )
        self._conn.commit()

    def write_blob(self, data: bytes, suffix: str = "") -> Path:
        name = f"{uuid.uuid4().hex}{suffix}"
        out = self.blob_dir / name
        out.write_bytes(data)
        return out


def _row_to_experiment(row: tuple) -> Experiment:
    exp_id, domain, v_json, a_json, m_json = row
    values_raw = json.loads(v_json)
    values = {k: InputValue.model_validate(body) for k, body in values_raw.items()}
    artifacts = {k: Path(v) for k, v in json.loads(a_json).items()}
    metadata = json.loads(m_json)
    return Experiment(
        id=exp_id, domain=domain, values=values, artifacts=artifacts, metadata=metadata
    )


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"unserializable: {type(obj).__name__}")


__all__ = ["ExperimentStore"]
