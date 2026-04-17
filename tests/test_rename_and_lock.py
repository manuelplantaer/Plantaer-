"""Rename coordinators + row locking semantics."""

from __future__ import annotations

import pytest

from plantaer.schema import (
    Category,
    Experiment,
    InputSpec,
    InputType,
    InputValue,
    MaterialDomain,
)
from plantaer.workspace import Workspace


@pytest.fixture
def ws(tmp_path) -> Workspace:
    w = Workspace.open(tmp_path / "wk")
    dom = MaterialDomain(
        name="Alpha",
        inputs=[
            InputSpec(
                name="temp_c",
                category=Category.PROCESSING,
                type=InputType.NUMERIC,
                bounds=(0.0, 1000.0),
                unit="C",
            ),
            InputSpec(
                name="atm",
                category=Category.ENVIRONMENT,
                type=InputType.CATEGORICAL,
                choices=["air", "argon"],
            ),
        ],
        targets=[
            InputSpec(name="yld", category=Category.TARGET, type=InputType.NUMERIC)
        ],
    )
    w.registry.add_material(dom)
    for i in range(5):
        w.store.upsert(
            Experiment(
                id=f"e{i}",
                domain="Alpha",
                values={
                    "temp_c": InputValue(value=200.0 + 50 * i, unit="C"),
                    "atm": InputValue(value="air" if i % 2 == 0 else "argon"),
                    "yld": InputValue(value=1.0 + 0.1 * i),
                },
            )
        )
    return w


# ---------------------------------------------------------------------------
# Rename material
# ---------------------------------------------------------------------------


def test_rename_material_moves_registry_and_experiments(ws):
    ws.rename_material("Alpha", "Alpha-v2")
    assert "Alpha-v2" in ws.registry.list_materials()
    assert "Alpha" not in ws.registry.list_materials()
    assert ws.store.count("Alpha-v2") == 5
    assert ws.store.count("Alpha") == 0
    # Domain.name field itself updated.
    assert ws.registry.get("Alpha-v2").name == "Alpha-v2"


def test_rename_material_rejects_collision(ws):
    # Create a second material.
    other = MaterialDomain(
        name="Beta",
        inputs=[
            InputSpec(name="x", category=Category.PROCESSING, type=InputType.NUMERIC)
        ],
        targets=[
            InputSpec(name="y", category=Category.TARGET, type=InputType.NUMERIC)
        ],
    )
    ws.registry.add_material(other)
    with pytest.raises(ValueError):
        ws.rename_material("Alpha", "Beta")


def test_rename_material_rejects_empty(ws):
    with pytest.raises(ValueError):
        ws.rename_material("Alpha", "   ")


def test_rename_material_to_same_name_is_noop(ws):
    ws.rename_material("Alpha", "Alpha")
    assert ws.store.count("Alpha") == 5


# ---------------------------------------------------------------------------
# Rename column (input + target)
# ---------------------------------------------------------------------------


def test_rename_column_migrates_values(ws):
    ws.rename_column("Alpha", "temp_c", "sinter_temp")
    dom = ws.registry.get("Alpha")
    assert "sinter_temp" in {s.name for s in dom.all_specs()}
    assert "temp_c" not in {s.name for s in dom.all_specs()}
    for exp in ws.store.iter_experiments("Alpha"):
        assert "sinter_temp" in exp.values
        assert "temp_c" not in exp.values
        # Value + unit preserved exactly.
        assert exp.values["sinter_temp"].unit == "C"


def test_rename_target_column_also_migrates(ws):
    ws.rename_column("Alpha", "yld", "yield_mpa")
    dom = ws.registry.get("Alpha")
    assert {t.name for t in dom.targets} == {"yield_mpa"}
    for exp in ws.store.iter_experiments("Alpha"):
        assert "yield_mpa" in exp.values
        assert "yld" not in exp.values


def test_rename_column_rejects_collision(ws):
    with pytest.raises(ValueError):
        ws.rename_column("Alpha", "temp_c", "atm")


def test_rename_column_rejects_unknown(ws):
    with pytest.raises(KeyError):
        ws.rename_column("Alpha", "does_not_exist", "whatever")


# ---------------------------------------------------------------------------
# Rename experiment id
# ---------------------------------------------------------------------------


def test_rename_experiment_id_updates_store(ws):
    ws.rename_experiment("Alpha", "e0", "sample_42")
    ids = {e.id for e in ws.store.iter_experiments("Alpha")}
    assert "sample_42" in ids
    assert "e0" not in ids


def test_rename_experiment_rejects_collision(ws):
    with pytest.raises(ValueError):
        ws.rename_experiment("Alpha", "e0", "e1")


# ---------------------------------------------------------------------------
# Lock / unlock semantics
# ---------------------------------------------------------------------------


def test_locked_flag_roundtrips_through_store(ws):
    e = ws.store.get("Alpha", "e0")
    assert e.locked is False
    ws.store.upsert(e.model_copy(update={"locked": True}))
    again = ws.store.get("Alpha", "e0")
    assert again.locked is True
    # Unlock round-trips too.
    ws.store.upsert(again.model_copy(update={"locked": False}))
    assert ws.store.get("Alpha", "e0").locked is False


def test_locked_default_is_false_for_new_rows(ws):
    assert all(not e.locked for e in ws.store.iter_experiments("Alpha"))


def test_pre_lock_sqlite_gets_migrated(tmp_path):
    """Opening a store whose DB predates the ``locked`` column still works."""
    import sqlite3

    db = tmp_path / "old.sqlite"
    # Hand-craft an old-style DB lacking the ``locked`` column.
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE experiments (
            id TEXT NOT NULL,
            domain TEXT NOT NULL,
            values_json TEXT NOT NULL,
            artifacts_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (domain, id)
        );
        """
    )
    conn.execute(
        "INSERT INTO experiments(id, domain, values_json, artifacts_json, metadata_json) "
        "VALUES ('e1', 'X', '{}', '{}', '{}')"
    )
    conn.commit()
    conn.close()

    from plantaer.store import ExperimentStore

    store = ExperimentStore(db, tmp_path / "blobs")
    got = store.get("X", "e1")
    assert got.locked is False
    # Upsert with locked=True still works after migration.
    store.upsert(got.model_copy(update={"locked": True}))
    assert store.get("X", "e1").locked is True
