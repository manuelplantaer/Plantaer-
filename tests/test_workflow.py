"""Autosave + suggestion-to-sheet workflow semantics."""

from __future__ import annotations

import pandas as pd
import pytest

from plantaer.schema import (
    Category,
    Experiment,
    InputSpec,
    InputType,
    InputValue,
    MaterialDomain,
)
from plantaer.ui.app import (
    _autosave,
    _add_suggestion_to_sheet,
    _values_changed,
)
from plantaer.ui.tabular import (
    LOCK_COL,
    experiments_to_frame,
    frame_to_experiments,
)
from plantaer.workspace import Workspace


@pytest.fixture
def mini_workspace(tmp_path) -> Workspace:
    ws = Workspace.open(tmp_path / "wk")
    dom = MaterialDomain(
        name="Mini",
        inputs=[
            InputSpec(
                name="x",
                category=Category.PROCESSING,
                type=InputType.NUMERIC,
                bounds=(0.0, 1.0),
            ),
            InputSpec(
                name="g",
                category=Category.ENVIRONMENT,
                type=InputType.CATEGORICAL,
                choices=["a", "b"],
            ),
        ],
        targets=[
            InputSpec(name="y", category=Category.TARGET, type=InputType.NUMERIC)
        ],
    )
    ws.registry.add_material(dom)
    for i in range(3):
        ws.store.upsert(
            Experiment(
                id=f"e{i}",
                domain="Mini",
                values={
                    "x": InputValue(value=0.1 + 0.3 * i),
                    "g": InputValue(value="a" if i % 2 == 0 else "b"),
                    "y": InputValue(value=float(i)),
                },
            )
        )
    return ws


# ---------------------------------------------------------------------------
# frame_to_experiments: skip empty rows
# ---------------------------------------------------------------------------


def test_frame_to_experiments_drops_empty_rows(mini_workspace):
    dom = mini_workspace.registry.get("Mini")
    df = pd.DataFrame(
        [
            {"id": "", LOCK_COL: False, "x": None, "g": None, "y": None},
            {"id": "real", LOCK_COL: False, "x": 0.5, "g": "a", "y": 1.0},
        ]
    )
    exps = frame_to_experiments(dom, df)
    assert len(exps) == 1
    assert exps[0].id == "real"


def test_frame_to_experiments_preserves_id_on_partial_rows(mini_workspace):
    dom = mini_workspace.registry.get("Mini")
    df = pd.DataFrame(
        [{"id": "partial", LOCK_COL: False, "x": 0.2, "g": None, "y": None}]
    )
    exps = frame_to_experiments(dom, df)
    assert len(exps) == 1
    assert exps[0].id == "partial"
    assert "x" in exps[0].values
    assert "y" not in exps[0].values


# ---------------------------------------------------------------------------
# Autosave: idempotent, honors lock, detects deletions
# ---------------------------------------------------------------------------


def test_autosave_no_change_is_idempotent(mini_workspace):
    dom = mini_workspace.registry.get("Mini")
    exps = list(mini_workspace.store.iter_experiments("Mini"))
    frame = experiments_to_frame(dom, exps)
    by_id = {e.id: e for e in exps}
    n_saved, n_deleted, errors = _autosave(
        mini_workspace, "Mini", dom, frame, by_id
    )
    assert n_saved == 0
    assert n_deleted == 0
    assert errors == []


def test_autosave_writes_edited_value(mini_workspace):
    dom = mini_workspace.registry.get("Mini")
    exps = list(mini_workspace.store.iter_experiments("Mini"))
    frame = experiments_to_frame(dom, exps)
    by_id = {e.id: e for e in exps}
    frame.loc[frame["id"] == "e0", "y"] = 99.0
    n_saved, n_deleted, errors = _autosave(
        mini_workspace, "Mini", dom, frame, by_id
    )
    assert n_saved == 1
    assert errors == []
    got = mini_workspace.store.get("Mini", "e0")
    assert got.values["y"].value == 99.0


def test_autosave_refuses_to_edit_locked_row(mini_workspace):
    dom = mini_workspace.registry.get("Mini")
    # Lock e0.
    e0 = mini_workspace.store.get("Mini", "e0")
    mini_workspace.store.upsert(e0.model_copy(update={"locked": True}))

    exps = list(mini_workspace.store.iter_experiments("Mini"))
    frame = experiments_to_frame(dom, exps)
    by_id = {e.id: e for e in exps}
    # Try to edit the locked row while it remains locked.
    frame.loc[frame["id"] == "e0", "y"] = 123.0
    n_saved, n_deleted, errors = _autosave(
        mini_workspace, "Mini", dom, frame, by_id
    )
    assert n_saved == 0
    assert any("locked" in err for err in errors)
    got = mini_workspace.store.get("Mini", "e0")
    assert got.values["y"].value == 0.0  # untouched


def test_autosave_allows_unlock_without_edit(mini_workspace):
    dom = mini_workspace.registry.get("Mini")
    e0 = mini_workspace.store.get("Mini", "e0")
    mini_workspace.store.upsert(e0.model_copy(update={"locked": True}))

    exps = list(mini_workspace.store.iter_experiments("Mini"))
    frame = experiments_to_frame(dom, exps)
    by_id = {e.id: e for e in exps}
    # Only untick the lock column, don't change values.
    frame.loc[frame["id"] == "e0", LOCK_COL] = False
    n_saved, n_deleted, errors = _autosave(
        mini_workspace, "Mini", dom, frame, by_id
    )
    assert n_saved == 1
    assert errors == []
    assert not mini_workspace.store.get("Mini", "e0").locked


def test_autosave_deletes_removed_rows(mini_workspace):
    dom = mini_workspace.registry.get("Mini")
    exps = list(mini_workspace.store.iter_experiments("Mini"))
    frame = experiments_to_frame(dom, exps)
    by_id = {e.id: e for e in exps}
    # Drop e1 from the frame.
    frame = frame[frame["id"] != "e1"].reset_index(drop=True)
    n_saved, n_deleted, errors = _autosave(
        mini_workspace, "Mini", dom, frame, by_id
    )
    assert n_deleted == 1
    assert mini_workspace.store.count("Mini") == 2


def test_autosave_refuses_to_delete_locked_row(mini_workspace):
    dom = mini_workspace.registry.get("Mini")
    e2 = mini_workspace.store.get("Mini", "e2")
    mini_workspace.store.upsert(e2.model_copy(update={"locked": True}))
    exps = list(mini_workspace.store.iter_experiments("Mini"))
    frame = experiments_to_frame(dom, exps)
    by_id = {e.id: e for e in exps}
    frame = frame[frame["id"] != "e2"].reset_index(drop=True)
    n_saved, n_deleted, errors = _autosave(
        mini_workspace, "Mini", dom, frame, by_id
    )
    assert n_deleted == 0
    assert any("locked" in err and "e2" in err for err in errors)
    assert mini_workspace.store.count("Mini") == 3


def test_autosave_skips_invalid_row_and_saves_others(mini_workspace):
    dom = mini_workspace.registry.get("Mini")
    exps = list(mini_workspace.store.iter_experiments("Mini"))
    frame = experiments_to_frame(dom, exps)
    by_id = {e.id: e for e in exps}
    # Invalid category on e1; valid numeric edit on e2.
    frame.loc[frame["id"] == "e1", "g"] = "NOT_A_CHOICE"
    frame.loc[frame["id"] == "e2", "y"] = 42.0
    n_saved, n_deleted, errors = _autosave(
        mini_workspace, "Mini", dom, frame, by_id
    )
    assert n_saved == 1
    assert errors  # at least one complaint about e1
    assert mini_workspace.store.get("Mini", "e2").values["y"].value == 42.0


# ---------------------------------------------------------------------------
# Suggestion -> sheet workflow
# ---------------------------------------------------------------------------


def test_add_suggestion_creates_untested_row(mini_workspace):
    dom = mini_workspace.registry.get("Mini")
    before = mini_workspace.store.count("Mini")
    new_id = _add_suggestion_to_sheet(
        mini_workspace,
        dom,
        suggestion_values={
            "x": InputValue(value=0.87),
            "g": InputValue(value="b"),
        },
        target_name="y",
        predicted_mean=4.2,
        predicted_std=0.3,
        direction="maximize",
    )
    assert mini_workspace.store.count("Mini") == before + 1
    added = mini_workspace.store.get("Mini", new_id)
    # Inputs present, target absent — this is the "awaiting results" state
    # that the model mask will exclude from training.
    assert "x" in added.values and "g" in added.values
    assert "y" not in added.values
    assert added.metadata["origin"] == "suggestion"
    assert added.metadata["predicted"]["y"]["mean"] == pytest.approx(4.2)
    assert added.metadata["predicted"]["y"]["direction"] == "maximize"


def test_full_loop_suggestion_then_fill_in_target(mini_workspace):
    """The user can add a suggestion, later fill in the measured target."""
    dom = mini_workspace.registry.get("Mini")
    new_id = _add_suggestion_to_sheet(
        mini_workspace,
        dom,
        suggestion_values={
            "x": InputValue(value=0.55),
            "g": InputValue(value="a"),
        },
        target_name="y",
        predicted_mean=3.1,
        predicted_std=0.4,
        direction="maximize",
    )
    # Simulate the user editing the proposal row with a measured target.
    exps = list(mini_workspace.store.iter_experiments("Mini"))
    frame = experiments_to_frame(dom, exps)
    frame.loc[frame["id"] == new_id, "y"] = 3.05
    by_id = {e.id: e for e in exps}
    n_saved, _, errors = _autosave(mini_workspace, "Mini", dom, frame, by_id)
    assert n_saved == 1
    assert errors == []
    measured = mini_workspace.store.get("Mini", new_id)
    assert measured.values["y"].value == 3.05
    # Suggestion metadata survives.
    assert measured.metadata["origin"] == "suggestion"


# ---------------------------------------------------------------------------
# _values_changed helper
# ---------------------------------------------------------------------------


def test_values_changed_detects_new_key(mini_workspace):
    dom = mini_workspace.registry.get("Mini")
    a = mini_workspace.store.get("Mini", "e0")
    b = a.model_copy(update={
        "values": {**a.values, "extra": InputValue(value=1.0)}
    })
    assert _values_changed(a, b)


def test_values_changed_detects_numeric_edit(mini_workspace):
    a = mini_workspace.store.get("Mini", "e0")
    b = a.model_copy(update={"values": {**a.values, "y": InputValue(value=999.0)}})
    assert _values_changed(a, b)


def test_values_changed_false_when_identical(mini_workspace):
    a = mini_workspace.store.get("Mini", "e0")
    b = mini_workspace.store.get("Mini", "e0")
    assert not _values_changed(a, b)
