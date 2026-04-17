"""Store round-trip tests."""

from __future__ import annotations

from plantaer.schema import Experiment, InputValue
from plantaer.store import ExperimentStore


def test_store_upsert_and_fetch(tmp_path, tabular_domain):
    store = ExperimentStore(tmp_path / "exp.sqlite", tmp_path / "blobs")
    exp = Experiment(
        id="e0",
        domain="ToyAlloy",
        values={
            "sinter_temp_c": InputValue(value=700.0, unit="C"),
            "hold_min": InputValue(value=30.0, unit="min"),
            "atmosphere": InputValue(value="argon"),
            "formula": InputValue(value={"Fe": 0.6, "Ni": 0.4}),
            "yield_strength": InputValue(value=450.0, unit="MPa"),
        },
    )
    store.upsert(exp)
    assert store.count("ToyAlloy") == 1
    got = store.get("ToyAlloy", "e0")
    assert got.values["atmosphere"].value == "argon"
    assert got.values["formula"].value == {"Fe": 0.6, "Ni": 0.4}


def test_store_bulk(tmp_path, tabular_domain, small_dataset):
    store = ExperimentStore(tmp_path / "exp.sqlite", tmp_path / "blobs")
    store.bulk_upsert(small_dataset)
    assert store.count("ToyAlloy") == len(small_dataset)
    ids = {e.id for e in store.iter_experiments("ToyAlloy")}
    assert ids == {e.id for e in small_dataset}
