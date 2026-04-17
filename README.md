# Plantaer

A materials-science foundation-model MVP. Declare any material, log experiments in a
per-material spreadsheet, get predictions with uncertainty, and receive suggested
next experiments from a Bayesian-optimization loop — agnostic to domain, input
modality, and data volume.

## Quickstart

```bash
pip install -e .[test]
pytest -q
streamlit run plantaer/ui/app.py
```

## What ships in v1

- `plantaer/schema.py` — typed experiment schema, 9 categories × 8 input types.
- `plantaer/store.py` + `plantaer/registry.py` — SQLite + JSON-backed per-material schemas, blob store.
- `plantaer/featurize/` — dispatcher routes by input type; tabular/composition/processing in v1.
- `plantaer/models/` — selector auto-picks GP (n<50), XGBoost (n<5k), MLP (n≥5k).
- `plantaer/design/bo.py` — qEI over the material's input schema; random-in-range fallback at tiny N.
- `plantaer/ui/` — Streamlit: material picker + one spreadsheet per material.

See `plan.md` for the full design document.
