"""Plantaer Streamlit app — material picker + per-material spreadsheet.

Run with::

    streamlit run plantaer/ui/app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from ..design import Objective, suggest_experiments
from ..models import fit_for_domain
from ..schema import (
    CATEGORY_ORDER,
    Category,
    InputSpec,
    InputType,
    InputValue,
    MaterialDomain,
)
from ..workspace import Workspace
from .tabular import experiments_to_frame, frame_to_experiments


WORKSPACE_ENV = "PLANTAER_WORKSPACE"


def _category_color(cat: Category) -> str:
    # Stable color per category; used for column header emojis / prefixes.
    palette = {
        Category.COMPOSITION: "#1f77b4",
        Category.STRUCTURE: "#9467bd",
        Category.COMPONENT: "#17becf",
        Category.PROCESSING: "#ff7f0e",
        Category.ENVIRONMENT: "#bcbd22",
        Category.GEOMETRY: "#8c564b",
        Category.CHARACTERIZATION: "#e377c2",
        Category.METADATA: "#7f7f7f",
        Category.TARGET: "#2ca02c",
    }
    return palette[cat]


@st.cache_resource
def _get_workspace(root: str) -> Workspace:
    return Workspace.open(root)


def _sidebar(ws: Workspace) -> str | None:
    st.sidebar.title("Plantaer")
    st.sidebar.caption(f"workspace: `{ws.root}`")
    materials = ws.registry.list_materials()
    if materials:
        sel = st.sidebar.selectbox("Material", materials, key="material_picker")
    else:
        sel = None
        st.sidebar.info("No materials registered yet.")

    with st.sidebar.expander("+ New material", expanded=not materials):
        _new_material_form(ws)
    return sel


def _new_material_form(ws: Workspace) -> None:
    with st.form("new_material"):
        name = st.text_input("Material name", placeholder="e.g. NMC cathodes")
        n_inputs = st.number_input("# input columns", min_value=1, max_value=50, value=3)
        n_targets = st.number_input("# target columns", min_value=1, max_value=10, value=1)
        st.caption("Declare each input (category + type drive featurization).")
        input_rows = []
        for i in range(int(n_inputs)):
            cols = st.columns([2, 2, 2, 1])
            input_rows.append(
                {
                    "name": cols[0].text_input(f"input #{i+1} name", key=f"in_n_{i}"),
                    "category": cols[1].selectbox(
                        f"input #{i+1} category",
                        [c.value for c in CATEGORY_ORDER if c != Category.TARGET],
                        key=f"in_c_{i}",
                    ),
                    "type": cols[2].selectbox(
                        f"input #{i+1} type",
                        [t.value for t in InputType],
                        key=f"in_t_{i}",
                    ),
                    "bounds": cols[3].text_input(
                        f"input #{i+1} bounds (lo,hi)", key=f"in_b_{i}"
                    ),
                }
            )
        target_rows = []
        for j in range(int(n_targets)):
            cols = st.columns([2, 1])
            target_rows.append(
                {
                    "name": cols[0].text_input(f"target #{j+1} name", key=f"tg_n_{j}"),
                    "bounds": cols[1].text_input(
                        f"target #{j+1} bounds (lo,hi)", key=f"tg_b_{j}"
                    ),
                }
            )
        submitted = st.form_submit_button("Create material")
        if submitted:
            try:
                domain = _materialize_form(name, input_rows, target_rows)
                ws.registry.add_material(domain)
                st.success(f"Created {domain.name!r}")
                st.rerun()
            except Exception as exc:
                st.error(f"{exc}")


def _materialize_form(name: str, input_rows, target_rows) -> MaterialDomain:
    if not name.strip():
        raise ValueError("material name required")
    inputs: list[InputSpec] = []
    for r in input_rows:
        if not r["name"].strip():
            continue
        bounds = _parse_bounds(r["bounds"])
        inputs.append(
            InputSpec(
                name=r["name"].strip(),
                category=Category(r["category"]),
                type=InputType(r["type"]),
                bounds=bounds,
            )
        )
    targets: list[InputSpec] = []
    for r in target_rows:
        if not r["name"].strip():
            continue
        bounds = _parse_bounds(r["bounds"])
        targets.append(
            InputSpec(
                name=r["name"].strip(),
                category=Category.TARGET,
                type=InputType.NUMERIC,
                bounds=bounds,
            )
        )
    if not inputs:
        raise ValueError("declare at least one input")
    if not targets:
        raise ValueError("declare at least one target")
    return MaterialDomain(name=name.strip(), inputs=inputs, targets=targets)


def _parse_bounds(text: str) -> tuple[float, float] | None:
    t = text.strip()
    if not t:
        return None
    parts = [p.strip() for p in t.replace(";", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"bounds must be lo,hi — got {text!r}")
    return float(parts[0]), float(parts[1])


def _sheet_page(ws: Workspace, material_name: str) -> None:
    domain = ws.registry.get(material_name)
    st.header(f"{material_name}")
    st.caption(
        f"{len(domain.inputs)} inputs • {len(domain.targets)} targets • "
        f"{ws.store.count(material_name)} experiments"
    )

    # Legend: column categories present in this material.
    present_cats = {s.category for s in domain.all_specs()}
    legend = " ".join(
        f":{_css_color(_category_color(c))}[■] {c.value}"
        for c in CATEGORY_ORDER
        if c in present_cats
    )
    st.markdown(legend)

    experiments = list(ws.store.iter_experiments(material_name))
    df = experiments_to_frame(domain, experiments)

    col_config = {"id": st.column_config.TextColumn("id", help="experiment id")}
    for s in domain.all_specs():
        header = f"[{s.category.value}] {s.name}" + (f" ({s.unit})" if s.unit else "")
        if s.type == InputType.NUMERIC:
            col_config[s.name] = st.column_config.NumberColumn(
                header,
                min_value=s.bounds[0] if s.bounds else None,
                max_value=s.bounds[1] if s.bounds else None,
                help=s.description,
            )
        elif s.type == InputType.CATEGORICAL and s.choices:
            col_config[s.name] = st.column_config.SelectboxColumn(
                header, options=s.choices
            )
        else:
            col_config[s.name] = st.column_config.TextColumn(header)

    edited = st.data_editor(
        df, num_rows="dynamic", column_config=col_config, key=f"editor_{material_name}",
        use_container_width=True,
    )

    save_col, clear_col, _ = st.columns([1, 1, 4])
    if save_col.button("Save sheet"):
        try:
            exps = frame_to_experiments(domain, edited)
            for exp in exps:
                exp.validate_against(domain)
                ws.store.upsert(exp)
            st.success(f"Saved {len(exps)} row(s).")
            st.rerun()
        except Exception as exc:
            st.error(f"{exc}")

    if clear_col.button("Reload from store"):
        st.rerun()

    st.divider()
    _predict_section(ws, domain)
    st.divider()
    _suggest_section(ws, domain)


def _predict_section(ws: Workspace, domain: MaterialDomain) -> None:
    st.subheader("Predict a candidate")
    st.caption(
        "Enter values for any subset of declared inputs; model auto-selected "
        "based on current data volume."
    )
    if ws.store.count(domain.name) == 0:
        st.info("No experiments logged yet — nothing to train on.")
        return
    cols = st.columns(max(1, min(4, len(domain.inputs))))
    values: dict[str, InputValue] = {}
    for i, spec in enumerate(domain.inputs):
        col = cols[i % len(cols)]
        v = _input_widget(col, spec, key=f"pred_{domain.name}_{spec.name}")
        if v is not None:
            values[spec.name] = InputValue(value=v, unit=spec.unit)
    if st.button("Predict", key=f"predict_btn_{domain.name}"):
        try:
            fitted = fit_for_domain(domain, ws.store.iter_experiments(domain.name))
            from ..featurize import build_matrices
            from ..schema import Experiment

            probe = Experiment(id="__probe", domain=domain.name, values=values)
            X, _, _ = build_matrices(domain, [probe])
            mean, std = fitted.predict(X)
            st.write(
                pd.DataFrame(
                    {
                        "target": fitted.target_names,
                        "predicted mean": mean[0],
                        "± std": std[0],
                    }
                )
            )
            st.caption(
                f"model: {type(fitted.estimator).__name__} • n_rows={fitted.n_rows} "
                f"• d={fitted.layout.dim}"
            )
        except Exception as exc:
            st.error(f"{exc}")


def _suggest_section(ws: Workspace, domain: MaterialDomain) -> None:
    st.subheader("Suggest next experiments")
    if not domain.targets:
        st.info("No targets declared.")
        return
    c1, c2, c3 = st.columns([2, 1, 1])
    target = c1.selectbox(
        "Objective target",
        [t.name for t in domain.targets],
        key=f"sugg_t_{domain.name}",
    )
    direction = c2.selectbox(
        "Direction", ["maximize", "minimize"], key=f"sugg_d_{domain.name}"
    )
    k = c3.number_input(
        "# suggestions", min_value=1, max_value=25, value=5, key=f"sugg_k_{domain.name}"
    )
    if st.button("Suggest", key=f"sugg_btn_{domain.name}"):
        try:
            fitted = fit_for_domain(domain, ws.store.iter_experiments(domain.name))
            suggestions = suggest_experiments(
                fitted,
                Objective(target=target, direction=direction),  # type: ignore[arg-type]
                n_suggestions=int(k),
            )
            rows = []
            for s in suggestions:
                row = {k: v.value for k, v in s.values.items()}
                row["_predicted_mean"] = s.predicted_mean
                row["_predicted_std"] = s.predicted_std
                row["_acquisition"] = s.acquisition
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            if fitted.n_rows < 5:
                st.warning(
                    "Tiny dataset — fell back to random in-bounds design. Log more "
                    "rows and retrain for EI-based suggestions."
                )
        except Exception as exc:
            st.error(f"{exc}")


def _input_widget(col, spec: InputSpec, key: str):
    if spec.type == InputType.NUMERIC:
        lo = float(spec.bounds[0]) if spec.bounds else 0.0
        hi = float(spec.bounds[1]) if spec.bounds else 1.0
        return col.number_input(spec.name, min_value=lo, max_value=hi, key=key)
    if spec.type == InputType.CATEGORICAL and spec.choices:
        return col.selectbox(spec.name, spec.choices, key=key)
    val = col.text_input(spec.name, key=key)
    return val or None


def _css_color(hex_color: str) -> str:
    # Streamlit markdown color tags accept named colors; approximate by picking
    # the nearest from the built-in palette.
    named = {
        "#1f77b4": "blue",
        "#9467bd": "violet",
        "#17becf": "blue",
        "#ff7f0e": "orange",
        "#bcbd22": "orange",
        "#8c564b": "red",
        "#e377c2": "violet",
        "#7f7f7f": "gray",
        "#2ca02c": "green",
    }
    return named.get(hex_color, "gray")


def main() -> None:
    st.set_page_config(page_title="Plantaer", layout="wide")
    root = os.environ.get(WORKSPACE_ENV, str(Path.cwd() / "plantaer_workspace"))
    ws = _get_workspace(root)
    material = _sidebar(ws)
    if material is None:
        st.title("Welcome to Plantaer")
        st.write(
            "Declare your first material in the sidebar to get started. Each "
            "material gets its own spreadsheet, predictor, and experiment-suggestion "
            "engine."
        )
        return
    _sheet_page(ws, material)


if __name__ == "__main__":
    main()
