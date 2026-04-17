"""Entry point for Streamlit Community Cloud.

Streamlit Cloud runs this file from the repo root, so `plantaer` is importable
as a package and the relative imports inside `plantaer/ui/app.py` resolve.
On first boot we seed a small demo workspace so the deployed app is useful
immediately — users can still create their own materials from the sidebar.
"""

from __future__ import annotations

import os
from pathlib import Path

from plantaer.ui.app import main
from plantaer.workspace import Workspace


def _ensure_seeded(root: str) -> None:
    ws_path = Path(root)
    ws = Workspace.open(ws_path)
    if ws.registry.list_materials():
        return
    # Lazy import so we don't force the example into the package API.
    from examples.seed_workspace import (
        _alloys_domain,
        _oxides_domain,
        _synth_alloys,
        _synth_oxides,
    )

    ws.registry.add_material(_alloys_domain())
    ws.registry.add_material(_oxides_domain())
    ws.store.bulk_upsert(_synth_alloys(120))
    ws.store.bulk_upsert(_synth_oxides(80))


if __name__ == "__main__" or True:
    # Streamlit imports this module rather than executing `__main__`, so we
    # run the seed + main unconditionally at import time.
    workspace_root = os.environ.get(
        "PLANTAER_WORKSPACE", str(Path.cwd() / "plantaer_workspace")
    )
    _ensure_seeded(workspace_root)
    main()
