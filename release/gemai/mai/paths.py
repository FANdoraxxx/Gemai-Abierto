"""Project-root discovery for a copied generator tree.

The stock tree lives at ``<project>/release/v6_cal``, so its modules can derive the project root by counting parents
(``Path(__file__).resolve().parents[k]``).  This copy lives deeper (``<project>/scratch/chorus3/gen``), which would make
every such count wrong (``official charts``, ``work/charts.pkl``, ``scratch/plan``, ``out/plan/targets.json`` ...).
Everything that needs the project root asks here instead: the first ancestor directory that contains ``work/charts.pkl``
and ``official charts`` (overridable with ``MAI_PROJECT_ROOT``).
"""
from __future__ import annotations

import os
from pathlib import Path


def project_root(start=None):
    env = os.environ.get("MAI_PROJECT_ROOT")
    if env and Path(env).is_dir():
        return Path(env)
    p = Path(start or __file__).resolve()
    for cand in [p] + list(p.parents):
        if (cand / "work" / "charts.pkl").is_file() and (cand / "official charts").is_dir():
            return cand
    # last resort: the stock layout (mai/x.py -> project = parents[3])
    return p.parents[3]


def tree_root(start=None):
    """Root of this generator tree (the directory that holds mai/, tools/, work/)."""
    return Path(start or __file__).resolve().parents[1]
