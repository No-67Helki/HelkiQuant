from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from qlib.data.dataset.processor import Processor


class FeatureWhitelist(Processor):
    """Keep only whitelisted feature columns.

    The whitelist JSON is produced by ``prune_features.py`` and has a ``kept``
    field. Non-feature groups such as labels are always preserved.
    """

    def __init__(self, whitelist_path: str, fields_group: str = "feature"):
        self.whitelist_path = whitelist_path
        self.fields_group = fields_group
        self._kept: set[str] | None = None

    def _load_kept(self) -> set[str]:
        if self._kept is None:
            path = Path(self.whitelist_path).expanduser()
            if not path.is_absolute():
                path = Path(__file__).resolve().parent / path
            data = json.loads(path.read_text(encoding="utf-8"))
            kept = data.get("kept", data if isinstance(data, list) else [])
            self._kept = {str(c) for c in kept}
        return self._kept

    def __call__(self, df: pd.DataFrame):
        kept = self._load_kept()
        if not isinstance(df.columns, pd.MultiIndex):
            cols = [c for c in df.columns if str(c) in kept]
            return df.loc[:, cols]

        cols = []
        for col in df.columns:
            group = col[0]
            name = col[-1]
            if group != self.fields_group or str(name) in kept:
                cols.append(col)
        return df.loc[:, cols]
