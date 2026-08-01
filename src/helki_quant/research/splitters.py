from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterator, Sequence

import pandas as pd


@dataclass(frozen=True)
class PurgedFold:
    fold: int
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str
    test_start: str
    test_end: str
    purge_days: int
    embargo_days: int

    def as_dict(self) -> dict:
        return asdict(self)


class PurgedWalkForwardSplitter:
    """Expanding walk-forward splitter with gaps around validation and test."""

    def __init__(
        self,
        *,
        min_train_days: int = 500,
        valid_days: int = 120,
        test_days: int = 60,
        step_days: int = 60,
        purge_days: int = 21,
        embargo_days: int = 5,
    ) -> None:
        if min(min_train_days, valid_days, test_days, step_days) <= 0:
            raise ValueError("window lengths must be positive")
        if purge_days < 0 or embargo_days < 0:
            raise ValueError("purge_days and embargo_days must be non-negative")
        self.min_train_days = int(min_train_days)
        self.valid_days = int(valid_days)
        self.test_days = int(test_days)
        self.step_days = int(step_days)
        self.purge_days = int(purge_days)
        self.embargo_days = int(embargo_days)

    def split(self, dates: Sequence) -> Iterator[PurgedFold]:
        calendar = pd.DatetimeIndex(pd.to_datetime(dates)).drop_duplicates().sort_values()
        valid_start = self.min_train_days + self.purge_days
        fold_no = 1
        while True:
            train_end = valid_start - self.purge_days - 1
            valid_end = valid_start + self.valid_days - 1
            test_start = valid_end + self.purge_days + self.embargo_days + 1
            test_end = test_start + self.test_days - 1
            if test_end >= len(calendar):
                break
            yield PurgedFold(
                fold=fold_no,
                train_start=str(calendar[0].date()),
                train_end=str(calendar[train_end].date()),
                valid_start=str(calendar[valid_start].date()),
                valid_end=str(calendar[valid_end].date()),
                test_start=str(calendar[test_start].date()),
                test_end=str(calendar[test_end].date()),
                purge_days=self.purge_days,
                embargo_days=self.embargo_days,
            )
            fold_no += 1
            valid_start += self.step_days
