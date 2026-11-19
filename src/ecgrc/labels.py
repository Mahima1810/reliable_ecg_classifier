"""AAMI beat-class mapping with explicit failure modes.

Rules:
- Every beat symbol must be in the standard map, or in the provisional map when provisional mapping is allowed.
- Non-beat symbols are listed and excluded from beat tables.
- Anything else raises UnknownSymbolError. Nothing is silently mapped to N, and Q is never mapped to N.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import label_config

CLASSES = ("N", "S", "V", "F")          # scored classes
EXCLUDED = ("Q",)                        # counted, never scored or trained on


class UnknownSymbolError(ValueError):
    pass


@dataclass(frozen=True)
class SymbolMap:
    standard: dict
    provisional: dict
    non_beat: frozenset
    allow_provisional: bool

    @classmethod
    def load(cls, allow_provisional: bool = False) -> "SymbolMap":
        cfg = label_config()
        return cls(standard=dict(cfg["standard"]), provisional=dict(cfg["provisional"]),
                   non_beat=frozenset(str(s) for s in cfg["non_beat"]), allow_provisional=allow_provisional)

    def classify(self, symbol: str, dataset: str = "", record: str = "", sample: int = -1) -> str | None:
        """Return the AAMI class for a beat symbol, or None for a non-beat symbol.

        Raises UnknownSymbolError for any symbol not covered by the maps.
        """
        if symbol in self.standard:
            return self.standard[symbol]
        if symbol in self.non_beat:
            return None
        if symbol in self.provisional:
            if not self.allow_provisional:
                raise UnknownSymbolError(
                    f"provisional symbol {symbol!r} in {dataset}/{record} at sample {sample}; "
                    "set allow_provisional=true in the experiment config to use it")
            return self.provisional[symbol]
        raise UnknownSymbolError(f"unmapped annotation symbol {symbol!r} in {dataset}/{record} at sample {sample}")

    def is_beat(self, symbol: str) -> bool:
        return symbol in self.standard or symbol in self.provisional


def check_map_consistency() -> list[str]:
    """Return a list of problems in the label config (empty when consistent)."""
    m = SymbolMap.load(allow_provisional=True)
    problems = []
    overlap = set(m.standard) & set(m.non_beat)
    if overlap:
        problems.append(f"symbols both beat and non-beat: {sorted(overlap)}")
    for sym, cls in m.standard.items():
        if cls not in CLASSES + EXCLUDED:
            problems.append(f"standard symbol {sym!r} maps to unknown class {cls!r}")
    for sym, cls in m.provisional.items():
        if cls not in CLASSES:
            problems.append(f"provisional symbol {sym!r} maps to unknown class {cls!r}")
    if set(m.standard) & set(m.provisional):
        problems.append("symbols in both standard and provisional maps")
    return problems
