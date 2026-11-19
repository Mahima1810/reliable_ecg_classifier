import pytest

from ecgrc.labels import CLASSES, UnknownSymbolError, SymbolMap, check_map_consistency


def test_map_is_internally_consistent():
    assert check_map_consistency() == []


def test_standard_aami_classes():
    m = SymbolMap.load()
    assert m.classify("N") == "N"
    assert m.classify("L") == "N"
    assert m.classify("R") == "N"
    assert m.classify("A") == "S"
    assert m.classify("V") == "V"
    assert m.classify("E") == "V"
    assert m.classify("F") == "F"


def test_q_is_never_normal():
    m = SymbolMap.load()
    for sym in ["/", "f", "Q", "?"]:
        assert m.classify(sym) == "Q"
        assert m.classify(sym) != "N"


def test_provisional_symbols_blocked_by_default():
    m = SymbolMap.load(allow_provisional=False)
    for sym in ["B", "n", "r"]:
        with pytest.raises(UnknownSymbolError):
            m.classify(sym, dataset="mitdb", record="100", sample=5)


def test_provisional_symbols_allowed_when_enabled():
    m = SymbolMap.load(allow_provisional=True)
    assert m.classify("B") == "N"
    assert m.classify("r") == "V"


def test_non_beat_symbols_return_none():
    m = SymbolMap.load()
    for sym in ["+", "~", "|", "T", "p"]:
        assert m.classify(sym) is None
        assert not m.is_beat(sym)


def test_unknown_symbol_raises_not_normal():
    m = SymbolMap.load()
    with pytest.raises(UnknownSymbolError):
        m.classify("Z", dataset="svdb", record="800", sample=10)


def test_scored_classes_exclude_q():
    assert "Q" not in CLASSES
    assert CLASSES == ("N", "S", "V", "F")
