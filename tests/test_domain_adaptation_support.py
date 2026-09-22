import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from domain_adaptation import split_support_query  # noqa: E402

# One subject, 60 s windows on a 30 s step: baseline 0-270 s, stress 300-570 s.
STARTS = np.arange(0, 600, 30.0)
FILES = np.array(["S1"] * len(STARTS))
Y = (STARTS >= 300).astype(int)
IDX = np.arange(len(STARTS))


def test_chronological_takes_first_windows_of_each_class():
    support, _ = split_support_query(IDX, Y, 2, np.random.default_rng(0), FILES, STARTS, "chronological")
    assert list(STARTS[support]) == [0, 30, 300, 330]


def test_gap_removes_every_query_window_near_support():
    support, query = split_support_query(IDX, Y, 2, np.random.default_rng(0), FILES, STARTS, "chronological", gap=30)
    for s in support:
        assert np.all(np.abs(STARTS[query] - STARTS[s]) >= 90)
    assert 420 in STARTS[query] and 120 in STARTS[query] and 390 not in STARTS[query] and 90 not in STARTS[query]


def test_gap_only_applies_within_the_same_recording_file():
    # f14_a: baseline at 0-270 s; f14_b: stress restarting its clock, first window at 200 s.
    files = np.array(["f14_a"] * 10 + ["f14_b"] * 10)
    starts = np.concatenate([np.arange(0, 300, 30.0), np.arange(200, 500, 30.0)])
    y = np.array([0] * 10 + [1] * 10)
    support, query = split_support_query(np.arange(20), y, 1, np.random.default_rng(0), files, starts, "chronological", gap=30)
    assert set(support) == {0, 10}
    assert {6, 7} <= set(query)  # f14_a at 180 s and 210 s: near f14_b's support time, but a different file
    assert 11 not in query and 12 not in query and 13 in query  # f14_b at 230 and 260 s excluded, 290 s kept


def test_random_mode_without_gap_keeps_neighbours_in_query():
    support, query = split_support_query(IDX, Y, 2, np.random.default_rng(0), FILES, STARTS, "random")
    assert len(support) == 4 and len(query) == len(IDX) - 4
