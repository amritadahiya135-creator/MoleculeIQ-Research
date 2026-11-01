import pytest
import math
from typing import List, Dict, Any
from tests.benchmarks.test_benchmarks import calculate_batch_throughput_stats


def test_three_passing_trials_pass():
    """Verify that when all three trials exceed 0.4 mol/s, the benchmark passes."""
    durations = [25.0, 30.0, 35.0]  # 0.80 mol/s, 0.667 mol/s, 0.571 mol/s
    n_mols = 20
    stats = calculate_batch_throughput_stats(durations, n_mols)

    assert stats["n_trials"] == 3
    assert stats["all_trials_pass_sla"] is True
    assert all(tp > 0.4 for tp in stats["raw_trial_throughputs"])
    assert stats["raw_min_molecules_per_second"] > 0.4
    assert stats["total_seconds_for_20"] == 30.0
    assert pytest.approx(stats["raw_molecules_per_second"], 1e-4) == 20.0 / 30.0
    assert stats["molecules_per_second"] == 0.7


def test_trial_at_exactly_0_4_fails():
    """Verify that a trial at exactly 0.4 mol/s strictly fails the > 0.4 SLA requirement."""
    # 20 molecules in 50.0 seconds -> exactly 0.4000 mol/s
    stats_single = calculate_batch_throughput_stats([50.0], n_molecules=20)
    assert stats_single["raw_trial_throughputs"][0] == 0.4
    assert not (stats_single["raw_trial_throughputs"][0] > 0.4)
    assert stats_single["all_trials_pass_sla"] is False

    # In a 3-trial run where 2 pass and 1 is exactly 0.4
    stats_multi = calculate_batch_throughput_stats([25.0, 30.0, 50.0], n_molecules=20)
    assert stats_multi["all_trials_pass_sla"] is False
    assert stats_multi["raw_min_molecules_per_second"] == 0.4
    failing = [tp for tp in stats_multi["raw_trial_throughputs"] if tp <= 0.4]
    assert len(failing) == 1
    assert failing[0] == 0.4


def test_trial_below_0_4_fails_even_if_median_passes():
    """Review of 80-second outlier: A trial below 0.4 mol/s must FAIL,

    preventing a passing median from concealing a severe performance regression.
    """
    # 2 healthy trials (~0.8 mol/s) and 1 extreme stall/regression trial (80s -> 0.25 mol/s)
    durations = [25.0, 26.0, 80.0]
    stats = calculate_batch_throughput_stats(durations, n_molecules=20)

    # Median summary passes (> 0.4 mol/s)
    assert stats["total_seconds_for_20"] == 26.0
    assert stats["raw_molecules_per_second"] > 0.4
    assert pytest.approx(stats["raw_molecules_per_second"], 1e-4) == 20.0 / 26.0

    # BUT per-trial performance integrity strictly FAILS
    assert stats["all_trials_pass_sla"] is False
    assert stats["raw_min_molecules_per_second"] == 20.0 / 80.0
    assert stats["raw_min_molecules_per_second"] < 0.4

    # The failing trial is fully identified and exposed
    failing_trials = [
        (i + 1, d, tp)
        for i, (d, tp) in enumerate(zip(stats["trial_seconds"], stats["raw_trial_throughputs"]))
        if tp <= 0.4
    ]
    assert len(failing_trials) == 1
    idx, dur, tp = failing_trials[0]
    assert idx == 3
    assert dur == 80.0
    assert tp == 0.25


def test_raw_values_control_acceptance_not_display_rounding():
    """Verify that raw unrounded values control acceptance, not display rounding."""
    # Case A: Borderline PASS
    # 20 molecules in 49.0s -> 20 / 49.0 = 0.408163... mol/s
    # Display rounded to 1 decimal is 0.4 (which would fail if asserted directly)
    # But raw is strictly > 0.4, so it must PASS
    stats_pass = calculate_batch_throughput_stats([49.0, 49.0, 49.0], n_molecules=20)
    assert stats_pass["molecules_per_second"] == 0.4  # rounded down
    assert stats_pass["raw_min_molecules_per_second"] > 0.4
    assert stats_pass["all_trials_pass_sla"] is True

    # Case B: Borderline FAIL
    # 20 molecules in 50.05s -> 20 / 50.05 = 0.399600... mol/s
    # Display rounded to 1 decimal is 0.4 (which might look like a pass if threshold was >= 0.4)
    # But raw is strictly < 0.4, so it must FAIL
    stats_fail = calculate_batch_throughput_stats([50.05, 50.05, 50.05], n_molecules=20)
    assert stats_fail["molecules_per_second"] == 0.4  # rounded up
    assert stats_fail["raw_min_molecules_per_second"] < 0.4
    assert stats_fail["all_trials_pass_sla"] is False


def test_diagnostic_visibility_on_failure():
    """Verify that all trials, raw values, and durations remain visible for diagnostics."""
    durations = [25.0, 52.0, 24.0]
    stats = calculate_batch_throughput_stats(durations, n_molecules=20)

    assert stats["all_trials_pass_sla"] is False
    assert len(stats["trial_seconds"]) == 3
    assert len(stats["raw_trial_throughputs"]) == 3
    assert stats["trial_seconds"] == [25.0, 52.0, 24.0]

    # Trial 2 is accurately flagged with duration 52.0s and throughput 20/52 mol/s
    failing = [
        {"trial": i + 1, "duration_s": dur, "tp": tp}
        for i, (dur, tp) in enumerate(zip(stats["trial_seconds"], stats["raw_trial_throughputs"]))
        if tp <= 0.4
    ]
    assert len(failing) == 1
    assert failing[0]["trial"] == 2
    assert failing[0]["duration_s"] == 52.0
    assert pytest.approx(failing[0]["tp"], 1e-5) == 20.0 / 52.0


def test_schema_backward_compatibility():
    """Verify that all backward-compatible and new diagnostic keys exist."""
    stats = calculate_batch_throughput_stats([25.0], n_molecules=20)
    expected_keys = {
        "raw_molecules_per_second",
        "molecules_per_second",
        "total_seconds_for_20",
        "n_trials",
        "trial_seconds",
        "trial_throughputs",
        "raw_trial_throughputs",
        "raw_median_seconds",
        "mean_seconds_for_20",
        "mean_molecules_per_second",
        "min_molecules_per_second",
        "max_molecules_per_second",
        "raw_min_molecules_per_second",
        "raw_max_molecules_per_second",
        "all_trials_pass_sla",
        "min_seconds_for_20",
        "max_seconds_for_20",
    }
    assert expected_keys.issubset(stats.keys())


@pytest.mark.parametrize(
    "invalid_durations, n_mols, exc_type",
    [
        ([], 20, ValueError),                    # Empty trials
        ([20.0], 0, ValueError),                 # Zero molecules
        ([20.0], -5, ValueError),                # Negative molecules
        ([0.0], 20, ValueError),                 # Zero duration
        ([-10.0], 20, ValueError),               # Negative duration
        ([float("nan")], 20, ValueError),        # NaN duration
        ([float("inf")], 20, ValueError),        # Inf duration
    ],
)
def test_input_validation_errors(invalid_durations, n_mols, exc_type):
    """Verify input validation handles edge cases and invalid inputs gracefully."""
    with pytest.raises(exc_type):
        calculate_batch_throughput_stats(invalid_durations, n_molecules=n_mols)
