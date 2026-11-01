import math
import time
import statistics
from typing import List, Dict, Any, Sequence
import pytest

from src.chemistry.validation import validate_smiles
from src.chemistry.descriptors import calculate_descriptors
from src.chemistry.fingerprints import generate_morgan_fingerprint
from src.models.inference import ToxicityInferenceService
from src.similarity.index import ChemicalSimilarityIndex
from src.services.orchestration import ScreeningOrchestrator

TEST_MOLECULES = [
    "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
    "CC(=O)Nc1ccc(O)cc1",      # Paracetamol
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",    # Ibuprofen
    "CN(C)C(=N)NC(=N)N",             # Metformin
    "c1ccccc1",                       # Benzene
    "CCN(CC)CC",                      # Triethylamine
    "O=C1NC(=O)NC(=O)C1(CC)c2ccccc2", # Phenobarbital
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C", # Testosterone
    "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O", # Atorvastatin
]


def calculate_batch_throughput_stats(
    trial_durations_seconds: Sequence[float],
    n_molecules: int,
) -> Dict[str, Any]:
    """Calculate repeatable, unrounded and display-rounded batch throughput metrics.

    Parameters
    ----------
    trial_durations_seconds : Sequence[float]
        Sequence of measured elapsed times in seconds for each batch trial.
    n_molecules : int
        Number of molecules evaluated in each batch trial.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing raw and display-rounded throughput and latency statistics.
    """
    if n_molecules <= 0:
        raise ValueError(f"n_molecules must be positive, got {n_molecules}.")
    if not trial_durations_seconds:
        raise ValueError("trial_durations_seconds cannot be empty.")

    durations: List[float] = []
    for d in trial_durations_seconds:
        d_val = float(d)
        if d_val <= 0.0 or not math.isfinite(d_val):
            raise ValueError(f"Trial duration must be positive and finite, got {d}.")
        durations.append(d_val)

    throughputs = [float(n_molecules) / d for d in durations]
    sorted_d = sorted(durations)
    sorted_tp = sorted(throughputs)

    med_duration = float(statistics.median(durations))
    med_tp = float(statistics.median(throughputs))
    mean_duration = float(statistics.mean(durations))
    mean_tp = float(statistics.mean(throughputs))

    return {
        # Raw unrounded throughput strictly evaluated against the performance SLA (> 0.4 mol/s)
        "raw_molecules_per_second": med_tp,
        # Display-rounded fields preserving existing dictionary schema
        "molecules_per_second": round(med_tp, 1),
        "total_seconds_for_20": round(med_duration, 2),
        # Multi-trial repeatability diagnostics
        "n_trials": len(durations),
        "trial_seconds": [round(d, 3) for d in durations],
        "trial_throughputs": [round(tp, 3) for tp in throughputs],
        "raw_trial_throughputs": throughputs,
        "raw_median_seconds": med_duration,
        "mean_seconds_for_20": round(mean_duration, 2),
        "mean_molecules_per_second": round(mean_tp, 2),
        "min_molecules_per_second": round(sorted_tp[0], 3),
        "max_molecules_per_second": round(sorted_tp[-1], 3),
        "raw_min_molecules_per_second": sorted_tp[0],
        "raw_max_molecules_per_second": sorted_tp[-1],
        "all_trials_pass_sla": all(tp > 0.4 for tp in throughputs),
        "min_seconds_for_20": round(sorted_d[0], 3),
        "max_seconds_for_20": round(sorted_d[-1], 3),
    }


def measure_latencies(fn, iterations: int = 5) -> Dict[str, float]:
    times: List[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms

    sorted_times = sorted(times)
    p50 = sorted_times[int(0.50 * len(sorted_times))]
    p95 = sorted_times[int(0.95 * len(sorted_times))]
    mean_val = statistics.mean(times)
    std_val = statistics.stdev(times) if len(times) > 1 else 0.0

    return {
        "mean_ms": round(mean_val, 2),
        "std_ms": round(std_val, 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "min_ms": round(sorted_times[0], 2),
        "max_ms": round(sorted_times[-1], 2),
    }


def run_comprehensive_benchmarks(n_batch_trials: int = 3) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}

    # 1. SMILES Validation
    def benchmark_validation():
        for smi in TEST_MOLECULES:
            validate_smiles(smi)
    val_stats = measure_latencies(benchmark_validation, iterations=5)
    results["SMILES_Validation_per_mol"] = {
        k: round(v / len(TEST_MOLECULES), 3) for k, v in val_stats.items()
    }

    # 2. Descriptor Calculation
    def benchmark_descriptors():
        for smi in TEST_MOLECULES:
            calculate_descriptors(smi)
    desc_stats = measure_latencies(benchmark_descriptors, iterations=5)
    results["Descriptors_per_mol"] = {
        k: round(v / len(TEST_MOLECULES), 3) for k, v in desc_stats.items()
    }

    # 3. Morgan Fingerprints
    def benchmark_fingerprints():
        for smi in TEST_MOLECULES:
            generate_morgan_fingerprint(smi)
    fp_stats = measure_latencies(benchmark_fingerprints, iterations=5)
    results["Morgan_FP_per_mol"] = {
        k: round(v / len(TEST_MOLECULES), 3) for k, v in fp_stats.items()
    }

    # 4. Inference Services Warmup & Benchmarks
    inf_service = ToxicityInferenceService()
    # Warmup
    inf_service.predict_clintox(TEST_MOLECULES[0])
    inf_service.predict_tox21(TEST_MOLECULES[0])

    def benchmark_clintox():
        for smi in TEST_MOLECULES:
            inf_service.predict_clintox(smi)
    clin_stats = measure_latencies(benchmark_clintox, iterations=5)
    results["ClinTox_Inference_per_mol"] = {
        k: round(v / len(TEST_MOLECULES), 3) for k, v in clin_stats.items()
    }

    def benchmark_tox21():
        for smi in TEST_MOLECULES:
            inf_service.predict_tox21(smi)
    tox21_stats = measure_latencies(benchmark_tox21, iterations=5)
    results["Tox21_12_Endpoint_per_mol"] = {
        k: round(v / len(TEST_MOLECULES), 3) for k, v in tox21_stats.items()
    }

    # 5. Chemical Similarity Search
    sim_idx = ChemicalSimilarityIndex()
    def benchmark_similarity():
        for smi in TEST_MOLECULES:
            sim_idx.search(smi, top_k=5)
    sim_stats = measure_latencies(benchmark_similarity, iterations=5)
    results["Similarity_Search_per_mol"] = {
        k: round(v / len(TEST_MOLECULES), 3) for k, v in sim_stats.items()
    }

    # 6. End-to-end Pipeline (without SHAP)
    orchestrator = ScreeningOrchestrator(inference_service=inf_service, similarity_index=sim_idx)
    def benchmark_pipeline_no_shap():
        for smi in TEST_MOLECULES:
            orchestrator.analyze_single_molecule(smi, compute_explanation=False)
    e2e_stats = measure_latencies(benchmark_pipeline_no_shap, iterations=5)
    results["E2E_SingleMol_per_mol"] = {
        k: round(v / len(TEST_MOLECULES), 3) for k, v in e2e_stats.items()
    }

    # 7. Batch Screening Throughput (Multi-trial repeated measurement)
    batch_workload = TEST_MOLECULES * 2  # 20 compounds
    trial_durations: List[float] = []
    for _ in range(max(1, n_batch_trials)):
        t_start = time.perf_counter()
        _ = orchestrator.screen_batch(batch_workload)
        trial_durations.append(time.perf_counter() - t_start)

    results["Batch_Throughput"] = calculate_batch_throughput_stats(
        trial_durations_seconds=trial_durations,
        n_molecules=len(batch_workload),
    )

    return results


def test_performance_benchmarks():
    """Verify performance metrics meet SLA targets."""
    results = run_comprehensive_benchmarks()
    print("\n=== MOLECULEIQ PERFORMANCE BENCHMARKS ===")
    for module, metrics in results.items():
        print(f"{module}: {metrics}")

    # SLA checks
    assert results["SMILES_Validation_per_mol"]["p50_ms"] < 25.0, "Validation p50 should be < 25ms"
    assert results["Descriptors_per_mol"]["p50_ms"] < 25.0, "Descriptors p50 should be < 25ms"
    assert results["Morgan_FP_per_mol"]["p50_ms"] < 25.0, "Fingerprint p50 should be < 25ms"
    assert results["ClinTox_Inference_per_mol"]["p50_ms"] < 1000.0, "ClinTox inference p50 should be < 1000ms"
    assert results["Similarity_Search_per_mol"]["p50_ms"] < 50.0, "Similarity p50 should be < 50ms"

    batch_stats = results["Batch_Throughput"]
    raw_throughputs = batch_stats.get(
        "raw_trial_throughputs",
        [batch_stats.get("raw_molecules_per_second", batch_stats["molecules_per_second"])],
    )
    trial_durations = batch_stats.get("trial_seconds", [batch_stats.get("total_seconds_for_20")])

    # Per-trial performance integrity: every scored trial must strictly exceed 0.4 mol/s
    failing_trials = [
        {"trial": i + 1, "duration_s": dur, "raw_throughput_mol_per_s": tp}
        for i, (dur, tp) in enumerate(zip(trial_durations, raw_throughputs))
        if tp <= 0.4
    ]

    assert len(failing_trials) == 0, (
        f"Batch throughput SLA violated: every scored trial must strictly exceed 0.4 mol/s. "
        f"Failing trials ({len(failing_trials)}/{len(raw_throughputs)}): {failing_trials}. "
        f"All trial throughputs (raw): {[round(tp, 4) for tp in raw_throughputs]} mol/s. "
        f"All trial durations: {trial_durations} s. "
        f"Median raw throughput: {batch_stats.get('raw_molecules_per_second', 0.0):.4f} mol/s "
        f"(display={batch_stats.get('molecules_per_second')} mol/s)."
    )

    # Median throughput check as secondary summary verification
    raw_median = batch_stats.get("raw_molecules_per_second", batch_stats["molecules_per_second"])
    assert raw_median > 0.4, (
        f"Median batch throughput should exceed 0.4 mol/s: raw={raw_median:.4f} mol/s, "
        f"display={batch_stats.get('molecules_per_second')} mol/s"
    )


if __name__ == "__main__":
    bench = run_comprehensive_benchmarks()
    for k, v in bench.items():
        print(f"{k}: {v}")
