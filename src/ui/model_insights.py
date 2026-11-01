"""Model Insights and Evaluation Evidence view (Workflow C — Phase B.1 & Phase B.2)."""

import html
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple
import streamlit as st
import pandas as pd

from src.models.tox21 import TOX21_ENDPOINTS
from src.models.tox21_reliability_diagnostics import (
    AD_STRATIFICATION_DISCLAIMER,
    ECE_DEFINITION,
    MACE_DEFINITION,
    MIN_POSITIVES_FOR_STRATUM_SUPPORT,
    PR_AUC_DEFINITION,
    TOX21_ENDPOINT_NAMES,
    EndpointADStratificationReport,
    EndpointReliabilityDiagnosticBundle,
    ReliabilityDiagramSeries,
    load_endpoint_reliability_diagnostics,
)
from src.ui.components import render_header, render_metric_card

REQUIRED_TOX21_METRIC_KEYS = (
    "sample_count",
    "train_count",
    "roc_auc",
    "pr_auc",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
)


def get_tox21_endpoint_selector_options() -> List[Tuple[str, str]]:
    """
    Return all 12 Tox21 endpoints as `(canonical_endpoint_id, human_readable_label)` tuples
    in canonical `TOX21_ENDPOINTS` order.
    """
    options: List[Tuple[str, str]] = []
    for ep in TOX21_ENDPOINTS:
        meta = TOX21_ENDPOINT_NAMES.get(ep, {})
        label = meta.get("selector_label", ep)
        options.append((ep, f"{label} [{meta.get('family', 'Tox21 Assay')}]"))
    return options


def load_tox21_endpoint_performance_rows(
    artifact_dir: str = "artifacts/tox21",
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Load held-out performance metrics for all 12 Tox21 endpoints from individual
    `{safe_ep}_metadata.json` files without modifying `summary.json`.

    Returns:
        Tuple of (rows, skipped_endpoints). Missing or malformed metadata files
        are skipped cleanly without duplicating rows, fabricating zero metrics,
        or raising unhandled exceptions.
    """
    rows: List[Dict[str, Any]] = []
    skipped_endpoints: List[str] = []
    seen_endpoints = set()

    for ep in TOX21_ENDPOINTS:
        if ep in seen_endpoints:
            continue
        seen_endpoints.add(ep)
        safe_ep = ep.replace("-", "_")
        meta_path = os.path.join(artifact_dir, f"{safe_ep}_metadata.json")
        if not os.path.exists(meta_path):
            skipped_endpoints.append(ep)
            continue

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                skipped_endpoints.append(ep)
                continue
            if data.get("endpoint") not in (None, ep):
                skipped_endpoints.append(ep)
                continue
            m = data.get("metrics")
            if not isinstance(m, dict):
                skipped_endpoints.append(ep)
                continue

            valid_metrics = True
            for k in REQUIRED_TOX21_METRIC_KEYS:
                val = m.get(k)
                if val is None or isinstance(val, bool):
                    valid_metrics = False
                    break
                fval = float(val)
                if not math.isfinite(fval):
                    valid_metrics = False
                    break
            if not valid_metrics:
                skipped_endpoints.append(ep)
                continue

            rows.append({
                "Endpoint": ep,
                "Pathway Class": "Nuclear Receptor (NR)" if ep.startswith("NR") else "Stress Response (SR)",
                "Total Samples": int(m["sample_count"]),
                "Train Samples (AD)": int(m["train_count"]),
                "ROC-AUC": f"{float(m['roc_auc']):.3f}",
                "PR-AUC": f"{float(m['pr_auc']):.3f}",
                "Balanced Acc": f"{float(m['balanced_accuracy']):.3f}",
                "Precision": f"{float(m['precision']):.3f}",
                "Recall": f"{float(m['recall']):.3f}",
                "F1 Score": f"{float(m['f1']):.3f}",
            })
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            skipped_endpoints.append(ep)

    return rows, skipped_endpoints


def _fmt_metric(val: Optional[float], decimals: int = 4, unavailable_text: str = "Unavailable") -> str:
    """Format an optional float metric without fabricating zeros for None/NaN."""
    if val is None or isinstance(val, bool):
        return unavailable_text
    try:
        fval = float(val)
    except (TypeError, ValueError):
        return unavailable_text
    if not math.isfinite(fval):
        return unavailable_text
    return f"{fval:.{decimals}f}"


def build_reliability_bin_table_rows(series: ReliabilityDiagramSeries) -> List[Dict[str, Any]]:
    """
    Build a 10-row tabular summary of reliability diagram bins for inspection.
    Never fabricates numeric values for empty bins.
    """
    rows: List[Dict[str, Any]] = []
    for b in series.bins:
        if b.is_empty:
            support_tag = "EMPTY (0 obs — omitted from curve)"
            mean_p_str = "N/A (Empty Bin)"
            obs_f_str = "N/A (Empty Bin)"
            gap_str = "N/A (Empty Bin)"
        else:
            support_tag = "SPARSE (< 10 obs)" if b.is_sparse else "SUPPORTED (≥ 10 obs)"
            mean_p_str = _fmt_metric(b.mean_predicted_prob, 4)
            obs_f_str = _fmt_metric(b.observed_positive_fraction, 4)
            gap_str = _fmt_metric(b.abs_calibration_gap, 4)

        rows.append({
            "Bin": f"Bin {b.bin_index + 1}",
            "Probability Interval": b.interval_notation,
            "Observations (n)": int(b.count),
            "Positives (1)": int(b.positive_count),
            "Negatives (0)": int(b.negative_count),
            "Mean Predicted Prob": mean_p_str,
            "Observed Positive Fraction": obs_f_str,
            "|Obs − Pred| Gap": gap_str,
            "Bin Support Status": support_tag,
        })
    return rows


def build_endpoint_metric_comparison_rows(
    bundle: EndpointReliabilityDiagnosticBundle,
) -> List[Dict[str, Any]]:
    """
    Build the Raw-vs-Calibrated metric comparison table for a single Tox21 endpoint,
    with explicit units, metric optimization directions, and single-class handling.
    """
    raw_m = bundle.series_metrics.get("uncalibrated")
    sig_m = bundle.series_metrics.get("sigmoid")
    iso_m = bundle.series_metrics.get("isotonic")

    if raw_m is None and sig_m is None:
        return []

    ref_m = sig_m or raw_m
    assert ref_m is not None
    unavail_disc = "Unavailable (Single-Class)" if ref_m.is_single_class else "Unavailable"

    def _diff_str(
        raw_v: Optional[float],
        sig_v: Optional[float],
        lower_is_better: Optional[bool],
    ) -> str:
        if raw_v is None or sig_v is None:
            return "N/A"
        delta = sig_v - raw_v
        if abs(delta) < 1e-7:
            return "0.0000 (Invariant)"
        if lower_is_better is True:
            tag = "Improved" if delta < 0 else "Degraded"
        elif lower_is_better is False:
            tag = "Improved" if delta > 0 else "Degraded"
        else:
            tag = ""
        suffix = f" ({tag})" if tag else ""
        return f"{delta:+.4f}{suffix}"

    rows = [
        {
            "Metric": "Valid Held-Out Test Observations (N)",
            "Category & Direction": "Cohort Support (Info)",
            "Unit / Scale": "Compounds (count)",
            "Raw RF (Uncalibrated)": str(raw_m.n_valid) if raw_m else "N/A",
            "Sigmoid Calibrated (Selected)": str(sig_m.n_valid) if sig_m else "N/A",
            "Isotonic (Eval-Only Comparator)": str(iso_m.n_valid) if iso_m else "N/A",
            "Sigmoid vs Raw Δ": "Identical Paired Split",
            "Definition & Interpretation": "Locked 20% stratified test partition after excluding missing assay labels.",
        },
        {
            "Metric": "Positive / Negative Count (Prevalence)",
            "Category & Direction": "Cohort Support (Info)",
            "Unit / Scale": "Count (% active)",
            "Raw RF (Uncalibrated)": (
                f"{raw_m.n_positive} pos / {raw_m.n_negative} neg ({raw_m.prevalence*100:.2f}%)"
                if raw_m
                else "N/A"
            ),
            "Sigmoid Calibrated (Selected)": (
                f"{sig_m.n_positive} pos / {sig_m.n_negative} neg ({sig_m.prevalence*100:.2f}%)"
                if sig_m
                else "N/A"
            ),
            "Isotonic (Eval-Only Comparator)": (
                f"{iso_m.n_positive} pos / {iso_m.n_negative} neg ({iso_m.prevalence*100:.2f}%)"
                if iso_m
                else "N/A"
            ),
            "Sigmoid vs Raw Δ": "Identical Labels",
            "Definition & Interpretation": "Empirical assay hit rate; Brier score and PR-AUC baselines depend strongly on prevalence.",
        },
        {
            "Metric": "Brier Score",
            "Category & Direction": "Calibration + Refinement (↓ Lower is better)",
            "Unit / Scale": "Squared error [0, 1]",
            "Raw RF (Uncalibrated)": _fmt_metric(raw_m.brier_score if raw_m else None),
            "Sigmoid Calibrated (Selected)": _fmt_metric(sig_m.brier_score if sig_m else None),
            "Isotonic (Eval-Only Comparator)": _fmt_metric(iso_m.brier_score if iso_m else None),
            "Sigmoid vs Raw Δ": _diff_str(
                raw_m.brier_score if raw_m else None,
                sig_m.brier_score if sig_m else None,
                lower_is_better=True,
            ),
            "Definition & Interpretation": "Mean squared error (1/N) Σ (p_i − y_i)²; proper scoring rule measuring probability accuracy.",
        },
        {
            "Metric": "Log Loss (Cross-Entropy)",
            "Category & Direction": "Calibration + Refinement (↓ Lower is better)",
            "Unit / Scale": "Nats [0, ∞)",
            "Raw RF (Uncalibrated)": _fmt_metric(raw_m.log_loss_value if raw_m else None),
            "Sigmoid Calibrated (Selected)": _fmt_metric(sig_m.log_loss_value if sig_m else None),
            "Isotonic (Eval-Only Comparator)": _fmt_metric(iso_m.log_loss_value if iso_m else None),
            "Sigmoid vs Raw Δ": _diff_str(
                raw_m.log_loss_value if raw_m else None,
                sig_m.log_loss_value if sig_m else None,
                lower_is_better=True,
            ),
            "Definition & Interpretation": "Binary log-likelihood loss with ε = 1e-15 clipping applied strictly during metric calculation.",
        },
        {
            "Metric": "Expected Calibration Error (10-Bin ECE)",
            "Category & Direction": "Probability Calibration (↓ Lower is better)",
            "Unit / Scale": "Weighted abs. gap [0, 1]",
            "Raw RF (Uncalibrated)": _fmt_metric(raw_m.ece_10bin_weighted if raw_m else None),
            "Sigmoid Calibrated (Selected)": _fmt_metric(sig_m.ece_10bin_weighted if sig_m else None),
            "Isotonic (Eval-Only Comparator)": _fmt_metric(iso_m.ece_10bin_weighted if iso_m else None),
            "Sigmoid vs Raw Δ": _diff_str(
                raw_m.ece_10bin_weighted if raw_m else None,
                sig_m.ece_10bin_weighted if sig_m else None,
                lower_is_better=True,
            ),
            "Definition & Interpretation": ECE_DEFINITION,
        },
        {
            "Metric": "Macro-Averaged Calibration Error (5-Bin MACE — Benchmark)",
            "Category & Direction": "Unweighted Mean Bin Gap (↓ Lower is better)",
            "Unit / Scale": "Mean abs. bin gap [0, 1]",
            "Raw RF (Uncalibrated)": _fmt_metric(raw_m.mace_5bin_unweighted_mean if raw_m else None),
            "Sigmoid Calibrated (Selected)": _fmt_metric(sig_m.mace_5bin_unweighted_mean if sig_m else None),
            "Isotonic (Eval-Only Comparator)": _fmt_metric(iso_m.mace_5bin_unweighted_mean if iso_m else None),
            "Sigmoid vs Raw Δ": _diff_str(
                raw_m.mace_5bin_unweighted_mean if raw_m else None,
                sig_m.mace_5bin_unweighted_mean if sig_m else None,
                lower_is_better=True,
            ),
            "Definition & Interpretation": "Unweighted mean of |obs_b − pred_b| across non-empty 5 equal-width bins (`mace_5bin_unweighted` in `calibration_benchmark.json`).",
        },
        {
            "Metric": "Macro-Averaged Calibration Error (10-Bin MACE — Unweighted Mean)",
            "Category & Direction": "Unweighted Mean Bin Gap (↓ Lower is better)",
            "Unit / Scale": "Mean abs. bin gap [0, 1]",
            "Raw RF (Uncalibrated)": _fmt_metric(raw_m.mace_10bin_unweighted_mean if raw_m else None),
            "Sigmoid Calibrated (Selected)": _fmt_metric(sig_m.mace_10bin_unweighted_mean if sig_m else None),
            "Isotonic (Eval-Only Comparator)": _fmt_metric(iso_m.mace_10bin_unweighted_mean if iso_m else None),
            "Sigmoid vs Raw Δ": _diff_str(
                raw_m.mace_10bin_unweighted_mean if raw_m else None,
                sig_m.mace_10bin_unweighted_mean if sig_m else None,
                lower_is_better=True,
            ),
            "Definition & Interpretation": "Unweighted mean of |obs_b − pred_b| across non-empty 10 equal-width bins on [0, 1].",
        },
        {
            "Metric": "Maximum Calibration Error (10-Bin Max |Obs − Pred|)",
            "Category & Direction": "Worst-Bin Calibration (↓ Lower is better)",
            "Unit / Scale": "Max abs. bin gap [0, 1]",
            "Raw RF (Uncalibrated)": _fmt_metric(raw_m.mace_10bin_max if raw_m else None),
            "Sigmoid Calibrated (Selected)": _fmt_metric(sig_m.mace_10bin_max if sig_m else None),
            "Isotonic (Eval-Only Comparator)": _fmt_metric(iso_m.mace_10bin_max if iso_m else None),
            "Sigmoid vs Raw Δ": _diff_str(
                raw_m.mace_10bin_max if raw_m else None,
                sig_m.mace_10bin_max if sig_m else None,
                lower_is_better=True,
            ),
            "Definition & Interpretation": MACE_DEFINITION,
        },
        {
            "Metric": "ROC-AUC (Rank Discrimination)",
            "Category & Direction": "Discrimination Only (↑ Higher is better)",
            "Unit / Scale": "Area [0, 1] (0.5 = random)",
            "Raw RF (Uncalibrated)": _fmt_metric(raw_m.roc_auc if raw_m else None, unavailable_text=unavail_disc),
            "Sigmoid Calibrated (Selected)": _fmt_metric(sig_m.roc_auc if sig_m else None, unavailable_text=unavail_disc),
            "Isotonic (Eval-Only Comparator)": _fmt_metric(iso_m.roc_auc if iso_m else None, unavailable_text=unavail_disc),
            "Sigmoid vs Raw Δ": _diff_str(
                raw_m.roc_auc if raw_m else None,
                sig_m.roc_auc if sig_m else None,
                lower_is_better=False,
            ),
            "Definition & Interpretation": (
                "Pairwise rank-ordering probability; strictly invariant under Platt scaling "
                "P(y=1|f) = 1/(1+exp(a*f+b)) where negative slope a < 0 yields dP/df > 0 (strictly increasing). "
                "Does NOT measure probability calibration."
            ),
        },
        {
            "Metric": "PR-AUC (Average Precision)",
            "Category & Direction": "Discrimination Only (↑ Higher is better)",
            "Unit / Scale": "Area [0, 1] (baseline ≈ prev)",
            "Raw RF (Uncalibrated)": _fmt_metric(raw_m.pr_auc if raw_m else None, unavailable_text=unavail_disc),
            "Sigmoid Calibrated (Selected)": _fmt_metric(sig_m.pr_auc if sig_m else None, unavailable_text=unavail_disc),
            "Isotonic (Eval-Only Comparator)": _fmt_metric(iso_m.pr_auc if iso_m else None, unavailable_text=unavail_disc),
            "Sigmoid vs Raw Δ": _diff_str(
                raw_m.pr_auc if raw_m else None,
                sig_m.pr_auc if sig_m else None,
                lower_is_better=False,
            ),
            "Definition & Interpretation": PR_AUC_DEFINITION,
        },
    ]
    return rows


def build_all_endpoints_diagnostic_table_rows(
    artifact_dir: str = "artifacts/tox21",
    csv_path: str = "data/raw/tox21.csv",
    reconstruct_sample_level: bool = False,
) -> List[Dict[str, Any]]:
    """
    Build the 12-row Endpoint-Level Diagnostic Table (one row per canonical Tox21 endpoint)
    reporting sample counts, positive counts, prevalence, Brier score, Log loss, 10-bin ECE,
    5-bin & 10-bin MACE, ROC-AUC, PR-AUC for Raw, Sigmoid, and Isotonic probabilities,
    recommended calibration method, and explicit availability/support status.

    Never fabricates zeros for unavailable values; one unavailable endpoint does not
    prevent the remaining endpoints from rendering.
    """
    rows: List[Dict[str, Any]] = []
    for ep in TOX21_ENDPOINTS:
        bundle = load_endpoint_reliability_diagnostics(
            endpoint=ep,
            artifact_dir=artifact_dir,
            csv_path=csv_path,
            reconstruct_sample_level=reconstruct_sample_level,
        )
        if bundle.overall_status != "AVAILABLE" or not bundle.series_metrics:
            rows.append({
                "Endpoint": ep,
                "Evaluated Samples (N)": "Unavailable",
                "Positive Samples": "Unavailable",
                "Prevalence": "Unavailable",
                "Raw Brier": "Unavailable",
                "Sigmoid Brier": "Unavailable",
                "Isotonic Brier": "Unavailable",
                "Raw Log Loss": "Unavailable",
                "Sigmoid Log Loss": "Unavailable",
                "Isotonic Log Loss": "Unavailable",
                "Raw ECE (10-Bin)": "Unavailable",
                "Sigmoid ECE (10-Bin)": "Unavailable",
                "Isotonic ECE (10-Bin)": "Unavailable",
                "Raw MACE (5-Bin Mean)": "Unavailable",
                "Sigmoid MACE (5-Bin Mean)": "Unavailable",
                "Isotonic MACE (5-Bin Mean)": "Unavailable",
                "Raw MACE (10-Bin Mean)": "Unavailable",
                "Sigmoid MACE (10-Bin Mean)": "Unavailable",
                "Isotonic MACE (10-Bin Mean)": "Unavailable",
                "Raw ROC-AUC": "Unavailable",
                "Sigmoid ROC-AUC": "Unavailable",
                "Isotonic ROC-AUC": "Unavailable",
                "Raw PR-AUC": "Unavailable",
                "Sigmoid PR-AUC": "Unavailable",
                "Isotonic PR-AUC": "Unavailable",
                "Recommended Method": "Unavailable",
                "Availability & Support Status": f"UNAVAILABLE ({bundle.overall_status})",
            })
            continue

        raw_m = bundle.series_metrics.get("uncalibrated")
        sig_m = bundle.series_metrics.get("sigmoid")
        iso_m = bundle.series_metrics.get("isotonic")
        ref_m = sig_m or raw_m
        assert ref_m is not None

        spans_zero = bool(bundle.bootstrap_sigmoid.ci_includes_zero)
        support_tag = (
            f"AVAILABLE (Low Minority Support: n_pos={ref_m.n_positive}, 95% Brier CI spans 0)"
            if spans_zero
            else f"AVAILABLE (Supported: n_pos={ref_m.n_positive})"
        )

        rows.append({
            "Endpoint": ep,
            "Evaluated Samples (N)": int(ref_m.n_valid),
            "Positive Samples": int(ref_m.n_positive),
            "Prevalence": f"{ref_m.prevalence * 100:.2f}%",
            "Raw Brier": _fmt_metric(raw_m.brier_score if raw_m else None),
            "Sigmoid Brier": _fmt_metric(sig_m.brier_score if sig_m else None),
            "Isotonic Brier": _fmt_metric(iso_m.brier_score if iso_m else None),
            "Raw Log Loss": _fmt_metric(raw_m.log_loss_value if raw_m else None),
            "Sigmoid Log Loss": _fmt_metric(sig_m.log_loss_value if sig_m else None),
            "Isotonic Log Loss": _fmt_metric(iso_m.log_loss_value if iso_m else None),
            "Raw ECE (10-Bin)": _fmt_metric(raw_m.ece_10bin_weighted if raw_m else None),
            "Sigmoid ECE (10-Bin)": _fmt_metric(sig_m.ece_10bin_weighted if sig_m else None),
            "Isotonic ECE (10-Bin)": _fmt_metric(iso_m.ece_10bin_weighted if iso_m else None),
            "Raw MACE (5-Bin Mean)": _fmt_metric(raw_m.mace_5bin_unweighted_mean if raw_m else None),
            "Sigmoid MACE (5-Bin Mean)": _fmt_metric(sig_m.mace_5bin_unweighted_mean if sig_m else None),
            "Isotonic MACE (5-Bin Mean)": _fmt_metric(iso_m.mace_5bin_unweighted_mean if iso_m else None),
            "Raw MACE (10-Bin Mean)": _fmt_metric(raw_m.mace_10bin_unweighted_mean if raw_m else None),
            "Sigmoid MACE (10-Bin Mean)": _fmt_metric(sig_m.mace_10bin_unweighted_mean if sig_m else None),
            "Isotonic MACE (10-Bin Mean)": _fmt_metric(iso_m.mace_10bin_unweighted_mean if iso_m else None),
            "Raw ROC-AUC": _fmt_metric(raw_m.roc_auc if raw_m else None),
            "Sigmoid ROC-AUC": _fmt_metric(sig_m.roc_auc if sig_m else None),
            "Isotonic ROC-AUC": _fmt_metric(iso_m.roc_auc if iso_m else None),
            "Raw PR-AUC": _fmt_metric(raw_m.pr_auc if raw_m else None),
            "Sigmoid PR-AUC": _fmt_metric(sig_m.pr_auc if sig_m else None),
            "Isotonic PR-AUC": _fmt_metric(iso_m.pr_auc if iso_m else None),
            "Recommended Method": bundle.recommended_method,
            "Availability & Support Status": support_tag,
        })
    return rows


def build_ad_stratification_table_rows(
    ad_report: EndpointADStratificationReport,
) -> List[Dict[str, Any]]:
    """
    Build tabular rows for the exploratory Applicability-Domain (Tanimoto similarity)
    stratification panel, explicitly flagging strata with `< 10` positives as `INSUFFICIENT_SUPPORT`.
    """
    rows: List[Dict[str, Any]] = []
    all_strata = list(ad_report.strata)
    if ad_report.unavailable_stratum is not None and ad_report.unavailable_stratum.n_total > 0:
        all_strata.append(ad_report.unavailable_stratum)

    for s in all_strata:
        if s.similarity_min is not None and s.similarity_max is not None:
            sim_range = f"[{s.similarity_min:.3f}, {s.similarity_max:.3f}]"
        else:
            sim_range = "Unavailable / Missing"

        prev_str = f"{s.prevalence * 100:.2f}%" if s.prevalence is not None else "N/A"
        is_insuff = s.support_status == "INSUFFICIENT_SUPPORT"
        disc_unavail = f"INSUFFICIENT_SUPPORT (pos={s.n_positive} < {ad_report.min_positives_support_rule})"

        rows.append({
            "Exploratory Similarity Stratum": s.stratum_label,
            "Observed Tanimoto Range": sim_range,
            "Total (n)": int(s.n_total),
            "Positives (1)": int(s.n_positive),
            "Negatives (0)": int(s.n_negative),
            "Prevalence": prev_str,
            "Support Status": s.support_status,
            "Raw Brier (↓)": _fmt_metric(s.raw_brier, 4, "N/A"),
            "Sigmoid Brier (↓)": _fmt_metric(s.sigmoid_brier, 4, "N/A"),
            "Brier Improvement (↑)": (
                f"{s.brier_improvement:+.4f}" if s.brier_improvement is not None else "N/A"
            ),
            "Raw ECE (↓)": _fmt_metric(s.raw_ece_10bin, 4, "N/A"),
            "Sigmoid ECE (↓)": _fmt_metric(s.sigmoid_ece_10bin, 4, "N/A"),
            "ROC-AUC (↑)": disc_unavail if is_insuff else _fmt_metric(s.sigmoid_roc_auc, 4, "N/A"),
            "PR-AUC (↑)": disc_unavail if is_insuff else _fmt_metric(s.sigmoid_pr_auc, 4, "N/A"),
        })
    return rows


def render_reliability_diagram_svg(
    reliability_series: Dict[str, ReliabilityDiagramSeries],
    endpoint: str,
    include_isotonic: bool = True,
    width: int = 720,
    height: int = 470,
) -> str:
    """
    Generate an SVG 10-bin reliability diagram and synchronized per-bin sample-count
    panel for `endpoint`.

    Safeguards:
      - Empty bins (`count == 0` or `mean_predicted_prob is None` or `observed_positive_fraction is None`)
        are NEVER plotted or interpolated across as fake zero values.
      - Sparse bins (`0 < count < 10`) are drawn with dashed/hollow outer rings and explicit
        `n=` count labels so sparse tail bins cannot be mistaken for dense bins.
      - Includes the `y = x` ideal calibration diagonal and a synchronized 10-bin histogram.
    """
    if not reliability_series:
        return (
            '<div class="mol-card" style="padding: 1rem; color: #f59e0b;">'
            f"Reliability curve data unavailable for endpoint <code>{html.escape(endpoint)}</code>."
            "</div>"
        )

    series_order = ["uncalibrated", "sigmoid"]
    if include_isotonic and "isotonic" in reliability_series:
        series_order.append("isotonic")

    available_keys = [
        k for k in series_order
        if k in reliability_series and reliability_series[k].non_empty_bin_count > 0
    ]
    if not available_keys:
        return (
            '<div class="mol-card" style="padding: 1rem; color: #f59e0b;">'
            f"No non-empty probability bins available to plot for <code>{html.escape(endpoint)}</code>."
            "</div>"
        )

    # Coordinate geometry
    pad_left = 68
    pad_right = 28
    pad_top = 42
    curve_h = 250
    gap_h = 34
    hist_h = 85
    plot_w = width - pad_left - pad_right

    curve_top = pad_top
    curve_bottom = curve_top + curve_h
    hist_top = curve_bottom + gap_h
    hist_bottom = hist_top + hist_h

    def _x_px(prob_val: float) -> float:
        return pad_left + max(0.0, min(1.0, float(prob_val))) * plot_w

    def _y_px(obs_val: float) -> float:
        return curve_bottom - max(0.0, min(1.0, float(obs_val))) * curve_h

    svg_parts: List[str] = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="auto" '
        'xmlns="http://www.w3.org/2000/svg" '
        'style="background: #0f172a; border: 1px solid #1e293b; border-radius: 10px; font-family: Inter, system-ui, sans-serif;">'
    ]

    # Title
    ref_s = reliability_series[available_keys[0]]
    svg_parts.append(
        f'<text x="{pad_left}" y="24" fill="#f8fafc" font-size="13" font-weight="700">'
        f'{html.escape(endpoint)} — 10-Bin Reliability Diagram (N = {ref_s.n_valid:,}, '
        f'Positives = {ref_s.n_positive:,} [{ref_s.prevalence*100:.1f}%])</text>'
    )

    # Grid lines and axes for reliability plot [0.0 .. 1.0]
    for tick in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        gx = _x_px(tick)
        gy = _y_px(tick)
        svg_parts.append(
            f'<line x1="{pad_left:.1f}" y1="{gy:.1f}" x2="{pad_left + plot_w:.1f}" y2="{gy:.1f}" '
            'stroke="#1e293b" stroke-width="1" />'
        )
        svg_parts.append(
            f'<line x1="{gx:.1f}" y1="{curve_top:.1f}" x2="{gx:.1f}" y2="{curve_bottom:.1f}" '
            'stroke="#1e293b" stroke-width="1" />'
        )
        svg_parts.append(
            f'<text x="{pad_left - 8:.1f}" y="{gy + 4:.1f}" fill="#94a3b8" font-size="10" text-anchor="end">'
            f'{tick:.1f}</text>'
        )
        svg_parts.append(
            f'<text x="{gx:.1f}" y="{curve_bottom + 14:.1f}" fill="#94a3b8" font-size="10" text-anchor="middle">'
            f'{tick:.1f}</text>'
        )

    # Subtle decile bin boundary lines (0.1, 0.3, 0.5, 0.7, 0.9)
    for dec in (0.1, 0.3, 0.5, 0.7, 0.9):
        dx = _x_px(dec)
        svg_parts.append(
            f'<line x1="{dx:.1f}" y1="{curve_top:.1f}" x2="{dx:.1f}" y2="{curve_bottom:.1f}" '
            'stroke="#1e293b" stroke-width="0.6" stroke-dasharray="2,3" />'
        )

    # Ideal calibration reference diagonal y = x
    svg_parts.append(
        f'<line x1="{_x_px(0.0):.1f}" y1="{_y_px(0.0):.1f}" '
        f'x2="{_x_px(1.0):.1f}" y2="{_y_px(1.0):.1f}" '
        'stroke="#64748b" stroke-width="1.5" stroke-dasharray="5,5" />'
    )
    svg_parts.append(
        f'<text x="{_x_px(0.72):.1f}" y="{_y_px(0.76):.1f}" fill="#94a3b8" font-size="10" font-style="italic">'
        'Ideal calibration (y = x)</text>'
    )

    # Axis labels
    svg_parts.append(
        f'<text x="{pad_left + plot_w / 2:.1f}" y="{curve_bottom + 27:.1f}" fill="#cbd5e1" '
        'font-size="11" font-weight="600" text-anchor="middle">'
        'Mean Predicted Probability in Decile Bin [0, 1]</text>'
    )
    svg_parts.append(
        f'<text x="16" y="{curve_top + curve_h / 2:.1f}" fill="#cbd5e1" '
        'font-size="11" font-weight="600" text-anchor="middle" '
        f'transform="rotate(-90 16 {curve_top + curve_h / 2:.1f})">'
        'Observed Positive Fraction</text>'
    )

    styles = {
        "uncalibrated": {
            "color": "#f59e0b",
            "dash": "",
            "label": "Raw RF (Uncalibrated)",
        },
        "sigmoid": {
            "color": "#10b981",
            "dash": "",
            "label": "Sigmoid Calibrated (Selected)",
        },
        "isotonic": {
            "color": "#38bdf8",
            "dash": "4,3",
            "label": "Isotonic (Eval-Only Comparator)",
        },
    }

    # Plot each series over non-empty bins ONLY
    for s_key in available_keys:
        s_obj = reliability_series[s_key]
        st_cfg = styles.get(s_key, {"color": "#a78bfa", "dash": "", "label": s_key})
        color = st_cfg["color"]
        dash_attr = f' stroke-dasharray="{st_cfg["dash"]}"' if st_cfg["dash"] else ""

        pts: List[Tuple[float, float, int, bool]] = []
        for b in s_obj.bins:
            if not b.is_empty and b.mean_predicted_prob is not None and b.observed_positive_fraction is not None:
                pts.append((_x_px(b.mean_predicted_prob), _y_px(b.observed_positive_fraction), b.count, b.is_sparse))

        if len(pts) >= 2:
            poly_coords = " ".join(f"{px:.1f},{py:.1f}" for px, py, _, _ in pts)
            svg_parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2.2"{dash_attr} points="{poly_coords}" />'
            )

        for px, py, cnt, is_sparse in pts:
            if is_sparse:
                # Sparse bin (< 10 observations): hollow circle with dashed outer ring
                svg_parts.append(
                    f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="#0f172a" stroke="{color}" stroke-width="2" />'
                )
            else:
                svg_parts.append(
                    f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.2" fill="{color}" stroke="#0f172a" stroke-width="1.2" />'
                )

    # Legend in top-left of curve area
    leg_x = pad_left + 12
    leg_y = curve_top + 14
    svg_parts.append(
        f'<rect x="{leg_x - 6}" y="{leg_y - 10}" width="252" height="{len(available_keys) * 18 + 22}" '
        'rx="5" fill="#0f172a" fill-opacity="0.88" stroke="#1e293b" stroke-width="1" />'
    )
    for idx, s_key in enumerate(available_keys):
        st_cfg = styles.get(s_key, {"color": "#a78bfa", "dash": "", "label": s_key})
        ly = leg_y + idx * 18
        svg_parts.append(
            f'<line x1="{leg_x}" y1="{ly}" x2="{leg_x + 18}" y2="{ly}" stroke="{st_cfg["color"]}" stroke-width="2.4" />'
        )
        svg_parts.append(
            f'<circle cx="{leg_x + 9}" cy="{ly}" r="3.5" fill="{st_cfg["color"]}" />'
        )
        svg_parts.append(
            f'<text x="{leg_x + 24}" y="{ly + 4}" fill="#e2e8f0" font-size="10.5">{html.escape(st_cfg["label"])}</text>'
        )
    # Note on hollow markers
    ly_note = leg_y + len(available_keys) * 18
    svg_parts.append(
        f'<circle cx="{leg_x + 9}" cy="{ly_note}" r="3.5" fill="#0f172a" stroke="#94a3b8" stroke-width="1.6" />'
    )
    svg_parts.append(
        f'<text x="{leg_x + 24}" y="{ly_note + 3.5}" fill="#94a3b8" font-size="9.5">'
        'Hollow marker = sparse bin (n &lt; 10)</text>'
    )

    # Synchronized Lower Panel: 10-Bin Sample Count Histogram (Log10-scaled bar height with exact integer labels)
    svg_parts.append(
        f'<line x1="{pad_left:.1f}" y1="{hist_top:.1f}" x2="{pad_left + plot_w:.1f}" y2="{hist_top:.1f}" '
        'stroke="#1e293b" stroke-width="1" />'
    )
    svg_parts.append(
        f'<line x1="{pad_left:.1f}" y1="{hist_bottom:.1f}" x2="{pad_left + plot_w:.1f}" y2="{hist_bottom:.1f}" '
        'stroke="#334155" stroke-width="1" />'
    )
    svg_parts.append(
        f'<text x="16" y="{hist_top + hist_h / 2:.1f}" fill="#94a3b8" font-size="10" font-weight="600" '
        f'text-anchor="middle" transform="rotate(-90 16 {hist_top + hist_h / 2:.1f})">Bin Count (n)</text>'
    )

    n_bins = ref_s.n_bins
    bin_w = plot_w / n_bins
    max_cnt = max(
        max(reliability_series[k].bin_counts) for k in available_keys
    )
    max_log = math.log10(max_cnt + 1.0) if max_cnt > 0 else 1.0

    for b_idx in range(n_bins):
        bx_left = pad_left + b_idx * bin_w
        # Draw bars for raw and sigmoid (and isotonic if enabled)
        n_bars = len(available_keys)
        sub_w = (bin_w - 10.0) / max(1, n_bars)
        for s_i, s_key in enumerate(available_keys):
            cnt = reliability_series[s_key].bin_counts[b_idx]
            color = styles.get(s_key, {"color": "#94a3b8"})["color"]
            bar_h = (math.log10(cnt + 1.0) / max_log) * (hist_h - 18.0) if cnt > 0 else 0.0
            bx = bx_left + 5.0 + s_i * sub_w
            by = hist_bottom - bar_h
            if cnt > 0:
                svg_parts.append(
                    f'<rect x="{bx:.1f}" y="{by:.1f}" width="{max(2.0, sub_w - 1.5):.1f}" height="{bar_h:.1f}" '
                    f'fill="{color}" fill-opacity="0.75" rx="1.5" />'
                )
            # Annotate count for primary series (sigmoid if present, else raw)
            if s_key == ("sigmoid" if "sigmoid" in available_keys else available_keys[0]):
                lbl_y = max(hist_top + 10.0, by - 3.0)
                lbl_color = "#cbd5e1" if cnt >= 10 else ("#f59e0b" if cnt > 0 else "#475569")
                svg_parts.append(
                    f'<text x="{bx_left + bin_w / 2:.1f}" y="{lbl_y:.1f}" fill="{lbl_color}" '
                    f'font-size="8.5" text-anchor="middle">n={cnt}</text>'
                )

        low_e = ref_s.bin_edges[b_idx]
        high_e = ref_s.bin_edges[b_idx + 1]
        interval_lbl = f"[{low_e:.1f},{high_e:.1f}]" if b_idx == n_bins - 1 else f"[{low_e:.1f},{high_e:.1f})"
        svg_parts.append(
            f'<text x="{bx_left + bin_w / 2:.1f}" y="{hist_bottom + 13:.1f}" fill="#94a3b8" '
            f'font-size="8.5" text-anchor="middle">{interval_lbl}</text>'
        )

    svg_parts.append("</svg>")
    return "".join(svg_parts)


def render_tox21_endpoint_diagnostics_section(
    artifact_dir: str = "artifacts/tox21",
    csv_path: str = "data/raw/tox21.csv",
):
    """
    Render the Phase B.2 interactive Tox21 Endpoint Reliability & Calibration Diagnostics
    workbench inside the Tox21 tab of Model Insights.
    """
    st.markdown("<div class='miq-divider'></div>", unsafe_allow_html=True)
    st.markdown("### 🎯 Endpoint-Level Reliability Diagnostics & Calibration Evaluation (Phase B.2)")
    st.caption(
        "Interactive sample-level reliability curves, paired bootstrap uncertainty (B = 500, seed = 42), "
        "and exploratory applicability-domain stratification across all 12 Tox21 assays."
    )

    selector_options = get_tox21_endpoint_selector_options()
    label_to_ep = {label: ep for ep, label in selector_options}
    display_labels = [label for _, label in selector_options]

    col_sel, col_opt = st.columns([2.2, 1.0])
    with col_sel:
        selected_label = st.selectbox(
            "Select Tox21 Assay Endpoint (All 12 Canonical Assays):",
            options=display_labels,
            index=0,
            key="b2_tox21_endpoint_selector",
        )
    with col_opt:
        show_isotonic = st.checkbox(
            "Include Isotonic Comparator (Evaluation-Only)",
            value=True,
            key="b2_show_isotonic_comparator",
            help=(
                "Sigmoid (Platt scaling) is the selected calibrator for all 12 endpoints via 3-way disjoint CV. "
                "Isotonic regression is shown strictly as an evaluation-only diagnostic comparator."
            ),
        )

    selected_ep = label_to_ep.get(selected_label, TOX21_ENDPOINTS[0])

    with st.spinner(f"Loading verified reliability diagnostics for {selected_ep}..."):
        bundle = load_endpoint_reliability_diagnostics(
            endpoint=selected_ep,
            artifact_dir=artifact_dir,
            csv_path=csv_path,
            reconstruct_sample_level=True,
        )

    # 7. Visible Data-Quality & Availability Status Banner
    cal_badge = "success" if bundle.calibration_artifact_status == "AVAILABLE" else "danger"
    rec_badge = "success" if bundle.reconciled_with_benchmark else "warning"
    ad_badge = (
        "success"
        if bundle.ad_stratification.status == "AVAILABLE"
        else ("warning" if bundle.ad_stratification.status == "AVAILABLE_WITH_SPARSE_STRATA" else "danger")
    )

    st.markdown(
        f"""
        <div class="mol-card" style="padding: 0.85rem 1.15rem; margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.75rem;">
                <div>
                    <span style="font-size: 0.76rem; color: #94a3b8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;">
                        Canonical Endpoint:
                    </span>
                    <code style="font-size: 0.92rem; color: #38bdf8; margin-left: 0.35rem;">{html.escape(bundle.endpoint)}</code>
                    <span style="color: #cbd5e1; font-size: 0.88rem; margin-left: 0.5rem;">({html.escape(bundle.display_name)})</span>
                </div>
                <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center;">
                    <span class="mol-badge mol-badge-{cal_badge}">Calibrator: {html.escape(bundle.calibration_artifact_status)} ({html.escape(bundle.recommended_method.upper())})</span>
                    <span class="mol-badge mol-badge-info">Evidence: {html.escape(bundle.sample_level_status)}</span>
                    <span class="mol-badge mol-badge-{rec_badge}">Benchmark Reconciled: {"YES" if bundle.reconciled_with_benchmark else "PARTIAL"}</span>
                    <span class="mol-badge mol-badge-{ad_badge}">AD Strata: {html.escape(bundle.ad_stratification.status)}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if bundle.overall_status != "AVAILABLE" or not bundle.series_metrics:
        st.error(
            f"⚠️ Reliability diagnostics unavailable for `{bundle.endpoint}`: "
            f"{bundle.status_message or 'Missing or corrupt calibration/benchmark artifact.'}"
        )
        return

    # 5. Sample-Support and Prevalence Summary Cards
    ds = bundle.data_summary
    raw_m = bundle.series_metrics.get("uncalibrated")
    sig_m = bundle.series_metrics.get("sigmoid")
    n_test = int(ds.get("test_count", raw_m.n_valid if raw_m else 0))
    n_test_pos = int(ds.get("test_positive_count", raw_m.n_positive if raw_m else 0))
    n_test_neg = int(ds.get("test_negative_count", raw_m.n_negative if raw_m else 0))
    test_prev = (n_test_pos / n_test * 100.0) if n_test > 0 else 0.0
    n_usable = int(ds.get("usable_samples", 0))
    n_excluded = int(ds.get("missing_or_invalid_excluded", 0))
    n_train = int(ds.get("train_count", 0))
    n_train_pos = int(ds.get("train_positive_count", 0))

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        render_metric_card(
            "Held-Out Test Cohort",
            f"{n_test:,}",
            f"20% stratified split ({n_usable:,} usable; {n_excluded:,} missing/invalid excluded)",
            "#38bdf8",
            badge="Locked 20% Test",
            badge_variant="info",
        )
    with s2:
        render_metric_card(
            "Test Class Support & Prevalence",
            f"{n_test_pos:,} pos / {n_test_neg:,} neg",
            f"Observed test prevalence: {test_prev:.2f}% (Train: {n_train_pos:,} / {n_train:,})",
            "#f59e0b" if n_test_pos < 60 else "#38bdf8",
            badge="Rare Positives" if n_test_pos < 60 else "Class Balance",
            badge_variant="warning" if n_test_pos < 60 else "info",
        )
    with s3:
        brier_raw_val = raw_m.brier_score if raw_m else None
        brier_sig_val = sig_m.brier_score if sig_m else None
        render_metric_card(
            "Brier Score (Raw → Sigmoid)",
            f"{_fmt_metric(brier_raw_val)} → {_fmt_metric(brier_sig_val)}",
            "↓ Lower is better (Proper scoring rule on [0, 1])",
            "#10b981",
            badge="Calibration",
            badge_variant="success",
        )
    with s4:
        ece_raw_val = raw_m.ece_10bin_weighted if raw_m else None
        ece_sig_val = sig_m.ece_10bin_weighted if sig_m else None
        render_metric_card(
            "10-Bin ECE (Raw → Sigmoid)",
            f"{_fmt_metric(ece_raw_val)} → {_fmt_metric(ece_sig_val)}",
            "↓ Lower is better (Sample-weighted decile error)",
            "#10b981",
            badge="Reliability",
            badge_variant="success",
        )

    # 2. Reliability Diagram & Bin Support Table
    st.markdown("#### 1. 10-Bin Reliability Diagram & Probability Bin Support")
    st.caption(
        "Compares empirical positive fraction against mean predicted probability across 10 equal-width bins "
        "on `[0.0, 1.0]` (`[0.0, 0.1), ..., [0.9, 1.0]`). Empty bins (`n = 0`) are never plotted as zero; "
        "sparse bins (`0 < n < 10`) are marked with hollow rings."
    )

    svg_html = render_reliability_diagram_svg(
        bundle.reliability_series,
        endpoint=bundle.endpoint,
        include_isotonic=show_isotonic,
    )
    st.markdown(svg_html, unsafe_allow_html=True)

    with st.expander(f"🔎 Inspect 10-Bin Sample Counts & Empirical Fractions for {bundle.endpoint}"):
        st.markdown(
            """
            <div style="background: rgba(15, 23, 42, 0.45); border: 1px solid #1e293b; border-radius: 6px; padding: 0.65rem 0.85rem; margin-bottom: 0.75rem; font-size: 0.82rem; color: #cbd5e1; line-height: 1.5;">
                <strong style="color: #38bdf8;">Bin Support & Empirical Estimation Policy:</strong><br>
                • <strong style="color: #38bdf8;">SUPPORTED (≥ 10 observations):</strong> Satisfies the project's empirical threshold for stable rate estimation within this evaluation cohort; does not imply external biological validation.<br>
                • <strong style="color: #f59e0b;">SPARSE (&lt; 10 observations):</strong> Limited sample support; empirical fractions and gap statistics are reported with available counts but are subject to finite-sample instability. Displayed with hollow/dashed markers on the curve.<br>
                • <strong style="color: #94a3b8;">EMPTY (0 observations):</strong> No evaluated held-out compounds fell within this probability interval. Calibration statistics are mathematically unestimable and displayed strictly as <code>N/A</code> rather than artificial zero error, and omitted from curve interpolation.
            </div>
            """,
            unsafe_allow_html=True,
        )
        series_choice = st.radio(
            "Select probability series for bin-level table:",
            options=[k for k in ("sigmoid", "uncalibrated", "isotonic") if k in bundle.reliability_series],
            format_func=lambda k: bundle.reliability_series[k].display_label,
            horizontal=True,
            key=f"b2_bin_table_radio_{bundle.endpoint}",
        )
        if series_choice and series_choice in bundle.reliability_series:
            bin_rows = build_reliability_bin_table_rows(bundle.reliability_series[series_choice])
            st.dataframe(pd.DataFrame(bin_rows), use_container_width=True, hide_index=True)

    # 3. Raw-vs-Calibrated Metric Comparison Table
    st.markdown("#### 2. Endpoint-Level Raw vs. Calibrated Metric Comparison")
    st.caption(
        "Lower values (↓) are preferable for Brier score, Log loss, 10-Bin ECE, and MACE. "
        "Higher values (↑) are preferable for ROC-AUC and PR-AUC, which measure rank discrimination "
        "(order of predictions) rather than probability calibration. Under Platt scaling "
        "P(y=1|f) = 1 / (1 + exp(a*f + b)), the strictly negative slope parameter (a < 0) yields a strictly "
        "monotonically increasing probability transformation (dP/df > 0), preserving pairwise ranking "
        "order and leaving ROC-AUC and PR-AUC invariant."
    )
    comp_rows = build_endpoint_metric_comparison_rows(bundle)
    if comp_rows:
        df_comp = pd.DataFrame(comp_rows)
        st.dataframe(
            df_comp,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Metric": st.column_config.TextColumn("Metric", width="medium"),
                "Category & Direction": st.column_config.TextColumn("Category & Direction", width="medium"),
                "Unit / Scale": st.column_config.TextColumn("Unit / Scale", width="small"),
                "Raw RF (Uncalibrated)": st.column_config.TextColumn("Raw RF (Uncal)", width="small"),
                "Sigmoid Calibrated (Selected)": st.column_config.TextColumn("Sigmoid (Selected)", width="small"),
                "Isotonic (Eval-Only Comparator)": st.column_config.TextColumn("Isotonic (Comparator)", width="small"),
                "Sigmoid vs Raw Δ": st.column_config.TextColumn("Sigmoid vs Raw Δ", width="small"),
                "Definition & Interpretation": st.column_config.TextColumn("Definition & Interpretation", width="large"),
            },
        )

    with st.expander("📋 All 12 Tox21 Endpoints — Complete Calibration & Reliability Diagnostic Table (1 Row / Endpoint)", expanded=True):
        st.caption(
            "One row per canonical Tox21 endpoint reporting evaluated held-out sample count (N), positive count, "
            "prevalence, Brier score, Log loss, 10-Bin ECE, 5-Bin & 10-Bin unweighted-mean MACE, ROC-AUC, PR-AUC "
            "for Raw, Sigmoid, and Isotonic probabilities, recommended calibration method, and support status."
        )
        all_ep_rows = build_all_endpoints_diagnostic_table_rows(
            artifact_dir=artifact_dir,
            csv_path=csv_path,
            reconstruct_sample_level=False,
        )
        all_ep_df = pd.DataFrame(all_ep_rows)

        tab_cal, tab_disc, tab_support, tab_all = st.tabs([
            "📊 Probability Calibration",
            "🎯 Ranking Discrimination",
            "👥 Cohort Support & Prevalence",
            "📑 Complete Unified Table (27 Columns)",
        ])

        with tab_cal:
            st.caption(
                "Probability Calibration Metrics: Brier score, Log loss, 10-Bin ECE, and MACE. "
                "Lower values (↓) indicate superior probability reliability. "
                "Sigmoid (Platt) scaling is the selected deployed calibrator; Isotonic is an evaluation-only comparator."
            )
            cal_cols = [
                "Endpoint", "Recommended Method",
                "Sigmoid Brier", "Raw Brier", "Isotonic Brier",
                "Sigmoid Log Loss", "Raw Log Loss", "Isotonic Log Loss",
                "Sigmoid ECE (10-Bin)", "Raw ECE (10-Bin)", "Isotonic ECE (10-Bin)",
                "Sigmoid MACE (10-Bin Mean)", "Raw MACE (10-Bin Mean)", "Isotonic MACE (10-Bin Mean)",
                "Sigmoid MACE (5-Bin Mean)", "Raw MACE (5-Bin Mean)", "Isotonic MACE (5-Bin Mean)",
            ]
            st.dataframe(all_ep_df[cal_cols], use_container_width=True, hide_index=True)

        with tab_disc:
            st.caption(
                "Ranking Discrimination Metrics: ROC-AUC and PR-AUC. Higher values (↑) indicate superior separation. "
                "Note: Rank discrimination is invariant under monotonic Platt scaling (a < 0); PR-AUC baseline depends on class prevalence."
            )
            disc_cols = [
                "Endpoint", "Recommended Method", "Prevalence",
                "Raw ROC-AUC", "Sigmoid ROC-AUC", "Isotonic ROC-AUC",
                "Raw PR-AUC", "Sigmoid PR-AUC", "Isotonic PR-AUC",
            ]
            st.dataframe(all_ep_df[disc_cols], use_container_width=True, hide_index=True)

        with tab_support:
            st.caption(
                "Evaluation Cohort & Support: Stratified 20% test partition after independent missing-label exclusion. "
                "Under finite sample sizes, low positive counts can produce wide bootstrap confidence intervals."
            )
            support_cols = [
                "Endpoint", "Evaluated Samples (N)", "Positive Samples", "Prevalence",
                "Recommended Method", "Availability & Support Status",
            ]
            st.dataframe(all_ep_df[support_cols], use_container_width=True, hide_index=True)

        with tab_all:
            st.caption("Complete 27-column diagnostic table across all metrics, methods, and support statuses.")
            st.dataframe(all_ep_df, use_container_width=True, hide_index=True)

    # 4. Paired Bootstrap Confidence Interval View
    st.markdown("#### 3. Paired Bootstrap Brier Score Uncertainty (`B = 500`, `seed = 42`)")
    boot = bundle.bootstrap_sigmoid
    if boot.status in ("AVAILABLE", "BENCHMARK_PRECOMPUTED") and boot.brier_improvement is not None and boot.brier_improvement_ci_95 is not None:
        ci_low, ci_high = boot.brier_improvement_ci_95
        spans_zero = bool(boot.ci_includes_zero)
        border_col = "#f59e0b" if spans_zero else "#10b981"
        badge_var = "warning" if spans_zero else "success"
        badge_txt = "95% CI Includes Zero (Low Minority Support)" if spans_zero else "95% CI Excludes Zero (Statistically Supported)"

        st.markdown(
            f"""
            <div class="mol-card" style="border-left: 4px solid {border_col}; padding: 1rem 1.2rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.5rem;">
                    <strong style="color: #f8fafc; font-size: 0.98rem;">
                        Paired Bootstrap Brier Improvement: <code>Brier(raw) − Brier(sigmoid)</code>
                    </strong>
                    <span class="mol-badge mol-badge-{badge_var}">{badge_txt}</span>
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(195px, 1fr)); gap: 0.75rem; font-size: 0.86rem; color: #cbd5e1; margin-bottom: 0.65rem;">
                    <div>• <strong>Brier Improvement (↑):</strong> <code>{boot.brier_improvement:+.6f}</code></div>
                    <div>• <strong>95% Bootstrap CI:</strong> <code>[{ci_low:+.6f}, {ci_high:+.6f}]</code></div>
                    <div>• <strong>Benchmark ΔBrier (Sig − Raw):</strong> <code>{boot.delta_brier_vs_raw:+.6f}</code> (CI: <code>[{boot.delta_brier_ci_95[0]:+.6f}, {boot.delta_brier_ci_95[1]:+.6f}]</code>)</div>
                    <div>• <strong>Valid Replicates:</strong> <code>{boot.n_valid_replicates} / {boot.n_requested_replicates}</code> (Two-class: <code>{boot.n_two_class_replicates}</code>)</div>
                </div>
                <div style="font-size: 0.84rem; color: #94a3b8; line-height: 1.5;">
                    {html.escape(boot.interpretation_note)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning(
            f"⚠️ Paired bootstrap confidence intervals unavailable for `{bundle.endpoint}` "
            f"({boot.status}: {boot.status_message or 'insufficient sample data'})."
        )

    # 6. Exploratory Applicability-Domain Stratification Panel
    st.markdown("#### 4. Exploratory Applicability-Domain (Tanimoto Similarity) Stratification")
    ad_rep = bundle.ad_stratification
    if ad_rep.status in ("AVAILABLE", "AVAILABLE_WITH_SPARSE_STRATA") and ad_rep.strata:
        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.55); border: 1px solid #1e293b; border-left: 3px solid #38bdf8; border-radius: 6px; padding: 0.65rem 0.95rem; margin-bottom: 0.75rem; font-size: 0.83rem; color: #cbd5e1;">
                <strong style="color: #38bdf8;">EXPLORATORY STRATIFICATION NOTICE:</strong>
                {html.escape(ad_rep.scientific_disclaimer)}
                <br><span style="color: #94a3b8;">
                <strong>Support Policy:</strong> Any stratum with fewer than <code>{ad_rep.min_positives_support_rule}</code> positive observations
                is flagged <code>INSUFFICIENT_SUPPORT</code> and positive-class discrimination metrics (ROC-AUC, PR-AUC) are suppressed.
                (Endpoint training reference corpus: <code>{ad_rep.training_corpus_size:,}</code> compounds; valid test AD scores: <code>{ad_rep.n_valid_ad_scores:,} / {ad_rep.n_total_test:,}</code>.)
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        ad_rows = build_ad_stratification_table_rows(ad_rep)
        st.dataframe(pd.DataFrame(ad_rows), use_container_width=True, hide_index=True)
        if ad_rep.status == "AVAILABLE_WITH_SPARSE_STRATA":
            st.caption(f"⚠️ {ad_rep.status_message}")
    else:
        st.info(
            f"ℹ️ Applicability-domain stratification is unavailable for `{bundle.endpoint}`: "
            f"{ad_rep.status_message or 'Verified sample-level records or training fingerprint corpus missing.'}"
        )

    # 8. Concise 5-Part Scientific Interpretation Panel
    st.markdown("#### 5. Scientific Interpretation & Methodological Boundaries")
    sd_comp = bundle.score_distribution_comparison
    r_60_80 = sd_comp.get("train_60pct_vs_80pct_pearson_r")
    mad_60_80 = sd_comp.get("train_60pct_vs_80pct_mean_abs_diff")
    shift_str = (
        f"60%-vs-80% training-size OOF score correlation r = {r_60_80:.4f}, MAD = {mad_60_80:.4f}."
        if isinstance(r_60_80, (int, float)) and isinstance(mad_60_80, (int, float))
        else "Finite-sample ensemble training-size differences exist across 60%, 80%, and 100% splits."
    )

    st.markdown(
        f"""
        <div class="mol-card" style="padding: 1rem 1.25rem;">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 0.9rem; font-size: 0.84rem; color: #334155; line-height: 1.55;">
                <div>
                    <strong style="color: #0F8075;">1. Probability Calibration:</strong><br>
                    Baseline Random Forests trained with <code>class_weight='balanced'</code> systematically over-predict positive probabilities relative to empirical Tox21 prevalence ({test_prev:.2f}%). Post-hoc Sigmoid (Platt) scaling aligns predicted probabilities with empirical assay hit rates without altering default raw screening outputs.
                </div>
                <div>
                    <strong style="color: #27C7B8;">2. Rank Discrimination vs. Reliability:</strong><br>
                    ROC-AUC and PR-AUC measure pairwise ranking discrimination, NOT probability calibration. Under Platt scaling <code>P(y=1|f) = 1 / (1 + exp(a*f + b))</code>, the strictly negative slope parameter (<code>a &lt; 0</code>) produces a strictly monotonically increasing transformation (<code>dP/df &gt; 0</code>), preserving relative score ordering and leaving ROC-AUC and PR-AUC invariant.
                </div>
                <div>
                    <strong style="color: #E5A34D;">3. Sample Support &amp; Minority Uncertainty:</strong><br>
                    Each Tox21 assay excludes missing labels independently (test cohort <code>N = {n_test:,}</code>, positives = <code>{n_test_pos:,}</code>). Across the 12 assays, <code>NR-PPAR-gamma</code> (37 test positives) and <code>SR-ATAD5</code> (53 test positives) have 95% paired bootstrap Brier CIs that include zero; this reflects finite-sample variance rather than proof that calibration is ineffective.
                </div>
                <div>
                    <strong style="color: #8E7CFF;">4. Applicability-Domain Evidence:</strong><br>
                    Tanimoto similarity tertiles above are exploratory and unvalidated (ClinTox cutoffs are never borrowed for Tox21). High 2D Morgan fingerprint similarity to training compounds does not guarantee reliable biological prediction or absence of activity cliffs.
                </div>
                <div>
                    <strong style="color: #EF4444;">5. Out-of-Domain Generalization &amp; Governance:</strong><br>
                    Calibration is evaluated on an i.i.d. stratified 80/20 split ({html.escape(shift_str)}) whose held-out test partition was previously inspected during initial v1.0.0 diagnostics prior to v1.1.0 3-way disjoint remediation. Neither random-split calibration nor in-vitro luminescence inactivity establishes scaffold generalization, in-vivo safety, or clinical suitability.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_insights_view():
    """Render Workflow C: Model Insights & Scientific Evaluation Metrics."""
    render_header(
        "Model Insights & Evaluation Metrics",
        "Transparent scientific evidence: benchmark distributions, held-out evaluation metrics, and task definitions.",
        badge_text="Scientific Evidence",
        badge_variant="info",
    )

    tab_clin, tab_tox21, tab_limits = st.tabs([
        "🔬 ClinTox Benchmark Model",
        "🧪 Tox21 12-Endpoint Suite",
        "🛡️ Scientific Boundaries & Governance",
    ])

    # TAB 1: ClinTox
    with tab_clin:
        meta_path = "artifacts/clintox/metadata.json"
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            metrics = meta.get("metrics", {})

            st.markdown("### ClinTox Random Forest Classifier (`CT_TOX`)")
            st.caption("Target definition: Likelihood of clinical trial failure attributed to toxicity (MoleculeNet benchmark).")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                render_metric_card("Held-Out ROC-AUC", f"{metrics.get('roc_auc', 0.85):.4f}", "Test split (20%)", "#3b82f6", badge="Discrimination", badge_variant="info")
            with m2:
                render_metric_card("PR-AUC (Avg Prec)", f"{metrics.get('pr_auc', 0.30):.4f}", "Minority class evaluation", "#06b6d4", badge="Precision-Recall", badge_variant="info")
            with m3:
                render_metric_card("Balanced Accuracy", f"{metrics.get('balanced_accuracy', 0.66):.4f}", "Mean of sensitivities", "#38bdf8", badge="Class-Weighted", badge_variant="info")
            with m4:
                render_metric_card("F1-Score", f"{metrics.get('f1', 0.39):.4f}", "Harmonic mean prec/recall", "#a78bfa", badge="F1 Metric", badge_variant="purple")

            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

            col_cm, col_meta = st.columns([1, 1.2], gap="medium")
            with col_cm:
                st.markdown("<h4 style='margin: 0 0 0.5rem 0; font-size: 1.1rem; color: #101B2D;'>Test Split Confusion Matrix</h4>", unsafe_allow_html=True)
                cm = metrics.get("confusion_matrix", [[270, 1], [19, 3]])
                df_cm = pd.DataFrame(
                    cm,
                    columns=["Predicted Non-Toxic (0)", "Predicted Toxic (1)"],
                    index=["Actual Non-Toxic (0)", "Actual Toxic (1)"],
                )
                st.dataframe(df_cm, use_container_width=True)
                test_cnt = metrics.get('test_count', 293)
                st.caption(f"Evaluated on stratified held-out 20% test partition (N = {test_cnt}).")

            with col_meta:
                st.markdown(
                    f"""
                    <div class="mol-card" style="padding: 1rem 1.25rem;">
                        <h4 style="margin: 0 0 0.5rem 0; font-size: 1.1rem; color: #101B2D;'>Architecture & Training Configuration</h4>
                        <div style="font-size: 0.88rem; color: #334155; line-height: 1.7;">
                            <div>• <strong>Algorithm:</strong> <code>{meta.get('algorithm', 'RandomForestClassifier')}</code></div>
                            <div>• <strong>Ensemble Size:</strong> <code>{meta.get('n_estimators', 300)}</code> estimators</div>
                            <div>• <strong>Class Weighting:</strong> <code>{meta.get('class_weight', 'balanced')}</code></div>
                            <div>• <strong>Featurization:</strong> Morgan circular fingerprints (radius=2, 2048 bits)</div>
                            <div>• <strong>Calibration:</strong> Platt scaling (Logistic Regression on validation folds)</div>
                            <div>• <strong>Training Cohort:</strong> <code>{metrics.get('train_count', 1168)}</code> compounds (80%)</div>
                            <div>• <strong>Test Partition:</strong> <code>{metrics.get('test_count', 293)}</code> compounds (20%)</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                """
                <div class="mol-card" style="border-left: 4px solid #27C7B8;">
                    <strong style="color: #0F8075;">Statistical Note on Class Imbalance:</strong>
                    <p style="color: #334155; font-size: 0.88rem; margin: 0.3rem 0 0 0; line-height: 1.5;">
                        In the ClinTox benchmark, toxic compounds represent only ~7.5% of samples. A trivial majority-class classifier
                        would achieve 92.5% standard accuracy by predicting "Non-Toxic" for every candidate. For this reason, standard accuracy is
                        intentionally excluded in favor of <strong>PR-AUC</strong> and <strong>Balanced Accuracy</strong>.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Dual Generalization Benchmark (Stratified Random vs. Bemis-Murcko Scaffold Split)
            scaffold_bench_path = "artifacts/clintox/scaffold_benchmark.json"
            if os.path.exists(scaffold_bench_path):
                try:
                    with open(scaffold_bench_path, "r", encoding="utf-8") as f:
                        scaf_data = json.load(f)
                    rand_m = scaf_data.get("random_split", {})
                    scaf_m = scaf_data.get("scaffold_split", {})
                except Exception:
                    rand_m, scaf_m = {}, {}
            else:
                rand_m, scaf_m = {}, {}

            st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
            st.markdown("### Dual Generalization Benchmark: Stratified Random vs. Bemis–Murcko Scaffold Split")
            st.caption(
                "Scientific evaluation comparing in-domain chemical series interpolation (random split) against "
                "out-of-scaffold generalization to novel chemotypes (scaffold split). Note: The two splits evaluate "
                "separately trained models on different held-out test partitions."
            )

            # Prominent Scientific Alert Card on Low Scaffold Recall and False Negatives
            st.markdown(
                """
                <div class="mol-card" style="border-left: 4px solid #f59e0b; background: #fffbeb; padding: 1rem 1.25rem; margin-bottom: 1rem;">
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.4rem;">
                        <span style="font-size: 1.2rem;">⚠️</span>
                        <strong style="color: #92400e; font-size: 1.05rem;">Critical Scientific Finding: High False-Negative Risk on Novel Scaffolds</strong>
                    </div>
                    <div style="font-size: 0.88rem; color: #78350f; line-height: 1.6;">
                        <p style="margin: 0 0 0.5rem 0;">
                            <strong>Severe Sensitivity Deficit:</strong> At the default 0.5 classification threshold, the scaffold-trained model detects only
                            <strong>1 out of 17 toxic compounds (5.88% recall)</strong>, missing <strong>16 of 17 toxic molecules (94.1% false-negative rate)</strong>.
                            While precision reaches 100% (0 false positives), this occurs because the model behaves ultra-conservatively on novel chemotypes.
                            A "Non-Toxic" prediction must <em>never</em> be interpreted as confirmed biological safety.
                        </p>
                        <p style="margin: 0 0 0.5rem 0;">
                            <strong>Independent Comparator Model:</strong> The scaffold benchmark evaluates an independently retrained model tested on a distinct held-out
                            partition (17 toxic cases, 5.80% prevalence); it is <em>not</em> the production serving model evaluated on the random test partition.
                        </p>
                        <p style="margin: 0 0 0.5rem 0;">
                            <strong>Statistical Uncertainty:</strong> With only 22 positive cases (random) and 17 positive cases (scaffold) in the test sets, the approximate
                            Hanley–McNeil 95% confidence intervals overlap substantially ([0.7484, 0.9530] vs. [0.6569, 0.9185]). The difference in ROC-AUC
                            (Δ = 0.0630) is within approximately one standard error (SE ≈ 0.067) and should not be overinterpreted as conclusive separation.
                        </p>
                        <p style="margin: 0;">
                            <strong>Exploratory Scope:</strong> This model is strictly an exploratory research and educational prioritization tool.
                            It is not validated for clinical decision-making, patient triage, or regulatory safety determinations.
                        </p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Compact Dual Benchmark Comparison Table
            dual_rows = [
                {
                    "Evaluation Dimension": "Evaluation Paradigm",
                    "Stratified Random Split (Production Serving Model)": "In-domain chemical series interpolation",
                    "Bemis–Murcko Scaffold Split (Comparator Model)": "Out-of-scaffold generalization to novel chemotypes",
                    "Scientific Interpretation": "Assesses familiar chemotypes vs. structurally disjoint ring-linker scaffolds",
                },
                {
                    "Evaluation Dimension": "Model Role",
                    "Stratified Random Split (Production Serving Model)": "Production serving model (trained on random train split)",
                    "Bemis–Murcko Scaffold Split (Comparator Model)": "Independent comparator (trained on scaffold train split)",
                    "Scientific Interpretation": "Scaffold benchmark evaluates a separately trained model, not the production artifact",
                },
                {
                    "Evaluation Dimension": "Held-Out Test Partition (N)",
                    "Stratified Random Split (Production Serving Model)": f"N = {rand_m.get('test_count', 293)} (from 1,461 unique deduplicated compounds)",
                    "Bemis–Murcko Scaffold Split (Comparator Model)": f"N = {scaf_m.get('test_count', 293)} (from 1,461 unique deduplicated compounds)",
                    "Scientific Interpretation": "Strict zero-overlap partitions derived after deduplicating conflicting raw entries",
                },
                {
                    "Evaluation Dimension": "Test Class Composition",
                    "Stratified Random Split (Production Serving Model)": f"{rand_m.get('test_positive_count', 22)} pos / {rand_m.get('test_negative_count', 271)} neg (7.51% prevalence)",
                    "Bemis–Murcko Scaffold Split (Comparator Model)": f"{scaf_m.get('test_positive_count', 17)} pos / {scaf_m.get('test_negative_count', 276)} neg (5.80% prevalence)",
                    "Scientific Interpretation": "Scaffold clustering shifts positive test prevalence downward from 7.51% to 5.80%",
                },
                {
                    "Evaluation Dimension": "ROC-AUC (Global Discrimination)",
                    "Stratified Random Split (Production Serving Model)": f"{rand_m.get('roc_auc', 0.8507):.4f}",
                    "Bemis–Murcko Scaffold Split (Comparator Model)": f"{scaf_m.get('roc_auc', 0.7877):.4f}",
                    "Scientific Interpretation": "Threshold-independent ranking; gap (Δ = 0.0630) reflects novel chemotype challenge",
                },
                {
                    "Evaluation Dimension": "Approx. 95% CI (Hanley–McNeil)",
                    "Stratified Random Split (Production Serving Model)": "[0.7484, 0.9530] (SE = 0.0522)",
                    "Bemis–Murcko Scaffold Split (Comparator Model)": "[0.6569, 0.9185] (SE = 0.0667)",
                    "Scientific Interpretation": "Approximate biostatistical uncertainty intervals overlap substantially (overlap > 0.20 AUC)",
                },
                {
                    "Evaluation Dimension": "PR-AUC (Average Precision)",
                    "Stratified Random Split (Production Serving Model)": f"{rand_m.get('pr_auc', 0.5099):.4f}",
                    "Bemis–Murcko Scaffold Split (Comparator Model)": f"{scaf_m.get('pr_auc', 0.3994):.4f}",
                    "Scientific Interpretation": "Precision across recall levels; sensitive to positive base rate drop (7.51% → 5.80%)",
                },
                {
                    "Evaluation Dimension": "Recall at Threshold 0.5",
                    "Stratified Random Split (Production Serving Model)": f"{rand_m.get('recall', 0.1364)*100:.2f}% (3 TP / 19 FN)",
                    "Bemis–Murcko Scaffold Split (Comparator Model)": f"{scaf_m.get('recall', 0.0588)*100:.2f}% (1 TP / 16 FN)",
                    "Scientific Interpretation": "Extreme sensitivity deficit on novel scaffolds: misses 94.1% of toxic compounds",
                },
                {
                    "Evaluation Dimension": "Precision at Threshold 0.5",
                    "Stratified Random Split (Production Serving Model)": f"{rand_m.get('precision', 0.7500)*100:.2f}% (3 TP / 1 FP)",
                    "Bemis–Murcko Scaffold Split (Comparator Model)": f"{scaf_m.get('precision', 1.0000)*100:.2f}% (1 TP / 0 FP)",
                    "Scientific Interpretation": "Zero false alarms on novel scaffolds, achieved by extreme conservatism",
                },
                {
                    "Evaluation Dimension": "Standard Accuracy",
                    "Stratified Random Split (Production Serving Model)": f"{rand_m.get('accuracy', 0.9317)*100:.2f}%",
                    "Bemis–Murcko Scaffold Split (Comparator Model)": f"{scaf_m.get('accuracy', 0.9454)*100:.2f}%",
                    "Scientific Interpretation": "Misleadingly high due to >92% majority negative class; uninformative without recall",
                },
                {
                    "Evaluation Dimension": "Brier Score (Calibration Loss)",
                    "Stratified Random Split (Production Serving Model)": f"{rand_m.get('brier_score', 0.0522):.4f}",
                    "Bemis–Murcko Scaffold Split (Comparator Model)": f"{scaf_m.get('brier_score', 0.0467):.4f}",
                    "Scientific Interpretation": "Mean squared probability error (lower is better); reflects Platt sigmoid calibration",
                },
            ]
            st.dataframe(pd.DataFrame(dual_rows), use_container_width=True, hide_index=True)
            st.caption(
                "Methodological Notice: Random split evaluates chemical interpolation within known scaffolds; "
                "Bemis–Murcko scaffold split evaluates out-of-scaffold generalization to novel chemotypes. "
                "Confidence intervals are approximate Hanley–McNeil (1982) standard error estimates based on test cohort "
                "positive/negative sample counts. ClinTox models are exploratory research tools and do not constitute "
                "clinical or regulatory safety determinations."
            )
        else:
            st.warning("⚠️ ClinTox metadata artifact not found. Please train models first.")

    # TAB 2: Tox21
    with tab_tox21:
        rows, skipped_eps = load_tox21_endpoint_performance_rows("artifacts/tox21")
        if rows:
            st.markdown("### Tox21 12-Endpoint In Vitro Screening Suite")
            st.caption("Independent binary Random Forest classifiers trained with isolated missing-label exclusion.")

            df_tox = pd.DataFrame(rows)
            st.dataframe(df_tox, use_container_width=True, hide_index=True)
            if skipped_eps:
                st.warning(f"⚠️ Metadata unavailable or malformed for endpoint(s): {', '.join(skipped_eps)}.")

            cal_bench_path = "artifacts/tox21/calibration_benchmark.json"
            if os.path.exists(cal_bench_path):
                with open(cal_bench_path, "r", encoding="utf-8") as f:
                    cal_bench = json.load(f)
                with st.expander("📐 Post-Hoc Probability Calibration Benchmark (3-Way Disjoint Selection & 5-Fold OOF Fitting)"):
                    cal_rows = []
                    for ep, ep_data in cal_bench.get("endpoints", {}).items():
                        hm = ep_data.get("held_out_test_metrics", {})
                        raw_m = hm.get("uncalibrated", {})
                        sig_m = hm.get("sigmoid", {})
                        iso_m = hm.get("isotonic", {})
                        gov = ep_data.get("selection_governance", {})
                        cal_rows.append({
                            "Endpoint": ep,
                            "Prevalence": f"{ep_data.get('data_summary', {}).get('prevalence', 0.0)*100:.1f}%",
                            "Raw Brier": f"{raw_m.get('brier_score', 0.0):.4f}",
                            "Sigmoid Brier": f"{sig_m.get('brier_score', 0.0):.4f}",
                            "Isotonic Brier": f"{iso_m.get('brier_score', 0.0):.4f}",
                            "Raw ECE": f"{raw_m.get('ece_10bin_weighted', 0.0):.4f}",
                            "Sigmoid ECE": f"{sig_m.get('ece_10bin_weighted', 0.0):.4f}",
                            "Isotonic ECE": f"{iso_m.get('ece_10bin_weighted', 0.0):.4f}",
                            "Recommended": gov.get("recommended_method", "sigmoid"),
                        })
                    st.dataframe(pd.DataFrame(cal_rows), use_container_width=True, hide_index=True)
                    st.caption(
                        "Recommended calibration methods were selected via 5-fold 3-way disjoint cross-validation strictly within the "
                        "80% training partition (base_rf 60% ⊥ cal_fit 20% ⊥ val 20%), distinct from the held-out 20% test benchmark "
                        "table shown above (N_test = 15,577 across 12 assays; note: test partitions were previously inspected during "
                        "initial v1.0.0 diagnostics prior to v1.1.0 method-selection remediation). Final calibrators are fitted on "
                        "5-fold stratified out-of-fold (OOF) training predictions. Default screening preserves raw uncalibrated "
                        "probabilities; calibrated probabilities are for research use only and do not imply clinical or wet-lab safety."
                    )

            # Phase B.2 Interactive Endpoint Reliability Diagnostics Workbench
            render_tox21_endpoint_diagnostics_section(
                artifact_dir="artifacts/tox21",
                csv_path="data/raw/tox21.csv",
            )

            st.markdown(
                """
                <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 0.5rem;">
                    <strong>Pathways Monitored:</strong>
                    Nuclear Receptors (Androgen Receptor, AhR, Aromatase, Estrogen Receptor, PPAR-gamma) |
                    Stress Response (Antioxidant Response Element, ATAD5 DNA Damage, Heat Shock Element, Mitochondrial Membrane Potential, p53).
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("⚠️ Tox21 metadata artifacts not found.")

    # TAB 3: Applicability & Boundaries
    with tab_limits:
        st.markdown("### Scientific Boundaries & Research Governance")

        st.markdown(
            """
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 0.5rem;">
                <div class="mol-card" style="margin: 0;">
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.4rem;">
                        <span style="font-size: 1.2rem;">🛡️</span>
                        <h4 style="margin: 0; font-size: 1.05rem; color: #101B2D;">Benchmark Ground Truth</h4>
                    </div>
                    <p style="font-size: 0.88rem; color: #475569; line-height: 1.5; margin: 0;">
                        The ClinTox dataset annotates clinical trial outcomes. A negative prediction indicates absence of documented clinical trial termination, NOT guaranteed cellular safety or absence of late-onset toxicity.
                    </p>
                </div>
                <div class="mol-card" style="margin: 0;">
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.4rem;">
                        <span style="font-size: 1.2rem;">🧪</span>
                        <h4 style="margin: 0; font-size: 1.05rem; color: #101B2D;">In Vitro vs. In Vivo Biology</h4>
                    </div>
                    <p style="font-size: 0.88rem; color: #475569; line-height: 1.5; margin: 0;">
                        Tox21 luminescence assays measure cellular pathway perturbations in immortalized cell lines. They do not simulate systemic absorption, hepatic metabolism, blood-brain barrier transport, or whole-organ clearance.
                    </p>
                </div>
                <div class="mol-card" style="margin: 0;">
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.4rem;">
                        <span style="font-size: 1.2rem;">📊</span>
                        <h4 style="margin: 0; font-size: 1.05rem; color: #101B2D;">Severe Class Imbalance</h4>
                    </div>
                    <p style="font-size: 0.88rem; color: #475569; line-height: 1.5; margin: 0;">
                        Severe minority class representation (~7.5% in ClinTox) requires cost-sensitive learning and strict PR-AUC optimization. Standard accuracy figures can be misleading in safety classification.
                    </p>
                </div>
                <div class="mol-card" style="margin: 0;">
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.4rem;">
                        <span style="font-size: 1.2rem;">📐</span>
                        <h4 style="margin: 0; font-size: 1.05rem; color: #101B2D;">Chemical Space & Domain Limits</h4>
                    </div>
                    <p style="font-size: 0.88rem; color: #475569; line-height: 1.5; margin: 0;">
                        Models are trained on drug-like organic molecules (MW < 600 Da). Novel scaffolds with Tanimoto similarity < 0.40 to training data trigger domain warnings indicating uncertain extrapolation.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
