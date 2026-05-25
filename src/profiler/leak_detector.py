"""
leak_detector.py — KVScope Leak & Anomaly Detection
=====================================================
KV cache "leaks" manifest in three ways:
  1. GROWTH LEAK    — cache grows super-linearly vs expected O(n*d*h)
  2. POST-EOS LEAK  — cache doesn't shrink after sequence ends (paged blocks not freed)
  3. FRAGMENTATION  — NVML 'used' >> torch.reserved >> torch.allocated (gaps accumulate)

This module provides statistical and heuristic detectors for all three.
Each detector returns a score in [0, 1] where 1 = definite leak.

Usage:
    detector = KVLeakDetector(model_type="gemma4")
    results = detector.analyze(tracer.snapshots_as_dataframe(), nvml_sampler)
    detector.print_report(results)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter


# ─── Result Types ─────────────────────────────────────────────────────────────

@dataclass
class LeakFinding:
    """A single detected anomaly."""
    detector: str             # Which detector found this
    severity: str             # "low" | "medium" | "high" | "critical"
    score: float              # 0–1 anomaly score
    description: str
    layers_affected: List[int] = field(default_factory=list)
    evidence: Dict = field(default_factory=dict)

    @property
    def emoji(self) -> str:
        return {"low": "🟡", "medium": "🟠", "high": "🔴", "critical": "🚨"}.get(self.severity, "❓")


@dataclass
class DetectorReport:
    model_type: str
    findings: List[LeakFinding] = field(default_factory=list)
    overall_leak_score: float = 0.0
    summary: str = ""

    @property
    def has_leaks(self) -> bool:
        return self.overall_leak_score > 0.3

    def critical_findings(self) -> List[LeakFinding]:
        return [f for f in self.findings if f.severity == "critical"]


# ─── Individual Detectors ─────────────────────────────────────────────────────

class GrowthCurveDetector:
    """
    Detects super-linear KV cache growth.

    Expected model: KV_size(step) = a * step + b  (linear in sequence length)
    If actual growth significantly exceeds this, memory is accumulating faster than
    expected — typically indicating:
      - vLLM PagedAttention block fragmentation
      - Cache not being properly evicted between requests
      - BnB or quantization overhead accumulating
    """

    def __init__(self, linearity_threshold: float = 0.95):
        self.linearity_threshold = linearity_threshold  # R² below this = anomalous

    def detect(self, df: pd.DataFrame, layer_type_filter: Optional[str] = None) -> LeakFinding:
        if layer_type_filter:
            df = df[df["layer_type"] == layer_type_filter]

        if df.empty or "step" not in df.columns:
            return LeakFinding(
                detector="growth_curve",
                severity="low",
                score=0.0,
                description="No data available for growth analysis.",
            )

        # ─── Sliding-window plateau guard ──────────────────────────────────
        # If we have explicit per-snapshot seq_len (k_seq_len), and the seq_len
        # caps out at a fixed value while step keeps incrementing, then the
        # cache is operating in sliding-window mode. That's a *feature*, not a
        # leak — the growth curve will look saturating, not linear. Score = 0.
        if "k_seq_len" in df.columns:
            seq_max = float(df["k_seq_len"].max())
            seq_p95 = float(df["k_seq_len"].quantile(0.95)) if len(df) > 0 else seq_max
            seq_p99 = float(df["k_seq_len"].quantile(0.99)) if len(df) > 0 else seq_max
            # Plateau heuristic: 95th and 99th percentiles within 5% of each
            # other AND of the max → distribution piles up at the cap.
            plateau_ratio = (seq_p99 - seq_p95) / max(seq_max, 1.0)
            if seq_max > 0 and plateau_ratio < 0.05 and seq_p95 / max(seq_max, 1) > 0.95:
                # Detect: sliding-window architecture
                return LeakFinding(
                    detector="growth_curve",
                    severity="low",
                    score=0.0,
                    description=(
                        f"Sliding-window plateau detected — KV seq_len caps at "
                        f"{int(seq_max)} tokens (p95={int(seq_p95)}, p99={int(seq_p99)}). "
                        f"Growth curve is saturating by design, not super-linear. "
                        f"Skipping linearity check."
                    ),
                    evidence={
                        "plateau_seq_len": int(seq_max),
                        "p95_seq_len": int(seq_p95),
                        "p99_seq_len": int(seq_p99),
                        "plateau_ratio": round(plateau_ratio, 4),
                        "skipped_reason": "sliding_window_plateau",
                    },
                )

        # Aggregate total KV MB per step
        step_totals = df.groupby("step")["total_mb"].sum().reset_index()
        if len(step_totals) < 5:
            return LeakFinding(
                detector="growth_curve",
                severity="low",
                score=0.0,
                description="Too few data points to assess growth linearity.",
            )

        x = step_totals["step"].values.astype(float)
        y = step_totals["total_mb"].values

        # Linear fit
        slope, intercept, r, p, se = stats.linregress(x, y)
        r2 = r ** 2

        # Fit a quadratic too — if it fits MUCH better, growth is super-linear
        coeffs2 = np.polyfit(x, y, 2)
        y_pred_quadratic = np.polyval(coeffs2, x)
        ss_res_q = np.sum((y - y_pred_quadratic) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2_quadratic = 1 - ss_res_q / ss_tot if ss_tot > 0 else 1.0

        # Super-linearity score: how much better does quadratic fit vs linear?
        quadratic_advantage = max(0.0, r2_quadratic - r2)

        # Residual analysis
        y_pred_linear = slope * x + intercept
        residuals = y - y_pred_linear
        max_positive_residual_mb = float(residuals.max())
        residual_trend_slope, *_ = stats.linregress(x, residuals)

        # ── Compute anomaly score with leak-specific gating ────────────────
        # A poor R² alone is not enough to flag a leak — many architectures
        # produce non-linear-but-bounded growth (sliding window, paged
        # eviction, KV compression). What actually indicates a leak is:
        #   (a) growth curve is super-linear (quadratic fits MUCH better), AND
        #   (b) residuals from the linear fit are TRENDING UPWARD (the gap is
        #       growing over time, not just noisy).
        # We require BOTH signals before scoring high.
        score = 0.0
        if quadratic_advantage > 0.05 and residual_trend_slope > 0.005:
            # Both super-linearity AND growing residual trend present
            score += quadratic_advantage * 2
            score += min(1.0, residual_trend_slope * 10)
        elif r2 < self.linearity_threshold and slope > 0:
            # Sub-linear / saturating growth (slope > 0 but R² low, no super-linear
            # signal) → mild signal at most, capped low.
            score += min(0.2, (self.linearity_threshold - r2))
        score = min(1.0, score)

        severity = "low"
        if score > 0.7:
            severity = "critical"
        elif score > 0.5:
            severity = "high"
        elif score > 0.3:
            severity = "medium"

        return LeakFinding(
            detector="growth_curve",
            severity=severity,
            score=round(score, 3),
            description=(
                f"KV growth linearity R²={r2:.3f} (threshold {self.linearity_threshold}). "
                f"Quadratic fit advantage: {quadratic_advantage:.3f}. "
                f"Residual trend: {residual_trend_slope:+.4f} MB/step. "
                f"Max positive residual: {max_positive_residual_mb:.2f} MB."
            ),
            evidence={
                "r2_linear": round(r2, 4),
                "r2_quadratic": round(r2_quadratic, 4),
                "quadratic_advantage": round(quadratic_advantage, 4),
                "slope_mb_per_step": round(float(slope), 4),
                "max_residual_mb": round(max_positive_residual_mb, 3),
                "residual_trend": round(float(residual_trend_slope), 5),
            },
        )


class PostEOSDetector:
    """
    Detects KV cache not being freed after sequence completion.

    After the EOS token is generated (or max_new_tokens is reached), the GPU
    memory used for KV cache should return close to the pre-generation baseline.
    If it doesn't, blocks are "leaked" — held in reserved memory but not freed.

    Requires post-generation NVML samples.
    """

    def __init__(self, leak_threshold_mb: float = 50.0):
        self.leak_threshold_mb = leak_threshold_mb

    def detect(
        self,
        baseline_mb: float,
        peak_mb: float,
        post_eos_mb: float,
        n_sequences: int = 1,
    ) -> LeakFinding:
        """
        Args:
            baseline_mb: NVML used_mb before generation started
            peak_mb: NVML used_mb at peak during generation
            post_eos_mb: NVML used_mb after generation + gc.collect() + cache.clear()
            n_sequences: How many sequences were generated (for proportional check)
        """
        expected_return = peak_mb - baseline_mb   # All of this should be freed
        actual_freed = peak_mb - post_eos_mb
        leaked_mb = expected_return - actual_freed

        if expected_return <= 0:
            return LeakFinding(
                detector="post_eos",
                severity="low",
                score=0.0,
                description="Baseline equals peak — no generation occurred or measurement error.",
            )

        leak_fraction = max(0.0, leaked_mb / expected_return)
        score = min(1.0, leak_fraction)

        severity = "low"
        if leaked_mb > self.leak_threshold_mb * 4:
            severity = "critical"
        elif leaked_mb > self.leak_threshold_mb * 2:
            severity = "high"
        elif leaked_mb > self.leak_threshold_mb:
            severity = "medium"

        return LeakFinding(
            detector="post_eos",
            severity=severity,
            score=round(score, 3),
            description=(
                f"Post-EOS unreleased memory: {leaked_mb:.1f} MB "
                f"({leak_fraction*100:.1f}% of peak KV overhead). "
                f"Expected freed: {expected_return:.1f} MB, actual freed: {actual_freed:.1f} MB."
            ),
            evidence={
                "baseline_mb": round(baseline_mb, 2),
                "peak_mb": round(peak_mb, 2),
                "post_eos_mb": round(post_eos_mb, 2),
                "leaked_mb": round(leaked_mb, 2),
                "leak_fraction": round(leak_fraction, 4),
                "n_sequences": n_sequences,
            },
        )


class FragmentationDetector:
    """
    Detects VRAM fragmentation: memory held by CUDA allocator but not used.

    Fragmentation = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()

    Normal fragmentation: <10% of reserved. High fragmentation indicates the
    allocator is holding large blocks that can't be coalesced — effectively
    a soft leak that prevents other allocations.

    This is especially pronounced with PagedAttention (vLLM) where block sizes
    don't perfectly align with tensor sizes.
    """

    def __init__(self, fragmentation_threshold: float = 0.15):
        self.fragmentation_threshold = fragmentation_threshold

    def detect(self, df: pd.DataFrame) -> LeakFinding:
        if "gpu_reserved_mb" not in df.columns or "gpu_alloc_mb" not in df.columns:
            return LeakFinding(
                detector="fragmentation",
                severity="low",
                score=0.0,
                description="GPU memory columns not available.",
            )

        if df.empty or "step" not in df.columns:
            return LeakFinding(
                detector="fragmentation",
                severity="low",
                score=0.0,
                description="No data available for fragmentation analysis.",
            )

        step_mem = df.groupby("step").agg(
            reserved=("gpu_reserved_mb", "max"),
            allocated=("gpu_alloc_mb", "max"),
        ).reset_index()

        step_mem["frag_mb"] = step_mem["reserved"] - step_mem["allocated"]
        step_mem["frag_ratio"] = step_mem["frag_mb"] / step_mem["reserved"].clip(lower=1)

        avg_frag_ratio = float(step_mem["frag_ratio"].mean())
        max_frag_mb = float(step_mem["frag_mb"].max())

        # Check if fragmentation grows with sequence length (bad sign)
        if len(step_mem) >= 5:
            frag_trend, *_ = stats.linregress(
                step_mem["step"].values,
                step_mem["frag_ratio"].values,
            )
        else:
            frag_trend = 0.0

        score = 0.0
        if avg_frag_ratio > self.fragmentation_threshold:
            score += (avg_frag_ratio - self.fragmentation_threshold) * 3
        if frag_trend > 0.001:  # Fragmentation ratio is growing
            score += frag_trend * 100
        score = min(1.0, score)

        severity = "low"
        if score > 0.6:
            severity = "critical"
        elif score > 0.4:
            severity = "high"
        elif score > 0.2:
            severity = "medium"

        return LeakFinding(
            detector="fragmentation",
            severity=severity,
            score=round(score, 3),
            description=(
                f"Avg fragmentation ratio: {avg_frag_ratio:.2%}. "
                f"Max fragmented: {max_frag_mb:.1f} MB. "
                f"Fragmentation trend: {'growing' if frag_trend > 0 else 'stable'}."
            ),
            evidence={
                "avg_frag_ratio": round(avg_frag_ratio, 4),
                "max_frag_mb": round(max_frag_mb, 2),
                "frag_trend_per_step": round(float(frag_trend), 6),
                "threshold": self.fragmentation_threshold,
            },
        )


class LayerAnomalyDetector:
    """
    Detects per-layer KV anomalies — identifies which specific layers
    contribute disproportionately to cache size or grow abnormally.

    For Gemma 4: global layers should dominate (full context), local should be tiny.
    For DeepSeek MLA: all layers should have similar small latent sizes.
    For GLM MoE: cache shouldn't vary across layers (attention is not sparse).
    """

    def detect(self, df: pd.DataFrame, model_type: str) -> LeakFinding:
        if df.empty or "step" not in df.columns:
            return LeakFinding(
                detector="layer_anomaly",
                severity="low",
                score=0.0,
                description="No data available for layer anomaly analysis.",
            )

        final_step = df["step"].max()
        final = df[df["step"] == final_step]
        layer_sizes = final.groupby("layer_idx")["total_mb"].sum()

        if len(layer_sizes) < 2:
            return LeakFinding(
                detector="layer_anomaly",
                severity="low",
                score=0.0,
                description="Too few layers to detect anomalies.",
            )

        mean_size = float(layer_sizes.mean())
        std_size = float(layer_sizes.std())

        # Z-score based outlier detection (threshold: 3 sigma)
        z_scores = (layer_sizes - mean_size) / (std_size + 1e-8)
        outlier_layers = layer_sizes[z_scores.abs() > 3].index.tolist()

        score = min(1.0, len(outlier_layers) / max(1, len(layer_sizes)) * 5)
        severity = "low" if score < 0.3 else "medium" if score < 0.6 else "high"

        return LeakFinding(
            detector="layer_anomaly",
            severity=severity,
            score=round(score, 3),
            description=(
                f"{len(outlier_layers)} layers are statistical outliers (>3σ from mean). "
                f"Mean layer KV: {mean_size:.2f}MB ± {std_size:.2f}MB."
            ),
            layers_affected=outlier_layers,
            evidence={
                "mean_layer_mb": round(mean_size, 3),
                "std_layer_mb": round(std_size, 3),
                "outlier_layers": outlier_layers,
                "max_layer_mb": round(float(layer_sizes.max()), 3),
                "min_layer_mb": round(float(layer_sizes.min()), 3),
            },
        )


class MLACompressionDriftDetector:
    """
    DeepSeek V4 specific: detects drift in MLA compression ratios.

    MLA compression ratio should be constant (it's architecturally fixed
    as kv_lora_rank / (n_heads * head_dim)). If the effective ratio changes
    across layers or steps, something is wrong with the KV projection.
    """

    def detect(self, df: pd.DataFrame) -> Optional[LeakFinding]:
        if "compression_ratio" not in df.columns:
            return None

        ratios = df["compression_ratio"].dropna()
        if len(ratios) < 10:
            return None

        ratio_std = float(ratios.std())
        ratio_mean = float(ratios.mean())
        cv = ratio_std / (ratio_mean + 1e-8)  # Coefficient of variation

        score = min(1.0, cv * 10)
        severity = "low" if score < 0.2 else "medium" if score < 0.5 else "high"

        return LeakFinding(
            detector="mla_compression_drift",
            severity=severity,
            score=round(score, 3),
            description=(
                f"MLA compression ratio: mean={ratio_mean:.4f}, std={ratio_std:.4f}, "
                f"CV={cv:.4f}. "
                f"{'Stable' if cv < 0.05 else 'Drifting — investigate KV projection'}."
            ),
            evidence={
                "ratio_mean": round(ratio_mean, 5),
                "ratio_std": round(ratio_std, 5),
                "ratio_cv": round(cv, 5),
                "ratio_min": round(float(ratios.min()), 5),
                "ratio_max": round(float(ratios.max()), 5),
            },
        )


# ─── Cross-Architecture Characterization Detectors ───────────────────────────
#
# Unlike the leak detectors above (which look for *anomalies* relative to a
# null hypothesis), the two detectors below produce *quantitative descriptors*
# that are directly comparable across architectures. They always run, always
# return a finding (severity stays "low" — they are descriptive, not diagnostic),
# and surface the numbers that go straight into the cross-architecture table
# in the paper:
#
#   - CacheDensityDetector   → bytes per generated token, per layer (KB/tok)
#   - LayerUniformityDetector → coefficient of variation of per-layer KV bytes
#
# These don't contribute to the leak score (weight 0). They exist so every
# profiled model produces a consistent set of cross-comparable numbers.


class CacheDensityDetector:
    """
    Quantifies KV cache density: bytes-per-token-per-layer at peak.

    For a model with L attention layers profiled, this reports:
        density_kb_per_tok = total_kv_bytes_at_peak / (final_seq_len * L_active)

    where L_active is the number of distinct layers that produced non-zero
    snapshots. This number is directly comparable across architectures and is
    the natural unit for "how memory-hungry is this model's cache".

    A well-formed dense GQA model with H_kv KV heads, head_dim D, dtype size B,
    using a plain DynamicCache should report exactly H_kv * D * B * 2 bytes/token
    per layer (the 2 is for separate K and V tensors).
    """

    def detect(self, df: pd.DataFrame) -> LeakFinding:
        if df.empty or "layer_idx" not in df.columns:
            return LeakFinding(
                detector="cache_density",
                severity="low",
                score=0.0,
                description="No snapshots — cannot compute cache density.",
            )

        # Per-snapshot bytes: prefer explicit k+v over total_mb (no rounding)
        df = df.copy()
        if "k_bytes" in df.columns and "v_bytes" in df.columns:
            df["_total_bytes"] = df["k_bytes"].fillna(0) + df["v_bytes"].fillna(0)
        elif "total_mb" in df.columns:
            df["_total_bytes"] = df["total_mb"].fillna(0) * (1024 ** 2)
        else:
            return LeakFinding(
                detector="cache_density",
                severity="low",
                score=0.0,
                description="No byte columns in snapshot dataframe.",
            )

        # Take the LAST snapshot per layer (regardless of how step is indexed).
        # This gives us each layer's final cache size at end-of-generation.
        # Use the row order in the dataframe to pick the latest sample per layer.
        if "timestamp_ms" in df.columns:
            df = df.sort_values("timestamp_ms")
        last_per_layer = df.groupby("layer_idx", as_index=False).tail(1)
        n_active_layers = int(last_per_layer["layer_idx"].nunique())
        if n_active_layers == 0:
            return LeakFinding(
                detector="cache_density",
                severity="low",
                score=0.0,
                description="No active layers found in snapshots.",
            )

        # seq_len: prefer the actual K tensor's S dim (k_seq_len). This is
        # robust against sliding-window plateaus and step-counter quirks.
        # If the column is missing (older runs), fall back to step.
        if "k_seq_len" in last_per_layer.columns and last_per_layer["k_seq_len"].max() > 0:
            # Use the max across layers (sliding-window layers may plateau at the
            # window size while dense layers grow to full seq_len). The "true"
            # generation length is the max.
            peak_seq_len = int(last_per_layer["k_seq_len"].max())
            seq_len_source = "k_seq_len"
        elif "step" in df.columns:
            peak_seq_len = max(1, int(df["step"].max()))
            seq_len_source = "step"
        else:
            peak_seq_len = 1
            seq_len_source = "fallback=1"

        total_bytes_at_end = float(last_per_layer["_total_bytes"].sum())
        bytes_per_layer_avg = total_bytes_at_end / n_active_layers

        # Weighted bytes-per-token-per-layer: each layer divides its OWN bytes
        # by its OWN seq_len (so sliding layers count correctly at S=window,
        # not at S=full_seq).
        if "k_seq_len" in last_per_layer.columns:
            per_layer_seq = last_per_layer["k_seq_len"].clip(lower=1)
        else:
            per_layer_seq = pd.Series([peak_seq_len] * len(last_per_layer))
        per_layer_density = (last_per_layer["_total_bytes"].values
                             / per_layer_seq.values)
        density_kb_per_tok_per_layer = float(per_layer_density.mean()) / 1024.0

        # Total density (sum across layers, divided by max seq) — for paper table
        bytes_per_token_total = total_bytes_at_end / max(peak_seq_len, 1)

        return LeakFinding(
            detector="cache_density",
            severity="low",
            score=0.0,  # descriptive, not diagnostic
            description=(
                f"Cache density: {density_kb_per_tok_per_layer:.2f} KB/token/layer "
                f"(avg over {n_active_layers} layers; total "
                f"{bytes_per_token_total/1024:.1f} KB/token; seq_len={peak_seq_len} "
                f"from {seq_len_source})."
            ),
            evidence={
                "bytes_per_token_per_layer": round(density_kb_per_tok_per_layer * 1024, 2),
                "bytes_per_token_total": round(bytes_per_token_total, 2),
                "n_active_layers": n_active_layers,
                "peak_seq_len": peak_seq_len,
                "seq_len_source": seq_len_source,
                "total_bytes_at_end": round(total_bytes_at_end, 2),
                "avg_bytes_per_layer": round(bytes_per_layer_avg, 2),
            },
        )


class LayerUniformityDetector:
    """
    Quantifies how uniformly KV bytes are distributed across attention layers
    at peak. Reports the coefficient of variation (CV = std / mean) of
    per-layer cache size.

    Interpretation:
      - CV ≈ 0       → uniform cache (vanilla DynamicCache, all layers identical)
      - CV ∈ (0, 0.5)→ mild heterogeneity (typical for GQA with shared KV)
      - CV > 0.5     → strongly heterogeneous (sliding+global, MLA, hybrid)
      - CV > 1.0     → bimodal/polar (e.g. sliding + dense, where sliding stays
                       small forever and dense grows linearly)

    A high CV is *not* a bug. It is an architectural fingerprint. The Gemma 4
    sliding-vs-global split should produce CV > 0.6 at long sequences; GLM-4.7
    plain DynamicCache should produce CV ≈ 0.
    """

    def detect(self, df: pd.DataFrame) -> LeakFinding:
        if df.empty or "layer_idx" not in df.columns:
            return LeakFinding(
                detector="layer_uniformity",
                severity="low",
                score=0.0,
                description="No per-layer snapshots — cannot compute layer uniformity.",
            )

        df = df.copy()
        if "k_bytes" in df.columns and "v_bytes" in df.columns:
            df["_total_bytes"] = df["k_bytes"].fillna(0) + df["v_bytes"].fillna(0)
        elif "total_mb" in df.columns:
            df["_total_bytes"] = df["total_mb"].fillna(0) * (1024 ** 2)
        else:
            return LeakFinding(
                detector="layer_uniformity",
                severity="low",
                score=0.0,
                description="No byte columns in snapshot dataframe.",
            )

        # Take the last snapshot per layer (latest measurement).
        if "timestamp_ms" in df.columns:
            df = df.sort_values("timestamp_ms")
        last_per_layer = df.groupby("layer_idx", as_index=False).tail(1)

        per_layer = last_per_layer.set_index("layer_idx")["_total_bytes"]
        if len(per_layer) < 2:
            return LeakFinding(
                detector="layer_uniformity",
                severity="low",
                score=0.0,
                description=f"Only {len(per_layer)} layer(s) — cannot compute CV.",
            )

        mean_bytes = float(per_layer.mean())
        std_bytes = float(per_layer.std(ddof=0))
        cv = std_bytes / mean_bytes if mean_bytes > 0 else 0.0

        # Categorize for the description
        if cv < 0.05:
            category = "uniform (plain DynamicCache pattern)"
        elif cv < 0.25:
            category = "mildly heterogeneous (typical GQA / MoE)"
        elif cv < 0.6:
            category = "heterogeneous (likely hybrid attention)"
        else:
            category = "strongly bimodal (sliding+dense or pre-allocated mix)"

        # Layer-type breakdown if we have it (mean per type)
        layer_type_breakdown = ""
        if "layer_type" in last_per_layer.columns:
            by_type = last_per_layer.groupby("layer_type")["_total_bytes"].agg(["mean", "count"])
            parts = [
                f"{lt}: {row['mean']/1024:.1f} KB (n={int(row['count'])})"
                for lt, row in by_type.iterrows()
            ]
            if parts:
                layer_type_breakdown = " | by type: " + ", ".join(parts)

        return LeakFinding(
            detector="layer_uniformity",
            severity="low",
            score=0.0,
            description=(
                f"Per-layer KV CV={cv:.3f} — {category}. "
                f"mean={mean_bytes/1024:.1f} KB, std={std_bytes/1024:.1f} KB across "
                f"{len(per_layer)} layers (last snapshot per layer)."
                f"{layer_type_breakdown}"
            ),
            evidence={
                "cv": round(cv, 4),
                "mean_kb": round(mean_bytes / 1024, 2),
                "std_kb": round(std_bytes / 1024, 2),
                "n_layers": int(len(per_layer)),
                "category": category,
            },
        )


# ─── Orchestrator ─────────────────────────────────────────────────────────────

class KVLeakDetector:
    """
    Runs all detectors and produces a unified report with overall leak score.
    """

    def __init__(self, model_type: str):
        self.model_type = model_type
        self.growth_detector = GrowthCurveDetector()
        self.fragmentation_detector = FragmentationDetector()
        self.layer_detector = LayerAnomalyDetector()
        # Cross-architecture characterization (always-on, descriptive only)
        self.density_detector = CacheDensityDetector()
        self.uniformity_detector = LayerUniformityDetector()

    def analyze(
        self,
        df: pd.DataFrame,
        baseline_mb: Optional[float] = None,
        peak_mb: Optional[float] = None,
        post_eos_mb: Optional[float] = None,
    ) -> DetectorReport:
        """
        Run all detectors.

        Args:
            df: DataFrame from KVCacheTracer.snapshots_as_dataframe()
            baseline_mb: NVML used_mb before generation (for post-EOS check)
            peak_mb: NVML used_mb at peak
            post_eos_mb: NVML used_mb after generation complete + cache.clear()
        """
        report = DetectorReport(model_type=self.model_type)

        # Guard: no data captured
        if df.empty or "step" not in df.columns:
            report.summary = "\u26a0\ufe0f  No KV cache snapshots were captured. Hooks may not have found KV tensors."
            report.overall_leak_score = 0.0
            return report

        # 1. Growth curve
        growth = self.growth_detector.detect(df)
        report.findings.append(growth)

        # 2. Fragmentation
        frag = self.fragmentation_detector.detect(df)
        report.findings.append(frag)

        # 3. Layer anomalies
        layer = self.layer_detector.detect(df, self.model_type)
        report.findings.append(layer)

        # 4. Post-EOS (only if memory measurements provided)
        if all(v is not None for v in [baseline_mb, peak_mb, post_eos_mb]):
            eos_detector = PostEOSDetector()
            eos = eos_detector.detect(baseline_mb, peak_mb, post_eos_mb)
            report.findings.append(eos)

        # 5. MLA compression drift (DeepSeek only)
        if self.model_type == "deepseek":
            mla = MLACompressionDriftDetector().detect(df)
            if mla:
                report.findings.append(mla)

        # 6. Cache density (descriptive, weight=0)
        report.findings.append(self.density_detector.detect(df))

        # 7. Layer uniformity (descriptive, weight=0)
        report.findings.append(self.uniformity_detector.detect(df))

        # Overall score: weighted max of individual scores.
        # cache_density and layer_uniformity have weight 0 — they are
        # descriptive cross-architecture metrics, not anomaly indicators.
        weights = {
            "post_eos": 0.35,
            "growth_curve": 0.30,
            "fragmentation": 0.20,
            "layer_anomaly": 0.10,
            "mla_compression_drift": 0.05,
            "cache_density": 0.0,
            "layer_uniformity": 0.0,
        }
        total_weight = 0.0
        weighted_score = 0.0
        for finding in report.findings:
            w = weights.get(finding.detector, 0.1)
            weighted_score += finding.score * w
            total_weight += w
        report.overall_leak_score = round(
            weighted_score / total_weight if total_weight > 0 else 0.0, 3
        )

        # Summary
        critical = [f for f in report.findings if f.severity in ("critical", "high")]
        if not critical:
            report.summary = f"✅ No significant KV cache leaks detected. Score: {report.overall_leak_score:.2f}"
        else:
            report.summary = (
                f"⚠️  {len(critical)} issues detected. "
                f"Overall leak score: {report.overall_leak_score:.2f}. "
                f"Check: {', '.join(f.detector for f in critical)}"
            )

        return report

    def print_report(self, report: DetectorReport):
        """Print a formatted report to stdout."""
        print("=" * 65)
        print(f"  KVScope Leak Detection Report — {report.model_type.upper()}")
        print("=" * 65)
        print(f"  {report.summary}")
        print(f"  Overall score: {report.overall_leak_score:.3f} (0=clean, 1=severe leak)")
        print()
        for finding in report.findings:
            print(f"  {finding.emoji} [{finding.detector}] score={finding.score:.3f} ({finding.severity})")
            print(f"     {finding.description}")
            if finding.layers_affected:
                print(f"     Layers: {finding.layers_affected}")
            print()
        print("=" * 65)
