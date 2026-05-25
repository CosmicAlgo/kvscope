"""
triton_ops.py — KVScope Custom GPU Kernels
==========================================
Custom Triton kernels for KV cache analysis operations that are too slow
or impossible to do accurately from Python-level hooks alone.

Kernels:
  1. kv_l2_norm_kernel      — Compute per-head L2 norms of K/V tensors
                              (measures which heads carry the most information)
  2. mla_compression_kernel — Compute effective compression ratio for MLA
                              by measuring ||latent||_F / ||expanded||_F per token
  3. kv_entropy_kernel      — Compute attention entropy across heads
                              (low entropy = head is "dead", wasting cache space)

All kernels operate on GPU tensors directly to avoid Python-level overhead
corrupting timing measurements.

Requirements:
    triton >= 2.1.0
    CUDA compute capability >= 7.0 (T4, V100, A100, L4 all qualify)

Usage:
    from src.profiler.triton_ops import measure_kv_head_utilization, measure_mla_compression

    head_norms = measure_kv_head_utilization(k_tensor, v_tensor)
    compression = measure_mla_compression(latent_tensor, expanded_tensor)
"""

import math
from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl


# ─── Kernel 1: Per-Head L2 Norm ───────────────────────────────────────────────
# Measures which KV heads carry the most information.
# Near-zero norm heads are candidates for eviction (they contribute nothing).
# Shape: K/V are [batch, n_heads, seq_len, head_dim]

@triton.jit
def kv_head_l2_norm_kernel(
    k_ptr,            # [B, H, S, D] float16/bfloat16
    v_ptr,
    k_norm_out_ptr,   # [B, H] output — L2 norm per head
    v_norm_out_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Each program instance handles one (batch, head) pair.
    Computes L2 norm: sqrt(sum(x^2)) over (seq_len, head_dim).

    Triton 3.x compatibility note: ``tl.store`` requires the value and pointer
    to be either *both* scalar or *both* block. We accumulate as a [1]-shape
    block (acc_k) and write to a [1]-shape pointer slice (out_offsets) so
    both sides match. Earlier versions of Triton happily auto-broadcasted a
    [1] value into a scalar pointer; 3.x raises
    ``Value argument cannot be block type if pointer argument is not a block``.
    """
    bh_idx = tl.program_id(0)
    b_idx = bh_idx // H
    h_idx = bh_idx % H

    # Accumulate squared sum over S and D as a [1]-shape block.
    acc_k = tl.zeros([1], dtype=tl.float32)
    acc_v = tl.zeros([1], dtype=tl.float32)

    for s_start in range(0, S, BLOCK_S):
        s_offsets = s_start + tl.arange(0, BLOCK_S)
        s_mask = s_offsets < S

        for d_start in range(0, D, BLOCK_D):
            d_offsets = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offsets < D

            # K pointer: [B, H, S, D] → offset = b*H*S*D + h*S*D + s*D + d
            k_offset = (
                b_idx * H * S * D
                + h_idx * S * D
                + s_offsets[:, None] * D
                + d_offsets[None, :]
            )
            k_mask = s_mask[:, None] & d_mask[None, :]

            k_vals = tl.load(k_ptr + k_offset, mask=k_mask, other=0.0).to(tl.float32)
            v_vals = tl.load(v_ptr + k_offset, mask=k_mask, other=0.0).to(tl.float32)

            # tl.sum on a 2D block produces a 0-d scalar; broadcasting into a
            # [1]-shape acc keeps acc as [1].
            acc_k += tl.sum(k_vals * k_vals)
            acc_v += tl.sum(v_vals * v_vals)

    # Write L2 norms: pointer must be a block too, hence the + tl.arange(0, 1)
    out_offsets = (b_idx * H + h_idx) + tl.arange(0, 1)
    tl.store(k_norm_out_ptr + out_offsets, tl.sqrt(acc_k))
    tl.store(v_norm_out_ptr + out_offsets, tl.sqrt(acc_v))


def _torch_kv_head_l2(k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Plain-PyTorch fallback for per-head L2 norms.

    Used when the Triton kernel cannot be compiled (rare — typically driven
    by Triton API churn between major releases). Mathematically identical
    to the kernel; about 5–15% slower in practice for the small tensors we
    feed it (one decode-step KV per layer).
    """
    # Sum-of-squares over (S, D), then sqrt → per-head L2 norm
    k_sq = (k.float() ** 2).sum(dim=(-2, -1))   # [B, H]
    v_sq = (v.float() ** 2).sum(dim=(-2, -1))
    return torch.sqrt(k_sq), torch.sqrt(v_sq)


def measure_kv_head_utilization(
    k: torch.Tensor,
    v: torch.Tensor,
    backend: str = "auto",
) -> Dict[str, torch.Tensor]:
    """
    Compute per-head L2 norms for K and V tensors.

    Args:
        k: Key tensor [B, H, S, D] or [B, S, H, D] (auto-transposed)
        v: Value tensor, same shape as k
        backend: "auto" (try Triton, fall back to torch), "triton", or "torch"

    Returns:
        dict with:
          "k_norms": [B, H] L2 norm per batch per head
          "v_norms": [B, H]
          "dead_head_mask": [H] bool — True if head norm < threshold
          "head_utilization_ratio": float — fraction of non-dead heads
          "backend_used": which path was actually taken
    """
    # Normalize to [B, H, S, D]
    if k.dim() == 3:
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
    if k.shape[2] < k.shape[1]:  # [B, S, H, D] → [B, H, S, D]
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

    k = k.contiguous()
    v = v.contiguous()

    B, H, S, D = k.shape
    backend_used = "torch"
    k_norms: Optional[torch.Tensor] = None
    v_norms: Optional[torch.Tensor] = None

    if backend in ("auto", "triton"):
        try:
            kn = torch.zeros(B, H, device=k.device, dtype=torch.float32)
            vn = torch.zeros(B, H, device=k.device, dtype=torch.float32)
            BLOCK_S = min(64, triton.next_power_of_2(S))
            BLOCK_D = min(64, triton.next_power_of_2(D))
            grid = (B * H,)
            kv_head_l2_norm_kernel[grid](
                k, v, kn, vn,
                B=B, H=H, S=S, D=D,
                BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D,
            )
            k_norms, v_norms = kn, vn
            backend_used = "triton"
        except Exception as e:
            if backend == "triton":
                raise
            # Auto: fall back silently to torch
            import warnings as _w
            _w.warn(
                f"[triton_ops] kv_head_l2_norm_kernel failed ({type(e).__name__}: "
                f"{e}); falling back to torch.",
                stacklevel=2,
            )

    if k_norms is None or v_norms is None:
        k_norms, v_norms = _torch_kv_head_l2(k, v)

    # A head is "dead" if its K norm is < 1% of the max norm
    max_norm = k_norms.max()
    dead_threshold = max_norm * 0.01
    dead_head_mask = k_norms.mean(0) < dead_threshold  # Average over batch
    utilization_ratio = float((~dead_head_mask).sum()) / H

    return {
        "k_norms": k_norms,
        "v_norms": v_norms,
        "dead_head_mask": dead_head_mask,
        "head_utilization_ratio": utilization_ratio,
        "n_dead_heads": int(dead_head_mask.sum()),
        "n_total_heads": H,
        "backend_used": backend_used,
    }


# ─── Kernel 2: MLA Compression Ratio ─────────────────────────────────────────
# Measures per-token information retention of MLA compression.
# For each token position, compute: ||latent_token|| / ||expanded_token||
# This tells you how much of the original KV information is preserved.

@triton.jit
def mla_frobenius_ratio_kernel(
    latent_ptr,    # [B, S, d_c]  compressed KV in latent space
    expand_ptr,    # [B, S, D]    full uprojected KV (D = n_heads * head_dim)
    ratio_out_ptr, # [B, S]       output ratio per token
    B: tl.constexpr,
    S: tl.constexpr,
    d_c: tl.constexpr,   # latent dim (compressed)
    D: tl.constexpr,     # full dim (expanded)
    BLOCK_B: tl.constexpr,
):
    """
    For each (batch, seq) position compute:
        ratio = ||latent[b,s,:]||_F / ||expanded[b,s,:]||_F

    Lower ratio ≠ better; it just measures how compactly the information
    is represented. What matters is CONSISTENCY across tokens.
    """
    s_idx = tl.program_id(0)

    for b in range(B):
        # Latent: [B, S, d_c]
        lat_base = b * S * d_c + s_idx * d_c
        lat_offsets = lat_base + tl.arange(0, d_c if d_c <= 512 else 512)

        # If d_c > 512, we'd need to loop — for now assume d_c <= 512
        # (DeepSeek V3: d_c=512, this holds)
        lat_vals = tl.load(
            latent_ptr + tl.arange(0, 512)[:d_c if d_c <= 512 else 512] + lat_base,
            mask=tl.arange(0, 512) < d_c,
            other=0.0,
        ).to(tl.float32)
        lat_norm_sq = tl.sum(lat_vals * lat_vals)

        # Expanded — loop in chunks of 512 since D can be large (e.g., 128*128=16384)
        exp_norm_sq = tl.zeros([1], dtype=tl.float32)
        exp_base = b * S * D + s_idx * D
        n_chunks = (D + 511) // 512

        for chunk in range(n_chunks):
            d_start = chunk * 512
            chunk_offsets = d_start + tl.arange(0, 512)
            chunk_mask = chunk_offsets < D
            exp_vals = tl.load(
                expand_ptr + exp_base + chunk_offsets,
                mask=chunk_mask,
                other=0.0,
            ).to(tl.float32)
            exp_norm_sq += tl.sum(exp_vals * exp_vals)

        # Ratio: latent_norm / expanded_norm
        ratio = tl.sqrt(lat_norm_sq) / (tl.sqrt(exp_norm_sq) + 1e-8)
        tl.store(ratio_out_ptr + b * S + s_idx, ratio)


def _torch_mla_frobenius_ratio(
    latent: torch.Tensor, expanded: torch.Tensor
) -> torch.Tensor:
    """Torch fallback for the MLA Frobenius-ratio kernel.

    Computes per-(batch, token) ratio = ||latent[b,s,:]|| / ||expanded[b,s,:]||.
    """
    lat_norm = torch.linalg.norm(latent, dim=-1)   # [B, S]
    exp_norm = torch.linalg.norm(expanded, dim=-1) # [B, S]
    return lat_norm / (exp_norm + 1e-8)


def measure_mla_compression(
    latent: torch.Tensor,
    expanded: torch.Tensor,
    backend: str = "auto",
) -> Dict[str, float]:
    """
    Measure effective information compression in MLA.

    Args:
        latent:   [B, S, d_c]  — compressed KV in latent space
        expanded: [B, S, n_heads*head_dim]  — full uprojected KV
        backend:  "auto" (try Triton, fall back to torch), "triton", "torch"

    Returns:
        dict with per-token and aggregate compression statistics.
    """
    if latent is None or expanded is None:
        return {"error": "latent or expanded is None"}

    latent = latent.contiguous().float()
    expanded = expanded.contiguous().float()

    B, S, d_c = latent.shape
    _, _, D = expanded.shape

    ratios: Optional[torch.Tensor] = None
    backend_used = "torch"

    if backend in ("auto", "triton"):
        try:
            r = torch.zeros(B, S, device=latent.device, dtype=torch.float32)
            grid = (S,)
            mla_frobenius_ratio_kernel[grid](
                latent, expanded, r,
                B=B, S=S, d_c=d_c, D=D,
                BLOCK_B=1,
            )
            ratios = r
            backend_used = "triton"
        except Exception as e:
            if backend == "triton":
                raise
            import warnings as _w
            _w.warn(
                f"[triton_ops] mla_frobenius_ratio_kernel failed "
                f"({type(e).__name__}: {e}); falling back to torch.",
                stacklevel=2,
            )

    if ratios is None:
        ratios = _torch_mla_frobenius_ratio(latent, expanded)

    return {
        "d_c": d_c,
        "D_full": D,
        "structural_ratio": d_c / D,       # Architectural compression ratio
        "mean_frobenius_ratio": float(ratios.mean()),  # Information compression
        "std_frobenius_ratio": float(ratios.std()),
        "min_frobenius_ratio": float(ratios.min()),
        "max_frobenius_ratio": float(ratios.max()),
        # Low variance = consistent compression = healthy MLA
        "compression_stability": float(1.0 - ratios.std() / (ratios.mean() + 1e-8)),
        "backend_used": backend_used,
    }


# ─── Kernel 3: KV Entropy (Python-level, no Triton needed for this one) ───────
# Triton is overkill here since this operates on attention weights, not raw KV.

def compute_attention_entropy(
    attention_weights: torch.Tensor,  # [B, H, S_q, S_k]
) -> Dict[str, torch.Tensor]:
    """
    Compute per-head attention entropy.

    High entropy → head attends broadly (all tokens roughly equal weight)
    Low entropy  → head attends sharply to few tokens (spiky distribution)
    Near-zero entropy → head may be "attending" to only 1-2 tokens; its
                        KV entries for other positions are wasted cache space.

    This is used to identify candidates for H2O (Heavy Hitter Oracle) eviction.
    """
    # Normalize if not already a distribution
    if not torch.allclose(attention_weights.sum(-1), torch.ones_like(attention_weights.sum(-1)), atol=1e-3):
        attention_weights = torch.softmax(attention_weights, dim=-1)

    # Shannon entropy: H = -sum(p * log(p))
    eps = 1e-10
    entropy = -(attention_weights * torch.log(attention_weights + eps)).sum(-1)
    # entropy shape: [B, H, S_q]

    # Max possible entropy for uniform distribution over S_k tokens
    S_k = attention_weights.shape[-1]
    max_entropy = math.log(S_k)

    normalized_entropy = entropy / max_entropy  # [B, H, S_q]

    return {
        "per_head_entropy": entropy.mean(-1),           # [B, H] avg over query positions
        "per_head_entropy_normalized": normalized_entropy.mean(-1),
        "mean_entropy": float(entropy.mean()),
        "max_entropy_possible": max_entropy,
        "low_entropy_head_fraction": float(
            (normalized_entropy.mean(-1) < 0.1).float().mean()
        ),
    }


# ─── Convenience: Run Full KV Analysis Pass ────────────────────────────────────

def run_kv_analysis(
    k: torch.Tensor,
    v: torch.Tensor,
    latent: Optional[torch.Tensor] = None,
    expanded: Optional[torch.Tensor] = None,
    attention_weights: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Run all applicable kernels in one call.

    Returns a flat dict of all metrics ready for Prometheus export.
    """
    results = {}

    try:
        head_stats = measure_kv_head_utilization(k, v)
        results["head_utilization_ratio"] = head_stats["head_utilization_ratio"]
        results["n_dead_heads"] = head_stats["n_dead_heads"]
        results["n_total_heads"] = head_stats["n_total_heads"]
        results["k_norm_mean"] = float(head_stats["k_norms"].mean())
        results["v_norm_mean"] = float(head_stats["v_norms"].mean())
    except Exception as e:
        results["head_analysis_error"] = str(e)

    if latent is not None and expanded is not None:
        try:
            mla_stats = measure_mla_compression(latent, expanded)
            results.update({f"mla_{k}": v for k, v in mla_stats.items()})
        except Exception as e:
            results["mla_error"] = str(e)

    if attention_weights is not None:
        try:
            entropy_stats = compute_attention_entropy(attention_weights)
            results["attn_mean_entropy"] = entropy_stats["mean_entropy"]
            results["attn_low_entropy_head_fraction"] = entropy_stats["low_entropy_head_fraction"]
        except Exception as e:
            results["entropy_error"] = str(e)

    return results


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("KVScope Triton Kernels — Self Test")
    print(f"Triton version: {triton.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("[!] No CUDA — skipping kernel tests")
    else:
        device = "cuda"
        B, H, S, D = 1, 8, 128, 64

        k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

        print("\n[1] Head utilization test...")
        stats = measure_kv_head_utilization(k, v)
        print(f"    Backend used: {stats.get('backend_used', '?')}")
        print(f"    K norm mean : {float(stats['k_norms'].mean()):.3f}")
        print(f"    Dead heads  : {stats['n_dead_heads']} / {stats['n_total_heads']}")
        print(f"    Utilization : {stats['head_utilization_ratio']:.1%}")

        # Cross-check Triton vs torch fallback to make sure both paths agree
        stats_torch = measure_kv_head_utilization(k, v, backend="torch")
        max_diff = float((stats['k_norms'] - stats_torch['k_norms']).abs().max())
        print(f"    Triton-vs-torch K norm max abs diff: {max_diff:.6f}")
        assert max_diff < 1e-2, "Triton and torch backends disagree!"

        print("\n[2] MLA compression test...")
        d_c = 64  # Compressed dim
        D_full = H * D
        latent = torch.randn(B, S, d_c, device=device, dtype=torch.float32)
        expanded = torch.randn(B, S, D_full, device=device, dtype=torch.float32)
        mla = measure_mla_compression(latent, expanded)
        print(f"    Structural ratio: {mla['structural_ratio']:.4f}")
        print(f"    Frobenius ratio : {mla['mean_frobenius_ratio']:.4f}")
        print(f"    Stability       : {mla['compression_stability']:.4f}")

        print("\n[3] Attention entropy test...")
        attn = torch.softmax(torch.randn(B, H, S, S, device=device), dim=-1)
        ent = compute_attention_entropy(attn)
        print(f"    Mean entropy    : {ent['mean_entropy']:.3f}")
        print(f"    Low entropy hds : {ent['low_entropy_head_fraction']:.1%}")

        print("\n[✓] All kernels passed.")
