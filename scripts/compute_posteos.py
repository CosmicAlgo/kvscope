import json
from pathlib import Path

R = Path("results/full_run_20260427T050945Z")

print(f"{'model':<14} {'prompt':>3} {'baseline':>8} {'overhead':>8} {'unrel':>8} {'score':>6}")
for fn in ["mha_baseline", "gemma4", "glm51", "gptoss"]:
    d = json.loads((R / f"{fn}_profile.json").read_text())
    for pp in d["per_prompt"]:
        m = pp.get("memory", {})
        b = m.get("baseline_mb", 0.0)
        peak = m.get("peak_mb", b)
        post = m.get("post_eos_mb", b)
        overhead = peak - b
        unrel = max(0.0, post - b)
        score = unrel / overhead if overhead > 1e-3 else 0.0
        score = max(0.0, min(1.0, score))
        print(f"{fn:<14} {pp['prompt_idx']:>3} {b:>8.0f} {overhead:>8.0f} {unrel:>8.0f} {score:>6.3f}")
