#!/usr/bin/env python3
"""Atomic environment capture for reproducibility & supply-chain audit."""

import hashlib, json, os, platform, re, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

def run(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except Exception as e:
        return -1, "", str(e)

def sh(cmd):
    _, out, _ = run(cmd.split(), timeout=30)
    return out.strip()

def sh_json(cmd, timeout=30):
    rc, out, _ = run(cmd, timeout=timeout)
    if rc != 0:
        return {"_error": out[:500]}
    try:
        return json.loads(out)
    except Exception as e:
        return {"_error": str(e), "_raw": out[:2000]}

def sha256_file(p):
    if not p.exists(): return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def pip_deep(pkg):
    rc, out, _ = run([sys.executable, "-m", "pip", "show", "-f", pkg], timeout=30)
    if rc != 0: return {"installed": False}
    meta = {}
    for line in out.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip().lower()] = v.strip()
    return {"installed": True, "version": meta.get("version"), "location": meta.get("location")}

def torch_info():
    try:
        import torch
        return {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "nccl": torch.cuda.nccl.version() if hasattr(torch.cuda.nccl, "version") else None,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
        }
    except Exception as e:
        return {"_error": str(e)}

def main():
    results_dir = Path("results")
    run_dirs = sorted(results_dir.glob("full_run_*"), key=lambda p: p.stat().st_mtime)
    out_dir = run_dirs[-1] if run_dirs else results_dir

    report = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "hostname": sh("hostname"),
        "user": sh("whoami"),
        "uname": platform.uname()._asdict(),
        "python": {"executable": sys.executable, "version": sys.version, "path": sys.path[:10]},
        "cuda": {"nvcc": sh("nvcc --version") if run(["nvcc", "--version"])[0] == 0 else None,
                 "nvidia_smi": sh_json(["nvidia-smi", "-q", "-x"])},
        "cpu": {"lscpu": sh("lscpu")},
        "memory": sh("cat /proc/meminfo")[:2000],
        "disk": sh("lsblk -d -o NAME,MODEL,SIZE,TYPE,ROTA") + "\n" + sh("df -Th"),
        "os": {"release": sh("cat /etc/os-release")},
        "torch": torch_info(),
        "git": {"sha": sh("git rev-parse HEAD"), "branch": sh("git rev-parse --abbrev-ref HEAD"),
                "status": sh("git status --short"), "remotes": sh("git remote -v")},
        "env": {k: os.environ.get(k) for k in ["PATH", "LD_LIBRARY_PATH", "CUDA_HOME", "VIRTUAL_ENV", "PYTHONPATH", "HF_TOKEN", "TRANSFORMERS_CACHE"]},
        "pip_all": sh("pip freeze --all").splitlines(),
        "pip_key_packages": {p: pip_deep(p) for p in [
            "torch", "transformers", "accelerate", "bitsandbytes", "triton", "numpy", "scipy", "sentencepiece", "protobuf", "datasets", "matplotlib", "seaborn",
        ]},
        "file_hashes": {
            "requirements.txt": sha256_file(Path("requirements.txt")),
            "setup.py": sha256_file(Path("setup.py")),
            "pyproject.toml": sha256_file(Path("pyproject.toml")),
        },
        "nvidia_processes": sh_json(["nvidia-smi", "pmon", "-s", "um", "-c", "1", "-f", "json"]),
    }

    out_path = out_dir / "environment_atomic.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Wrote atomic env report to {out_path}")

if __name__ == "__main__":
    main()
