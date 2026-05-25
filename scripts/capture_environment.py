#!/usr/bin/env python3
"""
Comprehensive Environment Capture Script

Captures all system, hardware, software, and experiment state for
perfect reproducibility. Run this after completing experiments to
save the exact environment snapshot before destroying the VM.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


def run_command(cmd: str, shell: bool = False) -> str:
    """Run a shell command and return stdout."""
    try:
        result = subprocess.run(
            cmd,
            shell=shell,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.stdout.strip()
    except Exception as e:
        return f"<error: {e}>"


def get_system_info() -> Dict[str, Any]:
    """Capture OS, kernel, and hardware info."""
    info = {
        "os": run_command("uname -s"),
        "hostname": run_command("hostname"),
        "kernel": run_command("uname -r"),
        "architecture": run_command("uname -m"),
        "cpu_model": run_command("cat /proc/cpuinfo | grep 'model name' | head -1 | cut -d: -f2"),
        "cpu_cores": run_command("nproc"),
        "ram_gb": run_command("free -g | grep Mem | awk '{print $2}'"),
        "uptime": run_command("uptime -p"),
    }
    return info


def get_gpu_info() -> Dict[str, Any]:
    """Capture detailed GPU and CUDA information."""
    info = {
        "nvidia_smi": run_command("nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,memory.used,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits"),
        "cuda_version": run_command("nvcc --version | grep release"),
        "driver_version": run_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
        "gpu_count": run_command("nvidia-smi --list-gpus | wc -l"),
    }
    return info


def get_python_env() -> Dict[str, Any]:
    """Capture Python version and all package versions."""
    import sys
    try:
        from importlib.metadata import distributions
    except ImportError:
        # Python < 3.8 fallback
        import pkg_resources
        distributions = pkg_resources.working_set

    info = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "python_executable": sys.executable,
        "packages": {},
    }

    # Get all installed packages with versions
    try:
        for dist in distributions():
            info["packages"][dist.metadata["Name"]] = dist.version
    except:
        # Fallback to pip list
        import subprocess
        result = subprocess.run(["pip", "list", "--format=json"], capture_output=True, text=True)
        if result.returncode == 0:
            import json
            for pkg in json.loads(result.stdout):
                info["packages"][pkg["name"]] = pkg["version"]

    return info


def get_git_info(repo_path: str) -> Dict[str, Any]:
    """Capture git branch, commit, and status."""
    if not os.path.exists(os.path.join(repo_path, ".git")):
        return {"error": "Not a git repository"}

    os.chdir(repo_path)
    info = {
        "branch": run_command("git branch --show-current"),
        "commit": run_command("git rev-parse HEAD"),
        "commit_short": run_command("git rev-parse --short HEAD"),
        "remote_url": run_command("git remote get-url origin"),
        "status": run_command("git status --porcelain"),
        "uncommitted_changes": bool(run_command("git status --porcelain")),
    }
    return info


def get_model_info(models_dir: str) -> Dict[str, Any]:
    """Capture information about downloaded models."""
    info = {"models_dir": models_dir, "models": {}}

    if not os.path.exists(models_dir):
        return info

    for model_name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_name)
        if not os.path.isdir(model_path):
            continue

        model_info = {
            "path": model_path,
            "size_gb": 0,
            "file_count": 0,
            "config": {},
        }

        # Count files and total size
        total_size = 0
        file_count = 0
        for root, dirs, files in os.walk(model_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    total_size += os.path.getsize(file_path)
                    file_count += 1
                except:
                    pass

        model_info["size_gb"] = round(total_size / (1024**3), 2)
        model_info["file_count"] = file_count

        # Try to read config.json if it exists
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    model_info["config"] = {
                        "model_type": config.get("model_type"),
                        "hidden_size": config.get("hidden_size"),
                        "num_hidden_layers": config.get("num_hidden_layers"),
                        "num_attention_heads": config.get("num_attention_heads"),
                        "num_key_value_heads": config.get("num_key_value_heads"),
                        "vocab_size": config.get("vocab_size"),
                        "max_position_embeddings": config.get("max_position_embeddings"),
                    }
            except:
                pass

        info["models"][model_name] = model_info

    return info


def get_results_info(results_dir: str) -> Dict[str, Any]:
    """Capture information about experiment results."""
    info = {"results_dir": results_dir, "results": {}}

    if not os.path.exists(results_dir):
        return info

    for file in os.listdir(results_dir):
        if file.endswith(".json"):
            file_path = os.path.join(results_dir, file)
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    info["results"][file] = {
                        "size_kb": round(os.path.getsize(file_path) / 1024, 2),
                        "model_type": data.get("model_type", "unknown"),
                        "max_tokens": data.get("max_tokens", "unknown"),
                        "num_results": len(data.get("results", [])),
                        "has_snapshots": any("snapshots" in r for r in data.get("results", [])),
                    }
            except:
                info["results"][file] = {"error": "Failed to parse"}

    return info


def get_storage_info() -> Dict[str, Any]:
    """Capture disk and storage information."""
    info = {
        "df_h": run_command("df -h /"),
        "models_disk_usage": run_command("du -sh /root/models 2>/dev/null"),
        "results_disk_usage": run_command("du -sh ./results 2>/dev/null"),
    }
    return info


def get_experiment_config() -> Dict[str, Any]:
    """Capture experiment configuration from run_profiling.sh."""
    config = {}

    # Read run_profiling.sh to extract key variables
    script_path = "experiments/run_profiling.sh"
    if os.path.exists(script_path):
        with open(script_path, "r") as f:
            content = f.read()
            # Extract key environment variables
            for var in ["MODELS_DIR", "RESULTS_DIR", "MAX_TOKENS", "CAPTURE_EVERY"]:
                if f"{var}=" in content:
                    for line in content.split("\n"):
                        if f"{var}=" in line and not line.strip().startswith("#"):
                            value = line.split("=")[1].strip().strip('"').strip("'")
                            config[var] = value
                            break

    return config


def main():
    """Main entry point."""
    print("[*] Capturing comprehensive environment snapshot...")

    # Determine paths
    repo_root = Path(__file__).parent.parent
    models_dir = os.environ.get("MODELS_DIR", "/root/models")
    results_dir = os.environ.get("RESULTS_DIR", "./results")

    snapshot = {
        "timestamp": datetime.utcnow().isoformat(),
        "capture_script": str(__file__),
        "repo_root": str(repo_root),
        "system": get_system_info(),
        "gpu": get_gpu_info(),
        "python_env": get_python_env(),
        "git": get_git_info(str(repo_root)),
        "models": get_model_info(models_dir),
        "results": get_results_info(results_dir),
        "storage": get_storage_info(),
        "experiment_config": get_experiment_config(),
    }

    # Save snapshot
    output_path = "./results/environment_snapshot.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)

    print(f"[*] Environment snapshot saved to: {output_path}")
    print(f"[*] Snapshot size: {os.path.getsize(output_path) / 1024:.2f} KB")

    # Also save a human-readable summary
    summary_path = "./results/environment_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("ENVIRONMENT SNAPSHOT SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {snapshot['timestamp']}\n")
        f.write(f"Repo: {snapshot['git']['remote_url']}\n")
        f.write(f"Branch: {snapshot['git']['branch']}\n")
        f.write(f"Commit: {snapshot['git']['commit_short']}\n\n")
        f.write(f"Python: {snapshot['python_env']['python_version']}\n")
        f.write(f"Python Path: {snapshot['python_env']['python_executable']}\n\n")
        f.write(f"GPU: {snapshot['gpu']['nvidia_smi']}\n")
        f.write(f"CUDA: {snapshot['gpu']['cuda_version']}\n\n")
        f.write(f"Models Directory: {snapshot['models']['models_dir']}\n")
        for model_name, model_info in snapshot['models']['models'].items():
            f.write(f"  - {model_name}: {model_info['size_gb']} GB, {model_info['file_count']} files\n")
        f.write(f"\nResults Directory: {snapshot['results']['results_dir']}\n")
        for result_name, result_info in snapshot['results']['results'].items():
            size_kb = result_info.get('size_kb', 'N/A')
            f.write(f"  - {result_name}: {size_kb} KB\n")
        f.write(f"\nStorage:\n")
        f.write(f"  {snapshot['storage']['models_disk_usage']}\n")
        f.write(f"  {snapshot['storage']['results_disk_usage']}\n")

    print(f"[*] Human-readable summary saved to: {summary_path}")
    print("[*] Done. Download these files before destroying the VM.")


if __name__ == "__main__":
    main()
