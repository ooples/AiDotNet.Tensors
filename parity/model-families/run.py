"""Run the 11-family AiDotNet vs Python torch.compile benchmark from pinned source."""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from prepare import DEFAULT_WORK, REVISIONS, prepare
from report import BATCHES, MODELS, summarize


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    with path.open("rb") as reader:
        return hashlib.file_digest(reader, "sha256").hexdigest()


def run_logged(cmd, root, env, log):
    print(log.stem, flush=True)
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.run(cmd, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
    return process.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", choices=REVISIONS, default="2026-11-19")
    parser.add_argument("--devices", nargs="+", choices=["cuda", "cpu"], default=["cuda", "cpu"])
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--launches", type=int, help="Fresh process pairs per model/device (default 2, smoke 1)")
    parser.add_argument("--smoke", action="store_true", help="Check execution with shorter training and inference")
    parser.add_argument("--verify-step1", action="store_true", help="Also compare the four supported models to PyTorch's first update")
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK, help="Dedicated source, build, cache and results directory")
    args = parser.parse_args()
    launches = args.launches if args.launches is not None else (1 if args.smoke else 2)
    if launches < 1 or len(set(args.devices)) != len(args.devices) or len(set(args.models)) != len(args.models):
        parser.error("Choose positive launches and unique models/devices.")
    import torch
    import torch._dynamo.config as dynamo
    import torchvision

    if dynamo.disable or dynamo.suppress_errors or os.environ.get("TORCH_COMPILE_DISABLE") == "1":
        parser.error("Enable torch.compile and disable silent compiler-error fallback for this benchmark.")
    if "cuda" in args.devices and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable in Python PyTorch. Install a CUDA wheel or select --devices cpu.")
    if os.name == "nt" and not shutil.which("cl"):
        parser.error("Open Developer PowerShell for Visual Studio (x64 C++ tools) so torch.compile can find cl.exe.")

    root = prepare(args.work, args.snapshot)
    output = root / "runs"
    output.mkdir(exist_ok=True)
    dest = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-"), dir=output))
    print(f"Results: {dest}", flush=True)
    project = root / "AiDotNet/benchmarks/AiDotNet.PyTorchParity"
    tensors = root / "AiDotNet.Tensors"
    py = project / "pytorch"
    build_dir = project / "bin/Release/net10.0"
    dll = build_dir / "AiDotNet.PyTorchParity.dll"
    env = os.environ.copy()
    env["AIDOTNET_CPU_THREADS"] = "8"
    if args.snapshot == "2026-11-19":
        for name in ["AIDOTNET_BLAS_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
            env[name] = "8"
        env["AIDOTNET_FUSED_DIAG"] = "1"
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(root / "inductor-cache"))
    env.setdefault("TRITON_CACHE_DIR", str(root / "triton-cache"))
    cfg = dict(snapshot=args.snapshot, smoke=args.smoke, models=args.models, devices=args.devices, launches=launches,
               epochs=2 if args.smoke else 3, train_batches=2 if args.smoke else 20, batch_size=64,
               inference_iterations=3 if args.smoke else 100, warmup_iterations=1 if args.smoke else 10,
               inference_batch_sizes=BATCHES, seed=1234, threads=8, receipts=True)
    write_json(dest / "protocol.json", cfg)
    hardware = dict(os=platform.platform(), cpu=platform.processor(), python=sys.version, torch=torch.__version__,
                    torchvision=torchvision.__version__, torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                    gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                    dotnet_sdk=subprocess.check_output(["dotnet", "--version"], text=True).strip(),
                    sources={name: pin["commit"] for name, pin in REVISIONS[args.snapshot].items()})
    if shutil.which("nvidia-smi"):
        smi = subprocess.run(["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv"],
                             capture_output=True, text=True)
        hardware["nvidia_smi"] = smi.stdout.strip()
    write_json(dest / "environment.json", hardware)
    build = ["dotnet", "build", str(project / "AiDotNet.PyTorchParity.csproj"), "-c", "Release",
             "-p:UseLocalTensors=true", f"-p:LocalTensorsRoot={tensors}", "-p:GeneratePackageOnBuild=false",
             "-p:UseSharedCompilation=false", "-m:1"]
    if run_logged(build, project, env, dest / "build.log"):
        sys.exit(f"Build failed. See {dest / 'build.log'}")
    local_tensor_dll = tensors / "src/AiDotNet.Tensors/bin/Release/net10.0/AiDotNet.Tensors.dll"
    hashes = {name: sha256(build_dir / name) for name in ["AiDotNet.Tensors.dll","AiDotNet.dll",dll.name]}
    if hashes["AiDotNet.Tensors.dll"] != sha256(local_tensor_dll):
        sys.exit("The benchmark did not load the local Tensors build output.")
    write_json(dest / "assemblies.json", hashes)
    if args.verify_step1:
        models = "mlp,cnn,lstm,transformer"
        ref = dest / "reference"
        command = [sys.executable, str(py / "export_reference.py"), "--models", models, "--out", str(ref)]
        if run_logged(command, py, env, dest / "export-reference.log"):
            sys.exit(f"Reference export failed. See {dest}")
        for device in args.devices:
            cmd = ["dotnet", str(dll), "--verify-step1", str(ref), "--models", models, "--device", device]
            log = dest / f"verify-{device}.log"
            if run_logged(cmd, project, dict(env, AIDOTNET_DIRECTGPU_BACKENDS="cuda") if device == "cuda" else env, log):
                sys.exit(f"Step-1 check failed for {device}. See {dest}")
            verdicts = log.read_bytes().decode("utf-8", errors="replace").splitlines()
            if not all(any(line.startswith(f"[step1] {model}: PASS ") for line in verdicts) for model in models.split(",")):
                sys.exit(f"Missing step-1 verdicts for {device}. See {log}")

    common = ["--epochs", str(cfg["epochs"]), "--train-batches", str(cfg["train_batches"]), "--batch-size", "64",
              "--inference-iterations", str(cfg["inference_iterations"]), "--warmup-iterations", str(cfg["warmup_iterations"]),
              "--inference-batches", "1,8,16", "--seed", "1234"]
    failures = []
    for launch in range(1, launches + 1):
        for device in args.devices:
            for model in args.models:
                for side, cmd, cwd in [("aidotnet", ["dotnet", str(dll)], project),
                                       ("torch-compile", [sys.executable, str(py / "benchmark.py"), "--compile", "--threads", "8"], py)]:
                    tag = f"{side}-{launch}-{device}-{model}"
                    command = cmd + ["--models", model, "--device", device, *common, "--output", str(dest / (tag + ".json"))]
                    started = datetime.now(timezone.utc).isoformat()
                    # A CUDA run of AiDotNet may use no other GPU backend: the harness reports every direct-GPU engine
                    # as cuda, so an inherited backend order (opencl first) would be compared as CUDA. With cuda as the
                    # only candidate, a CUDA failure leaves no GPU engine and the process fails instead.
                    run_env = dict(env, AIDOTNET_DIRECTGPU_BACKENDS="cuda") if side == "aidotnet" and device == "cuda" else env
                    code = run_logged(command, cwd, run_env, dest / (tag + ".log"))
                    write_json(dest / (tag + ".receipt.json"), dict(command=command, startedUtc=started,
                               finishedUtc=datetime.now(timezone.utc).isoformat(), exitCode=code))
                    if code:
                        failures.append(tag)
    if failures:
        sys.exit(f"Failed processes in {dest}: {', '.join(failures)}")
    with (dest / "summary.csv.tmp").open("w", encoding="utf-8", newline="") as output:
        summarize(dest, output)
    (dest / "summary.csv.tmp").replace(dest / "summary.csv")
    print(dest / "summary.csv")


if __name__ == "__main__":
    main()
