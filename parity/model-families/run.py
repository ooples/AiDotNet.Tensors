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

from power import on_external_power, require_external_power
from prepare import DEFAULT_WORK, REVISIONS, prepare
from report import BATCHES, MODELS, summarize

# The archived snapshots predate run_matrix.py, so this launcher reproduces their exact protocol. A snapshot whose
# AiDotNet revision ships run_matrix.py is run by that file instead, so there is one implementation of its protocol.
HISTORICAL = ("2026-11-13", "2026-11-19")


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


def without_platform(env):
    """The Visual Studio developer prompt sets Platform=x64, which moves dotnet build output to bin/x64/Release."""
    return {k: v for k, v in env.items() if k.upper() != "PLATFORM"}


def delegate(args, root):
    """Run a snapshot that ships run_matrix.py through that pinned file. A snapshot that pins a Tensors commit builds
    against that source; one that pins only AiDotNet uses the Tensors NuGet package that AiDotNet commit references."""
    runner = root / "AiDotNet/benchmarks/AiDotNet.PyTorchParity/run_matrix.py"
    output = root / "runs"
    output.mkdir(exist_ok=True)
    command = [sys.executable, str(runner), "--output", str(output), "--devices", *args.devices, "--models", *args.models]
    if "AiDotNet.Tensors" in REVISIONS[args.snapshot]:
        command += ["--tensors", str(root / "AiDotNet.Tensors")]
    if args.launches is not None:
        command += ["--launches", str(args.launches)]
    if args.smoke:
        command.append("--smoke")
    if args.allow_battery:
        command.append("--allow-battery")
    if args.scale:
        # Large-data mode: whole epochs over real and synthetic datasets, built by the pinned prepare_data.py.
        command += ["--scale", str(args.scale.resolve())]
        if args.scale_torch_from:
            command += ["--scale-torch-from", str(args.scale_torch_from.resolve())]
    if args.gpu_clock_lock:
        command += ["--gpu-clock-lock", str(args.gpu_clock_lock)]
    env = os.environ.copy()
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(root / "inductor-cache"))
    env.setdefault("TRITON_CACHE_DIR", str(root / "triton-cache"))
    sys.exit(subprocess.run(command, env=env).returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", choices=REVISIONS, default=max(REVISIONS))
    parser.add_argument("--devices", nargs="+", choices=["cuda", "cpu"], default=["cuda", "cpu"])
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--launches", type=int, help="Fresh process sets per model/device (historical default 2, current 5, smoke 1)")
    parser.add_argument("--smoke", action="store_true", help="Check execution with shorter training and inference")
    parser.add_argument("--verify-step1", action="store_true", help="Historical snapshots: also compare the four supported models "
                        "to PyTorch's first update (current snapshots always check it)")
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK, help="Dedicated source, build, cache and results directory")
    parser.add_argument("--allow-battery", action="store_true", help="Run even when the machine is on battery power")
    parser.add_argument("--scale", type=Path, metavar="DATA_ROOT",
                        help="Large-data comparison: whole epochs over the real and 1,000,000-sample synthetic datasets in "
                             "DATA_ROOT, built first by the snapshot's scale/prepare_data.py when missing")
    parser.add_argument("--scale-torch-from", type=Path, metavar="SUMMARY_CSV",
                        help="With --scale, run only the fastest PyTorch configuration per training row of this summary.csv")
    parser.add_argument("--gpu-clock-lock", type=int, metavar="MHZ",
                        help="Declare that the GPU's SM clock is locked at MHZ (nvidia-smi -lgc MHZ,MHZ as administrator); "
                             "the pinned runner refuses to start unless the GPU sustains it")
    args = parser.parse_args()
    if args.launches is not None and args.launches < 1 or len(set(args.devices)) != len(args.devices) or len(set(args.models)) != len(args.models):
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
    require_external_power(args.allow_battery)

    root = prepare(args.work, args.snapshot)
    runner = root / "AiDotNet/benchmarks/AiDotNet.PyTorchParity/run_matrix.py"
    if args.gpu_clock_lock and (args.snapshot in HISTORICAL or "--gpu-clock-lock" not in runner.read_text(encoding="utf-8")):
        parser.error(f"Snapshot {args.snapshot}'s runner has no --gpu-clock-lock; choose a later --snapshot")
    if args.scale:
        builder = root / "AiDotNet/benchmarks/AiDotNet.PyTorchParity/scale/prepare_data.py"
        if args.snapshot in HISTORICAL or not builder.is_file():
            parser.error(f"Snapshot {args.snapshot} has no large-data mode; choose a later --snapshot")
        # Idempotent: datasets whose meta.json exists are kept, so a second run reuses the verified files.
        missing = [n for n in ("emnist-digits", "emnist-seq32", "cinic10", "synth-mnist", "synth-seq32", "synth-rgb32")
                   if not (args.scale / n / "meta.json").is_file()]
        if missing and subprocess.run([sys.executable, str(builder), "--root", str(args.scale), "--datasets",
                                       ",".join(missing)]).returncode:
            sys.exit("Building the large-data datasets failed")
    if args.snapshot not in HISTORICAL:
        delegate(args, root)
    launches = args.launches if args.launches is not None else (1 if args.smoke else 2)
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
    # A CUDA row must not fall back to another GPU backend (OpenCL, HIP) if CUDA fails to initialise. The historical
    # harness reports every direct-GPU engine as "cuda" and records no backend name, so this pin is the guarantee.
    env["AIDOTNET_DIRECTGPU_BACKENDS"] = "cuda"
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
    if run_logged(build, project, without_platform(env), dest / "build.log"):
        sys.exit(f"Build failed. See {dest / 'build.log'}")
    # Every assembly in the benchmark output must be the one its own project just built.
    local_tensor_dll = tensors / "src/AiDotNet.Tensors/bin/Release/net10.0/AiDotNet.Tensors.dll"
    local_library = root / "AiDotNet/src/bin/Release/net10.0/AiDotNet.dll"
    if not dll.is_file():
        sys.exit(f"The build did not write {dll}. See {dest / 'build.log'}")
    hashes = {name: sha256(build_dir / name) for name in ["AiDotNet.Tensors.dll","AiDotNet.dll",dll.name]}
    if hashes["AiDotNet.Tensors.dll"] != sha256(local_tensor_dll) or hashes["AiDotNet.dll"] != sha256(local_library):
        sys.exit("The benchmark did not load this build's AiDotNet and Tensors assemblies.")
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
            if run_logged(cmd, project, env, log):
                sys.exit(f"Step-1 check failed for {device}. See {dest}")
            verdicts = log.read_bytes().decode("utf-8", errors="replace").splitlines()
            if not all(any(line.startswith(f"[step1] {model}: PASS ") for line in verdicts) for model in models.split(",")):
                sys.exit(f"Missing step-1 verdicts for {device}. See {log}")

    common = ["--epochs", str(cfg["epochs"]), "--train-batches", str(cfg["train_batches"]), "--batch-size", "64",
              "--inference-iterations", str(cfg["inference_iterations"]), "--warmup-iterations", str(cfg["warmup_iterations"]),
              "--inference-batches", "1,8,16", "--seed", "1234"]
    sides = [("aidotnet", ["dotnet", str(dll)], project),
             ("torch-compile", [sys.executable, str(py / "benchmark.py"), "--compile", "--threads", "8"], py)]
    failures = []
    for launch in range(1, launches + 1):
        for device in args.devices:
            for model in args.models:
                # The archived runs always started with AiDotNet; alternate so neither framework always runs first.
                for side, cmd, cwd in (sides if launch % 2 else sides[::-1]):
                    tag = f"{side}-{launch}-{device}-{model}"
                    command = cmd + ["--models", model, "--device", device, *common, "--output", str(dest / (tag + ".json"))]
                    require_external_power(args.allow_battery, f"before {tag}")
                    started = datetime.now(timezone.utc).isoformat()
                    code = run_logged(command, cwd, env, dest / (tag + ".log"))
                    receipt = dict(command=command, startedUtc=started, finishedUtc=datetime.now(timezone.utc).isoformat(),
                                   exitCode=code, externalPowerAfter=on_external_power())
                    write_json(dest / (tag + ".receipt.json"), receipt)
                    try:
                        if code:
                            raise ValueError("process failed")
                        # The check before the next process cannot cover this one: its own receipt does.
                        if receipt["externalPowerAfter"] is False and not args.allow_battery:
                            raise ValueError("the machine was on battery power when it finished")
                    except ValueError as error:
                        failures.append(f"{tag}: {error}")
    if failures:
        sys.exit(f"Failed processes in {dest}:\n" + "\n".join(failures))
    with (dest / "summary.csv.tmp").open("w", encoding="utf-8", newline="") as output:
        summarize(dest, output)
    (dest / "summary.csv.tmp").replace(dest / "summary.csv")
    print(dest / "summary.csv")


if __name__ == "__main__":
    main()
