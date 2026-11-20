"""Run the two CUDA trainers from the 17-second LSTM video."""
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "model-families"))
from prepare import prepare_sources

ORIGINAL = HERE / "original"
HASHES = json.loads((HERE / "evidence/sha256.json").read_text(encoding="utf-8"))


def check_hash(path, expected):
    with path.open("rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"{path} SHA-256 {actual}, expected {expected}")


@contextmanager
def worker(command, cwd, env, log):
    with log.open("w") as output:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT)
        try:
            yield process
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()


def ready(process, side):
    deadline = time.monotonic() + 900
    while not (side / "ready").exists():
        if process.poll() is not None or time.monotonic() >= deadline:
            raise RuntimeError(f"Trainer did not become ready. See {side}")
        time.sleep(0.05)


def metrics(side):
    rows = [json.loads(line) for line in (side / "live.jsonl").read_text().splitlines()]
    initial = json.loads((side / "ready").read_text())
    end = json.loads((side / "done").read_text())
    if ([r["step"] for r in rows] != list(range(0, 4097, 64)) or initial["parameters"] != 25482
        or not all(math.isfinite(r["loss"]) and 0 <= r["correct"] <= 320 for r in rows)
        or rows[0]["seconds"] != 0 or not all(a["seconds"] < b["seconds"] for a,b in zip(rows,rows[1:]))
        or end["correct"] != rows[-1]["correct"] or (side.name == "ai" and end["fused"] != 4104)):
        raise ValueError(f"Incomplete or inconsistent measurements in {side}")
    return dict(initial=initial, **rows[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=HERE / ".work")
    parser.add_argument("--record", action="store_true", help="Capture the dashboards and render an MP4 with the original recorder.")
    args = parser.parse_args()
    import torch
    import torch._dynamo.config as dynamo
    if not torch.cuda.is_available():
        parser.error("This workload requires an NVIDIA CUDA GPU and CUDA-enabled PyTorch.")
    if dynamo.disable or dynamo.suppress_errors or os.environ.get("TORCH_COMPILE_DISABLE") == "1":
        parser.error("Enable torch.compile and disable silent compilation-error fallback.")
    if args.record:
        from imageio_ffmpeg import get_ffmpeg_exe
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            pw.chromium.launch().close()
    for name, expected in HASHES["original_files"].items():
        check_hash(ORIGINAL / name, expected)
    sources = HERE / "sources"
    work = args.work.resolve()
    tree = prepare_sources(work / "sources", json.loads((sources / "revisions.json").read_text()), sources)
    runs = work / "runs"
    runs.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="take-", dir=runs))
    print(f"Run directory: {output}", flush=True)
    # Preserve the original recorder's relative directory layout inside this new take.
    pair = output / "reddit/working/paired"
    pair.mkdir(parents=True)
    for name in HASHES["original_files"]:
        shutil.copyfile(ORIGINAL / name, pair / name)
    project = pair / "Pair.csproj"
    xml = ET.parse(project)
    dashboard = tree / "AiDotNet/src/AiDotNet.Dashboard/AiDotNet.Dashboard.csproj"
    xml.find(".//ProjectReference").set("Include", str(dashboard))
    xml.write(project, encoding="utf-8")
    tensors = tree / "AiDotNet.Tensors"
    build = ["dotnet", "build", str(project), "-c", "Release", "-p:UseLocalTensors=true",
             f"-p:LocalTensorsRoot={tensors}", "-p:GeneratePackageOnBuild=false", "-p:UseSharedCompilation=false", "-m:1"]
    with (output / "build.log").open("w") as log:
        if subprocess.run(build, stdout=log, stderr=subprocess.STDOUT).returncode:
            sys.exit(f"Build failed. See {output / 'build.log'}")
    dll = pair / "bin/Release/net10.0/AiDotNet.PyTorchParity.dll"
    tensor_dll = "AiDotNet.Tensors.dll"
    check_hash(dll.with_name(tensor_dll), hashlib.sha256((tensors / "src/AiDotNet.Tensors/bin/Release/net10.0" / tensor_dll).read_bytes()).hexdigest())
    weights = output / "reference/lstm.weights.safetensors"
    weights.parent.mkdir()
    shutil.copyfile(HERE / weights.name, weights)
    fixture = pair / "fixture"
    subprocess.run([sys.executable, str(pair / "fixture.py")], check=True)
    for name, expected in HASHES["inputs"].items():
        check_hash(weights if name == weights.name else fixture / name, expected)

    take = output / "measurements"
    ai, pt = take / "ai", take / "torch"
    for d in (ai, pt):
        d.mkdir(parents=True)
    (pair / "current.txt").write_text(str(take), encoding="utf-8")
    env = dict(os.environ, AIDOTNET_CPU_THREADS="8", AIDOTNET_BLAS_THREADS="8", AIDOTNET_FUSED_DIAG="1")
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(work / "inductor-cache"))
    env.setdefault("TRITON_CACHE_DIR", str(work / "triton-cache"))
    py_models = tree / "AiDotNet/benchmarks/AiDotNet.PyTorchParity/pytorch"
    # Preload the pinned model before the historical trainer adds its author's checkout to sys.path.
    bootstrap = "import sys,os,runpy; sys.path.insert(0,os.getcwd()); import benchmark; sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
    py_cmd = [sys.executable, "-c", bootstrap, str(pair / "torch_train.py"), str(pt), str(fixture), str(weights)]
    ai_cmd = ["dotnet", str(dll), "train", str(ai), str(fixture), str(weights)]
    with worker(py_cmd, py_models, env, pt / "train.log") as trainer:
        ready(trainer, pt)
        if args.record:
            ff = pair.parent / ".media-tools/imageio_ffmpeg/binaries/ffmpeg-host.exe"
            ff.parent.mkdir(parents=True)
            shutil.copy2(get_ffmpeg_exe(), ff)
            subprocess.run([sys.executable, str(pair / "record_pair.py")], cwd=pair, env=env, check=True)
        else:
            (pt / "go").touch()
        if trainer.wait(timeout=900) != 0:
            sys.exit(f"PyTorch failed. See {pt / 'train.log'}")
    if not args.record:
        with worker(ai_cmd, pair, env, ai / "train.log") as trainer:
            ready(trainer, ai)
            (ai / "go").touch()
            if trainer.wait(timeout=900) != 0:
                sys.exit(f"AiDotNet failed. See {ai / 'train.log'}")
    result = {name: metrics(d) for name, d in [("AiDotNet", ai), ("PyTorch", pt)]}
    a, b = (v["initial"] for v in result.values())
    if a["initial"] != b["initial"] or abs(a["firstLoss"] - b["firstLoss"]) >= .002:
        raise ValueError("Starting predictions or first loss differ between frameworks")
    result["torch_over_aidotnet"] = result["PyTorch"]["seconds"] / result["AiDotNet"]["seconds"]
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name in ("AiDotNet", "PyTorch"):
        print(f"{name}: {result[name]['seconds']:.3f} s, {result[name]['correct']}/320 correct")
    print(f"PyTorch time / AiDotNet time: {result['torch_over_aidotnet']:.2f}x")
    print(output / "summary.json")
    if args.record:
        print(output / "reddit/cuda-lstm-side-by-side-review.mp4")


if __name__ == "__main__":
    main()
