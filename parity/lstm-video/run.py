"""Run the two CUDA trainers from the 17-second LSTM video."""
import argparse
import hashlib
import html
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "model-families"))
from power import require_external_power
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
    parser.add_argument("--repeats", type=int, default=1,
                        help="Run the pair this many times, alternating which framework goes first, and report the median")
    parser.add_argument("--torch-compile-mode", choices=["default", "reduce-overhead"], default="default",
                        help="torch.compile mode for the PyTorch trainer. The recording used default; reduce-overhead adds CUDA graphs "
                             "to whatever Dynamo compiles.")
    parser.add_argument("--torch-compile-rnn", action="store_true",
                        help="Let torch.compile trace the trainer's nn.LSTM (torch._dynamo.config.allow_rnn). Without it Dynamo "
                             "skips the model and PyTorch trains with eager cuDNN, as in the recording.")
    parser.add_argument("--allow-battery", action="store_true", help="Run even when the machine is on battery power")
    args = parser.parse_args()
    if args.repeats < 1 or (args.record and args.repeats != 1):
        parser.error("Choose a positive --repeats; --record captures a single take.")
    require_external_power(args.allow_battery)
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
    # The original recorder captions the clip with the GPU it was recorded on. The files under original/ stay
    # byte-identical, so only this take's staged copy is captioned with the GPU measured here.
    recorder = pair / "record_pair.py"
    caption = b"LIVE LSTM TRAINING  /  GTX 1660 Ti"
    source = recorder.read_bytes()
    if source.count(caption) != 1:
        sys.exit("The original recorder's GPU caption was not found; refusing to record with a wrong GPU name.")
    gpu_name = torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
    recorder.write_bytes(source.replace(caption, ("LIVE LSTM TRAINING  /  " + html.escape(gpu_name)).encode("utf-8")))
    project = pair / "Pair.csproj"
    xml = ET.parse(project)
    dashboard = tree / "AiDotNet/src/AiDotNet.Dashboard/AiDotNet.Dashboard.csproj"
    xml.find(".//ProjectReference").set("Include", str(dashboard))
    xml.write(project, encoding="utf-8")
    tensors = tree / "AiDotNet.Tensors"
    build = ["dotnet", "build", str(project), "-c", "Release", "-p:UseLocalTensors=true",
             f"-p:LocalTensorsRoot={tensors}", "-p:GeneratePackageOnBuild=false", "-p:UseSharedCompilation=false", "-m:1"]
    # The Visual Studio developer prompt sets Platform=x64, which moves the output to bin/x64/Release; the original
    # recorder reads bin/Release, so build without it.
    build_env = {k: v for k, v in os.environ.items() if k.upper() != "PLATFORM"}
    with (output / "build.log").open("w") as log:
        if subprocess.run(build, stdout=log, stderr=subprocess.STDOUT, env=build_env).returncode:
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

    env = dict(os.environ, AIDOTNET_CPU_THREADS="8", AIDOTNET_BLAS_THREADS="8", AIDOTNET_FUSED_DIAG="1")
    # Never let the "CUDA" trainer fall back to another GPU backend.
    env["AIDOTNET_DIRECTGPU_BACKENDS"] = "cuda"
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(work / "inductor-cache"))
    env.setdefault("TRITON_CACHE_DIR", str(work / "triton-cache"))
    py_models = tree / "AiDotNet/benchmarks/AiDotNet.PyTorchParity/pytorch"
    # Preload the pinned model before the historical trainer adds its author's checkout to sys.path. A non-default
    # compile mode and allow_rnn are applied here, so the original trainer still runs byte-for-byte. At exit the
    # bootstrap records how many graphs Dynamo compiled: zero means torch.compile ran the LSTM eagerly.
    bootstrap = ("import sys,os,runpy,functools,atexit,json; sys.path.insert(0,os.getcwd()); import benchmark, torch; "
                 "import torch._dynamo.config as dc; from torch._dynamo.utils import counters; "
                 "dc.allow_rnn=os.environ.get('PARITY_TORCH_ALLOW_RNN')=='1'; "
                 "mode=os.environ.get('PARITY_TORCH_COMPILE_MODE'); "
                 "torch.compile=functools.partial(torch.compile, mode=mode) if mode and mode!='default' else torch.compile; "
                 "sys.argv=sys.argv[1:]; out=os.path.join(sys.argv[1],'compiled_graphs.json'); "
                 "atexit.register(lambda: open(out,'w').write(json.dumps(dict(graphs=counters['stats']['unique_graphs'], "
                 "allow_rnn=dc.allow_rnn, mode=mode)))); "
                 "runpy.run_path(sys.argv[0],run_name='__main__')")
    env["PARITY_TORCH_COMPILE_MODE"] = args.torch_compile_mode
    env["PARITY_TORCH_ALLOW_RNN"] = "1" if args.torch_compile_rnn else "0"
    takes = []
    for repeat in range(1, args.repeats + 1):
        take = output / ("measurements" if args.repeats == 1 else f"measurements-{repeat}")
        ai, pt = take / "ai", take / "torch"
        for d in (ai, pt):
            d.mkdir(parents=True)
        (pair / "current.txt").write_text(str(take), encoding="utf-8")
        py_cmd = [sys.executable, "-c", bootstrap, str(pair / "torch_train.py"), str(pt), str(fixture), str(weights)]
        ai_cmd = ["dotnet", str(dll), "train", str(ai), str(fixture), str(weights)]
        if args.record:
            require_external_power(args.allow_battery, "before the PyTorch trainer")
            with worker(py_cmd, py_models, env, pt / "train.log") as trainer:
                ready(trainer, pt)
                ff = pair.parent / ".media-tools/imageio_ffmpeg/binaries/ffmpeg-host.exe"
                ff.parent.mkdir(parents=True)
                shutil.copy2(get_ffmpeg_exe(), ff)
                require_external_power(args.allow_battery, "before the recorder starts AiDotNet")
                subprocess.run([sys.executable, str(pair / "record_pair.py")], cwd=pair, env=env, check=True)
                if trainer.wait(timeout=900) != 0:
                    sys.exit(f"PyTorch failed. See {pt / 'train.log'}")
        else:
            # The recording ran PyTorch first; alternate the order between repeats.
            order = [("PyTorch", py_cmd, py_models, pt), ("AiDotNet", ai_cmd, pair, ai)]
            for name, command, cwd, side in (order if repeat % 2 else order[::-1]):
                require_external_power(args.allow_battery, f"before the {name} trainer")
                with worker(command, cwd, env, side / "train.log") as trainer:
                    ready(trainer, side)
                    (side / "go").touch()
                    if trainer.wait(timeout=900) != 0:
                        sys.exit(f"{name} failed. See {side / 'train.log'}")
        result = {name: metrics(d) for name, d in [("AiDotNet", ai), ("PyTorch", pt)]}
        a, b = (v["initial"] for v in result.values())
        if a["initial"] != b["initial"] or abs(a["firstLoss"] - b["firstLoss"]) >= .002:
            raise ValueError("Starting predictions or first loss differ between frameworks")
        result["torch_over_aidotnet"] = result["PyTorch"]["seconds"] / result["AiDotNet"]["seconds"]
        result["torchCompiledGraphs"] = json.loads((pt / "compiled_graphs.json").read_text())["graphs"]
        takes.append(result)
        print(f"Take {repeat}: AiDotNet {result['AiDotNet']['seconds']:.3f} s, PyTorch {result['PyTorch']['seconds']:.3f} s, "
              f"{result['torch_over_aidotnet']:.2f}x, PyTorch compiled {result['torchCompiledGraphs']} graph(s)", flush=True)
    seconds = {name: [t[name]["seconds"] for t in takes] for name in ("AiDotNet", "PyTorch")}
    ratios = [t["torch_over_aidotnet"] for t in takes]
    summary = dict(torchCompileMode=args.torch_compile_mode, torchCompileRnn=args.torch_compile_rnn, repeats=args.repeats,
                   takes=takes,
                   median_seconds={name: statistics.median(v) for name, v in seconds.items()},
                   torch_over_aidotnet_median=statistics.median(ratios),
                   torch_over_aidotnet_range=[min(ratios), max(ratios)])
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for name in ("AiDotNet", "PyTorch"):
        print(f"{name}: median {summary['median_seconds'][name]:.3f} s over {args.repeats} take(s), "
              f"{takes[-1][name]['correct']}/320 correct")
    print(f"PyTorch time / AiDotNet time: {summary['torch_over_aidotnet_median']:.2f}x "
          f"(range {min(ratios):.2f}-{max(ratios):.2f})")
    print(output / "summary.json")
    if args.record:
        print(output / "reddit/cuda-lstm-side-by-side-review.mp4")

if __name__ == "__main__":
    main()
