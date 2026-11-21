"""Validate the paired reports and produce per-launch training and inference measurements."""
import argparse
import csv
import json
import math
import sys
import zipfile
from pathlib import Path

MODELS = "mlp lstm transformer cnn resnet18 resnet50 vgg11 vit gru mobilenetv2 densenet121".split()
SIDES = ("aidotnet", "torch-compile")
BATCHES = [1, 8, 16]


def read_json(root, filename):
    return json.loads((root / filename).read_bytes().decode("utf-8-sig"))


def checked_metrics(report, side, model, device, protocol):
    ai = side == "aidotnet"
    def field(obj, cs, py):
        return obj[cs if ai else py]

    if field(report, "Framework", "framework") != ("AiDotNet" if ai else "PyTorch"):
        raise ValueError("wrong framework")
    rows = field(report, "Results", "results")
    if len(rows) != 1:
        raise ValueError("expected exactly one result per process")
    row = rows[0]
    if (field(row, "Model", "model"), field(row, "Device", "device")) != (model, device):
        raise ValueError("model/device mismatch")
    if ai:
        engine = report["Environment"]["EngineDevice"]
        if engine != {"cuda": "DirectGpuTensorEngine", "cpu": "CpuEngine"}[device]:
            raise ValueError(f"unexpected engine {engine}")
    elif report["device"] != device or (device == "cuda" and not report["cuda_available"]):
        raise ValueError("PyTorch device mismatch")
    threads = report["Environment"]["CpuThreads"] if ai else report["torch_num_threads"]
    if threads != protocol["threads"]:
        raise ValueError(f"unexpected thread count {threads}")
    # The Nov 13 harness predates the Protocol field. Raw epoch and batch data still exist.
    if protocol["snapshot"] != "2026-11-13":
        config = field(report, "Protocol", "protocol")
        for cs, py in [("BatchSize","batch_size"), ("Epochs","epochs"), ("TrainBatches","train_batches"),
                       ("InferenceIterations","inference_iterations"), ("WarmupIterations","warmup_iterations"),
                       ("Seed","seed"), ("InferenceBatchSizes","inference_batch_sizes")]:
            if field(config, cs, py) != protocol[py]:
                raise ValueError(f"protocol mismatch for {py}")
        if not ai and config["compile"] is not True:
            raise ValueError("PyTorch report did not request torch.compile")
    train = field(row, "Training", "training")
    steps = field(train, "StepsPerEpoch", "steps_per_epoch")
    epochs = field(train, "EpochSeconds", "epoch_seconds")
    if steps != protocol["train_batches"] or len(epochs) != protocol["epochs"]:
        raise ValueError("training schedule mismatch")
    for number in [*epochs, field(train, "FirstLoss", "first_loss"), field(train, "LastLoss", "last_loss")]:
        if not math.isfinite(number):
            raise ValueError("nonfinite training measurement")
    timings = [1000 * field(train, "SteadyStateEpochSecondsMedian", "steady_state_epoch_seconds_median") / steps]
    inference = field(row, "Inference", "inference")
    if [field(v, "BatchSize", "batch_size") for v in inference] != BATCHES:
        raise ValueError("inference batch sizes mismatch")
    timings.extend(field(v, "SteadyStateLatencyMsMedian", "steady_state_latency_ms_median") for v in inference)
    if not all(math.isfinite(t) and t > 0 for t in timings):
        raise ValueError("nonpositive or missing timing")
    return field(row, "Parameters", "parameters"), timings


def summarize(root, output):
    cfg = read_json(root, "protocol.json")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["model","device","launch","operation","batch_size","aidotnet_parameters","torch_parameters",
                     "aidotnet_ms","torch_compile_ms","torch_over_aidotnet"])
    for model in cfg["models"]:
        for device in cfg["devices"]:
            for launch in range(1, cfg["launches"] + 1):
                results = []
                for side in SIDES:
                    tag = f"{side}-{launch}-{device}-{model}"
                    try:
                        if cfg["receipts"] and read_json(root, tag + ".receipt.json")["exitCode"] != 0:
                            raise ValueError("benchmark process failed")
                        results.append(checked_metrics(read_json(root, tag + ".json"), side, model, device, cfg))
                    except (ValueError, KeyError, OSError, TypeError) as exc:
                        raise ValueError(f"{tag}: {exc}") from exc
                (a_count, ours), (t_count, theirs) = results
                for i, (a, t) in enumerate(zip(ours, theirs)):
                    writer.writerow([model,device,launch,"inference" if i else "train",
                                     BATCHES[i-1] if i else cfg["batch_size"],a_count,t_count,
                                     f"{a:.6f}",f"{t:.6f}",f"{t / a:.6f}"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, help="Run directory or historical evidence ZIP")
    args = parser.parse_args()
    # Write LF line endings on every platform, as the archived CSVs do; Windows would otherwise translate to CRLF.
    sys.stdout.reconfigure(newline="\n")
    if args.reports.is_dir():
        summarize(args.reports, sys.stdout)
    else:
        with zipfile.ZipFile(args.reports) as archive:
            summarize(zipfile.Path(archive), sys.stdout)
