#!/usr/bin/env python3
"""Resident PyTorch CUDA eager/graph peers for issue #836 operation families.

Graph rows replay a captured sequence of 1000 logical operations and normalize
per operation, matching the .NET runner and avoiding one-node graph submission
latency in the GPU execution comparison.
"""

import argparse
import csv
import glob
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time

import torch
import torch.nn.functional as functional


WARMUPS = 30
SAMPLES = 101
DEVICE_LAUNCHES = 50
GRAPH_OPERATIONS_PER_REPLAY = 1000


def foreign_python_processes():
    """Return other Python processes without requiring a third-party package."""
    current_pid = os.getpid()
    conflicts = []
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/fo", "csv", "/nh"],
            check=True,
            capture_output=True,
            text=True,
        )
        for row in csv.reader(result.stdout.splitlines()):
            if len(row) < 2:
                continue
            name = row[0].lower()
            try:
                pid = int(row[1])
            except ValueError:
                continue
            if pid != current_pid and name in ("python.exe", "python3.exe", "pythonw.exe"):
                conflicts.append(f"pid={pid} {row[0]}")
        return conflicts

    proc_root = "/proc"
    if not os.path.isdir(proc_root):
        raise RuntimeError("Cannot enumerate OS processes for clean evidence.")
    for entry in os.scandir(proc_root):
        if not entry.name.isdigit() or int(entry.name) == current_pid:
            continue
        try:
            with open(os.path.join(entry.path, "comm"), encoding="utf-8") as stream:
                name = stream.read().strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if name.lower() in ("python", "python3", "pythonw"):
            conflicts.append(f"pid={entry.name} {name}")
    return conflicts


def require_no_foreign_compute(label):
    """Fail closed when another Python or CUDA compute process is present."""
    python_conflicts = foreign_python_processes()
    if python_conflicts:
        raise RuntimeError(
            f"[{label}] foreign Python workload detected: "
            + "; ".join(python_conflicts)
        )

    result = subprocess.run(
        ["nvidia-smi", "pmon", "-c", "1", "-s", "u"],
        check=True,
        capture_output=True,
        text=True,
    )
    gpu_conflicts = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 9:
            continue
        try:
            pid = int(fields[1])
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        process_type = fields[2]
        try:
            sm_percent = int(fields[3])
        except ValueError:
            sm_percent = 0
        if process_type == "C" or ("C" in process_type and sm_percent > 5):
            gpu_conflicts.append(
                f"pid={pid} {fields[-1]} type={process_type} sm={fields[3]}%"
            )
    if gpu_conflicts:
        raise RuntimeError(
            f"[{label}] foreign GPU workload detected: "
            + "; ".join(gpu_conflicts)
        )


def configure_windows_triton():
    """Give torch.compile's Triton backend an explicit Windows toolchain."""
    if os.name != "nt":
        return
    if "CC" not in os.environ:
        candidates = glob.glob(
            r"C:\Program Files (x86)\Microsoft Visual Studio\*\BuildTools\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"
        ) + glob.glob(
            r"C:\Program Files\Microsoft Visual Studio\*\Community\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"
        )
        if candidates:
            os.environ["CC"] = sorted(candidates)[-1]
    spec = importlib.util.find_spec("triton")
    if spec is None or not spec.submodule_search_locations:
        return
    triton_root = os.path.join(
        next(iter(spec.submodule_search_locations)), "backends", "nvidia"
    )
    cuda_import_library = os.path.join(triton_root, "lib", "x64", "cuda.lib")
    if not os.path.isfile(cuda_import_library):
        return
    os.environ.setdefault("CUDA_PATH", triton_root)
    os.environ.setdefault("CUDA_HOME", triton_root)
    current_lib = os.environ.get("LIB", "")
    os.environ["LIB"] = os.path.dirname(cuda_import_library) + (
        os.pathsep + current_lib if current_lib else ""
    )


configure_windows_triton()


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(values):
    return (
        statistics.fmean(values),
        percentile(values, 0.50),
        percentile(values, 0.95),
        percentile(values, 0.99),
    )


def divide_round_up(value, divisor):
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    return (value + divisor - 1) // divisor


def measure_device(operation, logical_operations_per_call=1):
    warmup_calls = divide_round_up(WARMUPS, logical_operations_per_call)
    calls_per_sample = divide_round_up(DEVICE_LAUNCHES, logical_operations_per_call)
    logical_operations_per_sample = calls_per_sample * logical_operations_per_call
    for _ in range(warmup_calls):
        operation()
    torch.cuda.synchronize()
    timings = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(SAMPLES):
        start.record()
        for _ in range(calls_per_sample):
            operation()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000.0 / logical_operations_per_sample)
    return summarize(timings)


def measure_e2e(operation, logical_operations_per_call=1):
    warmup_calls = divide_round_up(WARMUPS, logical_operations_per_call)
    for _ in range(warmup_calls):
        operation()
    torch.cuda.synchronize()
    timings = []
    for _ in range(SAMPLES):
        start = time.perf_counter_ns()
        operation()
        torch.cuda.synchronize()
        timings.append(
            (time.perf_counter_ns() - start) /
            (1000.0 * logical_operations_per_call))
    return summarize(timings)


def maximum_error(actual, expected):
    if isinstance(actual, tuple):
        return max(
            (part.double() - oracle).abs().max().item()
            for part, oracle in zip(actual, expected)
        )
    return (actual.double() - expected).abs().max().item()


def output_storage_bytes(value):
    """Count required result storage once, excluding aliases within tuples."""
    tensors = value if isinstance(value, (tuple, list)) else (value,)
    seen = set()
    total = 0
    for tensor in tensors:
        if not torch.is_tensor(tensor):
            continue
        storage = tensor.untyped_storage()
        identity = (tensor.device.type, tensor.device.index, storage.data_ptr())
        if identity in seen:
            continue
        seen.add(identity)
        total += storage.nbytes()
    return total


def capture_graph(operation, logical_operations_per_replay=GRAPH_OPERATIONS_PER_REPLAY):
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            graph_output = operation()
    torch.cuda.current_stream().wait_stream(capture_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(logical_operations_per_replay):
            graph_output = operation()

    def replay():
        graph.replay()
        return graph_output

    return replay, graph, capture_stream


def make_case(name, run, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(20263000 + run * 100 + sum(ord(c) for c in name))

    def values(shape, scale=0.125, dtype=torch.float32):
        return (torch.rand(shape, generator=generator, device=device, dtype=dtype) * 2 - 1) * scale

    if name == "decode-gelu":
        m, k, n = 1, 512, 2048
        x, w, bias = values((m, k)), values((k, n), 0.0625), values((n,), 0.0625)

        def op():
            return functional.gelu(x @ w + bias, approximate="tanh")

        expected = functional.gelu(x.double() @ w.double() + bias.double(), approximate="tanh")
        return "M1 K512 N2048", 2.0 * m * k * n, op, expected

    if name == "gemm-fp32":
        m, k, n = 64, 256, 256
        left, right = values((m, k)), values((k, n), 0.0625)

        def op():
            return left @ right

        return "M64 K256 N256", 2.0 * m * k * n, op, left.double() @ right.double()

    if name == "fused-gelu":
        m, k, n = 64, 256, 256
        x, w, bias = values((m, k)), values((k, n), 0.0625), values((n,), 0.0625)

        def op():
            return functional.gelu(x @ w + bias, approximate="tanh")

        expected = functional.gelu(x.double() @ w.double() + bias.double(), approximate="tanh")
        return "M64 K256 N256", 2.0 * m * k * n, op, expected

    if name == "batched-gemm":
        batch, m, k, n = 4, 64, 256, 256
        left, right = values((batch, m, k)), values((batch, k, n), 0.0625)

        def op():
            return torch.bmm(left, right)

        return "B4 M64 K256 N256", 2.0 * batch * m * k * n, op, torch.bmm(left.double(), right.double())

    if name == "gemm-fp16":
        m, k, n = 16, 32, 16
        left, right = values((m, k), dtype=torch.float16), values((k, n), 0.0625, torch.float16)

        def op():
            return (left @ right).float()

        return "M16 K32 N16", 2.0 * m * k * n, op, left.double() @ right.double()

    if name in ("fused-gelu-fp16-m16-k512", "fused-gelu-fp16-m16-k1024"):
        m = 16
        k, n = ((512, 2048) if name.endswith("k512") else (1024, 4096))
        x = values((m, k), dtype=torch.float16)
        w = values((n, k), 0.0625, torch.float16)
        bias = values((n,), 0.0625)

        def op():
            return functional.gelu(x @ w.t() + bias, approximate="tanh")

        expected = functional.gelu(
            x.double() @ w.double().t() + bias.double(), approximate="tanh")
        return f"M16 K{k} N{n}", 2.0 * m * k * n, op, expected

    if name == "lora":
        batch, input_features, rank, output_features = 8, 256, 8, 256
        scale = 0.125
        x = values((batch, input_features))
        base = values((batch, output_features))
        a = values((input_features, rank), 0.0625)
        b = values((rank, output_features), 0.0625)

        def op():
            return base + scale * ((x @ a) @ b)

        expected = base.double() + scale * ((x.double() @ a.double()) @ b.double())
        work = 2.0 * batch * input_features * rank + 2.0 * batch * rank * output_features
        return "B8 I256 R8 O256", work, op, expected

    if name == "linear-ce-index":
        rows, hidden, vocab = 4, 16, 32
        x, w, bias = values((rows, hidden)), values((hidden, vocab), 0.0625), values((vocab,), 0.03125)
        target = torch.tensor([1, 7, 15, 31], device=device, dtype=torch.long)

        def op():
            return functional.cross_entropy(x @ w + bias, target)

        expected = functional.cross_entropy(x.double() @ w.double() + bias.double(), target)
        return "B4 K16 V32", 4.0 * rows * hidden * vocab, op, expected

    if name == "linear-backward-relu":
        m, k, n = 64, 256, 256
        grad = values((m, n))
        x = values((m, k))
        w = values((k, n), 0.0625)
        saved = values((m, n), 0.25)

        def op():
            masked = grad * (saved > 0)
            return masked @ w.t(), x.t() @ masked, masked.sum(dim=0)

        masked64 = grad.double() * (saved > 0)
        expected = (masked64 @ w.double().t(), x.double().t() @ masked64, masked64.sum(dim=0))
        work = 2.0 * m * n * (k + k) + m * n
        return "M64 K256 N256", work, op, expected

    if name == "dot":
        length = 4096
        left, right = values((length,)), values((length,))

        def op():
            return torch.dot(left, right)

        return "K4096", 2.0 * length, op, torch.dot(left.double(), right.double())

    if name == "outer":
        m, n = 64, 128
        left, right = values((m,)), values((n,))

        def op():
            return torch.outer(left, right)

        return "M64 N128", float(m * n), op, torch.outer(left.double(), right.double())

    if name == "batched-dot":
        batch, dimension = 4, 512
        left, right = values((batch, dimension)), values((batch, dimension))

        def op():
            return (left * right).sum(dim=1)

        return "B4 K512", 2.0 * batch * dimension, op, (left.double() * right.double()).sum(dim=1)

    if name == "strided-dot":
        length = 512
        left, right = values((length,)), values((length,))

        def op():
            return torch.dot(left, torch.flip(right, (0,)))

        return "A512 B512 reverse", 2.0 * length, op, torch.dot(left.double(), torch.flip(right.double(), (0,)))

    raise ValueError(name)


CASES = (
    "decode-gelu",
    "gemm-fp32",
    "fused-gelu",
    "batched-gemm",
    "gemm-fp16",
    "fused-gelu-fp16-m16-k512",
    "fused-gelu-fp16-m16-k1024",
    "lora",
    "linear-ce-index",
    "linear-backward-relu",
    "dot",
    "outer",
    "batched-dot",
    "strided-dot",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--json-lines", action="store_true")
    parser.add_argument("--only", choices=CASES)
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs must be positive")
    require_no_foreign_compute("dense-linear-pytorch-start")
    if not torch.cuda.is_available():
        print("CUDA-enabled Python PyTorch is required.", file=sys.stderr)
        return 2

    torch.set_grad_enabled(False)
    # Keep FP32 cells in the same numerical mode as the AiDotNet/cuBLAS and
    # direct-PTX FP32-accumulate routes. FP16 cells still use FP16 Tensor Cores.
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    software_fingerprint = {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "pytorch_cuda_version": torch.version.cuda,
        "device_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    for run in range(1, args.runs + 1):
        operations = (args.only,) if args.only else CASES
        for operation in operations:
            require_no_foreign_compute(f"{operation}-start")
            shape, work, eager, expected = make_case(operation, run, device)

            graph_operation, eager_graph, eager_capture_stream = capture_graph(eager)
            competitors = [
                ("PyTorch CUDA eager", eager, 1),
                ("PyTorch CUDA graph", graph_operation, GRAPH_OPERATIONS_PER_REPLAY),
            ]
            compiled = None
            compiled_graph = None
            compiled_capture_stream = None
            try:
                compiled = torch.compile(
                    eager, fullgraph=True, mode="max-autotune-no-cudagraphs")
                compiled_probe = compiled()
                torch.cuda.synchronize()
                compiled_graph_operation, compiled_graph, compiled_capture_stream = capture_graph(compiled)
                competitors.extend((
                    ("PyTorch compile max-autotune", compiled, 1),
                    ("PyTorch compile max-autotune graph", compiled_graph_operation,
                     GRAPH_OPERATIONS_PER_REPLAY),
                ))
                del compiled_probe
            except Exception as exception:
                print(json.dumps({
                    "status": "skip",
                    "run": run,
                    "operation": operation,
                    "shape": shape,
                    "method": "PyTorch compile max-autotune",
                    "reason": f"{type(exception).__name__}: {exception}",
                    **software_fingerprint,
                }, separators=(",", ":")))

            for method, measured, logical_operations_per_call in competitors:
                require_no_foreign_compute(f"{operation}-{method}-start")
                correctness_probe = measured()
                torch.cuda.synchronize()
                error = maximum_error(correctness_probe, expected)
                device_values = measure_device(measured, logical_operations_per_call)
                e2e_values = measure_e2e(measured, logical_operations_per_call)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                baseline = torch.cuda.memory_allocated()
                result = measured()
                torch.cuda.synchronize()
                peak_bytes = max(0, torch.cuda.max_memory_allocated() - baseline)
                output_bytes = output_storage_bytes(result)
                temporary_bytes = max(0, peak_bytes - output_bytes)
                require_no_foreign_compute(f"{operation}-{method}-end")
                median_us = device_values[1]
                record = {
                    "status": "ok",
                    "run": run,
                    "operation": operation,
                    "shape": shape,
                    "method": method,
                    "logical_operations_per_call": logical_operations_per_call,
                    "work_flops": work,
                    "device_mean_us": device_values[0],
                    "device_median_us": device_values[1],
                    "device_p95_us": device_values[2],
                    "device_p99_us": device_values[3],
                    "e2e_mean_us": e2e_values[0],
                    "e2e_median_us": e2e_values[1],
                    "e2e_p95_us": e2e_values[2],
                    "e2e_p99_us": e2e_values[3],
                    "tflops": work / (median_us * 1_000_000.0),
                    "gflops": work / (median_us * 1_000.0),
                    "managed_bytes": None,
                    "output_device_bytes": output_bytes,
                    "temporary_device_allocation_count": None,
                    "temporary_device_bytes": temporary_bytes,
                    "peak_device_bytes": peak_bytes,
                    "max_error": error,
                    "tolerance": 2e-3 if operation.startswith("fused-gelu-fp16-m16-") else 2e-4,
                    **software_fingerprint,
                }
                print(json.dumps(record, separators=(",", ":")))
                del result, correctness_probe
            del expected, eager_graph, eager_capture_stream
            if compiled_graph is not None:
                del compiled_graph
            if compiled_capture_stream is not None:
                del compiled_capture_stream
            if compiled is not None:
                del compiled
            require_no_foreign_compute(f"{operation}-end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
