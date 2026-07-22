#!/usr/bin/env python3
"""Resident CUDA competitors for issue #851 vision PTX families.

No host/device transfer is timed. Each available cell uses 30 warmups, 101
CUDA-event samples, 25 launches per sample, and three independent runs. A
missing torchvision operation is reported as ineligible rather than replaced
with a CPU result.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Callable

import torch
import torchvision
from torchvision import ops

WARMUPS = 30
SAMPLES = 101
LAUNCHES = 25


def require_no_foreign_compute(label: str) -> None:
    monitor = subprocess.run(["nvidia-smi", "pmon", "-c", "1", "-s", "u"],
                             check=True, capture_output=True, text=True, timeout=5)
    conflicts: list[str] = []
    for line in monitor.stdout.splitlines():
        cells = line.split()
        if not cells or cells[0].startswith("#") or len(cells) < 9:
            continue
        try:
            pid = int(cells[1])
        except ValueError:
            continue
        try:
            sm = int(cells[3])
        except ValueError:
            sm = 0
        process_type = cells[2].upper()
        if pid != os.getpid() and (process_type == "C" or
                                   ("C" in process_type and sm > 5)):
            conflicts.append(f"pid={pid} {cells[-1]} type={process_type} sm={sm}%")
    if conflicts:
        raise RuntimeError(f"[{label}] foreign GPU workload: " + "; ".join(conflicts))


@dataclass(frozen=True)
class Distribution:
    mean_us: float
    median_us: float
    p95_us: float
    p99_us: float


@dataclass(frozen=True)
class Case:
    family: str
    shape: str
    method: str
    launch: Callable[[], object]
    work: float
    flop_like: float
    algorithmic_bytes: float
    oracle: Callable[[], object] | None = None


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    position = q * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def summarize(values: list[float]) -> Distribution:
    return Distribution(
        statistics.fmean(values), percentile(values, 0.50),
        percentile(values, 0.95), percentile(values, 0.99))


def boxes(count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    top_left = torch.rand((count, 2), generator=generator, device="cuda") * 512
    extent = torch.rand((count, 2), generator=generator, device="cuda") * 128
    return torch.cat((top_left, top_left + extent), dim=1).contiguous()


def available(name: str) -> Callable | None:
    return getattr(ops, name, None)


def maximum_error(actual: object, reference: object) -> float:
    actual_values = actual if isinstance(actual, (tuple, list)) else (actual,)
    reference_values = reference if isinstance(reference, (tuple, list)) else (reference,)
    if len(actual_values) != len(reference_values):
        return math.inf
    error = 0.0
    for actual_value, reference_value in zip(actual_values, reference_values):
        actual_tensor = actual_value.detach().cpu().to(torch.float64)
        reference_tensor = reference_value.detach().cpu().to(torch.float64)
        if actual_tensor.shape != reference_tensor.shape:
            return math.inf
        if actual_tensor.numel():
            error = max(error, (actual_tensor - reference_tensor).abs().max().item())
    return error


def loss_backward_oracle(function: Callable, predicted: torch.Tensor,
                         target: torch.Tensor, grad_output: torch.Tensor) -> Callable[[], object]:
    def evaluate() -> object:
        cpu_predicted = predicted.detach().cpu().to(torch.float64).requires_grad_(True)
        cpu_target = target.detach().cpu().to(torch.float64)
        cpu_grad = grad_output.detach().cpu().to(torch.float64)
        loss = function(cpu_predicted, cpu_target, reduction="none")
        return torch.autograd.grad(loss, cpu_predicted, grad_outputs=cpu_grad)[0]
    return evaluate


def build_cases() -> tuple[list[Case], list[dict[str, str]]]:
    cases: list[Case] = []
    ineligible: list[dict[str, str]] = []
    a = boxes(256, 851)
    b = boxes(256, 852)
    pairwise = (
        ("BoxIoU", "box_iou", 20.0),
        ("GIoU", "generalized_box_iou", 32.0),
        ("DIoU", "distance_box_iou", 38.0),
        ("CIoU", "complete_box_iou", 56.0),
    )
    for family, name, flops in pairwise:
        function = available(name)
        if function is not None:
            cases.append(Case(family, "N=256,M=256", f"torchvision.ops.{name}",
                              lambda fn=function: fn(a, b), 256.0 * 256.0, flops, 36.0,
                              lambda fn=function: fn(a.cpu().double(), b.cpu().double())))
        else:
            ineligible.append({"family": family, "shape": "N=256,M=256",
                               "method": f"torchvision.ops.{name}",
                               "reason": "operation is absent from the installed torchvision build"})

    cases.append(Case("BoxArea", "N=4096", "torchvision.ops.box_area",
                      lambda: ops.box_area(boxes_area), 4096.0, 3.0, 20.0,
                      lambda: ops.box_area(boxes_area.cpu().double())))
    cases.append(Case("BoxConvert", "N=4096,xyxy->cxcywh", "torchvision.ops.box_convert",
                      lambda: ops.box_convert(boxes_area, "xyxy", "cxcywh"),
                      4096.0, 8.0, 32.0,
                      lambda: ops.box_convert(boxes_area.cpu().double(), "xyxy", "cxcywh")))

    loss_predicted = boxes(4096, 854).requires_grad_(True)
    loss_target = boxes(4096, 855)
    grad_output = torch.linspace(0.25, 1.0, 4096, device="cuda")
    loss_families = (
        ("GIoULoss", "generalized_box_iou_loss", 32.0),
        ("DIoULoss", "distance_box_iou_loss", 38.0),
        ("CIoULoss", "complete_box_iou_loss", 56.0),
    )
    ineligible.append({"family": "IoULoss", "shape": "N=4096",
                       "method": "torchvision.ops.iou_loss",
                       "reason": "torchvision exposes no aligned plain-IoU loss primitive"})
    ineligible.append({"family": "IoULossBackward", "shape": "N=4096",
                       "method": "torchvision autograd",
                       "reason": "no aligned plain-IoU loss primitive exists to differentiate"})
    for family, name, flops in loss_families:
        function = available(name)
        if function is None:
            ineligible.extend((
                {"family": family, "shape": "N=4096", "method": f"torchvision.ops.{name}",
                 "reason": "operation is absent from the installed torchvision build"},
                {"family": family + "Backward", "shape": "N=4096",
                 "method": f"torchvision autograd({name})",
                 "reason": "forward operation is absent from the installed torchvision build"},
            ))
            continue
        cases.append(Case(family, "N=4096", f"torchvision.ops.{name}",
                          lambda fn=function: fn(loss_predicted, loss_target, reduction="none"),
                          4096.0, flops, 36.0,
                          lambda fn=function: fn(loss_predicted.detach().cpu().double(),
                                                 loss_target.cpu().double(), reduction="none")))
        cases.append(Case(
            family + "Backward", "N=4096", f"torchvision autograd({name})",
            lambda fn=function: torch.autograd.grad(
                fn(loss_predicted, loss_target, reduction="none"), loss_predicted,
                grad_outputs=grad_output, retain_graph=False, create_graph=False)[0],
            4096.0, flops * 2.0, 56.0,
            loss_backward_oracle(function, loss_predicted, loss_target, grad_output)))

    scores = torch.linspace(1.0, 0.0, 256, device="cuda")
    labels = torch.arange(256, device="cuda") % 8
    cases.append(Case("NMS", "N=256", "torchvision.ops.nms",
                      lambda: ops.nms(a, scores, 0.5), 256.0 * 256.0, 20.0, 36.0,
                      lambda: ops.nms(a.cpu().double(), scores.cpu().double(), 0.5)))
    cases.append(Case("BatchedNMS", "N=256,C=8", "torchvision.ops.batched_nms",
                      lambda: ops.batched_nms(a, scores, labels, 0.5),
                      256.0 * 256.0, 20.0, 36.0,
                      lambda: ops.batched_nms(a.cpu().double(), scores.cpu().double(),
                                              labels.cpu(), 0.5)))

    masks = (torch.rand((256, 28, 28), device="cuda") > 0.9).to(torch.float32)
    cases.append(Case("MasksToBoxes", "256x28x28", "torchvision.ops.masks_to_boxes",
                      lambda: ops.masks_to_boxes(masks), 256.0 * 28.0 * 28.0, 2.0, 4.0,
                      lambda: ops.masks_to_boxes(masks.cpu().double())))

    feature = torch.randn((1, 256, 56, 56), device="cuda")
    roi_boxes = torch.zeros((256, 5), device="cuda")
    roi_index = torch.arange(256, device="cuda", dtype=torch.float32)
    roi_boxes[:, 1] = roi_index.remainder(24)
    roi_boxes[:, 2] = (roi_index * 3).remainder(24)
    roi_boxes[:, 3] = roi_boxes[:, 1] + 16
    roi_boxes[:, 4] = roi_boxes[:, 2] + 16
    cases.append(Case("RoIAlign", "K=256,C=256,7x7", "torchvision.ops.roi_align",
                      lambda: ops.roi_align(feature, roi_boxes, (7, 7),
                                            spatial_scale=0.25, sampling_ratio=2,
                                            aligned=True),
                      256.0 * 256.0 * 49.0, 32.0, 20.0,
                      lambda: ops.roi_align(feature.cpu().double(), roi_boxes.cpu().double(),
                                            (7, 7), spatial_scale=0.25, sampling_ratio=2,
                                            aligned=True)))
    cases.append(Case("RoIPool", "K=256,C=256,7x7", "torchvision.ops.roi_pool",
                      lambda: ops.roi_pool(feature, roi_boxes, (7, 7), spatial_scale=0.25),
                      256.0 * 256.0 * 49.0, 2.0, 8.0,
                      lambda: ops.roi_pool(feature.cpu().double(), roi_boxes.cpu().double(),
                                           (7, 7), spatial_scale=0.25)))

    ps_feature = torch.randn((1, 196, 56, 56), device="cuda")
    for family, name, sampling in (
        ("PsRoIAlign", "ps_roi_align", 2),
        ("PsRoIPool", "ps_roi_pool", None),
    ):
        function = available(name)
        if function is None:
            ineligible.append({"family": family, "shape": "K=256,C=196,7x7",
                               "method": f"torchvision.ops.{name}",
                               "reason": "operation is absent from the installed torchvision build"})
        elif sampling is None:
            cases.append(Case(family, "K=256,C=196,7x7", f"torchvision.ops.{name}",
                              lambda fn=function: fn(ps_feature, roi_boxes, (7, 7),
                                                     spatial_scale=0.25),
                              256.0 * 4.0 * 49.0, 2.0, 8.0,
                              lambda fn=function: fn(ps_feature.cpu().double(),
                                                     roi_boxes.cpu().double(), (7, 7),
                                                     spatial_scale=0.25)))
        else:
            cases.append(Case(family, "K=256,C=196,7x7", f"torchvision.ops.{name}",
                              lambda fn=function, sr=sampling: fn(
                                  ps_feature, roi_boxes, (7, 7),
                                  spatial_scale=0.25, sampling_ratio=sr),
                              256.0 * 4.0 * 49.0, 32.0, 20.0,
                              lambda fn=function, sr=sampling: fn(
                                  ps_feature.cpu().double(), roi_boxes.cpu().double(),
                                  (7, 7), spatial_scale=0.25, sampling_ratio=sr)))

    cross_a = torch.randn((1024, 3), device="cuda")
    cross_b = torch.randn((1024, 3), device="cuda")
    cases.append(Case("Cross3", "1024x3", "torch.cross",
                      lambda: torch.cross(cross_a, cross_b, dim=1), 1024.0, 9.0, 36.0,
                      lambda: torch.cross(cross_a.cpu().double(), cross_b.cpu().double(), dim=1)))

    mesh_x = torch.randn((1024,), device="cuda")
    mesh_y = torch.randn((256,), device="cuda")
    cases.append(Case("Meshgrid2D", "1024x256,ij", "torch.meshgrid",
                      lambda: torch.meshgrid(mesh_x, mesh_y, indexing="ij"),
                      1024.0 * 256.0 * 2.0, 0.0, 8.0,
                      lambda: torch.meshgrid(mesh_x.cpu().double(), mesh_y.cpu().double(),
                                             indexing="ij")))
    cases.append(Case("Meshgrid2D", "1024x256,xy", "torch.meshgrid",
                      lambda: torch.meshgrid(mesh_x, mesh_y, indexing="xy"),
                      1024.0 * 256.0 * 2.0, 0.0, 8.0,
                      lambda: torch.meshgrid(mesh_x.cpu().double(), mesh_y.cpu().double(),
                                             indexing="xy")))
    ineligible.append({"family": "IouFamilyBackward(A+B)", "shape": "N=256,M=256",
                       "method": "torchvision public autograd",
                       "reason": "no public pairwise IoU-family backward primitive matches both-owner ABI"})
    return cases, ineligible


boxes_area = boxes(4096, 853) if torch.cuda.is_available() else None


def measure(case: Case) -> tuple[Distribution, Distribution, int]:
    for _ in range(WARMUPS):
        case.launch()
    torch.cuda.synchronize()

    device_values: list[float] = []
    for _ in range(SAMPLES):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(LAUNCHES):
            case.launch()
        end.record()
        end.synchronize()
        device_values.append(start.elapsed_time(end) * 1000.0 / LAUNCHES)

    e2e_values: list[float] = []
    for _ in range(SAMPLES):
        start_ns = time.perf_counter_ns()
        case.launch()
        torch.cuda.synchronize()
        e2e_values.append((time.perf_counter_ns() - start_ns) / 1000.0)

    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    case.launch()
    torch.cuda.synchronize()
    temporary = max(0, torch.cuda.max_memory_allocated() - before)
    return summarize(device_values), summarize(e2e_values), temporary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA PyTorch is required; CPU results are intentionally unsupported.")

    properties = torch.cuda.get_device_properties(0)
    driver_getter = getattr(torch._C, "_cuda_getDriverVersion", None)
    print("environment_json=" + json.dumps({
        "gpu": torch.cuda.get_device_name(),
        "gpu_uuid": str(getattr(properties, "uuid", "unknown")),
        "compute_capability": [properties.major, properties.minor],
        "total_device_memory": properties.total_memory,
        "cuda_driver": driver_getter() if driver_getter is not None else None,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "os": platform.platform(),
        "warmups": WARMUPS,
        "samples": SAMPLES,
        "launches_per_device_sample": LAUNCHES,
    }, sort_keys=True))

    cases, ineligible = build_cases()
    for row in ineligible:
        print("vision_competitor_json=" + json.dumps({
            "status": "ineligible", **row
        }, sort_keys=True))

    for run in range(1, args.runs + 1):
        require_no_foreign_compute(f"vision-family-run-{run}-start")
        for case in cases:
            device, e2e, temporary = measure(case)
            error = None
            if case.oracle is not None:
                actual = case.launch()
                torch.cuda.synchronize()
                error = maximum_error(actual, case.oracle())
            gflops = (case.work * case.flop_like / (device.median_us * 1000.0)
                      if case.flop_like else 0.0)
            gbps = case.work * case.algorithmic_bytes / (device.median_us * 1000.0)
            print("vision_competitor_json=" + json.dumps({
                "status": "ok", "run": run, "family": case.family,
                "shape": case.shape, "method": case.method,
                "device": asdict(device), "end_to_end": asdict(e2e),
                "gflops": gflops, "algorithmic_gbps": gbps,
                "managed_bytes": None,
                "temporary_or_peak_device_bytes": temporary,
                "maximum_error": error,
            }, sort_keys=True))
        torch.cuda.synchronize()
        require_no_foreign_compute(f"vision-family-run-{run}-end")


if __name__ == "__main__":
    main()
