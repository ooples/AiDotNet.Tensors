"""Enumerate PyTorch's public feature surface from the INSTALLED torch package.

The list is generated, never hand-maintained, so a PyTorch release that adds a feature adds it
here, and the coverage test then fails until the feature is mapped or recorded as a gap.

Usage:
    python tools/parity/extract_pytorch_surface.py [--output parity/pytorch-surface.json]

Output is deterministic for a given torch version: items are sorted and no timestamp is written,
so the checked-in file changes only when torch or this extractor changes.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path


def _signature(obj) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return ""


def _public(names):
    return sorted(n for n in names if not n.startswith("_"))


def extract(torch) -> list[dict]:
    items: dict[tuple[str, str], dict] = {}

    def add(category: str, name: str, obj=None) -> None:
        key = (category, name)
        if key not in items:
            items[key] = {"category": category, "name": name, "signature": _signature(obj) if obj is not None else ""}

    # torch.nn.functional: the functional op surface.
    import torch.nn.functional as F

    for name in _public(dir(F)):
        obj = getattr(F, name)
        if callable(obj) and getattr(obj, "__module__", "").startswith("torch"):
            add("nn.functional", name, obj)

    # torch.nn modules: every public nn.Module subclass.
    for name in _public(dir(torch.nn)):
        obj = getattr(torch.nn, name)
        if inspect.isclass(obj) and issubclass(obj, torch.nn.Module) and obj is not torch.nn.Module:
            add("nn.Module", name, obj)

    # torch.optim optimizers and schedulers.
    for name in _public(dir(torch.optim)):
        obj = getattr(torch.optim, name)
        if inspect.isclass(obj) and issubclass(obj, torch.optim.Optimizer) and obj is not torch.optim.Optimizer:
            add("optim", name, obj)
    for name in _public(dir(torch.optim.lr_scheduler)):
        obj = getattr(torch.optim.lr_scheduler, name)
        base = getattr(torch.optim.lr_scheduler, "LRScheduler", None)
        if inspect.isclass(obj) and base is not None and issubclass(obj, base) and obj is not base:
            add("optim.lr_scheduler", name, obj)

    # torch.linalg and torch.fft namespaces.
    for ns_name in ("linalg", "fft"):
        ns = getattr(torch, ns_name)
        for name in _public(dir(ns)):
            obj = getattr(ns, name)
            if callable(obj) and not inspect.isclass(obj) and not inspect.ismodule(obj):
                add(f"torch.{ns_name}", name, obj)

    # Top-level torch functions (tensor-producing and tensor-consuming ops).
    for name in _public(dir(torch)):
        obj = getattr(torch, name)
        if inspect.isclass(obj) or inspect.ismodule(obj) or not callable(obj):
            continue
        module = getattr(obj, "__module__", "") or ""
        if module in ("torch", "torch.functional", "torch._C._VariableFunctions") or module.startswith("torch._refs") or module == "":
            add("torch", name, obj)

    # Tensor methods.
    for name in _public(dir(torch.Tensor)):
        obj = getattr(torch.Tensor, name)
        if callable(obj) and not isinstance(obj, property):
            add("Tensor", name, obj)

    # ATen operators: the finest-grained list of what the dispatcher can run.
    # dir() on the namespace also lists the namespace object's own attributes (`name`, `_dir`);
    # only OpOverloadPacket entries are operators.
    packet = torch._ops.OpOverloadPacket
    for name in _public(dir(torch.ops.aten)):
        if isinstance(getattr(torch.ops.aten, name, None), packet):
            add("aten", name)

    return [items[k] for k in sorted(items)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="parity/pytorch-surface.json")
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("DEFERRED: torch is not installed, so the PyTorch surface cannot be extracted.", file=sys.stderr)
        return 3

    items = extract(torch)
    extractor_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    counts: dict[str, int] = {}
    for item in items:
        counts[item["category"]] = counts.get(item["category"], 0) + 1

    document = {
        "torchVersion": torch.__version__.split("+")[0],
        "extractorSha256": extractor_sha,
        "counts": dict(sorted(counts.items())),
        "items": items,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=1, sort_keys=False) + "\n", encoding="utf-8", newline="\n")
    print(f"torch {document['torchVersion']}: {len(items)} items -> {output}")
    for category, count in document["counts"].items():
        print(f"  {category:22} {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
