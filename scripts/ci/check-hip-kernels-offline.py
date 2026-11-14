#!/usr/bin/env python3
"""Compile the HIP kernel sources offline for AMD targets, without an AMD GPU.

hiprtc compiles these sources at runtime on the user's device. A source that does not compile for a target leaves
every kernel in that module missing there, and nothing in CI runs on AMD hardware to notice. This gate extracts each
self-contained source (a C# verbatim string `return @"...";` or `=> @"...";` containing a __global__ kernel) from
the HIP backend, compiles it device-only with clang for every listed target, the way hiprtc sees it (HIP runtime and
fp16 headers implicitly included), and fails on any error.

It also requires the MFMA GEMM module to emit v_mfma instructions on every CDNA target, so a preprocessor guard that
silently drops the matrix-core path fails the gate instead of shipping scalar code.

Sources assembled at runtime (concatenation, interpolation) are not extracted; they are listed in the output.

Usage: check-hip-kernels-offline.py [--clang clang-20] [--rocm-path /usr] [--targets gfx908,...]
"""
import argparse
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
HIP_DIR = ROOT / "src" / "AiDotNet.Tensors" / "Engines" / "DirectGpu" / "HIP"
CDNA = ("gfx908", "gfx90a", "gfx940", "gfx941", "gfx942", "gfx950")
DEFAULT_TARGETS = "gfx908,gfx90a,gfx942,gfx950,gfx1030,gfx1100"
# (file stem, method) whose module must contain matrix-core instructions on CDNA targets.
MFMA_SOURCES = {("HipMfmaKernel", "GetMfmaSource")}

VERBATIM = re.compile(r'(\w+)\s*\(\)\s*(?:=>\s*@"((?:[^"]|"")*)"\s*;|\{\s*(?://[^\n]*\n\s*)*return\s+@"((?:[^"]|"")*)"\s*;)', re.S)
ANY_VERBATIM = re.compile(r'(?:return|=>)\s*@"((?:[^"]|"")*)"\s*;', re.S)


def extract():
    sources, dynamic = [], []
    for path in sorted(HIP_DIR.rglob("*.cs")):
        text = path.read_text(encoding="utf-8")
        found = False
        named = {m.start(): m for m in VERBATIM.finditer(text)}
        for i, m in enumerate(ANY_VERBATIM.finditer(text)):
            body = m.group(1).replace('""', '"')
            if "__global__" not in body:
                continue
            found = True
            method = next((n.group(1) for n in named.values() if n.start() <= m.start() < n.end()), f"source{i}")
            sources.append((path.stem, method, body))
        if not found and "__global__" in text:
            dynamic.append(path.relative_to(ROOT).as_posix())
    return sources, dynamic


def compile_one(clang, rocm_path, target, body):
    with tempfile.TemporaryDirectory() as tmp:
        src = pathlib.Path(tmp) / "k.hip"
        asm = pathlib.Path(tmp) / "k.s"
        src.write_text(body, encoding="utf-8", newline="\n")
        cmd = [clang, "-x", "hip", f"--offload-arch={target}", "--cuda-device-only", "-nogpulib",
               f"--rocm-path={rocm_path}", "-include", "hip/hip_runtime.h", "-include", "hip/hip_fp16.h",
               "-O3", "-S", "-o", str(asm), str(src)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            errors = [l for l in proc.stderr.splitlines() if "error" in l]
            return False, 0, errors[:8] or proc.stderr.splitlines()[-8:]
        return True, asm.read_text(encoding="utf-8", errors="replace").count("v_mfma"), []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clang", default="clang-20")
    ap.add_argument("--rocm-path", default="/usr")
    ap.add_argument("--targets", default=DEFAULT_TARGETS)
    args = ap.parse_args()
    targets = [t for t in args.targets.split(",") if t]

    sources, dynamic = extract()
    if not any((s, m) in MFMA_SOURCES for s, m, _ in sources):
        print("FAIL: the MFMA GEMM source was not found; update MFMA_SOURCES", file=sys.stderr)
        return 1

    failures = 0
    for stem, method, body in sources:
        for target in targets:
            ok, mfma, errors = compile_one(args.clang, args.rocm_path, target, body)
            label = f"{stem}.{method} {target}"
            if not ok:
                failures += 1
                print(f"FAIL {label}")
                for e in errors:
                    print(f"    {e}")
                continue
            if (stem, method) in MFMA_SOURCES and target in CDNA and mfma == 0:
                failures += 1
                print(f"FAIL {label}: compiled but emitted no v_mfma instructions")
                continue
            print(f"ok   {label}" + (f" (v_mfma: {mfma})" if (stem, method) in MFMA_SOURCES else ""))

    print(f"\n{len(sources)} sources x {len(targets)} targets, {failures} failure(s).")
    if dynamic:
        print("Not covered (sources assembled at runtime):")
        for d in dynamic:
            print(f"    {d}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
