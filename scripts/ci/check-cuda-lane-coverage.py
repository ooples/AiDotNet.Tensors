#!/usr/bin/env python3
"""Fail when a GPU test class falls outside the CUDA lane's test filter.

The CUDA lane selects tests by name (FullyQualifiedName~ terms), and a GPU test outside them never runs on CUDA: it is
not skipped, just never selected, so the skip check cannot see it either. A no-fallback GPU concat test in the Engines
namespace passed every CUDA run while the GPU path still fell back to CPU. This walks the test sources, finds each
top-level test class in a file that drives the GPU engine, and checks that some term of the filter selects it.

Usage: check-cuda-lane-coverage.py --filter "<dotnet test --filter expression>" path/to/tests [...]
"""
import argparse
import os
import re
import sys

# A file that drives a GPU engine or backend directly.
GPU_MARKER = re.compile(r"new DirectGpuTensorEngine\(|DirectGpuTensorEngineTestFixture|\bIsGpuAvailable\b|\bCudaBackend\b")
TEST_ATTRIBUTE = re.compile(r"\[(Fact|Theory|SkippableFact|SkippableTheory)\b")
FILE_NAMESPACE = re.compile(r"^namespace\s+([\w.]+)\s*;", re.M)
BLOCK_NAMESPACE = re.compile(r"^namespace\s+([\w.]+)\s*$", re.M)
CLASS = re.compile(r"^(?P<indent>[ \t]*)(?:(?:public|internal|sealed|static|partial|abstract)\s+)*class\s+(?P<name>\w+)", re.M)
# Classes the filter excludes by category anyway (benchmarks and perf gates).
EXCLUDED_CATEGORY = re.compile(r"\[Trait\(\s*\"Category\"\s*,\s*\"(Benchmark|Performance)\"\s*\)\]")


def top_level_classes(text):
    """Yields (fully qualified name, preceding text, body) for each top-level class of a C# file."""
    match = FILE_NAMESPACE.search(text)
    if match:
        namespace, indent = match.group(1), ""
    else:
        match = BLOCK_NAMESPACE.search(text)
        namespace, indent = (match.group(1), "    ") if match else ("", "")
    tops = [c for c in CLASS.finditer(text) if c.group("indent").replace("\t", "    ") == indent]
    for i, cls in enumerate(tops):
        name = f"{namespace}.{cls.group('name')}" if namespace else cls.group("name")
        end = tops[i + 1].start() if i + 1 < len(tops) else len(text)
        yield name, text[max(0, cls.start() - 400):cls.start()], text[cls.start():end]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", required=True, help="the lane's dotnet test --filter expression")
    parser.add_argument("roots", nargs="+")
    args = parser.parse_args()

    terms = re.findall(r"FullyQualifiedName~([^|&()]+)", re.sub(r"\s", "", args.filter))
    if not terms:
        print("::error::The filter has no FullyQualifiedName~ terms.")
        return 1

    missing = []
    for root in args.roots:
        for directory, subdirs, files in os.walk(root):
            subdirs[:] = [d for d in subdirs if d not in ("bin", "obj")]
            for file in files:
                if not file.endswith(".cs"):
                    continue
                path = os.path.join(directory, file)
                with open(path, encoding="utf-8-sig") as handle:
                    text = handle.read()
                if not GPU_MARKER.search(text) or not TEST_ATTRIBUTE.search(text):
                    continue
                for name, preamble, body in top_level_classes(text):
                    if not TEST_ATTRIBUTE.search(body) or EXCLUDED_CATEGORY.search(preamble.split("}")[-1]):
                        continue
                    # As the test name the filter sees: a class term may end in "." to match only that class.
                    if not any(term in name + "." for term in terms):
                        missing.append((name, os.path.relpath(path, root)))

    for name, path in missing:
        print(f"::error file={path}::{name} drives the GPU engine but the CUDA lane's filter does not select it. "
              "Move it under Engines.DirectGpu or add a FullyQualifiedName~ term for it.")
    print(f"{len(missing)} GPU test class(es) outside the CUDA lane filter ({len(terms)} filter terms).")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
