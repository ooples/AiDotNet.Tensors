"""One-shot Apple Metal compiler evidence for immutable exported production libraries."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
OUTPUT = Path(os.environ.get("METAL_VALIDATION_OUTPUT", "metal-payload-results")).resolve()
OUTPUT.mkdir(parents=True, exist_ok=False)
RECORD = {"schema": "residual-fusion-metal-compiler-v1", "runtime_executed": False,
          "performance_claim": False, "commands": [], "payload_results": [], "pass": False}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(argv, name):
    result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    log = OUTPUT / (name + ".log")
    log.write_text(result.stdout, encoding="utf-8")
    item = {"argv": argv, "exit_code": result.returncode, "log": log.name, "log_sha256": sha(log)}
    RECORD["commands"].append(item)
    return result


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


try:
    binding_path = ROOT / "source-binding.json"
    manifest_path = ROOT / "payload-manifest.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8-sig"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    RECORD.update(source_binding=binding, source_binding_sha256=sha(binding_path),
                  payload_manifest_sha256=sha(manifest_path), driver_sha256=sha(__file__),
                  workflow_sha256=sha(ROOT.parent.parent / ".github/workflows/residual-fusion-metal-payload.yml"),
                  github_run_id=os.environ.get("GITHUB_RUN_ID"), github_run_attempt=os.environ.get("GITHUB_RUN_ATTEMPT"),
                  github_sha=os.environ.get("GITHUB_SHA"), runner_os=os.environ.get("RUNNER_OS"),
                  runner_arch=os.environ.get("RUNNER_ARCH"))
    require(sha(manifest_path) == binding["payload_manifest_sha256"], "Export manifest bytes changed")
    require(manifest["source_commit"] == binding["source_commit"], "Source commits differ")
    require(manifest["assemblies_sha256"] == binding["assemblies_sha256"], "Validated assembly hashes differ")
    source = binding["source_commit"]
    if command(["git", "cat-file", "-e", source + "^{commit}"], "source-present").returncode:
        require(command(["git", "fetch", "--no-tags", "--depth=1", "origin", source], "fetch-exact-source").returncode == 0,
                "Could not fetch exact source commit")
    require(command(["git", "merge-base", "--is-ancestor", source, "HEAD"], "source-ancestor").returncode == 0,
            "Validation branch does not descend from the validated source")
    require(command(["git", "diff", "--exit-code", source, "HEAD", "--", "src", "tests"], "unchanged-production-tree").returncode == 0,
            "Production source or tests changed on the validation branch")
    archive = OUTPUT / "source.tar"
    require(command(["git", "archive", "--format=tar", "--output=" + str(archive), source, "--",
                     binding["source_archive_pathspec"]], "source-archive").returncode == 0, "Source archive failed")
    RECORD["source_archive_sha256"] = sha(archive)
    require(sha(archive) == binding["source_archive_sha256"] == manifest["source_archive_sha256"], "Source archive differs from the export")
    for name, entry in binding["payloads"].items():
        payload = ROOT / name
        require(sha(payload) == entry["sha256"] == manifest["payloads"][name]["sha256"], "Payload hash differs: " + name)
        require(payload.stat().st_size == entry["bytes"] == manifest["payloads"][name]["bytes"], "Payload length differs: " + name)
        shutil.copyfile(payload, OUTPUT / name)
    shutil.copyfile(manifest_path, OUTPUT / manifest_path.name)
    shutil.copyfile(binding_path, OUTPUT / binding_path.name)
    require(command(["xcode-select", "-p"], "xcode-select").returncode == 0, "Xcode is unavailable")
    require(command(["xcodebuild", "-version"], "xcode-version").returncode == 0, "Xcode version lookup failed")
    require(command(["xcrun", "--sdk", "macosx", "--show-sdk-path"], "sdk-path").returncode == 0, "macOS SDK lookup failed")
    require(command(["xcrun", "--sdk", "macosx", "--find", "metal"], "metal-path").returncode == 0, "Apple Metal compiler is unavailable")
    require(command(["xcrun", "--sdk", "macosx", "metal", "--version"], "metal-version").returncode == 0, "Apple Metal compiler version failed")
    for name, entry in binding["payloads"].items():
        air = OUTPUT / (Path(name).stem + ".air")
        argv = ["xcrun", "--sdk", "macosx", "metal", "-std=metal3.0", "-c"]
        if entry["strict"]:
            argv.append("-fno-fast-math")
        argv += [str(ROOT / name), "-o", str(air)]
        compiled = command(argv, Path(name).stem + "-compile")
        item = {"payload": name, "source_sha256": sha(ROOT / name), "strict": entry["strict"],
                "compile_exit_code": compiled.returncode, "pass": False}
        if compiled.returncode == 0 and air.is_file() and air.stat().st_size > 0:
            item.update(air=air.name, air_sha256=sha(air), air_bytes=air.stat().st_size)
            library = OUTPUT / (Path(name).stem + ".metallib")
            linked = command(["xcrun", "--sdk", "macosx", "metallib", str(air), "-o", str(library)], Path(name).stem + "-link")
            item["link_exit_code"] = linked.returncode
            if linked.returncode == 0 and library.is_file() and library.stat().st_size > 0:
                item.update(metallib=library.name, metallib_sha256=sha(library), metallib_bytes=library.stat().st_size, **{"pass": True})
        RECORD["payload_results"].append(item)
    RECORD["pass"] = len(RECORD["payload_results"]) == 2 and all(item["pass"] for item in RECORD["payload_results"])
except Exception as error:
    RECORD["error"] = type(error).__name__ + ": " + str(error)
finally:
    (OUTPUT / "compiler-results.json").write_text(json.dumps(RECORD, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"pass": RECORD["pass"], "output": str(OUTPUT), "error": RECORD.get("error")}))
sys.exit(0 if RECORD["pass"] else 1)
