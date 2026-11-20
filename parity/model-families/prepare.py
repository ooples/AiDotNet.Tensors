"""Materialize the exact historical source trees without changing an existing checkout."""
import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REVISIONS = json.loads((HERE / "sources/revisions.json").read_text(encoding="utf-8"))
DEFAULT_WORK = HERE / ".work"


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def prepare(work, snapshot):
    return prepare_sources(work.resolve() / snapshot, REVISIONS[snapshot], HERE / "sources")


def prepare_sources(root, revisions, sources):
    for name, pin in revisions.items():
        dest = root / name
        if not dest.exists():
            dest.mkdir(parents=True)
            git(dest, "init", "-q")
            git(dest, "config", "core.longpaths", "true")
            git(dest, "remote", "add", "origin", pin["url"])
        if not (dest / ".git").is_dir() or git(dest, "remote", "get-url", "origin") != pin["url"]:
            raise ValueError(f"Expected a dedicated checkout of {pin['url']} in {dest}")
        head = subprocess.run(["git", "-C", str(dest), "rev-parse", "--verify", "--quiet", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
        if not head:
            print(f"Fetching public {name} sources for {pin['commit'][:10]}...", flush=True)
            git(dest, "fetch", "--quiet", "--no-tags", "--depth=1", "origin",
                *(f"{sha}:refs/benchmark-inputs/{sha}" for sha in pin["fetch"]))
            if "bundle" in pin:
                bundle = str(sources / pin["bundle"])
                git(dest, "bundle", "verify", bundle)
                git(dest, "fetch", "--quiet", "--no-tags", bundle, pin["ref"])
            git(dest, "checkout", "--quiet", "--detach", pin["commit"])
        if git(dest, "rev-parse", "HEAD", "HEAD^{tree}").splitlines() != [pin["commit"], pin["tree"]]:
            raise ValueError(f"Wrong source revision in {dest}; choose a fresh --work directory")
        if git(dest, "status", "--porcelain", "--untracked-files=normal"):
            raise ValueError(f"Source checkout contains changes: {dest}")
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", choices=REVISIONS, default="2026-11-19")
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    args = parser.parse_args()
    print(prepare(args.work, args.snapshot))
