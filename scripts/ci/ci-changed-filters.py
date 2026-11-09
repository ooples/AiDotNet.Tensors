"""Decide which path-filtered workflows ci.yml runs for this pull request or push.

Reads .github/ci-path-filters.yml (filter -> event -> GitHub path globs), lists the files the event changed
through the GitHub API, and writes `<filter>=true|false` to $GITHUB_OUTPUT for every filter. The globs follow
GitHub's path-filter syntax: `*` and `?` stay inside one directory, `**` crosses directories.

Every filter is true when the changed files cannot be listed (an API failure, a push that creates a branch or
rewrites history, a pull request past the API's 3,000-file limit), or when the routing itself changed (ci.yml,
the filter file or this script): a filter that cannot be evaluated runs its workflow rather than skipping it.

Usage: ci-changed-filters.py <filters.yml>
Environment: GH_TOKEN, GITHUB_REPOSITORY, GITHUB_EVENT_NAME, GITHUB_OUTPUT, and PR_NUMBER (pull_request) or
PUSH_BEFORE and PUSH_AFTER (push).
"""
import json
import os
import re
import subprocess
import sys

import yaml

ROUTING_FILES = {".github/workflows/ci.yml", ".github/ci-path-filters.yml", "scripts/ci/ci-changed-filters.py"}
PR_FILE_LIMIT = 3000
COMPARE_FILE_LIMIT = 300


def glob_regex(pattern):
    """GitHub's path glob as an anchored regular expression."""
    out, i = [], 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def matches(patterns, files):
    """GitHub semantics: a file counts when the last pattern matching it is not a negation."""
    compiled = [(p.startswith("!"), glob_regex(p.lstrip("!"))) for p in patterns]
    for f in files:
        included = False
        for negated, regex in compiled:
            if regex.match(f):
                included = not negated
        if included:
            return True
    return False


def gh_json(*args):
    result = subprocess.run(["gh", "api", *args], capture_output=True, text=True, check=True)
    return result.stdout


def changed_files(event, repo):
    """The changed paths, or None when they cannot be listed completely."""
    try:
        if event == "pull_request":
            number = os.environ["PR_NUMBER"]
            out = gh_json("--paginate", f"repos/{repo}/pulls/{number}/files?per_page=100", "--jq", ".[].filename")
            files = [line for line in out.splitlines() if line]
            return None if len(files) >= PR_FILE_LIMIT else files
        if event == "push":
            before, after = os.environ.get("PUSH_BEFORE", ""), os.environ.get("PUSH_AFTER", "")
            if not before or set(before) == {"0"}:
                return None
            data = json.loads(gh_json(f"repos/{repo}/compare/{before}...{after}"))
            if data.get("status") not in ("ahead", "identical"):
                return None
            files = [f["filename"] for f in data.get("files", [])]
            return None if len(files) >= COMPARE_FILE_LIMIT else files
    except (subprocess.CalledProcessError, KeyError, json.JSONDecodeError) as error:
        print(f"::warning::Could not list the changed files ({error}); every filtered workflow runs.")
        return None
    return None


def main():
    filters = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
    event = os.environ["GITHUB_EVENT_NAME"]
    files = changed_files(event, os.environ["GITHUB_REPOSITORY"])
    if files is not None and ROUTING_FILES & set(files):
        print("The CI routing changed; every filtered workflow runs.")
        files = None
    results = {}
    for name, events in filters.items():
        patterns = (events or {}).get(event)
        results[name] = files is None or patterns is None or matches(patterns, files)
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        for name, run in results.items():
            output.write(f"{name}={'true' if run else 'false'}\n")
    width = max(map(len, results))
    for name, run in results.items():
        print(f"{name:<{width}}  {'runs' if run else 'skipped (no matching change)'}")


if __name__ == "__main__":
    main()
