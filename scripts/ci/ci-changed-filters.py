"""Decide which path-filtered workflows ci.yml runs for this pull request or push.

Reads .github/ci-path-filters.yml (filter -> event -> GitHub path globs), lists the files the event changed
through the GitHub API (a renamed file counts at its old path too, as GitHub's own path filters count it), and
writes `<filter>=true|false` to $GITHUB_OUTPUT for every filter. The globs follow GitHub's path-filter syntax:
`*` and `?` stay inside one directory, `**` crosses directories, `[...]` is a character class (`[!...]`
negated) and `+` repeats the character before it. A pattern outside that syntax runs its workflow.

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
    """GitHub's path glob as an anchored regular expression; ValueError for syntax it does not cover."""
    out, i = [], 0
    while i < len(pattern):
        c = pattern[i]
        if c == "[":
            close = pattern.find("]", i + 2)
            if close < 0:
                raise ValueError(f"unterminated character class in {pattern!r}")
            body = pattern[i + 1:close]
            negated = body.startswith("!")
            body = body[1:] if negated else body
            if not body or "/" in body:
                raise ValueError(f"unsupported character class in {pattern!r}")
            # A class never matches the separator: a body cannot hold "/", and a negated one excludes it.
            literal = "".join("\\" + ch if ch in "\\^]" else ch for ch in body)
            out.append("[" + ("^/" if negated else "") + literal + "]")
            i = close + 1
        elif c == "+":
            if not out:
                raise ValueError(f"a leading + in {pattern!r}")
            out.append("+")
            i += 1
        elif pattern.startswith("**/", i):
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
    """GitHub semantics: a file counts when the last pattern matching it is not a negation. A pattern this
    evaluator cannot read counts as a match, so its workflow runs instead of being skipped."""
    try:
        compiled = [(p.startswith("!"), glob_regex(p[1:] if p.startswith("!") else p)) for p in patterns]
    except (ValueError, re.error) as error:
        print(f"::warning::{error}; the workflow behind this filter runs.")
        return True
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
            out = gh_json("--paginate", f"repos/{repo}/pulls/{number}/files?per_page=100", "--jq",
                          '.[] | [.filename, (.previous_filename // "")] | @tsv')
            entries = [line.split("	") for line in out.splitlines() if line]
            if len(entries) >= PR_FILE_LIMIT:
                return None
            return [path for entry in entries for path in entry if path]
        if event == "push":
            before, after = os.environ.get("PUSH_BEFORE", ""), os.environ.get("PUSH_AFTER", "")
            if not before or set(before) == {"0"}:
                return None
            data = json.loads(gh_json(f"repos/{repo}/compare/{before}...{after}"))
            if data.get("status") not in ("ahead", "identical"):
                return None
            entries = data.get("files", [])
            if len(entries) >= COMPARE_FILE_LIMIT:
                return None
            return [path for f in entries for path in (f["filename"], f.get("previous_filename")) if path]
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
