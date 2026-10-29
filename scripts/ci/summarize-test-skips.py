#!/usr/bin/env python3
"""Summarise passed, failed and skipped tests from TRX files into the GitHub job summary (#1569).

A skipped test does not fail a build, so a GPU lane whose tests skip goes green without running them. This report makes
every skip visible, with its reason, on the run's summary page. By default it never changes the job's result; with
--fail-on-skip (a lane that has the device), any skip whose reason does not match --allowed-skip, and whose test name
does not match --allowed-skip-test, fails the job, so the lane cannot pass by not running. --allowed-skip-test is for
tests of another device (a HIP or Dawn device class on a CUDA lane), whose skip reasons do not name the backend.

Usage: summarize-test-skips.py --title "POCL (shard)" [--fail-on-skip] [--allowed-skip REGEX] [--allowed-skip-test REGEX] path/to/*.trx [...]
"""
import argparse
import glob
import os
import re
import sys
import xml.etree.ElementTree as ET

NS = {"t": "http://microsoft.com/schemas/VisualStudio/TeamTest/2010"}
MAX_LISTED = 200


def reason_of(result):
    for path in ("t:Output/t:ErrorInfo/t:Message", "t:Output/t:StdOut"):
        node = result.find(path, NS)
        if node is not None and node.text and node.text.strip():
            return " ".join(node.text.split())[:300]
    return "(no reason given)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--fail-on-skip", action="store_true",
                        help="fail when a test skips for a reason --allowed-skip does not match")
    parser.add_argument("--allowed-skip", default=None,
                        help="regex over skip reasons that are explicit, reasoned exemptions (e.g. 'requires sm_8')")
    parser.add_argument("--allowed-skip-test", default=None,
                        help="regex over test names whose skips are exempt (tests of a device this lane does not have)")
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()

    files = sorted({f for p in args.paths for f in glob.glob(p, recursive=True)})
    counts = {"Passed": 0, "Failed": 0, "NotExecuted": 0}
    skipped = []
    for path in files:
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as error:
            print(f"::warning::{path} is not valid TRX: {error}")
            continue
        for result in root.iter(f"{{{NS['t']}}}UnitTestResult"):
            outcome = result.get("outcome", "")
            counts[outcome] = counts.get(outcome, 0) + 1
            if outcome == "NotExecuted":
                skipped.append((result.get("testName", "?"), reason_of(result)))

    lines = [f"### Test outcomes: {args.title}", ""]
    if not files:
        lines.append("No TRX file was produced, so nothing ran or the run crashed before reporting.")
    else:
        lines.append(f"Passed **{counts['Passed']}**, failed **{counts['Failed']}**, skipped **{counts['NotExecuted']}** "
                     f"({len(files)} result file{'s' if len(files) != 1 else ''}).")
        if skipped:
            lines += ["", "| skipped test | reason |", "| --- | --- |"]
            for name, reason in skipped[:MAX_LISTED]:
                lines.append(f"| `{name}` | {reason.replace('|', '/')} |")
            if len(skipped) > MAX_LISTED:
                lines.append(f"| ... | {len(skipped) - MAX_LISTED} more |")
    text = "\n".join(lines) + "\n"

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(text)
    sys.stdout.write(text)
    if skipped:
        print(f"::notice title=Skipped tests ({args.title})::{len(skipped)} test(s) skipped; reasons are in the job summary.")
    if args.fail_on_skip:
        if not files:
            print(f"::error title=No test results ({args.title})::No TRX file was produced.")
            return 1
        # A run that executed nothing (discovery matched no tests, or every test was an allowed skip) proves nothing.
        executed = counts["Passed"] + counts["Failed"]
        if executed == 0:
            print(f"::error title=No tests executed ({args.title})::The result files hold no passed or failed test.")
            return 1
        allowed = re.compile(args.allowed_skip) if args.allowed_skip else None
        allowed_test = re.compile(args.allowed_skip_test) if args.allowed_skip_test else None
        unexpected = [(n, r) for n, r in skipped
                      if not (allowed and allowed.search(r)) and not (allowed_test and allowed_test.search(n))]
        if unexpected:
            for name, reason in unexpected[:50]:
                print(f"::error title=Unexpected skip::{name}: {reason}")
            print(f"{len(unexpected)} test(s) skipped on a lane that has the device; see the job summary.")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
