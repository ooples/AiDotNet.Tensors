"""Fails when a published performance claim is not backed by the committed benchmark reports (#1094).

    python tests/AiDotNet.Tensors.Benchmarks/check_benchmark_claims.py [--repo-description TEXT]

The README once advertised "3,200x faster than NumSharp" against a NumSharp release eight years older than the
current one, and the GitHub repository description kept that number after the README was corrected. This gate makes
every published speed number traceable to a report:

1. The README cites exactly one Results/<run>/ folder, and that folder holds the BenchmarkDotNet reports and a
   manifest.json (written by RunManifest.cs) naming the competitor versions that ran.
2. Those versions equal the benchmark project's PackageReferences, and every "<library> <version>" the README
   names agrees with them. Bumping a competitor without re-running fails here.
3. make_readme_tables.py regenerates every table from the reports. Every generated row must appear verbatim in the
   README (no edited numbers, no hidden losses), and every README table row with a speedup must be a generated row.
4. Every "N of M" and "all N" count in the README matches a library's generated wins and measurements.
5. Every speed multiplier (2.5x, 1.32x) anywhere in the README equals a generated speedup or its inverse.
6. Text with no report behind it -- the package description, the GitHub repository description -- carries no
   multipliers and no superlatives ("fastest", "beats", "faster than").
7. Benchmark write-ups kept for history (BENCHMARK_*.md) say so in their first lines.

docs/ holds engineering notes (bake-offs, investigations) and is not a published claim, so it is not scanned.
"""
import argparse, json, os, re, sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_readme_tables  # noqa: E402  (sibling script; isolated mode does not put its folder on the path)

ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
COMPETITORS = {  # PackageReference id -> the name the README uses
    'TorchSharp': 'TorchSharp', 'Microsoft.ML': 'ML.NET', 'TensorFlow.NET': 'TensorFlow.NET',
    'NumSharp': 'NumSharp', 'MathNet.Numerics': 'MathNet.Numerics'}
# "MathNet 5.0.0" in table headers is the same library as "MathNet.Numerics 5.0.0".
README_NAMES = {'TorchSharp': ['TorchSharp'], 'ML.NET': ['ML.NET'], 'TensorFlow.NET': ['TensorFlow.NET'],
                'NumSharp': ['NumSharp'], 'MathNet.Numerics': ['MathNet.Numerics', 'MathNet']}
MULTIPLIER = re.compile(r'(?<![\w.×])(\d+(?:[.,]\d+)?)\s?[x×](?![\w\d])')
SUPERLATIVE = re.compile(r'\b(fastest|beats?|outperform\w*|faster than|slower than|matches torchsharp)\b', re.I)
# "50% faster", "30 percent quicker": a relative speed claim without an "x".
PERCENT = re.compile(r'\b\d+(?:[.,]\d+)?\s?(?:%|percent)\s+(?:faster|slower|quicker|less (?:time|latency)|more throughput|speed\w*)', re.I)
HISTORICAL = ['BENCHMARK_BASELINE.md', 'tests/AiDotNet.Tensors.Benchmarks/BENCHMARK_RESULTS.md']

errors = []
def fail(msg): errors.append(msg)

def read(rel): return open(os.path.join(ROOT, rel), encoding='utf-8').read()

def strip_code_and_links(line):
    line = re.sub(r'`[^`]*`', '', line)            # inline code: paths, flags, type names
    return re.sub(r'\]\([^)]*\)', ']', line)        # link targets

def results_dir(readme):
    runs = set(re.findall(r'tests/AiDotNet\.Tensors\.Benchmarks/Results/([^/`)\s]+)/', readme))
    if len(runs) != 1:
        fail(f'README must cite exactly one Results/<run>/ folder; found {sorted(runs) or "none"}.')
        return None
    path = os.path.join(HERE, 'Results', runs.pop())
    if not os.path.isdir(path):
        fail(f'README cites {path}, which does not exist.')
        return None
    return path

def check_versions(readme, run):
    manifest_path = os.path.join(run, 'manifest.json')
    if not os.path.isfile(manifest_path):
        fail(f'{manifest_path} is missing: a run must record what it measured (RunManifest.cs writes it).')
        return
    manifest = {}
    for suite in json.load(open(manifest_path, encoding='utf-8')).get('runs', {}).values():
        manifest.update(suite.get('packages', {}))
    project = ET.parse(os.path.join(HERE, 'AiDotNet.Tensors.Benchmarks.csproj')).getroot()
    referenced = {e.get('Include'): e.get('Version') for e in project.iter('PackageReference')}
    for package, name in COMPETITORS.items():
        ran, ref = manifest.get(package), referenced.get(package)
        if ran is None:
            fail(f'{package}: manifest.json does not say which version ran.')
            continue
        if ref is not None and ref != ran:
            fail(f'{package}: the benchmark project references {ref} but the published run measured {ran}. '
                 'Re-run the suite and publish the new reports, or revert the bump.')
        for alias in README_NAMES[name]:
            for m in re.finditer(rf'(?<![\w.]){re.escape(alias)}(?![\w.]|\.Numerics) (\d+(?:\.\d+){{1,3}})', readme):
                if not (ran == m.group(1) or ran.startswith(m.group(1) + '.')):
                    fail(f'README names "{alias} {m.group(1)}", but the published run measured {package} {ran}.')

def check_tables(readme, run):
    generated, counts = make_readme_tables.generate(run, run)
    gen_rows = {l for l in generated.splitlines() if l.startswith('| ') and not l.startswith('| Operation')}
    readme_lines = readme.splitlines()
    readme_set = set(readme_lines)
    for row in sorted(gen_rows - readme_set):
        fail(f'Generated row missing from README (edited or dropped): {row}')
    for i, line in enumerate(readme_lines, 1):
        if line.startswith('| ') and re.search(r'\d\.\d+×', line) and line not in gen_rows:
            fail(f'README:{i} is a speedup row the reports do not produce: {line}')
    speedups = set()
    for row in gen_rows:
        for m in re.finditer(r'(\d+\.\d+)×', row):
            v = float(m.group(1)); speedups.add(f'{v:.2f}')
            if v > 0: speedups.add(f'{1 / v:.2f}')
    return counts, speedups

def check_counts(readme, counts):
    valid_pairs = {(w, w + l) for w, l in counts.values()}
    valid_all = {w for w, l in counts.values() if l == 0}
    summary = {lib: (w, l) for lib, (w, l) in counts.items()}
    for m in re.finditer(r'(?m)^\| ([^|]+?) (?:\([^|]*\) )?\| (\d+) \| (\d+) \|$', readme):
        lib = m.group(1).strip()
        key = next((k for k, names in README_NAMES.items() if lib in names), None)
        if key is not None and summary.get(key) != (int(m.group(2)), int(m.group(3))):
            fail(f'README summary says {lib} {m.group(2)} wins / {m.group(3)} losses; the reports give {summary.get(key)}.')
    prose = [strip_code_and_links(l) for l in readme.splitlines() if not l.startswith('|')]
    for line in prose:
        for m in re.finditer(r'\b(\d+) of (\d+)\b', line):
            if (int(m.group(1)), int(m.group(2))) not in valid_pairs:
                fail(f'README claims "{m.group(0)}", which matches no library\'s wins/measurements: {line.strip()[:160]}')
        for m in re.finditer(r'\ball (\d+)\b', line):
            if int(m.group(1)) not in valid_all:
                fail(f'README claims "{m.group(0)}", but no library has {m.group(1)} measurements and no losses.')
    return valid_pairs, valid_all

def check_readme_prose(readme, speedups, valid_pairs, valid_all):
    for i, raw in enumerate(readme.splitlines(), 1):
        if raw.startswith('| '): continue
        line = strip_code_and_links(raw)
        for m in MULTIPLIER.finditer(line):
            v = m.group(1).replace(',', '')
            if float(v) == 1: continue   # "above 1x means faster" is a definition, not a claim
            if f'{float(v):.2f}' not in speedups:
                fail(f'README:{i} says "{m.group(0)}", which is no speedup in the published reports: {raw.strip()[:160]}')
        if PERCENT.search(line):
            fail(f'README:{i} makes a percentage speed claim the reports do not state: {raw.strip()[:160]}')
        # A comparative word is backed only by a verified count in the same sentence, so one counted result
        # cannot carry an unrelated claim elsewhere on the line.
        for sentence in re.split(r'(?<=[.;!?])\s+', line):
            if not SUPERLATIVE.search(sentence): continue
            backed = any((int(a), int(b)) in valid_pairs for a, b in re.findall(r'\b(\d+) of (\d+)\b', sentence)) \
                or any(int(n) in valid_all for n in re.findall(r'\ball (\d+)\b', sentence))
            if not backed:
                fail(f'README:{i} makes a comparative claim with no counted result beside it: {sentence.strip()[:160]}')

def check_unbacked_text(label, text):
    if MULTIPLIER.search(text) or SUPERLATIVE.search(text) or PERCENT.search(text):
        fail(f'{label} makes a speed claim no report backs ({text.strip()[:200]!r}). '
             'Point readers to the README benchmarks instead of quoting numbers.')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo-description', help='the GitHub repository description (CI passes it from the API)')
    args = ap.parse_args()
    readme = read('README.md')
    run = results_dir(readme)
    if run is not None:
        check_versions(readme, run)
        counts, speedups = check_tables(readme, run)
        pairs, alls = check_counts(readme, counts)
        check_readme_prose(readme, speedups, pairs, alls)
    description = ET.parse(os.path.join(ROOT, 'src/AiDotNet.Tensors/AiDotNet.Tensors.csproj')).getroot().find('.//Description')
    check_unbacked_text('The NuGet <Description>', description.text if description is not None and description.text else '')
    if args.repo_description is not None:
        check_unbacked_text('The GitHub repository description', args.repo_description)
    for rel in HISTORICAL:
        if os.path.isfile(os.path.join(ROOT, rel)) and 'historical' not in '\n'.join(read(rel).splitlines()[:6]).lower():
            fail(f'{rel} is an old benchmark write-up; mark it historical in its first lines and point to the README.')
    if errors:
        print(f'{len(errors)} unbacked or stale performance claim(s):')
        for e in errors: print(' -', e)
        return 1
    print('Every published performance claim matches the committed benchmark reports.')
    return 0

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(main())
