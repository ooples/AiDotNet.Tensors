"""README benchmark tables from BenchmarkDotNet CSV reports: python make_readme_tables.py <vs-all results dir> <linalg results dir>

Both arguments can be the same folder (Results/<run>/ holds the reports of both suites)."""
import csv, glob, os, re, sys
from collections import defaultdict

def us(v):
    v = v.replace(',', '').strip()
    m = re.match(r'([\d.]+)\s*(ns|us|μs|µs|ms|s)', v)
    if not m: return None
    return float(m.group(1)) * {'ns': 1e-3, 'us': 1, 'μs': 1, 'µs': 1, 'ms': 1e3, 's': 1e6}[m.group(2)]

def fmt(x):
    if x is None: return '—'
    if x < 1: return f'{x * 1000:,.0f} ns'
    if x >= 1000: return f'{x / 1000:,.2f} ms'
    return f'{x:,.0f} µs' if x >= 10 else f'{x:,.1f} µs'

def sp(ai, other):
    if ai is None or other is None: return '—'
    r = other / ai
    return f'**{r:.2f}×**' if r >= 1 else f'{r:.2f}× (slower)'

LIBS = ['AiDotNet', 'TorchSharp', 'MlNet', 'TensorFlow', 'NumSharp', 'MathNet', 'TensorPrimitives', 'RawTensorPrimitives']
def split(method):
    m = method.strip("'")
    if ' - ' in m:
        op, lib = m.rsplit(' - ', 1); return lib.strip(), op.strip()
    for lib in LIBS:
        if m.startswith(lib + '_'): return lib, m[len(lib) + 1:]
    return None, m

ALIASES = {'tensoradd': 'add', 'tensormultiply': 'multiply', 'tensorsubtract': 'subtract', 'tensordivide': 'divide',
           'tensorexp': 'exp', 'tensorlog': 'log', 'tensorsqrt': 'sqrt', 'tensorabs': 'abs', 'tensorsum': 'sum',
           'reducesum': 'sum', 'tensormean': 'mean', 'reducemean': 'mean', 'tensormaxvalue': 'max', 'tensorminvalue': 'min',
           'tensormatmul': 'matmul'}
def norm(op): o = op.lower(); return ALIASES.get(o, o)

NAMES = {'add': 'Add', 'multiply': 'Multiply', 'subtract': 'Subtract', 'divide': 'Divide', 'exp': 'Exp', 'log': 'Log',
         'sqrt': 'Sqrt', 'abs': 'Abs', 'sum': 'Sum', 'mean': 'Mean', 'max': 'Max', 'min': 'Min', 'matmul': 'MatMul',
         'relu': 'ReLU', 'sigmoid': 'Sigmoid', 'tanh': 'Tanh', 'gelu': 'GELU', 'leakyrelu': 'LeakyReLU', 'mish': 'Mish',
         'softmax': 'Softmax', 'logsoftmax': 'LogSoftmax', 'layernorm': 'LayerNorm', 'groupnorm': 'GroupNorm',
         'batchnorm': 'BatchNorm', 'conv2d': 'Conv2D', 'maxpool2d': 'MaxPool2D', 'attentionqkt': 'Attention Q·Kᵀ',
         'sigmoidbackward': 'Sigmoid backward', 'tanhbackward': 'Tanh backward'}
def name(op):
    dbl = op.endswith('_double')
    base = op[:-7] if dbl else op
    return NAMES.get(base, base) + (' (double)' if dbl else '')

SHAPES = {'matmul_double': '256×256', 'softmax': '512×1024', 'logsoftmax': '512×1024', 'softmax_double': '512×1024',
          'layernorm': '32768×64', 'groupnorm': '32×64×32×32, 32 groups', 'batchnorm': '32×64×32×32',
          'conv2d': '1×16×64×64 → 32, 3×3', 'conv2d_double': '1×3×32×32 → 16, 3×3', 'maxpool2d': '1×32×64×64, 3×3 / 2', 'attentionqkt': '512×64 · 64×512'}
def shape(op, size, suite):
    if size and size != '?':
        n = int(size)
        if op == 'matmul': return f'{n}×{n}'
        return f'{n // 1000}K' if n < 1_000_000 else f'{n // 1_000_000}M'
    if suite == 'TensorFlowCpuComparisonBenchmarks' and op == 'conv2d': return '1×16×64×64 → 32, 3×3'
    return SHAPES.get(op, '1M')

def load(path):
    rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
    data = defaultdict(dict)  # (op, size) -> {(lib, job): us}
    for r in rows:
        lib, op = split(r['Method'])
        if lib is None: continue
        size = r.get('size') or r.get('N') or ''
        data[(norm(op), size)][(lib, r.get('Job', ''))] = us(r['Mean'])
    return data

def torch_table(path):
    data = load(path)
    out = ['| Operation | Shape | AiDotNet steady | TorchSharp steady | Speedup | AiDotNet cold call | TorchSharp cold call | Speedup |',
           '|---|---|--:|--:|--:|--:|--:|--:|']
    wins = losses = 0
    for (op, size), d in sorted(data.items(), key=lambda kv: (name(kv[0][0]), kv[0][1])):
        if ('TorchSharp', 'SteadyState') not in d or ('AiDotNet', 'SteadyState') not in d: continue
        a_s, t_s = d.get(('AiDotNet', 'SteadyState')), d.get(('TorchSharp', 'SteadyState'))
        a_c, t_c = d.get(('AiDotNet', 'ColdCall')), d.get(('TorchSharp', 'ColdCall'))
        for a, t in ((a_s, t_s), (a_c, t_c)):
            if a is not None and t is not None:
                if a <= t: wins += 1
                else: losses += 1
        out.append(f'| {name(op)} | {shape(op, size, "Torch")} | {fmt(a_s)} | {fmt(t_s)} | {sp(a_s, t_s)} | {fmt(a_c)} | {fmt(t_c)} | {sp(a_c, t_c)} |')
    return '\n'.join(out), wins, losses

def simple_table(path, other, label, suite):
    data = load(path)
    out = [f'| Operation | Shape | AiDotNet | {label} | Speedup |', '|---|---|--:|--:|--:|']
    wins = losses = 0
    for (op, size), d in sorted(data.items(), key=lambda kv: (name(kv[0][0]), kv[0][1])):
        a = next((v for (l, j), v in d.items() if l == 'AiDotNet'), None)
        t = next((v for (l, j), v in d.items() if l == other), None)
        if a is None or t is None: continue
        if a <= t: wins += 1
        else: losses += 1
        out.append(f'| {name(op)} | {shape(op, size, suite)} | {fmt(a)} | {fmt(t)} | {sp(a, t)} |')
    return '\n'.join(out), wins, losses

def linalg_table(path):
    rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
    data = defaultdict(dict)
    for r in rows:
        lib, op = split(r['Method'])
        if lib is None: continue
        data[(op, r.get('N') or r.get('Size') or r.get('size') or '')][lib] = us(r['Mean'])
    out = ['| Operation | N | AiDotNet | NumSharp 0.70.0 | MathNet 5.0.0 | Speedup vs NumSharp | Speedup vs MathNet |',
           '|---|--:|--:|--:|--:|--:|--:|']
    for (op, n), d in sorted(data.items(), key=lambda kv: (kv[0][0], int(kv[0][1]) if kv[0][1].isdigit() else 0)):
        a = d.get('AiDotNet')
        if a is None or ('NumSharp' not in d and 'MathNet' not in d): continue
        out.append(f"| {op} | {n} | {fmt(a)} | {fmt(d.get('NumSharp'))} | {fmt(d.get('MathNet'))} | {sp(a, d.get('NumSharp'))} | {sp(a, d.get('MathNet'))} |")
    return '\n'.join(out)

if __name__ == '__main__':
    vs, la = sys.argv[1], sys.argv[2]
    t, w, l = torch_table(glob.glob(os.path.join(vs, '*TorchSharp*report.csv'))[0])
    print(f'### vs TorchSharp — {w} wins, {l} losses\n\n{t}\n')
    t, w, l = simple_table(glob.glob(os.path.join(vs, '*MlNet*report.csv'))[0], 'MlNet', 'ML.NET 5.0.0', 'MlNet')
    print(f'### vs ML.NET — {w} wins, {l} losses\n\n{t}\n')
    t, w, l = simple_table(glob.glob(os.path.join(vs, '*TensorFlow*report.csv'))[0], 'TensorFlow', 'TensorFlow.NET 0.150.0', 'TensorFlowCpuComparisonBenchmarks')
    print(f'### vs TensorFlow.NET — {w} wins, {l} losses\n\n{t}\n')
    for f in sorted(glob.glob(os.path.join(la, '*report.csv'))):
        if not re.search(r'(LinearAlgebra|SmallMatrix|ElementWise)Benchmarks', f): continue
        print(f'### {os.path.basename(f).split(".")[-2].replace("-report", "")}\n\n{linalg_table(f)}\n')
