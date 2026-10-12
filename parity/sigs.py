"""Resolve the tensor-level IEngine overload behind each in-place parity gap."""
import os, re, glob, json, collections, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src = ''
for p in sorted(glob.glob(ROOT + '/src/AiDotNet.Tensors/Engines/IEngine*.cs')):
    src += open(p, encoding='utf-8').read() + '\n'
# strip comments
src = re.sub(r'//.*', '', src)
src = re.sub(r'/\*.*?\*/', '', src, flags=re.S)
DECL = re.compile(r'^\s*(?P<ret>[\w<>\[\],. ?]+?)\s+(?P<name>[A-Z]\w*)\s*(?P<gen><[^()]*?>)?\s*\((?P<params>[^;{]*?)\)\s*(where[^;{]*)?;', re.M)
methods = collections.defaultdict(list)
for m in DECL.finditer(src):
    ret = m.group('ret').strip()
    if ret.startswith(('return', 'new', 'throw')): continue
    methods[m.group('name')].append({'ret': ret, 'gen': (m.group('gen') or '').strip(), 'params': ' '.join(m.group('params').split())})

def split_params(ps):
    out, depth, cur = [], 0, ''
    for ch in ps:
        if ch in '<([': depth += 1
        if ch in '>)]': depth -= 1
        if ch == ',' and depth == 0: out.append(cur.strip()); cur = ''
        else: cur += ch
    if cur.strip(): out.append(cur.strip())
    return out

def tensor_overloads(name):
    res = []
    for o in methods.get(name, []):
        if not o['ret'].startswith('Tensor<'): continue
        ps = split_params(o['params'])
        selfidx = next((i for i, p in enumerate(ps) if re.match(r'(ReadOnly)?Tensor<', p)), None)
        if selfidx is None: continue
        res.append({**o, 'plist': ps, 'self': selfidx})
    return res

c = json.load(open(ROOT + '/parity/coverage.json', encoding='utf-8'))['entries']
gaps = [k for k, e in c.items() if e.get('status') == 'gap']
names = collections.defaultdict(list)
for k in gaps:
    cat, n = k.split(':', 1); names[n].append(cat)
out = {}
for n in sorted(names):
    if not n.endswith('_'): continue
    syms = sorted({c[f'{cat}:{n[:-1]}']['symbol'] for cat in ('Tensor', 'aten', 'torch', 'nn.functional') if c.get(f'{cat}:{n[:-1]}', {}).get('status') == 'mapped'})
    base = next((s.split('::')[1] for s in syms if s.startswith('AiDotNet.Tensors.Engines.IEngine::')), None)
    ov = []
    if base:
        ov = tensor_overloads(base) or tensor_overloads('Tensor' + base)
        chosen = base if tensor_overloads(base) else 'Tensor' + base
    out[n] = {'cats': names[n], 'syms': syms, 'base': base, 'engineName': chosen if ov else None, 'overloads': ov}
json.dump(out, open(os.path.join(ROOT, 'parity', 'build', 'sigs.json'), 'w'), indent=1)
for n, d in out.items():
    if d['overloads']:
        for o in d['overloads']:
            print(f"{n:22s} {d['engineName']}{o['gen']}({o['params']}) -> {o['ret']}  self#{o['self']}")
    else:
        print(f"{n:22s} ** UNRESOLVED ** syms={d['syms']}")
