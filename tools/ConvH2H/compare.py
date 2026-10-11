# Joins the torch and AiDotNet per-op timings: per-shape ratios and the count-weighted MobileNetV3 totals.
import json, sys
torch_rows = json.load(open(sys.argv[1])); ours = json.load(open(sys.argv[2]))
key = lambda r: (r['C'], r['H'], r['O'], r['K'], r['S'], r['G'])
t = {key(r): r for r in torch_rows}
tot = {'torch': [0, 0, 0], 'ours': [0, 0, 0]}
print(f"{'C':>4}{'HW':>4}{'O':>5}{'K':>2}{'S':>2}{'G':>5}{'n':>3} | {'fwd t/o us':>17} {'x':>5} | {'dIn t/o us':>17} {'x':>5} | {'dW t/o us':>17} {'x':>5}")
for r in ours:
    tr = t[key(r)]; n = r['count']
    for i, ph in enumerate(['fwd', 'din', 'dw']):
        tot['torch'][i] += n * tr[ph]; tot['ours'][i] += n * r[ph]
    print(f"{r['C']:>4}{r['H']:>4}{r['O']:>5}{r['K']:>2}{r['S']:>2}{r['G']:>5}{n:>3} | "
          + " | ".join(f"{tr[ph]:8.1f}/{r[ph]:8.1f} {r[ph] / tr[ph]:5.2f}" for ph in ['fwd', 'din', 'dw']))
names = ['forward', 'input grad', 'weight grad']
print('\nweighted totals over the network (us):')
for i in range(3):
    print(f"  {names[i]:12s} torch {tot['torch'][i]:9.1f}  ours {tot['ours'][i]:9.1f}  ratio {tot['ours'][i] / tot['torch'][i]:5.2f}x")
st, so = sum(tot['torch']), sum(tot['ours'])
print(f"  {'all conv':12s} torch {st:9.1f}  ours {so:9.1f}  ratio {so / st:5.2f}x")
