# PyTorch side of the #639 per-op head-to-head: forward, input-gradient and weight-gradient time of every MobileNetV3
# conv shape (mbv3_convs.json), median of `reps` timed calls after warmup, at a fixed thread count.
import json, sys, time, statistics, torch
import torch.nn.functional as F
threads, reps = int(sys.argv[1]), int(sys.argv[2])
torch.set_num_threads(threads)
torch.manual_seed(0)
def med(fn):
    for _ in range(10): fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1e6)
    return statistics.median(ts)
rows = []
for s in json.load(open(sys.argv[3])):
    x = torch.randn(s['N'], s['C'], s['H'], s['W'])
    w = torch.randn(s['O'], s['C'] // s['G'], s['K'], s['K'])
    y = F.conv2d(x, w, stride=s['S'], padding=s['P'], groups=s['G'])
    gy = torch.randn_like(y)
    fwd = med(lambda: F.conv2d(x, w, stride=s['S'], padding=s['P'], groups=s['G']))
    din = med(lambda: torch.nn.grad.conv2d_input(x.shape, w, gy, stride=s['S'], padding=s['P'], groups=s['G']))
    dw = med(lambda: torch.nn.grad.conv2d_weight(x, w.shape, gy, stride=s['S'], padding=s['P'], groups=s['G']))
    rows.append(dict(s, fwd=fwd, din=din, dw=dw))
json.dump(rows, open(sys.argv[4], 'w'), indent=1)
print('torch', torch.__version__, 'threads', threads, 'shapes', len(rows))
