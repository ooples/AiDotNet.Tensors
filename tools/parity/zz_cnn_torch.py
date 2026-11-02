# SCRATCH (probe branch only): torch reference for ZzCnnOpsBench (batch 64), min-of-3 per round, 15 rounds.
import sys, time, torch
import torch.nn.functional as F
threads = int(sys.argv[1]) if len(sys.argv) > 1 else torch.get_num_threads()
torch.set_num_threads(threads)
torch.manual_seed(9)
b = 64

def timed(f):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 0.3: f()
    r = []
    for _ in range(15):
        best = 1e9
        for _ in range(3):
            t = time.perf_counter(); f(); best = min(best, time.perf_counter() - t)
        r.append(best * 1e3)
    r.sort()
    return r[0], r[len(r) // 2]

def p(*s): return torch.randn(*s, requires_grad=True)
x1 = torch.randn(b, 1, 28, 28); w1 = p(32, 1, 3, 3); b1 = p(32)
x2 = p(b, 32, 28, 28)
x3 = p(b, 32, 14, 14); w3 = p(64, 32, 3, 3); b3 = p(64)
x4 = p(b, 64, 14, 14)
x5 = p(b, 3136); w5 = p(128, 3136); b5 = p(128)
x6 = p(b, 128); w6 = p(10, 128); b6 = p(10)
layers = [
    ("conv1 1->32 28", lambda: F.relu(F.conv2d(x1, w1, b1, padding=1)), [w1, b1]),
    ("pool1 32x28", lambda: F.max_pool2d(x2, 2), [x2]),
    ("conv2 32->64 14", lambda: F.relu(F.conv2d(x3, w3, b3, padding=1)), [x3, w3, b3]),
    ("pool2 64x14", lambda: F.max_pool2d(x4, 2), [x4]),
    ("fc1 3136->128", lambda: F.relu(F.linear(x5, w5, b5)), [x5, w5, b5]),
    ("fc2 128->10", lambda: F.linear(x6, w6, b6), [x6, w6, b6]),
]
for name, fwd, src in layers:
    dy = torch.randn_like(fwd())
    with torch.no_grad():
        f = timed(fwd)
    fb = timed(lambda: torch.autograd.grad(fwd(), src, dy))
    print(f"ZTC threads {threads} {name:<16} fwd {f[0]:.3f}/{f[1]:.3f}  fwd+bwd {fb[0]:.3f}/{fb[1]:.3f}  bwd~ {fb[0] - f[0]:.3f}", flush=True)
