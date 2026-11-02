# SCRATCH (probe branch only): torch reference for ZzMlpStepBench.
import sys, time, torch
import torch.nn.functional as F
threads = int(sys.argv[1]) if len(sys.argv) > 1 else torch.get_num_threads()
torch.set_num_threads(threads)
torch.manual_seed(3)

def med(v): v = sorted(v); return v[len(v) // 2]

def timed(f):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 0.25: f()
    r = []
    for _ in range(25):
        best = 1e9
        for _ in range(5):
            t = time.perf_counter(); f(); best = min(best, time.perf_counter() - t)
        r.append(best * 1e3)
    r.sort()
    return r[0], r[len(r) // 2]

model = torch.nn.Sequential(torch.nn.Linear(784, 512), torch.nn.ReLU(), torch.nn.Linear(512, 256), torch.nn.ReLU(), torch.nn.Linear(256, 10))
opt = torch.optim.SGD(model.parameters(), lr=0.01)
x = torch.randn(128, 784); y = torch.randn(128, 10)
f, b, o, s = [], [], [], []
for step in range(1200):
    t0 = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    loss = F.mse_loss(model(x), y)
    t1 = time.perf_counter()
    loss.backward()
    t2 = time.perf_counter()
    opt.step()
    t3 = time.perf_counter()
    if step >= 200:
        f.append((t1 - t0) * 1e3); b.append((t2 - t1) * 1e3); o.append((t3 - t2) * 1e3); s.append((t3 - t0) * 1e3)
print(f"ZTS threads {threads} step {med(s):.4f} fwd {med(f):.4f} bwd {med(b):.4f} opt {med(o):.4f}", flush=True)

a512 = torch.relu(model[0](x)).detach(); a256 = torch.relu(model[2](a512)).detach()
g512 = torch.randn(128, 512); g256 = torch.randn(128, 256)
r512 = timed(lambda: torch.ops.aten.threshold_backward(g512, a512, 0))
r256 = timed(lambda: torch.ops.aten.threshold_backward(g256, a256, 0))
out10 = model[4](a256).detach().requires_grad_(True)
mse = timed(lambda: torch.autograd.grad(F.mse_loss(out10, y), [out10]))
print(f"ZTS threads {threads} pieces relu512 {r512[0]:.4f}/{r512[1]:.4f} relu256 {r256[0]:.4f}/{r256[1]:.4f} mse-fwd+bwd {mse[0]:.4f}/{mse[1]:.4f}", flush=True)
a512g = a512.clone().requires_grad_(True); a256g = a256.clone().requires_grad_(True)
for name, fwd, srcs in [
    ("L1", lambda: torch.relu(model[0](x)), list(model[0].parameters())),
    ("L2", lambda: torch.relu(model[2](a512g)), [a512g] + list(model[2].parameters())),
    ("L3", lambda: model[4](a256g), [a256g] + list(model[4].parameters())),
]:
    dy = torch.randn_like(fwd())
    with torch.no_grad():
        fw = timed(fwd)
    fb = timed(lambda: torch.autograd.grad(fwd(), srcs, dy))
    print(f"ZTS threads {threads} {name} fwd {fw[0]:.4f}/{fw[1]:.4f} fwd+bwd {fb[0]:.4f}/{fb[1]:.4f} bwd~ {fb[0] - fw[0]:.4f}", flush=True)
