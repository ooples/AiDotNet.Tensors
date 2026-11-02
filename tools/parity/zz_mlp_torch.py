# SCRATCH (probe branch only): torch reference for ZzMlpGemmBench (batch 128), min-of-5 per round, 25 rounds.
import sys, time, torch
threads = int(sys.argv[1]) if len(sys.argv) > 1 else torch.get_num_threads()
torch.set_num_threads(threads)
torch.manual_seed(7)
m = 128

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

with torch.no_grad():
    for k, n in [(784, 512), (512, 256), (256, 10)]:
        x = torch.randn(m, k); w = torch.randn(n, k); b = torch.randn(n); dy = torch.randn(m, n)
        f = timed(lambda: torch.relu(torch.nn.functional.linear(x, w, b)))
        gw = timed(lambda: dy.t().mm(x))      # dW [n, k] as autograd forms it for nn.Linear
        gx = timed(lambda: dy.mm(w))          # dX [m, k]
        print(f"ZTM threads {threads} {k}->{n} fwd {f[0]:.4f}/{f[1]:.4f}  dW {gw[0]:.4f}/{gw[1]:.4f}  dX {gx[0]:.4f}/{gx[1]:.4f}", flush=True)
