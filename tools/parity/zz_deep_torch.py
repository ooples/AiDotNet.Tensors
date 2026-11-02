# SCRATCH (probe branch only): torch reference for ZzDeepConvBench, same shapes, min-of-3 per round, 15 rounds.
import time, torch
torch.set_num_threads(4)
torch.manual_seed(5)
convs = [(512, 512, 4, 1), (256, 256, 8, 1), (256, 512, 8, 2), (128, 256, 16, 2), (64, 128, 32, 2), (128, 128, 16, 1),
         (3, 64, 224, 2), (64, 64, 56, 1), (64, 64, 32, 1)]
b = 8
for cin, cout, hw, s in convs:
    k = 7 if cin == 3 else 3
    p = 3 if cin == 3 else 1
    x = torch.randn(b, cin, hw, hw, requires_grad=True)
    w = torch.randn(cout, cin, k, k, requires_grad=True)
    y = torch.nn.functional.conv2d(x, w, stride=s, padding=p)
    dy = torch.randn_like(y)
    ops = {
        'fwd': lambda: torch.nn.functional.conv2d(x.detach(), w.detach(), stride=s, padding=p),
        'dgrad': lambda: torch.ops.aten.convolution_backward(dy, x, w, None, [s, s], [p, p], [1, 1], False, [0, 0], 1, [True, False, False]),
        'wgrad': lambda: torch.ops.aten.convolution_backward(dy, x, w, None, [s, s], [p, p], [1, 1], False, [0, 0], 1, [False, True, False]),
    }
    for name, f in ops.items():
        with torch.no_grad():
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 0.25: f()
            r = []
            for _ in range(15):
                best = 1e9
                for _ in range(3):
                    t = time.perf_counter(); f(); best = min(best, time.perf_counter() - t)
                r.append(best * 1e3)
        r.sort()
        print(f"ZT {cin}->{cout} {hw}x{hw} s{s} {name:<5}  torch {r[0]:.3f}/{r[len(r)//2]:.3f}", flush=True)
