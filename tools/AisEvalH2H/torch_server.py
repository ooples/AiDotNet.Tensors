# Issue #436: the PyTorch side of driver.py's paired head-to-head. One JSON command per stdin line, one JSON reply per
# line on stdout (setup / dump / round / quit; see driver.py). Each setup builds the model in PyTorch's own layout from
# the driver's float32 files and prepares one variant: a thread count plus a mode (eager, TorchScript freeze +
# optimize_for_inference, or torch.compile with the Inductor C++ backend in its default or max-autotune mode).
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)


def load(directory, name, *shape):
    data = np.fromfile(os.path.join(directory, name + '.bin'), dtype=np.float32)
    assert data.size == int(np.prod(shape)), (name, data.size, shape)
    return torch.from_numpy(data.reshape(shape).copy())


class Mlp(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = torch.nn.ModuleList([torch.nn.Linear(i, o) for i, o in ((784, 512), (512, 128), (128, 10))])
        for n, fc in enumerate(self.fc, 1):
            fc.weight.copy_(load(d, f'w{n}', fc.in_features, fc.out_features).t())   # driver stores [in, out]
            fc.bias.copy_(load(d, f'b{n}', fc.out_features))

    def forward(self, x):
        return self.fc[2](F.relu(self.fc[1](F.relu(self.fc[0](x)))))


class MhaSdpa(torch.nn.Module):
    """Projections as x @ W (W stored [in, out]), heads split, scaled_dot_product_attention, output projection."""

    def __init__(self, d, transpose):
        super().__init__()
        self.heads = 4
        for name in ('wq', 'wk', 'wv', 'wo'):
            w = load(d, name, 64, 64)
            self.register_buffer(name, (w.t() if transpose else w).contiguous())

    def forward(self, x):
        b, s, m = x.shape
        h = self.heads

        def split(t):
            return t.view(b, s, h, m // h).transpose(1, 2)

        a = F.scaled_dot_product_attention(split(x @ self.wq), split(x @ self.wk), split(x @ self.wv))
        return a.transpose(1, 2).reshape(b, s, m) @ self.wo


class MhaModule(torch.nn.Module):
    """nn.MultiheadAttention (batch_first, no bias): its eval-mode fast path (native multi-head attention)."""

    def __init__(self, d, transpose):
        super().__init__()
        self.mha = torch.nn.MultiheadAttention(64, 4, bias=False, batch_first=True)
        ws = [load(d, n, 64, 64) for n in ('wq', 'wk', 'wv', 'wo')]
        # nn.MultiheadAttention computes x @ W^T with W stored [out, in].
        ws = [w if transpose else w.t() for w in ws]
        self.mha.in_proj_weight.copy_(torch.cat(ws[:3], 0))
        self.mha.out_proj.weight.copy_(ws[3])
        self.mha.eval()

    def forward(self, x):
        return self.mha(x, x, x, need_weights=False)[0]


class Lstm(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lstm = torch.nn.LSTM(32, 64, batch_first=True, bias=False)
        self.lstm.weight_ih_l0.copy_(load(d, 'w_ih', 256, 32))
        self.lstm.weight_hh_l0.copy_(load(d, 'w_hh', 256, 64))
        self.lstm.eval()

    def forward(self, x):
        return self.lstm(x)[0][:, -1]   # last step, as AiDotNet's returnSequences=false


SHAPES = {'mlp': (128, 784), 'mha': (128, 32, 64), 'lstm': (128, 32, 32)}


def build(model, directory, impl, transpose):
    if model == 'mlp':
        return Mlp(directory)
    if model == 'mha':
        return MhaModule(directory, transpose) if impl == 'nnmha' else MhaSdpa(directory, transpose)
    return Lstm(directory)


def prepare(module, x, mode):
    module.eval()
    if mode == 'eager':
        return module
    if mode == 'jit':
        traced = torch.jit.trace(module, (x,), check_trace=False)
        return torch.jit.optimize_for_inference(torch.jit.freeze(traced))
    if mode in ('compile', 'compile-max-autotune'):
        return torch.compile(module, mode=None if mode == 'compile' else 'max-autotune', dynamic=False)
    raise ValueError(mode)


def main():
    fn = None
    x = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        cmd = json.loads(line)
        try:
            if cmd['cmd'] == 'setup':
                torch.set_num_threads(int(cmd['threads']))
                d = cmd['dir']
                x = load(d, 'x', *SHAPES[cmd['model']])
                module = build(cmd['model'], d, cmd.get('impl', ''), bool(cmd.get('transpose', False)))
                t0 = time.perf_counter()
                compiled = prepare(module, x, cmd['mode'])
                with torch.inference_mode():
                    for _ in range(3):
                        compiled(x)
                fn = compiled
                reply = {'ok': True, 'prepare_s': time.perf_counter() - t0, 'threads': torch.get_num_threads()}
            elif cmd['cmd'] == 'dump':
                with torch.inference_mode():
                    out = fn(x).contiguous().numpy().astype(np.float32)
                out.tofile(cmd['path'])
                reply = {'ok': True, 'count': int(out.size)}
            elif cmd['cmd'] == 'round':
                iters = int(cmd['iters'])
                times = []
                with torch.inference_mode():
                    for _ in range(3):
                        fn(x)
                    for _ in range(iters):
                        t0 = time.perf_counter()
                        fn(x)
                        times.append((time.perf_counter() - t0) * 1000.0)
                times.sort()
                median = times[iters // 2] if iters % 2 else (times[iters // 2 - 1] + times[iters // 2]) / 2
                p95 = times[min(iters - 1, -(-95 * iters // 100) - 1)]
                reply = {'median_ms': median, 'p95_ms': p95, 'min_ms': times[0]}
            elif cmd['cmd'] == 'quit':
                return
            else:
                reply = {'error': 'unknown command ' + cmd['cmd']}
        except Exception as exception:   # report and stay alive: the driver drops the variant
            reply = {'error': f'{type(exception).__name__}: {exception}'[:2000]}
        sys.stdout.write(json.dumps(reply) + '\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
