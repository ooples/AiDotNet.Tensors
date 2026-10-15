# Issue #436: AiDotNet's fused CPU inference primitives (MlpForward, MultiHeadAttentionForward, LstmSequenceForward)
# against real PyTorch at its best CPU settings, at the AIsEval shapes.
#
#   1. Writes each model's input and weights as float32 files, shared by both sides, and a float64 NumPy reference.
#   2. Checks both sides' outputs against the reference: a variant that is fast but wrong never competes.
#   3. Searches PyTorch's settings per model: thread counts with eager, then TorchScript (freeze +
#      optimize_for_inference) and torch.compile (Inductor C++, default and max-autotune) at the best eager thread
#      counts; for MHA both scaled_dot_product_attention and nn.MultiheadAttention's fast path.
#   4. Paired, interleaved rounds of AiDotNet against PyTorch's best variant: A B B A per round, each a full timing
#      window in its own long-lived process, so machine drift hits both sides alike; the ratio is per round.
#
# Usage: python driver.py --aidotnet-dir <Tensors worktree> [--out results.json] [--rounds 10] [--iters 200]
# The PyTorch side runs inside the MSVC environment (vcvars64) so Inductor's C++ backend can compile.
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VCVARS = r'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat'


# ---------------------------------------------------------------- data and float64 references

def write(directory, name, array):
    np.ascontiguousarray(array, dtype=np.float32).tofile(os.path.join(directory, name + '.bin'))


def make_data(model, directory, seed=436):
    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(seed)

    def u(*shape, scale=1.0):
        return (rng.uniform(-1, 1, size=shape) * scale).astype(np.float32)

    if model == 'mlp':
        t = {'x': u(128, 784), 'w1': u(784, 512, scale=784 ** -0.5), 'b1': u(512, scale=0.1),
             'w2': u(512, 128, scale=512 ** -0.5), 'b2': u(128, scale=0.1),
             'w3': u(128, 10, scale=128 ** -0.5), 'b3': u(10, scale=0.1)}
    elif model == 'mha':
        t = {'x': u(128, 32, 64), **{n: u(64, 64, scale=0.125) for n in ('wq', 'wk', 'wv', 'wo')}}
    else:
        t = {'x': u(128, 32, 32), 'w_ih': u(256, 32, scale=0.125), 'w_hh': u(256, 64, scale=0.125)}
    for name, array in t.items():
        write(directory, name, array)
    return {k: v.astype(np.float64) for k, v in t.items()}


def reference(model, t, mha_transpose=False):
    if model == 'mlp':
        h = np.maximum(t['x'] @ t['w1'] + t['b1'], 0)
        h = np.maximum(h @ t['w2'] + t['b2'], 0)
        return h @ t['w3'] + t['b3']
    if model == 'mha':
        x = t['x']
        b, s, m = x.shape
        w = {n: (t[n].T if mha_transpose else t[n]) for n in ('wq', 'wk', 'wv', 'wo')}

        def split(a):
            return a.reshape(b, s, 4, m // 4).transpose(0, 2, 1, 3)

        q, k, v = split(x @ w['wq']), split(x @ w['wk']), split(x @ w['wv'])
        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(m // 4)
        scores = np.exp(scores - scores.max(-1, keepdims=True))
        scores /= scores.sum(-1, keepdims=True)
        return (scores @ v).transpose(0, 2, 1, 3).reshape(b, s, m) @ w['wo']
    x = t['x']
    hidden = 64
    h = np.zeros((x.shape[0], hidden))
    c = np.zeros_like(h)
    sig = lambda z: 1 / (1 + np.exp(-z))
    for step in range(x.shape[1]):
        g = x[:, step] @ t['w_ih'].T + h @ t['w_hh'].T   # PyTorch gate order i, f, g, o
        i, f, gg, o = (g[:, n * hidden:(n + 1) * hidden] for n in range(4))
        c = sig(f) * c + sig(i) * np.tanh(gg)
        h = sig(o) * np.tanh(c)
    return h


def relative_error(expected, path):
    actual = np.fromfile(path, dtype=np.float32).astype(np.float64)
    if actual.size != expected.size:
        return float('inf')
    return float(np.abs(actual - expected.ravel()).max() / max(np.abs(expected).max(), 1e-30))


# ---------------------------------------------------------------- the two servers

class Server:
    def __init__(self, name, argv, env=None, json_commands=False):
        self.name = name
        self.json_commands = json_commands
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, bufsize=1, env=env)

    def ask(self, command):
        line = json.dumps(command) if self.json_commands else command
        self.proc.stdin.write(line + '\n')
        self.proc.stdin.flush()
        while True:
            reply = self.proc.stdout.readline()
            if not reply:
                raise RuntimeError(f'{self.name} exited')
            reply = reply.strip()
            if reply.startswith('{'):   # skip any banner the process prints
                return json.loads(reply)

    def close(self):
        try:
            self.ask({'cmd': 'quit'} if self.json_commands else 'quit')
        except Exception:
            pass
        self.proc.wait(timeout=30)


def msvc_env():
    out = subprocess.run(f'cmd /s /c ""{VCVARS}" >nul && set"', capture_output=True, text=True, shell=True).stdout
    env = dict(os.environ)
    for line in out.splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            env[k] = v
    return env


# ---------------------------------------------------------------- search and paired rounds

def search(torch_srv, model, directory, expected, transpose, threads, iters, log):
    def try_variant(variant):
        setup = torch_srv.ask({'cmd': 'setup', 'model': model, 'dir': directory, 'transpose': transpose, **variant})
        if 'error' in setup:
            log(f'    {variant}: setup failed: {setup["error"][:200]}')
            return None
        path = os.path.join(directory, 'torch_out.bin')
        torch_srv.ask({'cmd': 'dump', 'path': path})
        error = relative_error(expected, path)
        if not error <= 1e-3:
            log(f'    {variant}: WRONG (relative error {error:.3g}); excluded')
            return None
        median = min(torch_srv.ask({'cmd': 'round', 'iters': iters})['median_ms'] for _ in range(2))
        log(f'    {variant}: {median:.4f} ms (prepare {setup["prepare_s"]:.1f} s, error {error:.2g})')
        return median

    results = []
    impls = ['sdpa', 'nnmha'] if model == 'mha' else ['']
    for impl in impls:
        for t in threads:
            m = try_variant({'mode': 'eager', 'threads': t, 'impl': impl})
            if m is not None:
                results.append((m, {'mode': 'eager', 'threads': t, 'impl': impl}))
    best_threads = []
    for _, v in sorted(results, key=lambda r: r[0]):
        if v['threads'] not in best_threads:
            best_threads.append(v['threads'])
    for impl in impls:
        for mode in ('jit', 'compile', 'compile-max-autotune'):
            for t in best_threads[:2]:
                v = {'mode': mode, 'threads': t, 'impl': impl}
                m = try_variant(v)
                if m is not None:
                    results.append((m, v))
    results.sort(key=lambda r: r[0])
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--aidotnet-dir', required=True)
    ap.add_argument('--data', default=r'E:\torchbench\aiseval-h2h')
    ap.add_argument('--out', default=os.path.join(HERE, 'results.json'))
    ap.add_argument('--models', default='mlp,mha,lstm')
    ap.add_argument('--rounds', type=int, default=10)
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--search-iters', type=int, default=100)
    ap.add_argument('--threads', default='1,2,4,8,16,32,64,128')
    ap.add_argument('--pause', type=float, default=0.3, help='seconds between switching sides (idle OpenMP spin ends)')
    args = ap.parse_args()
    threads = [int(t) for t in args.threads.split(',') if int(t) <= os.cpu_count()]

    def log(message):
        print(message, flush=True)

    aidn = Server('aidotnet', ['dotnet', 'run', '--project',
                               os.path.join(args.aidotnet_dir, 'tests', 'AiDotNet.Tensors.Benchmarks'),
                               '-c', 'Release', '--no-build', '--', '--aiseval-h2h-serve'])
    torch_srv = Server('torch', [sys.executable, os.path.join(HERE, 'torch_server.py')], env=msvc_env(),
                       json_commands=True)
    summary = {'host_cpus': os.cpu_count(), 'rounds': args.rounds, 'iters': args.iters, 'models': {}}
    try:
        import torch
        summary['torch'] = torch.__version__
        for model in args.models.split(','):
            directory = os.path.join(args.data, model)
            data = make_data(model, directory)
            log(f'== {model}')
            r = aidn.ask(f'setup {model} {directory}')
            if 'error' in r:
                raise RuntimeError(f'AiDotNet setup: {r["error"]}')
            aidn_out = os.path.join(directory, 'aidn_out.bin')
            aidn.ask(f'dump {aidn_out}')
            transpose = False
            expected = reference(model, data)
            error = relative_error(expected, aidn_out)
            if model == 'mha' and not error <= 1e-3:
                # Settle the projection convention from AiDotNet's output (x @ W or x @ W^T).
                alt = reference(model, data, mha_transpose=True)
                if relative_error(alt, aidn_out) <= 1e-3:
                    transpose, expected, error = True, alt, relative_error(alt, aidn_out)
            log(f'  AiDotNet output vs float64 reference: {error:.3g}' + ('  (x @ W^T)' if transpose else ''))
            if not error <= 1e-3:
                summary['models'][model] = {'error': f'AiDotNet output wrong: {error:.3g}'}
                continue

            log('  PyTorch search:')
            ranked = search(torch_srv, model, directory, expected, transpose, threads, args.search_iters, log)
            if not ranked:
                summary['models'][model] = {'error': 'no correct PyTorch variant'}
                continue
            best = ranked[0][1]
            log(f'  PyTorch best: {best} ({ranked[0][0]:.4f} ms)')
            torch_srv.ask({'cmd': 'setup', 'model': model, 'dir': directory, 'transpose': transpose, **best})

            def timed(side):
                time.sleep(args.pause)
                if side == 'a':
                    return aidn.ask(f'round {args.iters}')
                return torch_srv.ask({'cmd': 'round', 'iters': args.iters})

            rounds = []
            for n in range(args.rounds):
                a1, b1, b2, a2 = timed('a'), timed('b'), timed('b'), timed('a')
                ai = (a1['median_ms'] + a2['median_ms']) / 2
                pt = (b1['median_ms'] + b2['median_ms']) / 2
                rounds.append({'aidotnet_ms': ai, 'torch_ms': pt, 'speedup': pt / ai,
                               'aidotnet_p95_ms': max(a1['p95_ms'], a2['p95_ms']),
                               'torch_p95_ms': max(b1['p95_ms'], b2['p95_ms'])})
                log(f'  round {n + 1:2}: AiDotNet {ai:.4f} ms  PyTorch {pt:.4f} ms  speedup {pt / ai:.3f}x')
            s = sorted(r['speedup'] for r in rounds)
            k = len(s)
            summary['models'][model] = {
                'torch_best': best, 'torch_search': [{'median_ms': m, **v} for m, v in ranked],
                'aidotnet_relative_error': error, 'rounds': rounds,
                'median_speedup': (s[(k - 1) // 2] + s[k // 2]) / 2, 'min_speedup': s[0], 'max_speedup': s[-1],
                'aidotnet_median_ms': float(np.median([r['aidotnet_ms'] for r in rounds])),
                'torch_median_ms': float(np.median([r['torch_ms'] for r in rounds])),
            }
            m = summary['models'][model]
            log(f'  => {model}: speedup median {m["median_speedup"]:.3f}x (rounds {s[0]:.3f}..{s[-1]:.3f}); '
                f'AiDotNet {m["aidotnet_median_ms"]:.4f} ms vs PyTorch {m["torch_median_ms"]:.4f} ms')
    finally:
        aidn.close()
        torch_srv.close()
        with open(args.out, 'w') as f:
            json.dump(summary, f, indent=2)
        log(f'wrote {args.out}')


if __name__ == '__main__':
    main()
