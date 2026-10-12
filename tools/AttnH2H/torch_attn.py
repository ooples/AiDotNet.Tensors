# PyTorch side of the #1141 head-to-head: one fp32 CUDA attention training step (SDPA forward, sum(out * gy),
# backward to q, k, v, synchronize). Usage: torch_attn.py <batch> <heads> <headDim> <reps> <seq,seq,...>
import sys, time, torch
import torch.nn.functional as F
b, h, d, reps = map(int, sys.argv[1:5]); seqs = [int(x) for x in sys.argv[5].split(',')]
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
dev = 'cuda'
print(f'torch={torch.__version__} B={b} H={h} D={d} reps={reps}')
print('api   S      causal  median_ms  min_ms')
for s in seqs:
    for causal in (False, True):
        q, k, v = (torch.randn(b, h, s, d, device=dev, requires_grad=True) for _ in range(3))
        gy = torch.randn(b, h, s, d, device=dev)
        def step():
            for t in (q, k, v): t.grad = None
            o = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
            (o * gy).sum().backward()
            torch.cuda.synchronize()
        for _ in range(5): step()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); step(); ts.append((time.perf_counter() - t0) * 1000)
        ts.sort()
        print(f'sdpa  {s:<6} {str(causal):<7} {ts[reps // 2]:9.3f} {ts[0]:7.3f}')
