# JIT GEMM microkernel PoC (Phase 0/1)

Archived proof-of-concept for the custom AVX2 JIT GEMM kernel (see
`../jit-gemm-issue-draft.md`). Files are `.txt`-suffixed so they are NOT built as
part of the Tensors solution. To run: copy `JitGemmPoc.cs.txt` → `Program.cs` and
`JitGemmPoc.csproj.txt` → `*.csproj` into a scratch dir and `dotnet run -c Release`.

Validated results on AVX2 (no AVX-512):
- emit+run+correct AVX2 machine code via `delegate* unmanaged` (maxErr ~1e-6)
- 6x16 microkernel: 118 GFLOP/s single-core vs RyuJIT 90 (1.3x)
- multi-core on one pool: 527 GFLOP/s @16t (matches oneDNN's 543, no oversubscription)
- naive packed driver on small-K transformer shapes is pack/prologue-bound (47/30/32)
  -> Phase 2 needs an unpacked stride-reading kernel for small-K.

## Phase 2 update: UNPACKED kernel beats oneDNN (UnpackedKernel.cs.txt)

Stride-reading 6x16 (no pack, no scatter), measured correctly (threads spawned
once, reps internal) on the transformer shapes, 16t, on our own pool:
- QKV  [4092,64]x[64,192]  699 GFLOP/s  (oneDNN 543, managed 91)
- FFN1 [4092,64]x[64,128]  684 GFLOP/s  (oneDNN 564, managed 113)
- FFN2 [4092,128]x[128,64] 604 GFLOP/s  (oneDNN 541, managed 66)
Correct (maxErr ~1e-6). 3-base-pointer trick for the 6 strided A rows
(rows 0/2/4 in RCX/RBX/RSI + index RAX=lda*4). This is the small-K hot path.
Next: wire into CpuEngine via PersistentParallelExecutor + edge kernels +
shape->kernel JIT cache, then the 16-shape end-to-end benchmark.

## Phase 2 integration (JitGemmAvx2 wired into dispatch) — end-to-end result

JitGemmAvx2 (emit-once unpacked kernel + PPE-parallel driver w/ serial-small
fallback + managed M%6/N%16 edges) wired into SimdGemm.Sgemm, FusedLinear,
TensorMatMul2D (AIDOTNET_JIT_GEMM=1, default OFF). 16-shape bench, min-of-6:
- MLP b8 -38%, b32/b128 ~-2-3%  (clean big GEMMs -> JIT wins)
- CNN/LSTM ~neutral
- transformer REGRESSES (b8 +30%, b1 +24%): the per-6x16-tile prologue (10 XMM
  save/restore) is paid per tile; the transformer's many-small-tile GEMMs (b8
  QKV = 504 tiles) pay it 504x while the managed kernel pays 0.
- net 5/16 (== baseline). Off-by-default, no shipped regression.

NEXT (the transformer win): macro-kernel — prologue once per row-panel, loop the
N/16 col-blocks internally (amortizes the 10-XMM save/restore ~12x). Then the
kernel's isolated 600-700 GFLOP/s should translate to the transformer end-to-end.

## Macro-kernel experiment: REJECTED (made transformer worse)

Tried a macro-kernel (one row-panel × all N/16 col-blocks per call) to amortize
the per-tile prologue. Isolated 566 GFLOP/s (below the per-tile 699 — the per-cb
register resets + serialized col-block loop cost more than the prologue saves).
End-to-end it REGRESSED the transformer further (+41-51% vs per-tile's +5-30%).
Reverted to the per-tile kernel.

## KEY CONCLUSION: the transformer is NOT GEMM-throughput-bound end-to-end

The JIT kernel is 600-700 GFLOP/s in ISOLATION (beats oneDNN) yet does not flip
the transformer end-to-end (per-tile: net-neutral/mild-regress; macro: worse).
Why: isolated GFLOPS (same GEMM 600x, B hot in cache, threads spawned once) does
NOT equal end-to-end throughput (each GEMM run once, cold-ish B, interleaved with
SDPA/LayerNorm/softmax/residuals that evict cache, + per-GEMM dispatch). The
transformer's time is the aggregate of many ops + dispatch, not GEMM throughput.
- MLP (3 big GEMMs, weights reused across the batch, little interleaving) IS
  GEMM-bound -> JIT wins (b8 -38%).
- Transformer is NOT -> no GEMM kernel (ours at 699 or oneDNN at 543) flips it.
Next lever for the transformer is whole-forward-pass fusion / dispatch reduction,
not a faster GEMM. The JIT kernel stays an off-by-default MLP/GEMM-bound win.

## FINAL: the wins came from the OP fixes, not the GEMM JIT (6/16 -> 8/16)

End-to-end op profiling (the decisive step) showed the transformer's pure GEMMs
are only ~9%; the costs were SDPA (scalar softmax), LayerNorm (serial), and the
two-thread-pool contention. Fixing those OPS — not the GEMM — delivered the wins:

| fix | effect |
|---|---|
| SIMD softmax (Exp8) in SDPA      | SDPA 3.15 -> 1.42 ms/call |
| LayerNorm row-count parallel gate | 0.33 -> 0.13 ms/call (2.5x) |
| SDPA -> PersistentParallelExecutor (one pool) | SDPA 1.42 -> 0.86 ms/call |
| net (16-shape, min-of-6)         | **6/16 -> 8/16**; transformer now wins b1+b8(+b32) |

The JIT GEMM (beats oneDNN isolated at 699, 376 GFLOP/s per-call) is NET-NEGATIVE
end-to-end (8/16 -> 6/16): it wins large-M (transformer FFN/QKV M=4096) but loses
medium/small-M (MLP M<=128, transformer b32), and the transformer isn't GEMM-bound
so the isolated GFLOP/s don't translate. Kept as an off-by-default, M-gated
(AIDOTNET_JIT_MIN_M=512) opt-in for genuinely GEMM-bound large-M workloads.

Remaining losses (8/16, all op/shape-specific, not GEMM-GFLOPS): MLP (small-M
GEMM vs MKL's small-GEMM JIT), LSTM/transformer large-batch (aggregate ops).
