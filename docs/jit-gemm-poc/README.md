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
