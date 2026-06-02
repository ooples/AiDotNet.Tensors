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
