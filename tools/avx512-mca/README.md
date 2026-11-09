# AVX-512 direct-conv loop model (#1445)

`Run-Avx512Mca.ps1` models the AVX-512 direct-conv inner loops on Sapphire Rapids, Ice Lake Server and Zen 4. It needs no AVX-512 hardware:

1. **Compile.** crossgen2 compiles verbatim copies of the loops (`Tiles.cs`) for `x86-64-v4`, with the framework in the version bubble, so the code matches what the tier-1 JIT emits on an AVX-512 host.
2. **Model.** llvm-mca models each hot loop.

```powershell
./Run-Avx512Mca.ps1 -Crossgen2 <crossgen2.exe> -LlvmMca <llvm-mca.exe> -Framework <dotnet>/shared/Microsoft.NETCore.App/10.0.x
```

- **crossgen2:** from the `microsoft.netcore.app.crossgen2.win-x64` NuGet package, `tools/crossgen2.exe`.
- **llvm-mca:** from any LLVM release.

## Results (crossgen2 10.0.14, LLVM 23.1.5)

Cycles per loop iteration.

| loop | FMAs | port-5 shuffles | Zen 4 (llvm-mca) | Intel server (port count) |
|---|---:|---:|---:|---:|
| weight gradient, AVX2 tile (4 x 3 x 8 lanes) | 12 (ymm) | 0 | 9.85 | — |
| weight gradient, 512-bit, joined loads (before) | 24 | 6 | 38.06 | 15 |
| weight gradient, 512-bit, sixteen-wide packing (after) | 24 | 0 | **31.18** | **12** |
| forward / strided dX sextet, blocked weights (before) | 12 | 4 | 22.05 | 8 |
| forward / strided dX sextet, interleaved weights (after) | 12 | 0 | **18.06** | **6** |

**How the joins compile.** The JIT lowers each `Vector512.Create(Vector256, Vector256)` join of two loads into two `vinsertf32x8`. On Intel server cores those run on port 5, which is also the second 512-bit FMA port. So a loop of F FMAs and S shuffles takes (F + S) / 2 cycles there instead of F / 2.

**Why the Intel column is a port count.** LLVM's Sapphire Rapids and Ice Lake Server models issue 512-bit FMAs on port 0 only. Their figures (24 cycles for 24 FMAs either way) therefore can't show the port-5 contention, so the Intel column counts ports instead.

**Zen 4.** Its model is used as is:
- Per multiply-add, the 512-bit weight-gradient loop drops from 0.099 to 0.081 cycles.
- The AVX2 tile takes 0.103 cycles, so on Zen 4 the 512-bit tile wins only without the joins.
