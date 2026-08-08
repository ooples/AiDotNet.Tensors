# Direct-PTX tensor-layout, copy, transpose, packing, and dtype-conversion blueprint

Date: 2026-07-22

Tracking issue: #845

Parent blueprint: `2026-07-20-fused-attention-championship-blueprint.md`

## Verdict

This increment establishes the issue-#845 assembly line and implements one exact
contiguous Ampere candidate: a linear **FP32-to-FP16 dtype conversion**. Each
thread loads one FP32x4 vector, applies four round-to-nearest-even
`cvt.rn.f16.f32` conversions, and commits one packed FP16x4 vector — zero shared
memory, zero local bytes, zero global intermediates, no temporary allocation, no
division/remainder, no stride, and no scalar shape parameters. Correctness is
verified on the local RTX 3080 (SM86) as **bit-identical** to the CPU
`(Half)float` round-to-nearest-even reference across all admitted sizes,
including zero, subnormal, and non-finite inputs.

The backend requires prewarm before CUDA graph capture and pins every captured
module until backend disposal so bounded-cache eviction cannot invalidate a
graph executable. Production fails closed with `cast-fp16-performance-gate-not-met`
(`IsPromotedShape => false`). This draft does not close #845. Its 20-cell
manifest assigns the remaining widen, BF16, copy, transpose, permute, reshape,
interleave, block-quantize/dequantize, pad, slice, concat, and public-routing
families.

## Formal contiguous ABI

| Tensor | Extent | Access | Alignment |
|---|---:|---|---:|
| input  | `[size]` FP32 vector | read  | 16 B |
| output | `[size]` FP16 vector | write | 16 B |

The launch has two 64-bit pointer parameters. Each thread owns four contiguous
elements: one `ld.global.ca.v4.f32`, four `cvt.rn.f16.f32`, one
`st.global.v4.u16`. Admitted sizes: `65536, 262144, 1048576, 4194304`; Ampere
only; else exact-reason fallback.

## Fair-comparison notes (apples-to-apples)

The resident PyTorch competitor script measures four variants — eager,
CUDA-graph, `max-autotune-no-cudagraphs` compile, and compile+graph — of
`.to(float16)`. Because the cast has defined round-to-nearest-even semantics that
match PyTorch's own conversion, the comparison is exact: identical bytes in,
identical FP16 bit patterns out. The C# release-gate harness feeds the
**strongest** competitor (minimum median across the AiDotNet
`convert_fp32_to_fp16` kernel and PyTorch) into `DirectPtxReleaseGate`.

A bare cast is memory-bound (`size*4` read + `size*2` write), so an isolated
replacement is unlikely to clear 1.10x against a well-tuned vectorized cast; the
durable wins in this family come from **fusing** the cast into the producer or
consumer (cast-on-store from a GEMM epilogue, or cast-on-load into a kernel) so
the FP16 buffer is never separately materialized.

**Protocol hardening (family-wide):** lock GPU clocks (`nvidia-smi -lgc`) and
interleave the C#/Python capture halves to prevent the thermal-drift artifact the
pointwise blueprint already had to exclude a run for.

## Correctness and runtime proof

Focused tests enforce pointer-only PTX with exact vector load, four conversions,
one packed store, no `.shared`/`.local`/`shfl`/`bar.sync`/stride/`.param .u32`, a
closed unpromoted size domain, manifest completeness with exactly one
experimental cell and no promoted cell, and (on a validated Ampere device)
bit-exact FP16 output including non-finite and subnormal inputs, with zero local
and static-shared bytes and at least three active blocks per SM.

## Reproduction

```powershell
dotnet run --project tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj `
  -c Release -- --direct-ptx-cast-fp16 3

python tests/AiDotNet.Tensors.Benchmarks/BaselineRunners/py/run_direct_ptx_cast_fp16_competitors.py --runs 3
```

## Next layout increments

1. Add the FP16->FP32 widen and the FP32<->BF16 conversions (same vector shape).
2. Add the shared-memory 32x32 tiled transpose (coalesced read + coalesced
   transposed write) — the canonical layout win.
3. Add block-quantize/dequantize (Q8_0) packing cells reusing the ggml block ABI.
4. Fuse cast-on-store / cast-on-load into producer and consumer kernels so the
   FP16 buffer is never separately materialized; add independently measured Ada,
   Hopper, and Blackwell modules.
