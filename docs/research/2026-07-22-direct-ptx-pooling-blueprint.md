# Direct-PTX pooling, interpolation, padding, grid-sample, and spatial-transform blueprint

Date: 2026-07-22

Tracking issue: #842

Parent blueprint: `2026-07-20-fused-attention-championship-blueprint.md`

## Verdict

This increment establishes the issue-#842 assembly line and implements one exact
contiguous Ampere FP32 candidate: **global average pooling**,
`output[bc] = mean over H*W of input[bc, :]`. It is **not promoted**. Each warp
owns one (batch, channel) row of `spatial = H*W` elements, keeps every lane value
in registers from the sole global load through a butterfly-shuffle sum, scales by
the baked `1/spatial` immediate, and lane zero stores one FP32 element — zero
shared/local, one global load and one global store per row. Correctness is
verified on the local RTX 3080 (SM86) against a double-precision oracle within a
`1e-4 * (|mean| + 1)` relative tolerance across all admitted shapes.

Production fails closed with `global-avgpool-performance-gate-not-met`
(`IsPromotedShape => false`). This draft does not close #842. Its 20-cell manifest
assigns the remaining global-max, adaptive/windowed avg and max pool, backward,
nearest/bilinear upsample, interpolate, pad, grid-sample, ROI-align,
spatial-transform, pixel-shuffle, and public-routing families.

## Honest positioning (launch-overhead win, not a fusion win)

Global average pooling is a bare mean reduction, and the existing
`global_avgpool2d` kernel is already a single-pass reduction — so this is **not**
a fusion win (there is no adjacent op to fold in and no materialized intermediate
to remove). It is, however, a real **launch/dispatch-overhead** win: the
exact-shape pointer-only ABI has no generic-kernel dispatch overhead, and in a
single diagnostic capture on the RTX 3080 the direct kernel beat the existing
`global_avgpool2d` kernel on the median at every admitted shape by roughly
**1.4–1.6x** (e.g. 35.8us -> 21.8us at (2048,128)) and PyTorch `mean(dim=-1)` by
1.3–1.6x, at ~5e-8 relative error — three shapes cleared all gates and one held
on P95 contention noise. This is the same overhead-only advantage the bare
row-sum kernel showed; the larger, structural pooling wins still live in the
**windowed** and **fused** cases (avg/max pool with an activation, or GAP feeding
an FC) where an intermediate would otherwise be materialized. Promotion still
requires three clean, separately launched captures, so the kernel stays disabled.

## Formal contiguous ABI

| Tensor | Extent | Access | Alignment |
|---|---:|---|---|---:|
| input  | `[batch*channels, H*W]` FP32 row-major | read  | 16 B |
| output | `[batch*channels]` FP32 vector          | write | 16 B |

Two 64-bit pointer parameters. The NCHW input is addressed as
`[batch*channels, H*W]` (contiguous), so no stride or dimension parameters reach
the kernel. Admitted buckets: `(rows=batch*channels, spatial=H*W) = (256,128),
(2048,64), (2048,128), (8192,128)`, `spatial` a multiple of 32; Ampere only. The
common 7x7/14x14 (spatial 49/196, not multiples of 32) need a predicated-tail
strided variant, tracked as a follow-up.

## Fair-comparison notes (apples-to-apples)

The resident PyTorch competitor is `x.mean(dim=-1)` over the flattened spatial
axis, scored against a double-precision oracle. The C# release-gate harness feeds
the **strongest** competitor (minimum median across the AiDotNet
`global_avgpool2d` kernel and PyTorch) into `DirectPtxReleaseGate`.

**Protocol hardening (family-wide):** lock GPU clocks (`nvidia-smi -lgc`) and
interleave the C#/Python capture halves to prevent the thermal-drift artifact the
pointwise blueprint already had to exclude a run for.

## Correctness and runtime proof

Focused tests enforce pointer-only PTX with one vector load, a five-step
butterfly reduction, the baked `0f3C000000` (=1/128) scale, one scalar store, no
`.shared`/`.local`/`bar.sync`/stride/`.param .u32`, a closed unpromoted shape
domain, manifest completeness with exactly one experimental cell and no promoted
cell, and (on a validated Ampere device) mean parity within relative tolerance
with zero local and static-shared bytes and at least three active blocks per SM.

## Reproduction

```powershell
dotnet run --project tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj `
  -c Release -- --direct-ptx-global-avgpool 3
```

## Next pooling increments

1. Add global max pool (same warp-row shape, max reduction) and the
   broadcast-scaled GAP backward.
2. Add a predicated-tail strided variant for non-multiple-of-32 spatial extents
   (7x7, 14x14) — the common ResNet GAP shapes.
3. Add windowed avg/max pool with baked kernel/stride, and fuse pooling with an
   activation so the pooled tensor is not separately materialized.
4. Add FP16/BF16 families and independently measured Ada, Hopper, and Blackwell
   modules; never infer promotion from Ampere.
