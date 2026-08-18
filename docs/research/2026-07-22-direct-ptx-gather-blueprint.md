# Direct-PTX embedding, gather/scatter, indexing, segment, and sparse-update blueprint

Date: 2026-07-22

Tracking issue: #844

Parent blueprint: `2026-07-20-fused-attention-championship-blueprint.md`

## Verdict

This increment establishes the issue-#844 assembly line and implements one exact
contiguous Ampere FP32 candidate: a warp-per-row **embedding gather**,
`output[i,:] = source[indices[i],:]`. It is **not promoted**. Each warp loads its
INT32 index once, then streams the source row to the output row in a single
vectorized coalesced copy — zero shared memory, zero local bytes, zero global
intermediates, no temporary allocation, no division/remainder, and no scalar
shape parameters. Correctness is verified on the local RTX 3080 (SM86) as an
**exact** copy (bit-identical to the CPU reference) across all admitted shapes.

The backend requires prewarm before CUDA graph capture and pins every captured
module until backend disposal so bounded-cache eviction cannot invalidate a
graph executable. Production fails closed with `gather-performance-gate-not-met`
(`IsPromotedShape => false`). This draft does not close #844. Its 22-cell
manifest assigns the remaining index-select, embedding-backward, scatter,
scatter-add, scatter-reduce, gather-nd, take-along-axis, one-hot, segment,
sparse-update, masked-select, and public-routing families.

## Formal contiguous ABI

| Tensor | Extent | Access | Mode | Alignment |
|---|---:|---|---|---:|
| indices | `[numIndices]` INT32 vector          | read  | exact    | 16 B |
| source  | `[*, featureSize]` FP32 row-major      | read  | at-least | 16 B |
| output  | `[numIndices, featureSize]` FP32        | write | exact    | 16 B |

The launch has three 64-bit pointer parameters. The source table is an
**at-least** view: its row count is deliberately not part of the module identity,
because the kernel indexes rows through the runtime index values. Indices are
trusted in range, exactly as the established `embedding_forward` kernel requires.
Each warp owns an output row, loads `indices[row]`, computes the source byte
offset with one `mul.wide.s32`, and copies one FP32x4 (feature=128) or FP32x2
(feature=64) vector. Admitted buckets: `(numIndices,featureSize) = (256,128),
(2048,64), (2048,128), (8192,128)`; Ampere only; else exact-reason fallback.

## Fair-comparison notes (apples-to-apples)

The resident PyTorch competitor script measures four variants — eager,
CUDA-graph, `max-autotune-no-cudagraphs` compile, and compile+graph — of
`index_select`. Because gather is a pure copy, correctness is exact (zero error)
for every path, so the comparison is unambiguous: identical bytes moved,
NVIDIA-only, graph-vs-graph symmetric. The C# release-gate harness feeds the
**strongest** competitor (minimum median across the AiDotNet `embedding_forward`
kernel and PyTorch) into `DirectPtxReleaseGate`.

Gather is memory-bound, so the durable wins in this family come from removing
launch/dispatch overhead at small and medium index counts (where the exact-shape
pointer-only ABI has an edge over the generic kernel) and, later, from **fusing**
the gather with its consumer (e.g. gather + layernorm, or scatter-add gradient
accumulation) so an intermediate is never materialized.

**Protocol hardening (family-wide):** lock GPU clocks (`nvidia-smi -lgc`) and
interleave the C#/Python capture halves to prevent the thermal-drift artifact the
pointwise blueprint already had to exclude a run for.

## Correctness and runtime proof

Focused tests enforce pointer-only PTX with exact index/vector load and store
counts, no `.shared`/`.local`/`shfl`/`bar.sync`/stride/`.param .u32`, a closed
unpromoted shape domain, manifest completeness with exactly one experimental cell
and no promoted cell, and (on a validated Ampere device) bit-exact gather results
with zero local and static-shared bytes and at least three active blocks per SM.

## Reproduction

```powershell
dotnet run --project tests/AiDotNet.Tensors.Benchmarks/AiDotNet.Tensors.Benchmarks.csproj `
  -c Release -- --direct-ptx-gather 3

python tests/AiDotNet.Tensors.Benchmarks/BaselineRunners/py/run_direct_ptx_gather_competitors.py --runs 3
```

## Next gather increments

1. Add the scatter direction (`scatter_rows`) and atomic `scatter_add_rows` /
   `embedding_backward` with warp-owned rows and per-element atomics.
2. Add `index_select` along non-row axes with a baked axis stride, and
   `take_along_axis` elementwise indexing.
3. Fuse gather with an immediate consumer (LayerNorm / bias-add) so the gathered
   rows never touch global memory twice.
4. Add FP16/BF16 vector families and independently measured Ada, Hopper, and
   Blackwell modules; never infer promotion from Ampere.
