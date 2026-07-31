# The kernel conveyor

Three stages every generated kernel passes through, in the same order, with the same
gates, driven by a loop over `CodegenKernelCatalog` rather than by per-kernel code.
Adding a kernel means adding one catalog entry; the stages then apply to it with no
new code. That is the property that makes ~800 kernels tractable.

```text
--kernel-verify [name|all]    emit -> run on device -> compare vs fp64 interpretation
--kernel-release [name|all]   emit -> driver-linked cubin -> nvdisasm machine-code audit
--kernel-bench [name|all]     time with the Phase 0.5 calibrated protocol
```

## Why the stages share one source of truth

The oracle in verify and the kernel in release/bench are generated from the **same**
`CodegenKernelSpec`. A spec change therefore cannot leave the reference and the
implementation disagreeing. That is not hypothetical: a hand-written grouped
deformable backward kernel passed three structural gates while computing zeros,
because its thread count and its reference were maintained separately.

## Historical commissioning snapshot, 2026-07-24

The six-row tables below record the first conveyor commissioning run; they are not
current release evidence. The catalog now contains 13 operations and a tuned release
can contain 16 cubins because split reductions emit partial and combine kernels. Current
claims come from the protocol-stamped files under `artifacts/`. For post-drift absolute
timings, see `BENCH_CLOCK_DRIFT.md`.

### Stage 1 — verify (tolerance 2e-3, vs fp64 interpretation of the same spec)

| kernel | regs | lowering | guards elided | max rel dev | result |
|---|---|---|---|---|---|
| depthwise_conv2d_3x3_bias_relu | 40 | unroll | 56 | 0.000E+000 | PASS |
| depthwise_conv2d_3x3 | 38 | unroll | 55 | 0.000E+000 | PASS |
| conv2d_1x1_bias_relu | 36 | unroll | 53 | 0.000E+000 | PASS |
| conv2d_3x3_bias_relu | 40 | loop x1 | 65 | 0.000E+000 | PASS |
| maxpool2d_2x2 | 18 | unroll | 20 | 0.000E+000 | PASS |
| conv_transpose2d_3x3_stride2 | 40 | unroll | 49 | 0.000E+000 | PASS |

6 passed, 0 failed.

### Stage 2 — release (driver-linked cubin, nvdisasm SASS audit, gate = zero spills)

| kernel | regs | SASS instr | LDG | STG | spill ld/st | gate |
|---|---|---|---|---|---|---|
| depthwise_conv2d_3x3_bias_relu | 38 | 168 | 19 | 1 | 0/0 | PASS |
| depthwise_conv2d_3x3 | 36 | 168 | 18 | 1 | 0/0 | PASS |
| conv2d_1x1_bias_relu | 40 | 584 | 129 | 1 | 0/0 | PASS |
| conv2d_3x3_bias_relu | 40 | 184 | 19 | 1 | 0/0 | PASS |
| maxpool2d_2x2 | 18 | 88 | 4 | 1 | 0/0 | PASS |
| conv_transpose2d_3x3_stride2 | 40 | 440 | 18 | 1 | 0/0 | PASS |

6 zero-spill, 0 spilling. Manifest: `artifacts/codegen-cubins/codegen-cubins.tsv`,
one row per kernel with the content-addressed cubin SHA-256 and source key.

### Stage 3 — bench (Phase 0.5 protocol: device-filling, batched regions, 3 runs)

| kernel | blocks | us/launch | P95/median | run spread |
|---|---|---|---|---|
| depthwise_conv2d_3x3_bias_relu | 25,088 | 100.3 | 1.43 | 2.6% |
| depthwise_conv2d_3x3 | 25,088 | 96.5 | 1.98 | 7.2% |
| conv2d_1x1_bias_relu | 3,136 | 79.6 | 1.79 | 3.7% |
| conv2d_3x3_bias_relu | 1,568 | 142.4 | 1.51 | 2.1% |
| maxpool2d_2x2 | 25,088 | 160.2 | 1.28 | 2.4% |
| conv_transpose2d_3x3_stride2 | 12,544 | 122.4 | 1.35 | 4.6% |

These historical absolute timings are not claims against a competitor. The 7.2% row
was re-measured after clock observation/retry was added; `BENCH_CLOCK_DRIFT.md` is the
authoritative account of that correction.

## What building the conveyor found

**The emitter could only fully unroll.** Running stage 2 over six kernels instead of
one immediately hit `Reduction trip count 288 exceeds the unroll limit 64`. A dense
3x3 convolution over 32 input channels -- an ordinary ResNet layer -- could not be
generated at all. The single-kernel bake-off never revealed this because depthwise
3x3 is 9 trips.

The fix is strip-mining: peel outer reduction axes into runtime loops until the
remaining suffix fits the unroll limit, so the inner taps stay unrolled where index
folding and guard elision pay off, while the channel walk becomes a real loop. The
fully-unrolled path is byte-identical to before, so already-released cubins are
unchanged. `conv2d_3x3_bias_relu` now emits 184 SASS instructions at 40 registers
with zero spills for 288 trips of work.

**Strip-mining was about to ship unverified.** The verify shape used C=4 (36 trips,
unrolled) while the released shape used C=32 (288 trips, strip-mined), so the loop
lowering would have been released having only ever been checked in its unrolled form
-- the same failure as the stride-2 transposed cubins released with no numerical
coverage, one abstraction level up.

Stage 1 now gates on it: **the verify shape must exercise the same lowering as the
shape that gets released**, reported in the `lowering` column and enforced by
`EveryCatalogEntry_VerifiesTheLoweringItReleases`. The conv2d_3x3 verify shape moved
to C=8 (72 trips) so it strip-mines like the shape it releases.

## Adding a kernel

Add one entry to `CodegenKernelCatalog.Build()` with a spec at two shapes: a small
one whose fp64 reference is cheap, and a device-filling one for timing. The lowering
gate will tell you if the small shape fails to exercise what the large one releases.
