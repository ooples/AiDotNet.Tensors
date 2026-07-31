# Cross-cutting guarantees: ratchet, determinism, architecture

Three properties the conveyor needs that are not per-kernel evidence.

## 1. A static-metric ratchet that runs without a GPU

The metrics that gate a release — registers, SASS instructions, spills — come from
ptxas and nvdisasm and need a device. CI has none. So the ratchet runs on the metrics
the **emitter** produces without a device, which move in the same direction as the
device metrics and move first:

| metric | why it is the leading indicator |
|---|---|
| PTX lines | codegen bloat before ptxas ever sees it |
| loads | the LDG count, pre-scheduling |
| vector loads | silently losing `ld.global.v4.f32` is invisible in wall clock |
| elided guards | interval analysis quietly regressing to conservative guards |
| looped axes | a kernel dropping from full unroll to strip-mined |

The baseline is checked in at
`tests/AiDotNet.Tensors.Tests/Engines/Codegen/codegen-static-baseline.tsv`:

| kernel | ptx lines | loads | v4 loads | elided guards | looped axes |
|---|---|---|---|---|---|
| depthwise_conv2d_3x3_bias_relu | 270 | 19 | 0 | 56 | 0 |
| depthwise_conv2d_3x3 | 262 | 18 | 0 | 55 | 0 |
| conv2d_1x1_bias_relu | 793 | 81 | 16 | 293 | 0 |
| conv2d_3x3_bias_relu | 285 | 19 | 0 | 65 | 1 |
| maxpool2d_2x2 | 90 | 4 | 0 | 20 | 0 |
| conv_transpose2d_3x3_stride2 | 358 | 18 | 0 | 49 | 0 |
| depthwise_conv2d_3x3_bwd_data | 262 | 18 | 0 | 55 | 0 |
| conv2d_1x1_bwd_data | 1073 | 128 | 0 | 388 | 0 |
| conv2d_3x3_bwd_data | 277 | 18 | 0 | 64 | 1 |

**Improvements fail too.** A ratchet that silently accepts a metric going down cannot
distinguish an optimisation from a kernel that stopped doing work — the emitter that
loses a bounds guard and the emitter that loses a load both look like progress.

### It was tested by breaking it

Changing `FullUnrollLimit` from 64 to 32 — a plausible-looking tuning change — was
caught immediately:

```
Codegen static metrics moved. If the change is intended, update ...
  conv2d_1x1_bias_relu
    baseline Metrics { PtxLines = 793, Loads = 81, VectorLoads = 16, ElidedGuards = 293, LoopedAxes = 0 }
    actual   Metrics { PtxLines =  77, Loads =  3, VectorLoads =  0, ElidedGuards =  11, LoopedAxes = 1 }
```

That one-constant change silently cost the kernel all 16 of its vector loads and
pushed it into a runtime loop. Wall clock would have moved a few percent — inside the
range that reads as noise.

## 2. Determinism

The same spec must produce byte-identical PTX across repeat calls and across emitter
instances. Cubins are content-addressed on the PTX text, so nondeterminism defeats the
artifact cache and makes a released hash unreproducible.
`Emission_IsByteIdenticalAcrossRunsAndInstances` checks both, for all nine kernels.

## 3. Architecture parameterisation

**A real bug, found by writing the test.** `.version 7.1` was hardcoded while
`.target sm_XX` was parameterised. That pairing is self-contradictory: ptxas rejects
`.version 7.1` with `.target sm_90`, because sm_89 and sm_90 were not introduced until
ISA 7.8. Emitting for anything past Ampere produced invalid PTX, and nothing caught it
because every test ran on the sm_86 box.

`PtxIsaVersionFor` now derives the version from the capability:

| target | ISA |
|---|---|
| sm_70 – sm_86 | 7.1 |
| sm_87 | 7.4 |
| sm_89, sm_90 | 7.8 |

7.1 is a floor rather than the exact minimum for older targets, because that is the
version the shipped sm_86 cubins were built with and lowering it would change their
content hash for no benefit. Since sm_86 still resolves to 7.1, this change leaves
every released artifact byte-identical — confirmed by re-running the on-device
conveyor: 9/9 verify at 0.000E+000.

`ArchitectureAffectsOnlyTheHeader` additionally requires that changing the target
changes **only the first two lines**. A kernel body that varied with the architecture
would mean the arch parameter is steering codegen, and none of the sm_86 evidence
would transfer to another card.
