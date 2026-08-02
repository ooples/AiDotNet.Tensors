# Thread coarsening: the lever the bake-off pointed at

The cuDNN bake-off found dense convolution running at 2% of peak bandwidth and 5% of
peak FP32 while cuDNN reached 28%, and identified the cause as **load count, not load
cost**: one thread per output means every operand is re-read once per output that needs
it. Coarsening gives each thread several adjacent outputs along the contiguous axis, so
one load feeds several FMAs.

Measured 2026-07-25 on an idle RTX 3080. `--kernel-coarsen-ab` reproduces it.

## Paired in-process A/B, 4 outputs per thread vs 1

| kernel | lanes | loads/output | 1-per-thread us | coarsened us | speedup |
|---|---|---|---|---|---|
| conv2d_1x1_bwd_data | 4 | 32.0 | 80.2 | 47.9 | **1.713x** |
| conv2d_1x1_bias_relu | 4 | 20.2 | 72.9 | 44.1 | **1.656x** |
| depthwise_conv2d_3x3_bias_relu | 4 | 11.5 | 98.4 | 73.5 | **1.343x** |
| depthwise_conv2d_3x3_bwd_data | 4 | 11.2 | 94.8 | 74.1 | **1.293x** |
| depthwise_conv2d_3x3 | 4 | 11.2 | 95.4 | 75.0 | **1.283x** |
| conv_transpose2d_3x3_stride2 | 4 | 11.2 | 122.3 | 101.1 | **1.208x** |
| conv2d_3x3_bwd_data | 4 | 11.2 | 142.9 | 127.0 | **1.124x** |
| conv2d_3x3_bias_relu | 4 | 11.5 | 138.9 | 127.5 | **1.089x** |
| maxpool2d_2x2 | 4 | 4.0 | 158.1 | 159.6 | 0.995x |

Eight of nine improve. Numerics are unchanged: 9/9 verify at exactly 0.000E+000, and
all nine remain zero-spill.

**maxpool being neutral is a consistency check, not a failure.** The bake-off measured
it at 106% of the bandwidth roofline, i.e. already at the hardware limit, so it
predicted there was nothing to win there. Coarsening returned 0.995x. The roofline
model made a falsifiable prediction and it held.

## Coarsening ALONE regressed the kernel it was supposed to help most

First measurement of the dense 1x1: **0.944x — slower.** Loads per output had fallen
from 81 to 68.2, so the reuse was real, and the kernel still got worse.

The reason is that coarsening cuts thread count by 4, and thread count is what hides
memory latency. The 1x1's input walks the reduction axis with stride H*W, so every
input load is a separate uncoalesced transaction; with a quarter of the threads there
was a quarter of the overlap to hide them behind. The reuse gain was smaller than the
occupancy loss.

The fix is the other half of the same idea. An operand that is **unit-stride in the
coarsened axis** has its four lane values at four consecutive addresses, and lane 0
starts at a multiple of four, so the four scalar loads become one `ld.global.v4.f32`:

| | loads/output | speedup |
|---|---|---|
| coarsening only | 68.2 | 0.944x |
| + lane-vectorised activation operand | 20.2 | **1.656x** |

This is precisely the operand that Phase 2's reduction-axis vectorisation could never
reach, and it is why that earlier work bought only 1.037x. **Coarsening is what makes
activation-operand vectorisation possible; vectorisation is what makes coarsening
pay.** Neither is worth much alone.

## Effect on the cuDNN comparison

| kernel | before | after | note |
|---|---|---|---|
| depthwise_conv2d_3x3_bias_relu | 2.37x | **3.03x** | 69% -> 93% of bandwidth roofline |
| depthwise_conv2d_3x3_bwd_data | 2.21x | **2.78x** | 70% -> 94% |
| depthwise_conv2d_3x3 | 1.69x | **2.14x** | 71% -> 93% |
| maxpool2d_2x2 | 1.56x | 1.42x | unchanged; at the roofline either way |
| conv_transpose2d_3x3_stride2 | 0.87x | **0.98x** | reached parity |
| conv2d_1x1_bias_relu | 0.33x | **0.56x** | still behind |
| conv2d_1x1_bwd_data | 0.30x | 0.32x | still behind |
| conv2d_3x3_bias_relu | 0.24x | 0.25x | still behind |
| conv2d_3x3_bwd_data | 0.18x | 0.16x | still behind |

Losses drop from five to four and the depthwise family now sits at **93-94% of the
memory roofline**, which is close enough to the hardware limit that no further compiler
work will move it.

**Dense convolution is still 4-6x behind, and coarsening was not enough to fix it.**
That is the honest result. Registers cannot hold a large enough tile to get the reuse
cuDNN gets; the next step is staging tiles in shared memory, which every kernel in the
catalog still reports as completely unused (LDS 0 / STS 0).

## Three bugs this found, all in the same class

Each was a case of the *header* and the *body* of the generated kernel disagreeing --
which is the exact defect class the index-map IR was built to eliminate, reappearing one
level up.

1. **The loop label was written to the header instead of the body**, so `LOOP0:` landed
   immediately after the opening brace, ahead of its own counter's initialisation. The
   backward branch re-zeroed the counter every iteration and the kernel never
   terminated -- a GPU hang, which presented as the whole verify stage producing no
   output at all.
2. **Virtual-register declarations were hardcoded** at `%p<256>` / `%f<512>`. Those were
   written as generous bounds and silently became ceilings: coarsening pushed the
   transposed convolution to `%p256`, one past the declared range, and ptxas reported
   it as `Arguments mismatch for instruction 'setp'` -- an undeclared register, not a
   malformed instruction. Declarations are now written after the body from the counts
   actually used, so they cannot be outgrown.
3. **The per-lane axis views were cloned before the strip-mine loop assigned its
   counter register**, so a strip-mined kernel emitted an empty operand
   (`mad.lo.s32 %r16, , 9, %r15`).

The launch grid is now derived from the same coarsened thread count as the in-kernel
guard, and `LaunchBlocks_CoverExactlyTheCoarsenedThreadCount` asserts both, because a
coarsened kernel launched with an uncoarsened grid would compute every output four
times, and the reverse would silently skip three quarters of the output.
