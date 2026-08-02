# Phase 2: the instruction classes we never emitted

The emitter issued only scalar `ld.global.f32`. Three instruction classes the hardware
offers were never reached: **vector loads** (`ld.global.v4.f32`), **async copy**
(`cp.async` / LDGSTS), and **tensor cores** (HMMA / IMMA).

This closes the first of the three. The other two remain open, and are named at the
bottom rather than implied to be done.

## What was implemented

A binding is read with `ld.global.v4.f32` when — and only when — its **unit-stride
dimension is indexed by exactly the innermost reduction axis, and that axis spans the
whole dimension**. Under that condition four consecutive reduction trips read four
consecutive floats, the group start is a multiple of four, and `cudaMalloc`'s 256-byte
alignment makes the instruction's 16-byte alignment requirement a guarantee rather
than a hope. The vector is fetched on the first trip of each group; the other three
trips reuse the components already in registers.

The condition is checked structurally, not guessed. Bindings that read a *gathered
window* are deliberately excluded: `input[n, c, oh+kh-1, ow+kw-1]` has a per-thread
base address with no alignment guarantee, so vectorising it would be unsafe.
`GatheredWindows_AreNotVectorised` locks that in.

## Measured, 2026-07-25, idle RTX 3080

### Static effect on the one kernel that qualifies

| metric | scalar | vectorised | change |
|---|---|---|---|
| SASS instructions | 584 | 440 | **−24.7%** |
| LDG | 129 | 81 | **−37.2%** |
| registers | 40 | 40 | — |
| spills | 0/0 | 0/0 | — |
| numerics vs fp64 oracle | 0.000E+000 | 0.000E+000 | unchanged |

### Wall-clock, paired in-process A/B

| kernel | v4 loads | scalar us | vector us | speedup |
|---|---|---|---|---|
| conv2d_1x1_bias_relu | 16 | 78.4 | 75.1 | **1.037x** |
| the other 8 catalog kernels | 0 | — | — | no unit-stride reduction axis |

## The finding that matters: static metrics overstated the win by ~7x

A 37% cut in LDG and a 25% cut in instructions bought **3.7%** of wall clock.

Running the kernel in separate processes before and after suggested ~9.6% (80.7 ->
73.6 us). The paired in-process A/B — the only comparison Phase 0.5 showed to be
trustworthy — says 1.037x, which is barely outside the 1.03x claimable band above the
1.05% noise floor. **The cross-process number was inflated by roughly 2.6x and should
not have been reported.**

The reason the instruction saving does not convert: `conv2d_1x1` is not limited by
issuing weight loads. Its *input* access `input[n, c, oh, ow]` walks the reduction axis
`c` with stride H*W, so every input load is a separate uncoalesced transaction, and
those dominate. Removing three quarters of the *weight* loads removes a cost that was
mostly hidden behind them.

## Applicability is 1 in 9, and that is structural

Only `conv2d_1x1_bias_relu` qualifies. The others fail the condition for real reasons:

* depthwise and dense 3x3 index their weights by the tap axis, whose extent is 3 — not
  a multiple of the vector width;
* the transposed and gathered bindings have no alignment guarantee;
* `conv2d_1x1_bwd_data` is instructive: the **adjoint transposes the access pattern**,
  so the weight operand that was unit-stride in the forward pass is strided in the
  backward one. The derivation is correct; vectorisation simply does not survive it.

The limitation is that vectorising along the *reduction* axis can only ever reach the
weight operand. Reaching the activation operand needs vectorising along the contiguous
*output* axis, which means each thread computing four adjacent outputs — thread
coarsening / register blocking. That is a change to the thread-to-output mapping, not
a load-selection change, and it is the higher-value follow-on.

## Still not emitted

* **`cp.async` / LDGSTS** — overlaps global-to-shared transfer with compute. Needs a
  shared-memory staging model, which the emitter does not yet have (LDS/STS are both
  zero across all nine kernels).
* **Tensor cores (HMMA / IMMA)** — needs the reduction expressed as a tiled matrix
  product with the fragment layouts those instructions require.
* **Thread coarsening** — the prerequisite for vectorising the activation operand, and
  on this evidence the one most likely to convert into wall clock.
