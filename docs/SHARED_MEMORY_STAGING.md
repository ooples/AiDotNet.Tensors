
---

# The input half: two-dimensional lowering

Staging the activation operand needs a different block shape, and that lowering is now
built and correct. It is **off by default**, because measured it is not yet profitable.

## Why it needs a 2D block

The activation is invariant in the reuse axis, so threads differing only in that axis
want the same input. A flat block is the wrong arrangement for that: its threads walk the
spatial axes at **one** value of the reuse axis, so no two threads in it share an input.

A 2D block fixes the arrangement — x over the contiguous axis so stores stay coalesced,
y over the reuse axis so one staged input row serves the whole column.

## What it costs today

| lowering | dense 3x3 | why |
|---|---|---|
| flat, weights staged | **61.4 us** | current default |
| two-dimensional, no staging | 74.3 us | correct, but loses weight staging and wastes half a warp per block |

Two measurable reasons, both addressable:

**1. It gives up weight staging, and staging the old way under 2D is WRONG.** Under a
flat block every thread shares one reuse-axis group, so the weights are block-invariant
and one staged slice serves everyone. Under 2D, y varies over that axis, so each row
needs its own slice. Staging the block-invariant way returned **5.277 and 1.112e1 instead
of zero** on the two dense kernels — caught immediately by verifying against the fp64
interpretation.

The correct model under 2D is that each operand is invariant in one **dimension**, not in
the block:

```
input   varies in x, invariant in y  -> stage indexed by x, shared down the column
weights varies in y, invariant in x  -> stage indexed by y, shared across the row
```

So staging must be indexed by the dimension the operand varies in. That is the remaining
work.

**2. The block is warp-ragged.** 7 tiles of `ow` by 16 tiles of `k` is 112 threads, which
is 3.5 warps: every block wastes half a warp. Padding x to 8 with a guard makes it 128,
exactly 4 warps.

## What it is worth once finished

Per c-step a block would stage 3 halo rows x 30 columns of input (90 floats) plus 4x9
weights — **504 bytes** — and the arithmetic gives:

| quantity | value |
|---|---|
| global loads per thread per step | 1.12 (from 36) |
| **loads/MAC** | **0.0078** (now 0.258; cuDNN ~0.03) |
| LDS reads/MAC | 0.500 |
| time if LDS-bound | **~15 us** |
| time if compute-bound | 7.5 us |

Against cuDNN's 41.0 us that is roughly **2.7x ahead** on the kernel we currently lose at
0.67x. The lowering committed here is the scaffolding for it; it stays off until it earns
its way, because shipping a 74.3 us path in place of a 61.4 us one would be a regression
dressed as progress.
