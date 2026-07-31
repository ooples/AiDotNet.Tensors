# Backward kernels, derived rather than written

A forward operator is

```
out[F(p)] = sum over r of  data[G(p,r)] * weight[H(p,r)]
```

and its gradient with respect to the data is the **adjoint** of that same map:

```
dData[j] = sum over every (p,r) with G(p,r) = j  of  dOut[F(p)] * weight[H(p,r)]
```

which is another operator of exactly the same shape. So a backward kernel does not
need to be authored, verified, released and benchmarked a second time — it can be
derived from the forward spec and then carried through the same conveyor.

`CodegenAdjoint.BackwardData(forward, dataInput)` does the derivation. It is
mechanical once index maps are first-class:

* an axis the data map **determines** (given the backward output index) becomes a
  parallel axis of the backward kernel;
* an axis the data map leaves **free** becomes a reduction axis — that is precisely
  the set being summed over;
* a forward gather window `oh*stride + kh - pad` inverts to the exact-division map
  `(ih + pad - kh)/stride`, which the affine layer already models as
  `TransposedWindow`.

That last point is worth stating plainly: **the exactness predicate is not a special
case bolted on for transposed convolution. It is what an adjoint index map is.** The
same code that made stride-2 transposed convolution expressible is what makes every
strided backward pass expressible.

## Why this is where the IR pays for itself

Hand-written backward kernels are where the bugs live. The shipped grouped-deformable
backward kernel computed zeros for its offset and mask gradients because its thread
count was maintained by hand, separately from its reference, and all three structural
gates passed on the broken code. A derived adjoint cannot drift from the forward
operator it came from, because there is nothing to keep in sync.

## The test that matters

Checking a derived backward kernel against its own interpreter proves only that the
derivation is self-consistent. The **dot-product identity** is independent of how the
adjoint was constructed:

```
<forward(x), y>  ==  <x, backward(y)>
```

It holds for the true adjoint of a linear operator and for nothing else, so it catches
a wrong index map, a wrong reduction set, and a missing exactness predicate alike.
All four cases pass to within 1e-9 relative:

| forward operator | what the adjoint has to do | dot-product identity |
|---|---|---|
| depthwise 3x3 | invert a window to a transposed window | PASS |
| dense 1x1 | move a forward PARALLEL axis (output channels) into the reduction set | PASS |
| dense 3x3 | both at once: a moved axis AND two inverted windows | PASS |
| depthwise 3x3 stride 2 | exactness predicate is load-bearing — one in four positions contributes | PASS |

Operators whose adjoint is not an index-map transform are refused loudly rather than
mis-lowered: a non-sum reduction, an activation (whose backward needs the forward
pre-activation value), or a bias/scale (which makes the operator affine, not linear).

## Measured, 2026-07-25, idle RTX 3080

Three derived backward kernels went through the full conveyor with no per-kernel code.

### Verify — against the fp64 interpretation

| kernel | regs | lowering | max rel dev | result |
|---|---|---|---|---|
| depthwise_conv2d_3x3_bwd_data | 38 | unroll | 0.000E+000 | PASS |
| conv2d_1x1_bwd_data | 36 | unroll | 0.000E+000 | PASS |
| conv2d_3x3_bwd_data | 40 | loop x1 | 0.000E+000 | PASS |

9 of 9 catalog kernels pass (6 forward, 3 derived).

### Release — driver-linked cubin, SASS audit, gate = zero spills

| kernel | regs | SASS instr | LDG | STG | spill | gate |
|---|---|---|---|---|---|---|
| depthwise_conv2d_3x3_bwd_data | 36 | 168 | 18 | 1 | 0/0 | PASS |
| conv2d_1x1_bwd_data | 44 | 576 | 128 | 1 | 0/0 | PASS |
| conv2d_3x3_bwd_data | 40 | 176 | 18 | 1 | 0/0 | PASS |

9 of 9 zero-spill.

### Bench — Phase 0.5 calibrated protocol

| kernel | blocks | us/launch | run spread | forward counterpart |
|---|---|---|---|---|
| depthwise_conv2d_3x3_bwd_data | 25,088 | 96.6 | 2.4% | 96.8 |
| conv2d_1x1_bwd_data | 3,136 | 81.9 | 1.2% | 80.7 |
| conv2d_3x3_bwd_data | 784 | 145.3 | 0.5% | 140.3 |

Each backward kernel costs within a few percent of the forward operator it was derived
from, which is what an adjoint should cost: the same arithmetic with the index map
transposed. `depthwise_conv2d_3x3_bwd_data` is byte-for-byte the same instruction and
register count as its forward kernel (36 regs, 168 SASS), because a stride-1 pad-1 3x3
adjoint is structurally the forward operator with the taps flipped.

## Adding a backward kernel

Do not write one. Add a catalog entry that applies `CodegenAdjoint.BackwardData` to a
linear forward spec, and the conveyor does the rest. If the forward operator has a
bias or activation, differentiate the epilogue-free version — the derivation refuses
the affine case rather than quietly returning the wrong gradient.
