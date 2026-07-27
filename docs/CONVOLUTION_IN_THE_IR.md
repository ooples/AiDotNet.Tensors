# Convolution reaches the front end

## What was missing

`CodegenOpKind` had **no convolution at all**. A convolution arrived as `Opaque` or not at
all, which meant the thirteen catalog kernels carrying every measured win on this branch
were reachable only as hand-written `CodegenKernelSpec` instances. No graph could ask for
one, so nothing a model actually runs could reach them.

The gap was purely upstream. The index-map layer has expressed convolution exactly since
the catalog was written — `Window(spatial, tap, stride, pad)` for the forward direction,
`TransposedWindow` for the adjoint — and `CodegenAdjoint` already derives the backward maps
from it. What did not exist was a way to *ask*.

## What was added

| piece | what it does |
|---|---|
| `CodegenOpKind.Conv2D` | dense NCHW cross-correlation |
| `CodegenOpKind.DepthwiseConv2D` | one filter per channel, no channel reduction |
| `CodegenOpKind.ConvTranspose2D` | the adjoint window, `(ih + pad - kh) / stride` |
| `CodegenOpCategory.Convolution` | a sliding-window contraction — a reduction with windowed maps |
| `CodegenConvAttributes` | stride and padding, as a typed record |
| `CodegenLowering.LowerConv2D<T>` | builds the graph, optionally with bias and ReLU |

`CodegenGraphToSpec` translates all three into the same spec form everything else uses:

```
dense:      out[n,k,oh,ow] = Σ_{c,kh,kw} in[n,c,W(oh,kh),W(ow,kw)] · w[k,c,kh,kw]
depthwise:  out[n,c,oh,ow] = Σ_{kh,kw}   in[n,c,W(oh,kh),W(ow,kw)] · w[c,kh,kw]
transposed: the same, with TransposedWindow
```

Attributes are a typed record rather than an `int[]` because these four numbers are not
interchangeable and a transposed pair is invisible inside an array.

## Refusals

A convolution that cannot be expressed exactly is declined with a reason, never
approximated:

| form | reason |
|---|---|
| no `CodegenConvAttributes` | stride and padding unknown; guessing them changes the operator |
| declared output ≠ derived output | a mismatched extent reads a shifted window and still runs |
| weights not `[K,C,kh,kw]` / `[C,kh,kw]` | the contraction would be against the wrong axis |
| channel counts that disagree | the reduction would contract the wrong extent |
| geometry producing an empty output | rejected at lowering, before such a graph exists |

The output shape is **derived** from the geometry inside `LowerConv2D` rather than accepted
from the caller, so a graph cannot declare an extent its own stride and padding do not
produce.

## The bias is `[1,K,1,1]`, not `[K]`

Broadcasting is right-aligned, so a bare `[K]` aligns against the **width** axis and means
a per-column bias — a different operator that would still translate and still run. Writing
the channel axis out keeps the broadcast rule uniform instead of adding a
convolution-specific exception to it. This was caught by the catalog-agreement test, not by
inspection.

## How it is checked

**Against the catalog, in fp64.** A spec translated *from a graph* is interpreted against
the hand-written catalog spec — the one already verified on the device at `0.000E+000` — on
identical operands, element for element. That is the strongest available bar: it would fail
for a translation that swapped a stride for a pad, which reads a shifted window and emits
perfectly. Covers depthwise 3×3 with and without epilogue, dense 1×1, and dense 3×3.

**On the device.** `--frontend-check` now runs all four convolution forms end to end,
graph → PTX → GPU:

```
graph                                  elements       rel dev   ref   result
depthwise 3x3 + bias + relu             100,352    0.000E+000  fp64    PASS   17.1 us   98blk x256
dense 1x1 + bias + relu                 100,352    0.000E+000  fp64    PASS   13.1 us   32blk x196
dense 3x3 + bias + relu                  25,088    0.000E+000  fp64    PASS   14.7 us   32blk x196
conv-transpose 3x3 stride 2              30,752    0.000E+000  fp64    PASS   24.0 us   61blk x256

front end: 13 passed, 0 failed
SM clock across the run: 1770->1770 MHz (+0.0%)
```

**The adjoint still derives.** `CodegenAdjoint.BackwardData` and `BackwardWeights` are
applied to a graph-built convolution and produce gradients shaped like the data and the
weights respectively — so training can use these kernels, which was the point of reaching
the front end.

## Still not done

- **The catalog specs are not yet replaced by graph-built ones.** The two agree in fp64,
  but the catalog still authors its specs by hand. Switching it over would make the
  agreement structural rather than tested.
- **`CodegenLowering` does not yet produce these ops from real `CompiledStep` chains.**
  `LowerConv2D` is a factory a caller invokes; nothing in the engine's own lowering path
  emits a `Conv2D` node yet.
- **No CPU emitter handles convolution**, so the device check falls back to the `fp64`
  reference for every convolution row — same situation as matmul and reductions. See
  `FRONT_END_REDUCTIONS.md`.
