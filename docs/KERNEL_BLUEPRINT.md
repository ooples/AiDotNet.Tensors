# The kernel blueprint

How to add a GPU kernel to this repository so that it either beats the strongest existing
path apples-to-apples, or is correctly withheld. Every rule below is here because it was
learned the expensive way on branch `agent/direct-ptx-conv-promotion-841`; the measurement
that produced each one is cited so it can be re-checked rather than believed.

Applies to every open direct-PTX PR.

---

## 0. Before writing a kernel: is the work already done?

**Check whether AiDotNet already has a fused kernel for this operator.** Not cuDNN — *ours*.

This is first because it is the cheapest possible finding and it has already caught two
kernels on this campaign:

- PR #874 built a fused SGD-momentum kernel, then measured it against the existing
  `sgd_momentum_update` and found a **tie (0.73–1.05×)**, because that kernel was already
  single-pass. The gate correctly held on every shape. The work was real; the win was not
  available.
- I then queued "fused Adam" as the next target on the strength of that PR's prose, and a
  one-line grep showed `adam_update`, `adamw_update`, `adam_multi_tensor_update`,
  `nadam_update`, `adamax_update` and `sparse_adam_update` all already exist. The task was
  deleted before any code was written, but it should never have been created.

```bash
grep -rn "your_operator" src/AiDotNet.Tensors/Engines/DirectGpu/CUDA/Kernels/
grep -rn "public.*YourOperator" src/AiDotNet.Tensors/Engines/DirectGpu/CUDA/CudaBackend*.cs
```

If a fused kernel exists, the remaining opportunity is a **chain** it does not fuse
(clipping + unscale + step), not the operator itself.

---

## 1. Where wins actually come from

Measured across thirteen kernels against cuDNN in its CUDA-graph lane, true fp32, locked
clocks. The pattern is consistent enough to plan from:

| source of win | measured | why it holds |
|---|---|---|
| **fusion the competitor structurally cannot do** | 1.35× | cuDNN cannot fuse through its own call, so each epilogue stage costs it a launch and a full tensor round trip, against one instruction in a loop we already run |
| **specialisation where the competitor is generic** | 2.08–2.99× | depthwise is a special case cuDNN handles with general machinery |
| **memory-bound operators at a roofline** | 1.41× | max-pool at DRAM 94.6%: nobody beats the memory system, and we reach it |
| dense convolution / GEMM at large C,K | **0.33–0.65×** | cuDNN's home ground. Do not pick this fight |
| an operator our own kernels already fuse | **1.00×** | there is nothing left to fuse |

**Aim at the first three. The last two are where effort goes to die.**

### Why dense GEMM lost, refined

The dense-GEMM row was originally read as "cuBLAS is better tuned". It was two separate
deficits, and separating them changed what the next lever is:

1. **Instruction selection.** The emitter emitted no tensor-core instruction at all, so every
   generated matmul ran on the FP32 pipes against a competitor on the tensor cores — roughly
   twenty times the arithmetic throughput. That is now fixed (`PtxTensorCoreEmitter`), and it
   is worth 2.05–6.77× against our own previous lowering.
2. **Operand locality.** It was not enough. One warp per 16×16 tile with no staging moves
   `O(M·N·K)` operand bytes instead of `O(M·N·K / tile)`, and the measurement shows the cliff
   plainly: 11.8 TFLOP/s at 2048³ falling to **3.0** at 4096³, as the reused bands outgrow
   L2. cuBLAS holds 57.6 TFLOP/s at the same shape.

Standing against cuBLAS is **0.35× / 0.23× / 0.05×** at 1024³ / 2048³ / 4096³ — a loss, and a
widening one. See `TENSOR_CORES.md`. The lesson generalises: *having the right instruction is
necessary and not sufficient; check the data path before claiming a category is closed.*

---

## 2. Diagnose with stall reasons, never with throughput percentages

This is the single most expensive lesson on the branch. Two levers were designed, built,
verified correct, and measured useless because a percentage was read as a cause.

`--kernel-limiter` reports both. The throughput columns say which unit is *busiest*. The
stall columns say what the kernel is *waiting on*, and only the second is a cause.

| stall | meaning | lever |
|---|---|---|
| `mio_throttle` | load/store queue full | fewer memory instructions: vectorise, stage |
| `long_scoreboard` | waiting on global memory latency | `cp.async`, prefetch, more occupancy |
| `short_scoreboard` | waiting on shared memory | reduce shared dependency |
| `wait` | waiting on arithmetic results | more independent accumulators |
| nothing dominant | balanced | **no code-generator lever — needs a different algorithm** |

### What happened when this was ignored

Dense 3×3 sat at "L1 59%", which was read as "too many global loads". Per-dimension
shared-memory staging was built to cut them. It worked, verified at `0.000E+000`, and:

| | L1 | DRAM | SM |
|---|---|---|---|
| baseline | 64.08% | 4.31% | 53.24% |
| with staging | **77.45%** | 3.36% | 44.43% |

L1 went **up**. On NVIDIA hardware shared memory *is* L1TEX, so `ld.shared` is counted by
the very metric the lever was meant to relieve. The obvious replacement — register reuse —
moved it not at all: 64.27 / 64.11 / 64.20 at coarsening 2 / 4 / 8.

The stall breakdown said what neither could: **`mio_throttle` 3.03%**. The load pipe was
never the bottleneck, so no amount of load reduction could ever have helped. Across all
thirteen kernels `mio_throttle` never exceeds 3.8%, while `long_scoreboard` runs 66–88% on
seven of them.

**Rule: name the stall counter that justifies a lever before writing the lever.**

---

## 3. Apples-to-apples means the strongest path, including ours

A ratio is only as honest as its denominator.

- **Compare against the strongest competitor form.** PyTorch eager allocates an output per
  call and pays full launch overhead; the CUDA-graph lane removes both. Use the graph lane.
- **Compare at true fp32.** PyTorch defaults to `allow_tf32=True`, which routes convolution
  to tensor cores at 10-bit mantissa — a different operation from the exact fp32 we verify
  against an fp64 oracle. Set `allow_tf32=False`. (Measured: TF32 was not even faster here,
  27.55 µs vs 24.80 µs, because it pays layout transforms.)
- **Include our own existing kernel in the comparison.** This is what caught the SGD tie.
  The competitor is the best thing a user could otherwise call, and sometimes that is us.
- **Lock clocks and sample them at both ends.** A 2025→1770 MHz swing inside one kernel's
  three runs produced the intermittent 7.5% spreads that were briefly mistaken for a
  property of the kernel.
- **Refuse to report timings on a busy GPU.** A run taken while another process held 84% of
  the SMs produced a 64 µs 16K-element ReLU and a 466 µs 512-element reduction. Those
  numbers were discarded, not published. The harness now suppresses timings and prints why.

---

## 4. Correctness gates, in the order they catch things

1. **fp64 oracle, at the shape you release.** Verifying a small proxy and releasing a large
   one shipped two unexercised lowerings here — the strip-mined loop and shared staging —
   because both are chosen from extents that differ between shapes.
2. **Relative tolerance, scaled to the arithmetic.** An absolute `2e-3` rejected a *correct*
   split whose deviation was `8.575` — which is `5.6E-004` relative, ordinary fp32
   accumulation over 100,352 terms. That false negative hid a measured 17×. And a fixed
   `1e-6` is the wrong *shape* for a reduction: measuring one operator at two lengths gave
   `8.316E-007` over 64 terms and `3.335E-006` over 256, a 4.01× rise for 4× the terms. The
   bound is `max(1e-6, n · 1.2e-7)`.
3. **An independent reference for anything the oracle shares an assumption with.** The
   oracle and the emitter both filled out-of-range taps with zero. Zero is the identity of
   *addition*; under a maximum it is a candidate, so a padded max-pool returned 0 instead of
   the largest negative value — **in both implementations, in agreement, at `0.000E+000`**.
   A shared oracle cannot catch a shared mistake. Check against arithmetic written out by
   hand.
4. **Conventions, read from the consumer.** Max-pool indices are the *spatial* index
   `ih*inWidth + iw`, because the backward kernel decodes them that way; ties keep the first
   maximum. Either wrong compiles, produces plausible pooled values, and corrupts every
   gradient.
5. **Zero register spills**, audited through `nvdisasm` on the cubin you ship.

---

## 5. Choose lowerings by measurement, not by model

A static cost model picked lowerings four times on this branch and lost to the hardware
every time it was checked: it predicted a 2.78× occupancy penalty where 1.46× was measured,
picked a 4×8 tile slower than the 4×4 it replaced, called a transposed-conv tile worse on
two separate metrics when it was 1.12× faster, and preferred two split axes where one won by
1.41×.

`--kernel-autotune` emits candidates, requires them to agree numerically, times them, and
records the winner. Two bugs it exposed are worth repeating because both were silent:

- The agreement check used an **absolute** tolerance and rejected a correct 17× split.
- The winner cache was **written by catalog name and read by spec name**, which differ for
  every depthwise entry — so those kernels ran untuned while the cache reported them tuned.
  A cache miss is indistinguishable from "the modelled choice already won".

---

## 6. Promotion is per family, on that family's own evidence

One flag for a whole family set would route work to a kernel measured at 0.33×.
`DirectPtxConvolutionPromotion` records a decision per family, consulted *in addition to*
the feature flag and the architecture predicate, and an unrecognised family defaults to
withheld. Each exclusion carries its reason, and the dense-3×3 one names the balanced stall
profile so nobody retries the tuning that has already failed twice.

**A promotion nobody can reach is a note, not a speedup.** Depthwise and max-pool were
promoted for a full session before either had a call site; their wins existed only inside
the benchmark harness. Ship the dispatch with the promotion.

---

## 7. What the generated-kernel layer can express today

Reaching for it is cheaper than a hand-written kernel when the operator fits:

| feature | covers |
|---|---|
| index maps with `Window` / `TransposedWindow` | convolution, depthwise, transposed, pooling |
| `MatMul` + transposes, `BatchMatMul` | linear layers, LoRA |
| `ReduceSum` / `ReduceMax` / `ReduceMean` | reductions, global pooling |
| `PreReduce` (`Exp`, `Square`) + signed `PreBias` | softmax denominators, variance, squared error |
| activations: ReLU, Sigmoid, Tanh, Swish, GELU, Reciprocal, Rsqrt | epilogues, RMSNorm |
| `ReduceScale` | means, loss normalisation, `1/denominator` |
| extra outputs, **any number** (`ArgMaxIndex`, `AffineOfPrimary`) | pooling indices, optimizer state, multi-state steps |
| `CodegenAdjoint` | backward kernels derived, never hand-authored |
| `CodegenSplitReduction` | small-output long-reduction kernels |
| `ElementType` per binding (fp16 / bf16 storage, fp32 arithmetic) | mixed-precision inference, narrow activations against wide weights |

Storage is a property of each **binding**, not of the kernel, which is what lets one
kernel read fp16 activations against fp32 weights — the common decode shape. Arithmetic
stays fp32 regardless: an fp16 accumulator over a long reduction loses roughly three
decimal digits, and is a different operator rather than a cheaper one. Two details are
easy to get wrong and are covered by tests: bf16 is the *top half of the fp32 pattern*, so
it shifts rather than converting (no `cvt.f32.bf16` exists on this architecture), and
narrowing must round to nearest even — truncating biases every value toward zero. A narrow
binding is also excluded from the vector path, since `v4.f32` scales by four bytes and
would read twice the intended span. Measured on device at `0.000E+000` against an fp64
oracle that is given the *quantised* inputs, so the row reports the kernel rather than
fp16's rounding of its operands.

| tensor cores (`PtxTensorCoreEmitter`, `wmma` m16n16k16) | fp16 GEMM with a fused element-wise epilogue |
| `CodegenIndirectIndex` (gather/scatter, int32 index tensors) | embedding lookup and its backward, one-hot projection, sparse accumulation |
| `CodegenAlgebra` (complex, quaternion) | FFT-class kernels, complex linear algebra, rotation composition |

The tensor-core path is a *separate* emitter with its own recogniser, because `wmma` is
warp-collective and its fragment layout is opaque: a lowering that assumes which lane holds
which element still assembles, still runs, and is wrong. Anything it cannot express exactly
is refused **with a reason** and falls back to the scalar emitter. It is correct today and
still 0.35×–0.05× against cuBLAS; the remaining gap is operand locality, not instruction
selection, and it is not promoted anywhere until staging lands. See `TENSOR_CORES.md`.

Data-dependent indexing lives on the **binding**, not on the affine expression, and that
placement is the whole design. An affine expression is a closed-form function of the axes;
the bounds predicate, the index folding and the tensor-core recogniser all rely on it, and an
index fetched from memory has none of those properties. Three rules follow, each of which
protects against a failure that does not announce itself:

- **The loaded index is clamped unconditionally** before it reaches address arithmetic. That
  is not the caller's out-of-range policy — it is what stops a malformed index tensor forming
  an address outside the allocation. Predicating the load alone does not do it, because the
  address is computed either way.
- **A scatter store is atomic**, decided by structure rather than by the caller: an output
  dimension addressed at run time cannot be proven injective. A plain store passes every
  single-threaded check and then loses gradients when warps collide — a different wrong
  answer per run, which reads as flakiness. Verified at 4096-way contention onto one row.
- **An index tensor is a distinct element type** (`Int32`) and is refused as an arithmetic
  operand *at the spec*, not at the load site. The first version of that guard sat in
  `EmitLoad` and a test caught a kernel reading the index buffer as fp32 anyway: the emitter
  has scalar, vectorised, staged and coarsened load paths, so a guard on one is a guard on
  none.

Complex and quaternion arithmetic take a **separate path inside the same emitter**, and the
separation is the same judgement the tensor-core one made: the real path's value is a single
fp32 register threaded through the operand cache, shared staging, split reductions, coarsened
lanes and argmax tracking. Widening that everywhere would risk thirteen verified kernels to
gain nothing, because none of those mechanisms mean anything here — there is no order to
maximise over, the vector path assumes unit-stride scalars, and a complex operand's two
floats are already adjacent. The real path is untouched; only the product and the accumulator
widen.

Two things this got right only because they were checked properly:

- **The multiplication table is verified against the defining relations**, never against a
  second copy of the same formula. `i² = j² = k² = ijk = -1`, `ij = k`, norm multiplicativity
  `|ab| = |a||b|`, and associativity. A single wrong sign yields a kernel that runs and
  returns a rotation that is subtly not a rotation, and a self-consistent check would agree
  with it.
- **A real activation over a non-real algebra is refused**, not applied component-wise.
  Component-wise ReLU on a complex number is a *different operator*, not the same one
  generalised. Same for `Max` (no order) and pre-reduction transforms.

The output count is not capped, and the legacy single-argmax pair is now the *same*
mechanism — it folds into the extras list rather than living beside it, because two ways to
say "an extra output" is how one of them ends up unmaintained. Two outputs bound to one
parameter are refused at construction: both stores would land on the same buffer with no
ordering between them, giving plausible values and a different answer per run.

**Nothing on this list is now a reason to hand-write a kernel.** All five gaps that were
recorded here as unfixed — fp16/bf16, tensor cores, gather/scatter, complex/hypercomplex,
and three-or-more outputs — are closed and verified on device. What remains open is
*performance*, specifically tensor-core operand locality (§1), not expressiveness.

---

## 8. The checklist

Before opening a kernel PR:

- [ ] No existing AiDotNet kernel already fuses this operator (§0)
- [ ] The win category is one of the three that pay (§1)
- [ ] A named stall counter justifies each lever (§2)
- [ ] The competitor is the strongest form, at true fp32, on an idle GPU with locked clocks (§3)
- [ ] Verified against an fp64 oracle at the released shape, with a tolerance scaled to the arithmetic (§4)
- [ ] Anything the oracle could share a mistake with is checked against hand-written arithmetic (§4)
- [ ] Conventions read from the consumer, not assumed (§4)
- [ ] Zero spills in the shipped cubin (§4)
- [ ] Lowering chosen by `--kernel-autotune`, not by a model (§5)
- [ ] Promotion decided per family, with the dispatch site shipped alongside (§6)
- [ ] Losses reported as plainly as wins

The last line is the one that makes the rest worth anything. This branch reports six wins
and four losses, and the losses are what redirected the work.
