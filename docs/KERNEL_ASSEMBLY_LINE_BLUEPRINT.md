# Blueprint: the kernel assembly line, and the breakthrough it enables

Target: a kernel library that beats every mainstream stack on raw performance, and a
production model where adding a kernel is a **declaration**, not an implementation —
so the same pattern and the same performance reach all ~800 kernels and every open PR.

Status: Phase 0 shipped (index-map IR + PTX emitter + verified bake-off). Everything
below Phase 1 is proposed.

---

## 1. The thesis

> **A kernel stops being an artifact you write and becomes an artifact you declare.
> The declaration generates the kernel, its numerical oracle, its bounds proofs, its
> machine code, its safety audit and its benchmark. And because every kernel is a
> declaration in one algebra, the whole model can then be compiled into a single
> GPU-resident program instead of a sequence of launches.**

Two layers, and the ordering is the point:

* **Layer 1 — the assembly line.** One spec algebra, one lowering pipeline, seven
  auto-derived artifacts per kernel. This is what makes 800 kernels tractable at all.
* **Layer 2 — the breakthrough.** Those specs compose into **one persistent kernel**
  with a device-side work scheduler. No launch boundary between ops, weights resident
  across ops, no pipeline drain.

You cannot megakernel 800 hand-written kernels. You *can* megakernel 800 specs.
Layer 1 is the thing that makes Layer 2 possible, and Layer 2 is the thing nobody
else can copy quickly — because they have no unified spec layer to do it from.

### Why this is not incremental

Every mainstream stack — cuDNN, cuBLAS, CUTLASS, Triton, TVM, torch.compile —
shares four structural limits:

| limit | who has it | what it costs |
|---|---|---|
| kernel = launched function | everyone | full machine drain/refill per op |
| compiler stops at PTX | Triton, TVM, Inductor, us today | `ptxas` owns regalloc + scheduling; everyone clusters at the same ceiling |
| fusion breaks at the vendor library | Inductor, XLA | conv/GEMM boundary is a hard wall |
| specialisation happens at runtime | Triton, torch.compile | JIT warmup, and generic fallbacks when the search is too costly |

We attack all four, in the order that makes each next one cheaper.

---

## 2. Layer 1 — the assembly line

### 2.1 One declaration, seven derived artifacts

A kernel author writes **only** the middle column.

```
                    ┌─ 1. kernel PTX/cubin        (PtxAffineEmitter)
                    ├─ 2. fp64 numerical oracle   (Spec.Interpret)
   CodegenKernelSpec├─ 3. bounds proofs           (interval analysis, derived)
   (10-40 lines)    ├─ 4. launch grid             (IterationSpace.TotalThreads)
                    ├─ 5. correctness test        (generated: emitter vs oracle)
                    ├─ 6. SASS audit + cubin hash (existing #863 pipeline)
                    └─ 7. benchmark + competitor  (generated harness)
```

**The spec is the test.** `Interpret()` is the semantic definition, so the oracle is
free and can never drift from the kernel. Today we hand-wrote 43 oracle tests and
they are the only reason two shipped defects were caught at all; under the assembly
line they cost zero lines.

**The spec is the proof.** Bounds guards and the launch grid are *derived*, so the
two defect classes that shipped in #841 — a guard that disagreed with the grid, and a
guard that was simply forgotten — are unrepresentable.

### 2.2 Spec algebra: what must be expressible

Phase 0 covers affine indexing + reduce + epilogue. To reach 800 kernels the algebra
needs, in rough priority order:

1. **Affine / quasi-affine indexing** — shipped (conv, transposed conv, gather,
   broadcast, transpose, im2col).
2. **Tiling and memory-space placement** — declare that a tensor region is staged in
   shared memory or registers. Today 35 of 43 kernels use *zero* shared memory; that
   is a scheduling gap, not a design choice.
3. **Data-dependent gather** — deformable offsets, embedding lookups, paged KV.
   Deliberately outside the affine layer; needs an explicit `Gather` node so the
   emitter declines rather than mis-lowers.
4. **Cooperative reductions** — warp shuffle / block tree, chosen by the scheduler.
5. **Tensor-core tiles** — MMA fragments as a first-class tile type.
6. **Multi-output and in-place** — backward kernels producing several gradients.

### 2.3 The scheduler is separate from the algebra

Compute is declared once; the *schedule* (tile sizes, unroll factors, memory
placement, vectorisation width) is searched. This is the TVM compute/schedule split,
and we already own both halves of the machinery: `FusionPatternRegistry` plus four
autotuners plus a content-addressed cubin store to memoise into.

Concrete first scheduler wins, all measured as missing today:

* **Stage warp-uniform operands.** 19 LDG for a 9-tap filter whose taps are
  warp-uniform.
* **Vectorise contiguous access.** `v2/v4` loads on the contiguous axis.
* **Raise occupancy where it is capped.** 12 of 43 kernels exceed 42 registers.

---

## 3. Layer 2 — the breakthrough: one resident program

### 3.1 What it is

Compile a whole model region into **one persistent kernel**. Blocks are long-lived
workers that pull work items from a device-resident queue and execute a compiled
schedule. The host enqueues one launch per *step*, not per *op*.

```
   today            one launch per op, full drain between ops
                    HBM round-trip per op, weights re-read every time

   resident         one launch per step
                    device-side scheduler walks the op DAG
                    weights stay in SMEM/registers across ops
                    producer→consumer handoff never touches HBM
```

### 3.2 Why it is a real advantage and not a trick

* **CUDA graphs do not do this.** They amortise the *launch API*, not the pipeline
  drain and not the HBM round-trip. This is the most common misconception about the
  idea and the reason the win is available.
* **Inductor cannot do this** past a cuBLAS/cuDNN call, because it does not own the
  library kernel.
* **Persistent kernels exist but are hand-written**, one-off, per-model. Doing it
  *compiler-generated from a spec algebra* is the copyable part.

### 3.3 What it demands from Layer 1

Cross-op residency requires a global view of register and shared-memory budget, which
requires every op to be described in one algebra with a schedulable memory model.
That is exactly Layer 1. This is why the ordering is not negotiable.

---

## 4. Layer 3 — breaking the `ptxas` ceiling

Everyone stops at PTX. Owning SASS means owning register allocation, instruction
scheduling and the dual-issue/stall control bits — the last place a large factor is
still available on compute-bound kernels.

* Precedent: MaxAs (Maxwell) hand-scheduled SASS beat cuBLAS; CuAssembler covers
  Turing/Ampere by round-tripping `nvdisasm` to recover encodings.
* We already emit exactly that input — the audit runs
  `nvdisasm --print-code --print-instruction-encoding`.
* Sequencing: only worth it **after** Layer 1, because a SASS backend behind a spec
  algebra upgrades 800 kernels at once, whereas a SASS backend behind hand-written
  kernels upgrades one.

---

## 5. Layer 4 — the shipped specialisation database

Everyone else JITs at runtime or ships generic kernels. We already commit
content-addressed cubins. Scale that deliberately:

* thousands of exact specialisations, constants baked, **zero runtime JIT**;
* autotune results memoised into the same content-addressed store, so tuning is
  cross-run, cross-machine and shippable;
* the store is keyed by PTX hash, so a spec change invalidates exactly the affected
  artifacts and nothing else.

This is an architectural advantage most frameworks structurally cannot copy: they
have no artifact identity model, so they cannot ship a million pre-tuned kernels.

---

## 6. Phase plan with hard gates

| phase | deliverable | gate |
|---|---|---|
| **0. shipped** | index-map IR, PTX emitter, C#/Rust bake-off | generated == hand-written on regs/instrs/LDG; byte-identical cubins ✔ |
| **0.5 measurement** | device-filling shapes, interleaved in-process A/B, 3 clean runs | the harness can resolve a 1.10x difference with P95 < 2x median |
| **1. conveyor MVP** | spec → 7 artifacts, one command; 3 kernels migrated | each migrated kernel matches its old oracle and is no worse on every static metric |
| **2. scheduler** | SMEM staging, vectorisation, occupancy targeting | LDG on the depthwise kernel drops 19 → ~10; measured win on a device-filling shape |
| **3. resident program** | 2–3 op chain in one persistent kernel | ≥1.10x median vs the same chain as separate launches, **and** vs a PyTorch composition |
| **4. assembly line** | migrate by family, PR by PR | per family: all specs pass, no static-metric regression, evidence auto-published |
| **5. SASS backend** | own regalloc + scheduling | ≥1.10x vs our own PTX path on a compute-bound kernel |

**Phase 0.5 is not optional and not glamorous.** Five runs of the same comparison
this session spanned 0.62x–2.71x, and two kernels with bit-identical output measured
1.57x apart from sequential ordering alone. Until the harness can resolve 1.10x,
every claim in phases 2–5 is unfalsifiable.

---

## 7. The conveyor: how a kernel enters the line

### 7.1 New kernel

```
1. write the spec            (10-40 lines, one file)
2. dotnet run -- --kernel-verify <spec>     # oracle + interval proofs, NO GPU needed
3. dotnet run -- --kernel-release <spec>    # cubin + hash + SASS audit + manifest row
4. dotnet run -- --kernel-bench <spec>      # interleaved A/B + competitor lane
```

Nothing is hand-written: no PTX, no oracle, no bounds guard, no launch grid, no test.

### 7.2 Migrating an existing kernel or open PR

The migration is mechanical *because the old kernel is the reference*:

```
1. express the kernel as a spec
2. run the OLD kernel and the NEW spec against the same fp64 oracle   -> must agree
3. compare static metrics old vs new (regs / instrs / LDG / STG / spills)
4. accept only if no metric regresses; investigate if any does
5. delete the hand-written emitter; the spec replaces ~200-300 LOC each
```

This is exactly the protocol already validated on depthwise Conv2D 3x3: identical
numerics, identical registers/LDG, fewer instructions, ~20x less source.

Applied across the estate: **24,031 LOC of hand-written PTX becomes roughly 800
specs**, and the ~252,000 lines that finishing by hand would have required never get
written.

### 7.3 Ordering the 23 open PRs

Migrate by **family**, not by PR age, because a family shares an index-map shape and
therefore shares a spec template. Suggested order — cheapest template first, highest
leverage first within that:

1. pointwise / activation / normalisation *(pure affine, no reduction)*
2. convolution family *(template exists today)*
3. GEMM / matmul tiles *(needs Phase 2 tiling)*
4. attention *(needs cooperative reductions + tensor-core tiles)*
5. data-dependent gather: deformable, embeddings, paged KV *(needs Gather node)*

---

## 8. What would make a competitor copy us

Ranked by how hard it is for them to replicate:

1. **Compiler-generated resident programs.** Requires a unified spec algebra they do
   not have; the hard part is the algebra, not the persistent kernel.
2. **Spec-derived verification.** "Your oracle cannot drift from your kernel because
   both come from one declaration" is a correctness argument no competing stack makes.
3. **Shipped specialisation database.** Needs an artifact identity model.
4. **SASS ownership.** Replicable in principle; expensive, per-architecture, and
   nobody wants to maintain it — which is exactly why it stays a moat.

---

## 9. Honest risks

* **Measurement first.** Phase 0.5 gates everything. This is the single largest risk
  to the whole plan and it is unglamorous.
* **The algebra can become a straitjacket.** Every op that does not fit becomes an
  `Opaque` escape hatch, and enough escape hatches means we rebuilt the old world.
  Track the ratio of specs to opaques as a health metric.
* **Resident programs have real limits**: register/SMEM budget is shared across the
  whole fused region, so a bad global allocation is worse than separate launches.
  Needs a real allocator, not a heuristic.
* **SASS is per-architecture and undocumented.** Ampere today; Blackwell is another
  encoding table. Do not start it until Layer 1 makes it leverage 800 kernels.
* **We currently lose to cuDNN on dense conv** (3x3 measured 1.16x slower). The plan
  must beat them on *structure* — residency and fusion — not by out-tuning fifteen
  years of vendor SASS work at their own game.

---

## 10. Immediately next

1. **Phase 0.5**: device-filling shapes + interleaved in-process A/B, three runs, and
   prove the harness resolves 1.10x. Everything downstream depends on it.
2. **Phase 1 MVP**: the four conveyor commands, and migrate three kernels of
   different families to prove the template generalises.
3. **Phase 3 spike in parallel**: a two-op resident chain, hand-assembled from two
   specs, purely to measure whether the residency win is as large as predicted before
   committing the compiler work to it.
