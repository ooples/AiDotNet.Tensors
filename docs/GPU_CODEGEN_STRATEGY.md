# GPU codegen strategy: how we actually beat cuDNN/PyTorch

Status: decided 2026-07-24. Supersedes the "hand-write one PTX kernel per shape" default.

This document exists because the #841 direct-PTX campaign produced a working but
structurally wrong process. The numbers below are measured on this repo and this box
(RTX 3080, sm_86, driver 13030, CUDA 13.3), not estimated.

---

## 1. Diagnosis of the current approach

### 1.1 Hand-written PTX is not "close to the machine"

PTX is a **virtual ISA**, not machine code. `ptxas` owns register allocation and
instruction scheduling. Direct evidence from the #841 campaign: declared register
budgets of 32 / 40 / 48 produced **actual** SASS register counts of 40 / 55 / 72 / 80.
We did not choose those numbers; the compiler did.

So hand-written PTX buys neither machine-level control nor language-level safety.
The real machine code is SASS.

### 1.2 The cost, measured

| metric | value |
|---|---|
| PTX emitter files | 65 |
| PTX emitter LOC | 24,031 |
| `s.AppendLine` PTX-emission calls | 5,016 |
| distinct kernels produced | 43 **shape-locked specializations** |

Untyped string concatenation has a demonstrated failure mode. During the grouped
deformable refactor an index expression was hand-recomputed and lost a `* DeformGroups`
factor. Result: the in-kernel bounds guard retired half the threads and the tail of
`dOffset`/`dMask` was never written (`expected 0.00884, actual 0`).

**Every structural gate passed on the broken kernel** — cubin export, PTX<->cubin
identity, and the nvdisasm SASS zero-spill audit. None of them check numerics. A typed
IR with index maps makes that class of defect unrepresentable.

### 1.3 We already own a compiler, and the PTX work bypasses it

```
Compilation/Codegen/Ir/   CodegenGraph, CodegenNode, CodegenOpKind,
                          CodegenLowering, SymbolicShapePropagation
IKernelEmitter.Emit(CodegenGraph, CodegenElementType) -> CodegenEmitResult
backends:                 AvxCs, Glsl, Hip, Msl, Triton, Wgsl, CudaGraph, Aot
                          ^ no PTX/CUDA emitter
also:                     FusionPatternRegistry, CpuFusionPass, 4 autotuners
                          CpuJit/X86Emitter.cs - 773 LOC emitting raw x86 bytes
```

The codebase already emits real machine code on CPU. The GPU path regressed to string
concatenation and cannot reach the fusion pass or the autotuners.

### 1.4 The IR's actual gap

`CodegenOpKind` covers pointwise, activations, reductions, matmul, softmax, SDPA and
movement. It has **no conv/gather op and no iteration space or affine index maps**.
The existing GPU emitters are thin pointwise walkers (HipEmitter 91 LOC,
TritonEmitter 196 LOC).

**This is the single blocking abstraction.** Without index maps we can neither generate
conv-class kernels from IR nor fuse a conv with its epilogue generically. With them,
both fall out of the same work. Triton solves it with `tl.arange` + masks, TVM with
compute/schedule split, MLIR with affine maps.

---

## 2. Competitive analysis: where PyTorch is structurally weak

Stack: eager ATen (hand-written CUDA C++) -> **cuDNN/cuBLAS** for heavy math ->
`torch.compile`/Inductor -> Triton -> LLVM NVPTX -> PTX -> ptxas -> SASS.

Exploitable flaws, ordered by leverage:

1. **Eager = one launch + a full HBM round-trip per op.** Dominates memory-bound work.
2. **Inductor cannot fuse *through* cuBLAS/cuDNN.** It fuses pointwise/reductions, then
   hands heavy math to a closed vendor library. The GEMM/conv boundary is a hard
   fusion wall. This is the structural gap we exploit.
3. **Triton shares our ceiling** — tile-level over LLVM NVPTX + ptxas, so no register or
   scheduling control, and a coarse autotune grid.
4. **Generality tax** — cuDNN serves all shapes, so runtime dispatch and generic guards.
   We specialize exactly: bake shape constants, delete bounds checks, precompute offsets.
5. **Toolchain fragility** — measured in PR #886 on this box, `torch.compile` was *slower
   than eager* (366-532 us vs 215 us) because triton could not find a full CUDA toolkit.

### 2.1 Our own scoreboard says where we win

| our kernel | vs cuDNN |
|---|---|
| 3x3 conv, after heavy optimization | **1.16x slower** |
| 1x1 fused conv+bias+ReLU | **1.60x faster** (1.09x vs cuDNN doing *less* work) |

We lose head-to-head on dense math NVIDIA has tuned in SASS for 15 years. We win where
fusion and specialization are forced. Chasing cuDNN on conv3x3 is fighting on their
ground.

**Corollary:** cuDNN has no kernel at all for the HRE operators (Born-retrieval readout,
content-conditioned spectral gating, DSB pilot-reinjection, Pyramid). Any PyTorch
implementation is a *composition* of many eager ops - flaws #1 and #2 simultaneously.
That is where a 10x gap is available, not 1.1x.

---

## 3. Decided plan

### Track 1 - Fusion frontier (primary perf thesis)
Persistent whole-block mega-kernels that remove HBM round-trips PyTorch cannot.
**Benchmark against a PyTorch *composition*, not a single cuDNN call** - the composition
is the honest peer for a fused block.

### Track 2 - Compiler: PtxEmitter behind the existing IR (floor), Rust evaluated head-to-head
- Extend the IR with an iteration space + affine index maps (section 1.4).
- Add `PtxEmitter : IKernelEmitter` so kernels are generated, not hand-written; this
  reuses `FusionPatternRegistry` and the autotuners and kills the untyped-index bug class.
- **Before porting all kernels**, run the small-scale bake-off in section 4 against a
  Rust implementation and decide on measured evidence.

### Track 3 - HRE novel operators (mandatory, independent)
Fused kernels for Born-retrieval, spectral gating, DSB, Pyramid. Proceeds regardless of
Tracks 1-2 because it is where the competition must compose and where the 100x thesis
lives.

### Track 4 - SASS (option, not committed)
True machine-level control. Precedent: MaxAs (Maxwell) reportedly beat cuBLAS near
theoretical peak; CuAssembler covers Turing/Ampere by round-tripping `nvdisasm` to
recover encodings and control bits. **We already emit exactly that input** - the audit
runs `nvdisasm --print-code --print-instruction-encoding`. Escalate only after Track 2,
and note it is a last-10-30% play on compute-bound kernels, which is where we currently
*lose* to cuDNN.

### Track 5 - Existing 43 kernels
Run the #863 timing sweep on all 43 once the GPU is idle, publish honest median/P95/
TFLOPs, then triage which survive into the new architecture.
(Blocked: a long-running `qwen_ffn_weight_pq.py` holds the GPU.)

---

## 4. Bake-off protocol: C# PtxEmitter vs Rust compiler

**Target kernel:** depthwise Conv2D 3x3 + bias + ReLU, N2/C8/H8/W8, sm_86.
Chosen because it has affine gather indexing, a 9-tap reduction, and an epilogue - the
exact shape of "conv + epilogue fusion" PyTorch must split - and it already has a
verified fp64-oracle test and a released cubin.

**Measured baseline (hand-written, committed cubin
`ce22c03507f3282162b753340f98e7bcd71c0ee24b81c8b6b6f2f7928c73ff3d`):**

| metric | hand-written |
|---|---|
| registers | 40 |
| SASS instructions | 336 |
| LDG / STG | **19** / 1 |
| LDS / STS | 0 / 0 |
| local load/store (spill) | 0 / 0 |
| source cost | 208 LOC, 61 `AppendLine` |

**Finding already visible in the baseline:** 19 global loads for a 9-tap depthwise conv,
with zero shared-memory use. The 9 filter taps are warp-uniform yet are re-loaded from
global per thread instead of being hoisted into registers. A scheduler should cut LDG
from 19 toward ~10. The human writing 61 `AppendLine` calls did not catch it. This is
the compiler's value proposition in measured numbers.

**Gate metrics — static, so they do NOT need an idle GPU:**
1. numerics vs the existing fp64 CPU oracle (must match, non-negotiable)
2. registers, SASS instruction count, LDG/STG/LDS/STS, spills (`nvdisasm`)
3. source cost to express the kernel (LOC / constructs)
4. wall-clock median + P95 vs cuDNN — **added when the GPU frees**, per #863 gate 10

Decide C# vs Rust on 1-3 first; confirm with 4. Do not port the remaining kernels until
this gate is passed.

---

## 5. Standing rules carried forward from #863/#886

- A kernel is not "done" on green structural gates. Cubin export, PTX<->cubin identity
  and SASS zero-spill **do not check numerics**. Always re-run the fp64-oracle suite
  after touching an emitter; it costs ~25 s for 55 tests.
- Release the cubin for the shape you actually verified. Exporting a stride=2 cubin while
  only stride=1 was tested left a codegen branch with zero numerical coverage.
- Status fields must describe the code. `ExperimentalDirectPtx` means a dispatch hook
  exists; `KernelReleasedNotRouted` means kernel + cubin exist but the public API still
  takes the baseline path 100% of the time.
- No perf number in a PR that was not measured on an idle GPU.
