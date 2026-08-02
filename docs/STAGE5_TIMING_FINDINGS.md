# Stage 5 timing: why the sweep as specified cannot produce actionable data

Measured 2026-07-25 on an idle RTX 3080 (sm_86, driver 610.47, CUDA 13.3,
torch 2.12.1+cu130 / cuDNN 9.2). cuDNN was made loadable by putting torch's bundled
`torch/lib` on PATH, so our in-process baseline and the PyTorch lane use the *same*
cuDNN build.

## 1. The completed codegen bake-off (final)

Depthwise Conv2D 3x3 + bias + ReLU, N2/C8/H8/W8.

| gate metric | hand-written | C# PtxEmitter | Rust ptxgen | verdict |
|---|---|---|---|---|
| numerics vs fp64 oracle | pass | pass, exact (0E0) | pass, exact (0E0) | tie |
| registers / thread | 40 | 40 | 40 | tie |
| SASS instructions | 168 | **160** | **160** | generated −4.8% |
| LDG / STG | 19 / 1 | 19 / 1 | 19 / 1 | tie |
| LDS / STS | 0 / 0 | 0 / 0 | 0 / 0 | tie |
| spills (local ld/st) | 0 / 0 | 0 / 0 | 0 / 0 | tie |
| wall-clock median, 3 interleaved runs | 21.4 / 21.8 / 21.4 us | 20.7 / 21.9 / 20.8 us | same emitter output | parity, ~1.02x |
| source cost, this kernel | 208 LOC, 61 `AppendLine` | 10-line spec | 10-line spec | ~20x less |
| source cost, IR + emitter | n/a | 738 code lines | 544 code lines | Rust −26% |
| cubin equality | — | byte-identical to Rust | byte-identical to C# | `943a8e863b1a96d2…` |

Gate was "no worse than baseline on registers, instructions and LDG": **PASS**, and
strictly better on instructions after interval-analysis bounds inference (56 guards
elided). C# selected for production on integration cost; Rust retained for a
possible SASS backend.

## 2. The screen: our fused 1x1 vs cuDNN, five observations

N1/C64/H16/W16/K64, `PtxFusedConv2DNchwK1Kernel` vs PyTorch cuDNN CUDA-graph.

| observation | ours median | ours P95 | cuDNN-graph median | cuDNN-graph P95 | ratio |
|---|---|---|---|---|---|
| A | 67.7 | 155.1 | 41.98 | 213.0 | **0.62x** |
| B | 25.2 | 265.0 | 28.67 | 92.2 | 1.14x |
| C | 38.2 | 222.9 | 103.42 | 332.8 | 2.71x |
| D | 37.3 | 98.4 | 63.49 | 233.5 | 1.70x |
| E | 32.2 | 92.5 | 63.49 | 274.4 | 1.97x |

**The verdict changes sign across runs.** The competitor alone spans 28.7–103.4 us
(3.6x); ours spans 25.2–67.7 us (2.7x); every P95 is 2.5–8x its own median.

Reporting only C/D/E would show a clean 1.70–2.71x win that also satisfies the
P95 half of the #863 gate. That would be cherry-picking: A and B were measured
first, on the same idle device, and contradict it.

## 3. Root cause: every released shape is too small to measure

The released cubin shapes were chosen for correctness-test convenience and to
satisfy the `N*...%256` grid constraint. They are tiny:

* bake-off kernel: 1024 threads = **4 blocks**
* screen kernel: N1/C64/16x16/K64 = 16,384 outputs

An RTX 3080 has 68 SMs. A 4-block launch occupies about 6% of the device, and the
measured 21–38 us is dominated by launch and synchronisation latency, not kernel
execution. Two kernels producing *bit-identical* output measured 1.57x apart on
this hardware purely from measurement ordering (see section 4).

**Consequence: running the Stage 5 sweep across all 43 released shapes would
produce 43 rows of launch-overhead noise, not triage data.** It would look like
evidence and would not be. The sweep is therefore not run as specified.

## 4. Measurement-protocol rules earned the hard way

1. **Never compare sequentially.** Timing all of A then all of B reported the
   generated kernel 1.57x slower than the hand-written one; interleaving
   sample-by-sample showed parity. Their outputs are bit-identical, so the entire
   difference was clock/thermal drift between blocks.
2. **One run is not a data point.** Five observations of the same comparison
   spanned 0.62x to 2.71x.
3. **Report P95 next to median always.** A P95 of 8x the median means the median
   is not describing the distribution.
4. **Cross-process comparisons are weaker still.** Our number comes from a .NET
   process and the competitor from a Python subprocess; they are not interleaved
   and cannot be.

## 5. What Stage 5 should become

Before any sweep produces usable numbers, the released specialisations need
**production-representative shapes**, not test-convenience shapes. Concretely:

* re-export each family at a shape that fills the device (hundreds of blocks,
  e.g. the ResNet c64 N32/C64/56x56/K64 already used by the register-blocked cell);
* keep the small shapes as the correctness specialisations;
* time with interleaved A/B inside one process wherever both sides can run there;
* require three clean runs and report the full spread, not the best.

This is a precondition for Stage 3 (the fusion pilot) as well: its gate is
">= 1.10x median vs a PyTorch composition", and at current shapes the noise floor
is several times that threshold.

## 6. Phase 0.5 — the harness is now calibrated (measured 2026-07-24)

Sections 2-4 established that the harness could not resolve the 1.10x gate. Rather
than assume a protocol fix works, `--bench-calibrate` measures the instrument against
two known-truth comparisons on a device-filling shape (N32/C64/56x56 = 25,088 blocks,
vs the 4 blocks of the released shapes):

* **null test** — the same kernel against itself. True ratio is exactly 1.000x, so
  any deviation is the harness's own noise floor.
* **known-ratio test** — the same kernel at C=64 vs C=70. True work ratio is exactly
  70/64 = 1.09375x, deliberately close to the 1.10x gate.

Both variants use the *generated* kernel, so kernel differences cannot confound the
measurement of the instrument.

### What actually fixed it, in the order it was measured

| protocol | null err | ratio err | worst P95/median | gates |
|---|---|---|---|---|
| device-filling shape + interleaved, 1 launch per timed region | 2.44% | 5.36% | **5.50** | 0/3 |
| + 50 launches per timed region | 2.47% | 2.81% | 1.50 | 1/3 |
| + counterbalanced slot order | **3.78%** | 2.75% | 1.60 | 2/3 |
| + paired within-sample ratio estimator | **1.05%** | **2.43%** | 1.80 | **3/3 PASS** |

Three findings worth keeping:

1. **Batching launches per timed region was the single biggest win** (tail 5.50 ->
   1.50). Timing one launch with a CPU stopwatch around a synchronize measures launch
   API + sync latency + OS scheduler jitter; at 50 launches per region the sync cost
   is amortised and kernel execution dominates.
2. **Counterbalancing slot order made the null test worse, not better** (2.47% ->
   3.78%). The hypothesis that slot-1-vs-slot-2 position bias drove the residual was
   wrong; alternating merely mixed two slightly different distributions into each
   array and destabilised their medians.
3. **The estimator was the real defect.** `median(A)/median(B)` compares two
   distributions gathered across the whole run, so clock drift during the run leaks
   directly into the ratio. Taking the ratio *within* each sample pair -- two regions
   microseconds apart -- cancels drift, and the median over pairs is outlier-robust.
   This alone took the noise floor from 3.78% to 1.05%.

### The number that matters

**Noise floor 1.05%.** Differences below roughly 2-3% are not claimable on this rig;
a 1.10x gate is ~10x the noise floor and is therefore resolvable. The known-ratio
test lands at 1.067-1.076x against a true 1.09375x -- a consistent ~2% underestimate,
inside the 3% gate but a reminder that the harness is slightly conservative, which is
the safe direction for a performance claim.

Phase 0.5 gate: **MET**. Downstream perf claims may now be made with this protocol,
and only with this protocol.
