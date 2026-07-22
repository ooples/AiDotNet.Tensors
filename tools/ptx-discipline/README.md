# Direct-PTX discipline harness

A local + CI guard that mechanically enforces the quality rules every
hand-emitted PTX kernel in `src/AiDotNet.Tensors/Engines/DirectGpu/CUDA/Ptx`
must follow, so quality cannot silently regress ("drift") from one kernel to
the next.

## Run it

```bash
bash tools/ptx-discipline/check-ptx-discipline.sh      # Git Bash / Linux / CI
pwsh tools/ptx-discipline/check-ptx-discipline.ps1     # Windows wrapper
```

Exit code `0` = pass. Non-zero = a violation was found; the offending
`file:line` and the rule are printed.

## The rules

Each rule encodes a concrete defect that was previously caught in review, so
the harness is a regression test against those specific mistakes.

| # | Rule | Why |
|---|------|-----|
| 1 | No `shfl.sync.bfly.b32` on an `.f32` register | `shfl.*.b32` is a bit-manipulation op; its operands are `.b32` registers. Reinterpret the float through a `.b32` register (`mov.b32`) first. Passing an `.f32` register relies on assembler leniency and diverges from the house idiom. |
| 2 | Every `HasValidated*` predicate pins an exact `(major, minor)` | A kernel is measured/promoted on a specific SM. A whole-family compare (`== DirectPtxArchitectureFamily.Ampere`) would let it engage on hardware (e.g. SM80/SM87) it was never benchmarked on. |
| 3 | No kernel hardcodes `IsPromotedShape => true` | Promotion is earned by clearing `DirectPtxReleaseGate` across clean runs, never by a literal. Kernels ship fail-closed. |
| 4 | No kernel emits `.local` memory | The family is register-resident. (`DirectPtxResourceBudget.Validate` also enforces zero local bytes at JIT time; this catches it earlier, in source.) |
| 5 | No **new** whole-family architecture gate | *Baseline rule.* Existing family-level gates (decode / paged-prefill / attention-backward / rmsnorm) are tracked debt in `family-gate-baseline.txt`. The guard fails only if the count **increases**, so new kernels cannot add drift; lower the baseline as debt is paid down. |

## The per-kernel checklist (not yet mechanically enforced)

The guard covers what is cheaply checkable from source. A new PTX kernel PR
should also satisfy, and reviewers should confirm:

- An exact-shape ABI (`Require`) check and input/output aliasing rejection.
- A coverage manifest cell assigning its public backend API.
- A backend **integration** test (zero managed allocation, CPU-oracle parity,
  no-spill audit) **and** a **fallback** test (unsupported shape and disabled
  gate both route to the existing kernel).
- An architecture-matrix fail-closed test for its `HasValidated*` predicate.
- A benchmark experiment feeding the `DirectPtxReleaseGate` (>=1.10x median vs
  the strongest competitor, over three clean independent runs) before any
  promotion claim.

## Updating the baseline

When you migrate an existing family-level gate to an exact-SM `HasValidated*`
predicate, decrement the number in `family-gate-baseline.txt` accordingly. The
goal is to drive it to `0`.
