# PTX evolution campaigns, RTX 3080 (sm_86), 2026-11-01/02

These are the three approved campaigns from T-3 (#1573): GEMM-large, then GEMM in-graph, then conv 3x3, in that order. Each had a $50 budget. They were run with `--ptx-campaign` from `tests/AiDotNet.Tensors.Benchmarks/DirectPtxDenseLinearFullExperiment.PtxCampaign.cs` using AiDotNet.Evolution.Ptx 0.7.0. The proposal model was `claude-opus-5-5`, called through `claude -p`. Every vendor comparison is strict FP32: deterministic mode, cuBLAS in PEDANTIC math. Each comparison uses paired ABBA rounds on CUDA events, run both eagerly and as CUDA-graph replays. The raw records are the `*.campaign.jsonl` files next to this README.

## Measure first (no model calls)

These numbers come from `--ptx-campaign measure`, which uses production routing's verdict. "Speedup" is the vendor's time divided by our kernel's time, so 1.0 means parity and above 1.0 means we win.

| shape (MxNxK) / op | eager | graph |
|---|---|---|
| 2048³ | 0.984 | 0.968 |
| 4096³ | 1.012 | 1.015 |
| 1024³ | 1.147 | 1.098 |
| 2048x4096x4096 | 0.961 | 0.967 |
| 2048x11008x4096 | 0.998 | 0.968 |
| 2048x4096x11008 | 1.085 | 1.079 |
| 128x4096x4096 | 0.874 | 0.852 |
| 3136x64x576 | 0.765 | 0.734 |
| conv 3x3, 64->64 ch, 16x16, vs `backend.Conv2D` (serves NVRTC fused Winograd F(2,3), not cuDNN; see correction below) | 1.60 | 0.528 |

## Results

| campaign | target | spend | calls (ok / failed) | best vs incumbent | verdict |
|---|---|---|---|---|---|
| GEMM-large | 2048³ | $21.83 | 4 / 4 | 1.0005x | not promoted; median 1.000x, lower 5% 0.949x |
| GEMM in-graph | 128x4096x4096 | $50.00 | 7 / 9 | 1.017x | not promoted; median 1.018x, lower 5% 1.012x |
| conv 3x3 | 64->64 ch, 16x16 | $7.27 | 16 / 0 | 1.441x | **qualifies**: median 1.299x, lower 5% 1.161x |

Total spend: $79.10 of the $150 approved.

### Conv 3x3 winner against `backend.Conv2D` (same buffers, output checked first)

| | incumbent | winner | `backend.Conv2D` |
|---|---|---|---|
| eager | 14.98 us (1.57x cuDNN) | 10.13 us (1.92x) | 19.05 us |
| CUDA graph | 13.60 us (0.52x) | 8.48 us (0.83x) | 7.08 us |

The winner is `conv3x3-winner.ptx`, from generation 14. Its stated changes:
- It unrolls 4 input channels per iteration, so one load-latency wait covers four channels.
- It rotates the FMAs across 6 accumulators instead of one serial chain.
- It predicates the out-of-range taps instead of zeroing them.

Block geometry and shared memory are unchanged. Its maximum relative difference from `backend.Conv2D` is 2.5e-7. As a graph replay, `backend.Conv2D` still wins at this shape.

**Correction.** This section originally labeled the baseline "cuDNN". In fact the driver timed `backend.Conv2D`, and the conv forward slot serves `nvrtc.conv2d.winograd_f23_fused` at this shape, not cuDNN. Each route timed separately against the winner (same run, `--vendor-check-only=1 --seed-ptx=conv3x3-winner.ptx`, RTX 3080, fp32, graph replay; eager in parentheses):

| route | route us | winner us | winner speedup |
|---|---|---|---|
| `nvrtc.conv2d.winograd_f23_fused` (the slot's pick) | 7.08 (18.2) | 8.47 (10.2) | 0.84x (1.81x) |
| `cudnn.conv2d.default` (fixed implicit-precomp GEMM on main) | 22.6 (124) | 8.48 (10.4) | 2.64x (11.6x) |
| `cudnn.conv2d.algo1` | 22.6 (120) | 8.48 (10.1) | 2.66x (11.8x) |

For reference, the best cuDNN algorithm (algo6, Winograd) runs 13.9 us in graph. #1657 makes the cuDNN default use cuDNN's own heuristic pick (14.1 us here).

### GEMM in-graph: the campaign's own kernels against cuBLAS

The verdict's re-tune picked a different kernel (`b128x64x32` split), so the campaign incumbent was timed directly with `--vendor-check-only=1`. At 128x4096x4096 it runs 0.85x cuBLAS both eagerly and in-graph (289 us vs 236 us), and its output is bit-identical to strict cuBLAS. The best rewrite improved on it by 1.7%. That is not enough to close the 15% gap, and below the promotion bar.

## What went wrong, and the driver fixes (all on this branch)

1. **The seed failed its own correctness check, so the run made 0 proposals.** Async-copy GEMM configurations (`as2r`/`as3r`) stage tiles in dynamic shared memory. The contract launched them with none, which gave CUDA_ERROR_ILLEGAL_ADDRESS. The contract now passes `config.SharedBytes`, and the driver checks the seed before any call ("seed_check" record).
2. **Paid replies were discarded at the output-token cap.** `PtxProgramEvolutionOptions.MaxOutputTokens` defaults to 16,384, and the CLI counts the model's reasoning as output. In the GEMM-large run, 3 of the 4 parsed replies ran to 16.5k-21.5k tokens and were rejected after they had been paid for. The cap is now 65,536 (`--max-output-tokens`); the dollar caps bound the spend.
3. **Calls timed out.** Once full replies were admitted, `ClaudeCodeChatClientOptions.Timeout` (5 minutes) ended 9 of the 16 paid in-graph calls. It now defaults to 20 minutes (`--call-timeout-min`). The conv campaign ran with both fixes: 16 of 16 calls succeeded, and spend was $7.27.
4. **Failed calls left no record of why.** The search record now has `call_failures`.
5. **The vendor comparison could be empty or vacuous.** The GEMM verdict went through production routing's own re-tune, so it described a different kernel, or nothing. The verdict now times the campaign's incumbent and winner directly. The output check writes the generated kernel into its own buffer: a kernel that wrote nothing into the vendor's buffer would otherwise match.

GEMM-large ran before fixes 2-5, and GEMM in-graph before 3-5. Both stopped at their generation limit (8 and 16).
