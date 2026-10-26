# FP16 accumulation and GPU "unlock" projects (#1549)

GeForce cards issue FP16 tensor-core MMA with **FP32 accumulation** at a fraction of the rate they issue it with **FP16 accumulation**. The fraction is half on Ampere and Ada, and NVIDIA lists an eighth for the RTX 5090. Datacenter and workstation parts issue both forms at the same rate.

PyTorch and cuBLAS accumulate in FP32, so on a GeForce card they run at the lower rate. AiDotNet.Tensors keeps an FP32 total but accumulates each K tile in FP16. This roughly doubles the math ceiling, needs no driver or firmware changes, and is on by default.

## What Tensors does by default

`GemmFp16In32fOut` and `GemmFp16HalfOut` route large FP16 GEMMs through a direct-PTX tensor-core kernel that:

1. accumulates each K tile (16–64 products) in FP16 (`mma.sync…f16.f16.f16.f16`);
2. adds the tile's partial sums into FP32 accumulators before the next tile;
3. after the main loop, checks every accumulator for a non-finite value. Finite FP16 partials cannot overflow the FP32 total, so a non-finite accumulator means a partial passed 65504 or an operand was already non-finite. The block votes with `bar.red.or.pred`, and a block that saw one redoes its tile with FP32 accumulation.

So the output never depends on operand magnitude:
- overflow-sized operands give the exact FP32-accumulated result;
- a NaN operand stays NaN;
- clean blocks are bit-identical to an FP16-accumulate run.

The error budget is FP16 rounding of each tile's partial sum (2⁻¹¹ relative per tile), on top of the usual FP16 operand rounding.

A shape is routed only if the kernel beats cuBLAS on this device:
- **When it's decided:** on the first eager call of a shape with at least 2²⁹ multiply-adds.
- **How:** paired ABBA rounds against cuBLAS. The median must be ≥ 1.02x and no round may lose. A winning median with one losing round is measured once more, and that second set must pass on its own.
- **Lifetime:** the verdict lasts for the process.
- **Evolved routes:** `CudaBackend.TuneFp16AccumulateRouteAsync(m, n, k, …)` evolves a kernel for one shape and persists its verdict through the autotune cache.

Measured on an RTX 3080 (FP16 in, FP32 out, operands in [-1, 1)), verdict vs cuBLAS:

| shape (M×N×K) | evolved route | quick pick |
|---|---|---|
| 4096³ | 1.315x | 1.276x |
| 2048³ | 1.363x | 1.282x |
| 4096×768×3072 | 1.065x | 1.227x median |
| 4096×3072×768 | 1.110x | 1.159x |
| 1024×768×3072 | 1.491x | 1.542x |
| 12544×128×1152 | 1.544x | 1.631x |
| 1024³ | 1.298x | 1.274x |

## FP16 flash attention

`FlashAttention` on CUDA runs the FP16 flash kernel **by default wherever the caller already chose FP16**: inside an `AutocastScope(PrecisionMode.Float16)`, or when the query is an FP16-resident activation. This matches PyTorch, which keeps FP32 SDPA exact and uses its FP16 kernels for FP16 tensors and autocast. Plain FP32 attention stays on the exact FP32 kernel; with inputs in [-1, 1] the FP16 kernel's output error is 1e-4 to 4.4e-4.

With the direct-PTX attention opt-in (`AIDOTNET_DIRECT_PTX_ATTENTION=1`, or the master `AIDOTNET_DIRECT_PTX=1`), FP32 attention takes the FP16 kernels too: Q/K/V are converted to FP16 and one of two kernels runs:
- **The online kernel** handles its validated short shapes (S ≤ 128, D = 64).
- **`PtxFlashAttentionKernel`** handles every other length and D = 128. It is a FlashAttention-2 style forward:
  - grouped-query heads, and causal masking with an offset;
  - FP32 output and the per-row log-sum-exp that the existing device backward reads;
  - one K and one V buffer in FlashAttention-2 order, so occupancy is bound by registers rather than shared memory;
  - 4 or 8 warps per block depending on shape.

Accumulation follows the GEMM rules above:
- Q·Kᵀ and P·V accumulate in FP16 per 64-key tile, wherever FP16 accumulation issues faster.
- P·V partials are promoted into FP32 outputs.
- A block whose scores or outputs go non-finite redoes its rows with FP32 accumulation. Overflow-sized scores and P·V partials reproduce the FP32 kernel exactly.

RTX 3080, FP16 Q/K/V in [-1, 1), scale 1/√D, against PyTorch 2.14's best FP16 SDPA backend (cuDNN or memory-efficient; no flash backend on Windows). Runs were interleaved; each figure is the best of two.

| shape (B×H×S×D) | PyTorch | Tensors | speedup |
|---|---|---|---|
| 8×16×512×64 | 178.8 µs | 117.7 µs | 1.52x |
| 8×16×1024×64 | 601.6 µs | 411.3 µs | 1.46x |
| 4×16×2048×64 | 1177.1 µs | 799.8 µs | 1.47x |
| 2×16×4096×64 | 2355.2 µs | 1520.3 µs | 1.55x |
| 8×16×1024×128 | 1137.1 µs | 811.0 µs | 1.40x |
| 2×16×4096×128 | 4488.2 µs | 3138.3 µs | 1.43x |
| 8×16×512×64 causal | 135.0 µs | 85.2 µs | 1.58x |
| 8×16×1024×64 causal | 408.6 µs | 231.8 µs | 1.76x |
| 4×16×2048×64 causal | 745.1 µs | 411.0 µs | 1.81x |
| 2×16×4096×64 causal | 1388.8 µs | 780.3 µs | 1.78x |
| 8×16×1024×128 causal | 689.9 µs | 497.4 µs | 1.39x |
| 2×16×4096×128 causal | 2447.8 µs | 1674.8 µs | 1.46x |

**Why not 2x:**
- PyTorch's cuDNN path already runs at 57–61 TFLOP/s, close to the card's FP32-accumulate ceiling (62).
- The FP16-accumulate ceiling is about twice that (123).
- The flash kernel reaches 85–92 TFLOP/s at D = 64, about 75% of that ceiling. The rest goes to the FP32 softmax and the per-tile synchronization.

## Rate detection

The FP16-accumulate route only pays where FP32 accumulation is the slower form, so Tensors measures this before tuning anything:
- `PtxMmaRateProbe` keeps eight independent `mma.sync.m16n8k16` chains per warp in registers, on every SM, for each accumulate form.
- The time ratio between the two forms is the issue-rate ratio.
- RTX 3080: 61.9 against 123.2 TFLOP/s, ratio 1.99.

| measured ratio | rate class | what the route does |
|---|---|---|
| < 1.3 | `fp32acc-full` | Nothing to double: every shape keeps cuBLAS without tuning. |
| ≈ 2, ≈ 8, … | `fp32acc-1/2`, `fp32acc-1/8`, … | Tunes and routes as above. |

The rate class is part of every persisted verdict. A verdict measured before a card was unlocked, or relocked, is ignored, and the shape is measured again.

You can check your own card:

```csharp
using var backend = new CudaBackend();
var (fp32Acc, fp16Acc, fullRate) = backend.MeasureTensorCoreAccumulateRates();
Console.WriteLine($"FP32 accumulate {fp32Acc:F0} TFLOP/s, FP16 accumulate {fp16Acc:F0} TFLOP/s, full rate: {fullRate}");
```

### Environment variables

| variable | effect |
|---|---|
| `AIDOTNET_FP16_ACCUMULATE=0` | Turns the FP16-accumulate route off: every FP16 GEMM uses cuBLAS's FP32 accumulation. |
| `AIDOTNET_FP16_ACCUMULATE=1` | Tunes shapes even on a device whose FP32 accumulation is already full rate. |

## The unlock projects

Several open-source projects raise the FP32-accumulate rate of GeForce cards to the FP16-accumulate rate:
- [Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock) (Ampere);
- [JustAResearcher/gpu-unlock-ada](https://github.com/JustAResearcher/gpu-unlock-ada) (Ada);
- [KohakuBlueleaf/KohakUnlock](https://github.com/KohakuBlueleaf/KohakUnlock) (Turing).

From their sources, they load a privileged kernel module before the NVIDIA driver. The module writes the SMs' speed-select registers and resets the function (FLR), so the driver initializes a card that issues FP32-accumulate MMA at full rate. Their reported GEMM gains range from about +12% to +94% depending on shape. Memory-bound work such as LLM decode does not change.

**Tensors does not ship, install or apply any unlock.**
- Running one means loading unsigned code into the kernel and writing undocumented GPU registers, which risks hangs, data loss or worse.
- It may void warranties.
- The projects carry their own licences, and NVIDIA's driver and firmware terms apply.

Whether to run one is your decision. If you do:
- Tensors detects the higher FP32-accumulate rate on the next process start. The FP16-accumulate route then steps aside, since cuBLAS's FP32 accumulation is now as fast, and persisted FP16-accumulate verdicts from before the unlock are ignored.
- cuBLAS, cuDNN and the FP32-accumulate tensor-core kernels pick up the higher rate directly.
- No Tensors setting is needed either way.

The FP16-accumulate route gets most of the same doubling on a **stock** card, in software, without any of these risks. That is why it is on by default.