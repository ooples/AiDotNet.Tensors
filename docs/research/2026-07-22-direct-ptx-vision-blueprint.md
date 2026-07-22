# Direct PTX vision, detection, ROI, and geometry blueprint

This document is the implementation and evidence contract for issue #851. The
code is deliberately disabled by default and unpromoted. Managed builds and
static emitter tests are evidence; NVIDIA correctness, timing, JIT-resource,
and Nsight results remain pending until the resident-GPU commands below run.
No winner or zero-spill result is claimed before those artifacts exist.

## Literal scope ledger

| Issue phrase / public route | Existing NVIDIA route | Direct-PTX assignment |
|---|---|---|
| Box IoU / pairwise IoU | `detection_box_iou`, `parity210_pairwise_iou` | exact pairwise IoU module; self-pairwise uses legal read/read alias |
| GIoU / DIoU / CIoU | detection NVRTC family | one baked metric per exact pairwise shape |
| Box area / format conversion | detection NVRTC family | vector-load area and nine baked XYXY/XYWH/CXCYWH conversions |
| IoU-family losses and backward | IoU NVRTC family | aligned forward and deterministic register-only backward modules |
| Pairwise IoU-family backward | two deterministic NVRTC reductions | separate owner-A and owner-B modules; fixed reduction order |
| NMS / batched NMS | `resident_nms` | score selection, tie handling, class filtering, IoU, suppression, and compaction in one resident module |
| Masks-to-boxes | `parity210_masks_to_boxes` | per-mask register reduction with explicit empty-mask semantics |
| ROIAlign / ROIPool / PS variants | ROI NVRTC family | baked geometry, scale, sampling, alignment, and channel mapping |
| Cross products | `parity210_cross3` | exact contiguous extent-three vector kernel |
| Mesh/coordinate and sampling-grid utilities | both public `TensorMeshgrid` overloads use Repeat/Tile composition | an explicit DirectGpu tuple route plus two direct load/store grid-broadcast modules for each `ij`/`xy` result pair |
| Detection decode/filter stages | no separate public #851 entry point | fused into NMS where semantics permit; no invented API |
| Pixel distance transform | no public AiDotNet operation or CUDA kernel | no route exists to port; DIoU is assigned above and is not misreported as an image distance-transform API |
| Crop, resize, affine-grid, grid-sample, and gradients | explicitly owned by spatial child #842 | excluded from this PR to preserve one issue/one production owner |
| Mesh Laplacian and irregular mesh adjacency | mesh/graph resident kernels | explicitly owned by sparse/graph child #852 |
| Spiral neighborhood generation | `ISpiralIndicesBackend.GenerateSpiralIndices` | explicitly owned by the mesh/spiral scope in specialized child #854 |

`DirectPtxVisionCoverageManifest` contains only implemented and routed #851
cells. It contains no planned or fallback-only row.

## Architecture and specialization matrix

Only GA102/SM86 is admitted. SM80, Ada, Hopper, Blackwell, unknown devices,
and every unlisted shape fail closed to the established CUDA backend. They are
separate tuning domains and cannot inherit SM86 evidence.

| Family | Admitted FP32 specializations |
|---|---|
| Pairwise IoU/GIoU/DIoU/CIoU | `(N,M)`: `(256,256)`, `(1024,256)`, `(1024,1024)`, `(4096,256)` |
| Box area, conversion, aligned losses/backward | `N`: `256`, `1024`, `4096`; conversion format pair baked |
| Pairwise backward | `N,M` each `256` or `1024`; variant and owner baked |
| NMS | `N`: `256`, `1024`; threshold exactly `0.5`; batched flag baked |
| Masks-to-boxes | `[256,28,28]`, `[64,64,64]` |
| ROI | normal `[1,256,56,56]`, PS `[1,196,56,56]`, `K=256`, output `7x7`; spatial scale exactly `0.25`; align sampling ratio `2` |
| Cross3 | `(outer,inner)`: `(256,1)`, `(1024,1)`, `(256,64)` |
| Meshgrid2D | `(N0,N1)`: `(256,256)`, `(1024,256)`; output index and `ij`/`xy` baked |

Every module has a versioned `DirectPtxKernelBlueprint`, exact logical and
physical extents, FP32 dtype, layout identity, 16-byte pointer alignment,
an explicit zero byte offset, exact allocation extent, access mode, alias rejection, block geometry,
resource budget, and semantic map. Static tests emit and validate all 120
members of this closed matrix. The JIT audit adds PTX SHA-256, GPU UUID,
SM, driver fingerprint, registers, static shared bytes, local bytes, maximum
threads, occupancy, and JIT log.

## Contiguity and ABI

Production dispatch first selects a closed `DirectPtxVisionSpec`; arbitrary
dimensions never trigger JIT and record an operation-specific
`specialization-not-admitted` reason before the established backend runs.
`DirectPtxTensorView.Create` then validates the
device pointer and exact allocation once. The PTX ABI contains one pointer per
tensor and no shapes, strides, formats, thresholds, modes, or alignment flags.
High-level tensor routes use the repository's canonical contiguous GPU buffer
materialization before reaching the backend. Noncanonical views therefore
copy through the established contiguous path or fall back; the admitted PTX
never performs a dynamic stride/layout check.

Output storage may not overlap any input. Read-only box inputs may alias, which
is required by standalone self-pairwise IoU. NMS `suppressed` is the sole
read/write contract and must be zeroed by the caller before a fresh operation.

## Fused dataflow and global traffic

| Family | Global reads | Register/shared work | Global writes |
|---|---|---|---|
| Pairwise metrics/loss | two 16-byte box rows; one upstream scalar for backward | geometry, union, enclosing/distance/aspect terms and fixed-order reductions remain in registers | one scalar or one four-coordinate gradient |
| Box area/conversion/Cross3 | vector/scalar source components once | full transform in registers | each result once |
| NMS | boxes, scores, classes, and suppression state | deterministic controller fuses selection, lower-index tie rule, class test, IoU, threshold, suppression, and stable compaction | suppression updates, selected indices, one count; evidence reports the `4*N`-byte suppression workspace |
| Masks-to-boxes | every mask pixel once per admitted mask | min/max bounds in registers; no bounds tensor | one 16-byte box |
| ROI family | ROI row once per output and only contributing feature cells | bin geometry, bilinear weights, sample accumulation or max/average reduction in registers | one output scalar |
| Meshgrid2D | one selected coordinate per output | output axis and indexing mode are compile-time constants | one dense-grid scalar |

No module materializes a geometry, area, intersection, union, bin, weight, or
broadcast intermediate in global memory. The current specializations declare
zero static and dynamic shared memory. `cp.async`, Tensor Cores, and shared
bank analysis are therefore not claimed: these scalar/indexing kernels lack a
measured reusable tile. If resident measurements show ROI or mask reuse can
pay for tiling, that becomes a new separately measured blueprint version.

## Determinism and numerical semantics

- Degenerate box widths/heights clamp to zero; non-positive union produces
  zero. NMS NaNs and equal scores retain the established lower-index ordering.
- Pairwise owner gradients use one thread per owner and a fixed ascending
  other-index order, avoiding atomics.
- Aligned and pairwise backward v1 differentiates the emitted metric with a
  symmetric register-only `0.001` perturbation. This eliminates temporary
  tensors and nondeterministic atomics, but remains unpromoted until its error
  matrix passes against the established analytical CUDA route.
- Empty masks produce `[0,0,0,0]`. ROI invalid batch rows and empty bins write
  zero. RoIPool emits CUDA `roundf` halfway-away-from-zero semantics rather
  than PTX ties-to-even conversion. Meshgrid exactly preserves `ij` and `xy`
  output ordering, and atomically validates both output ABIs before launch.

## Runtime, gates, cache, and graph capture

The family gate is `AIDOTNET_DIRECT_PTX_VISION=1`; the master gate also works.
Every operation additionally has a disabled-by-default snake-case gate such as
`AIDOTNET_DIRECT_PTX_VISION_NMS`, `..._ROI_ALIGN`, or `..._MESHGRID_2D`.
BoxIoU retains `AIDOTNET_DIRECT_PTX_VISION_BOX_IOU`.

Each backend instance is bound to one GPU and driver fingerprint. Within that
instance, the cache is a bounded LRU keyed by every baked dimension, flag,
scalar bit pattern, and operation. Prewarm performs emission, JIT, resource
admission, and module load outside capture. Cold capture fails with an exact
reason; a prewarmed module is pinned while its captured graph can reference it.
Backend disposal unloads both vision caches and the shared runtime. Normal
disabled fallback uses precomputed reason strings and adds no per-call managed
allocation.

Version 1 has one block geometry per exact specialization, so it has no honest
runtime tuning choice and performs no timing-based autotuning. The closed
admission table is the deterministic plan set; competing geometries must be
introduced as new bounded blueprint versions and promoted from offline clean
runs rather than tuned or allocated during graph capture.

JIT admission rejects nonzero local bytes, resource-budget violations, and
insufficient active blocks. That is not proof of zero executed spills; only
the Nsight workflow below can supply that proof.

## Resident evidence protocol

All methods use precreated tensors on one NVIDIA GPU and exclude host/device
transfer. The C# family runner compares direct PTX, direct graph replay,
current AiDotNet NVRTC, and current AiDotNet graph replay. The separate Python
runner covers eligible torchvision/PyTorch implementations without mixing
framework contexts. CPU, MKL, and OpenBLAS are excluded.

Each cell uses 30 warmups, 101 samples, 25 launches per CUDA-event sample, and
three independent runs. JSON records device and E2E mean/median/P95/P99,
modeled GFLOPS and algorithmic GB/s, managed bytes/call, temporary or peak
device bytes, maximum error, registers, shared/local bytes, occupancy, hash,
and the environment fingerprint.

## Grouped evidence table -- pending GPU execution

The rows are grouped by operation. The fastest valid median will be bolded
only after three complete runs and all correctness/resource gates pass.

| Operation / shape | NVIDIA resident method | mean us | median us | P95 us | P99 us | E2E median us | GFLOPS | GB/s | managed B/call | temp/peak VRAM B | max error | regs | shared B | local B | blocks/SM | zero spills | result |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| BoxIoU / four shapes | Direct PTX; graph; AiDotNet; graph; torchvision; torchvision graph; torch.compile | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | hold |
| GIoU, DIoU, CIoU / 256x256 | Direct PTX; graph; AiDotNet; graph; eligible torchvision | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | hold |
| BoxArea, BoxConvert / N=4096 | Direct PTX; graph; AiDotNet; graph; torchvision | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | hold |
| Four aligned losses + backward / N=4096 | Direct PTX; graph; AiDotNet; graph; eligible PyTorch/torchvision | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | hold |
| Pairwise backward A+B / 256x256 | Direct PTX paired launch; graph; AiDotNet paired deterministic NVRTC; graph | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | hold |
| NMS and batched NMS / N=256 | Direct PTX; graph; AiDotNet; graph; torchvision | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | hold |
| MasksToBoxes / 256x28x28 | Direct PTX; graph; AiDotNet; graph; torchvision | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | hold |
| ROI four variants / canonical shapes | Direct PTX; graph; AiDotNet; graph; eligible torchvision | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | hold |
| Cross3 / 1024x3 | Direct PTX; graph; AiDotNet; graph; PyTorch | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | hold |
| Meshgrid2D / 1024x256 `ij` and `xy`, both outputs | Direct PTX paired launch; graph; AiDotNet Repeat/Tile pair; graph; PyTorch | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | hold |

## Commands prepared, not executed here

```powershell
$env:AIDOTNET_DIRECT_PTX_VISION = '1'
dotnet run --project tests/AiDotNet.Tensors.Benchmarks -c Release -- --direct-ptx-vision-box-iou 3
dotnet run --project tests/AiDotNet.Tensors.Benchmarks -c Release -- --direct-ptx-vision-family 3
python tests/AiDotNet.Tensors.Benchmarks/BaselineRunners/py/run_direct_ptx_vision_box_iou_competitors.py --runs 3
python tests/AiDotNet.Tensors.Benchmarks/BaselineRunners/py/run_direct_ptx_vision_competitors.py --runs 3
tests/AiDotNet.Tensors.Benchmarks/Profiling/run-direct-ptx-ncu.ps1 -Target vision-box-iou
```

Promotion requires at least 1.10x direct-PTX median speedup over the strongest
eligible competitor, direct P95 no worse than competitor P95 plus 10%, zero
hot managed allocation, zero avoidable temporary VRAM, the operation-specific
error tolerance, JIT local bytes equal to zero, and Nsight counters proving
zero executed local/spill traffic. Any missing or contradictory artifact is a
hold, never a pass.
