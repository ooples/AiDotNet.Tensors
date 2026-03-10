# Benchmark Baseline (CPU + GPU)

Date: 2026-03-09
Hardware: AMD Ryzen 9 3950X, DDR4, Windows 11 Pro
Runtime: .NET 10.0, BenchmarkDotNet 0.15.8
Branch: perf/cpu-gpu-optimizations (commit 6b362e6)

## Baseline Results (Median values, 1M floats unless noted)

| Operation | AiDotNet (us) | TorchSharp (us) | Ratio | Alloc | Status |
|-----------|--------------|----------------|-------|-------|--------|
| **MatMul 256x256** | 109 | 149 | **0.73x** | 262KB | WIN |
| **MatMul 512x512** | 569 | 583 | **0.98x** | 1MB | TIED |
| **MatMul Double 256** | 219 | 237 | **0.92x** | 524KB | WIN |
| **Add 100K** | 74 | 64 | 1.16x | 192B | Close |
| **Add 1M** | 556 | 339 | 1.64x | 192B | GAP |
| Add 1M (1-thread) | 556 | 543 | 1.02x | - | TIED |
| **Multiply 100K** | 32 | 76 | **0.42x** | 192B | WIN |
| **Multiply 1M** | 790 | 431 | 1.83x | 192B | GAP |
| **Subtract 1M** | 728 | 350 | 2.08x | 4MB | GAP |
| **Divide 1M** | 805 | 292 | 2.76x | 4MB | GAP |
| **Exp 1M** | 3,302 | 218 | **15.1x** | 4MB | P0 GAP |
| **Log 1M** | 4,230 | 378 | **11.2x** | 4MB | P0 GAP |
| **Sqrt 1M** | 507 | 364 | 1.39x | 4MB | GAP |
| **Abs 1M** | 436 | 295 | 1.48x | 4MB | GAP |
| **ReLU 1M** | 270 | 184 | 1.47x | 32B | GAP |
| **Sigmoid 1M** | 231 | 241 | **0.96x** | 3KB | WIN |
| RawTP Sigmoid 1M | 6,005 | - | - | 0 | (ref) |
| **Tanh 1M** | 519 | 456 | 1.14x | 4MB | Close |
| **GELU 1M** | 541 | 349 | 1.55x | 4MB | GAP |
| **Mish 1M** | 3,800 | 2,809 | 1.35x | 4MB | GAP |
| **LeakyReLU 1M** | 844 | 803 | 1.05x | 4MB | Close |
| **Sum 1M** | 316 | 606 | **0.52x** | 208B | WIN |
| RawTP Sum 1M | 480 | - | - | 0 | (ref) |
| **Mean 1M** | 316 | 310 | 1.02x | 208B | TIED |
| **MaxValue 1M** | 256 | 195 | 1.31x | 0 | GAP |
| **MinValue 1M** | 305 | 186 | 1.64x | 0 | GAP |
| **Softmax 512x1024** | 9,288 | 123 | **75.5x** | 6MB | P0 GAP |
| LogSoftmax 512x1024 | NA (error) | 117 | - | - | BROKEN |
| **Conv2D (16ch 64x64)** | 435 | 402 | 1.08x | 525KB | Close |
| Conv2D ZeroAlloc | 443 | 402 | 1.10x | 168B | Close |
| **BatchNorm (64ch 32x32)** | 8,057 | 876 | **9.2x** | 16MB | P0 GAP |
| LayerNorm | NA (error) | NA (error) | - | - | BROKEN |
| **MaxPool2D (3x3)** | 618 | 116 | **5.3x** | 137KB | P0 GAP |
| **SigmoidBackward 1M** | 2,638 | 278 | **9.5x** | 8MB | P0 GAP |
| **TanhBackward 1M** | 2,418 | 221 | **10.9x** | 8MB | P0 GAP |
| **Attention Q@K^T** | 3,587 | 203 | **17.7x** | 3MB | P0 GAP |
| **Add Double 1M** | 979 | 433 | 2.26x | 8MB | GAP |
| **Sigmoid Double 1M** | 1,104 | 730 | 1.51x | 8MB | GAP |

## P0 Priority Gaps (>5x slower)

| Operation | Gap | Root Cause |
|-----------|-----|------------|
| Softmax | 75.5x | Scalar loops, massive allocations (6MB) |
| Attention Q@K^T | 17.7x | Scalar triple loop, no SimdGemm routing |
| Exp | 15.1x | Scalar TensorPrimitives delegation, 4MB alloc |
| Log | 11.2x | Scalar TensorPrimitives delegation, 4MB alloc |
| TanhBackward | 10.9x | ToArray + scalar ToDouble/FromDouble per element |
| SigmoidBackward | 9.5x | ToArray + scalar ToDouble/FromDouble per element |
| BatchNorm | 9.2x | Scalar numOps, no SIMD, 16MB allocations |
| MaxPool2D | 5.3x | Scalar nested loops with params int[] indexer |

## Wins (faster than TorchSharp)

| Operation | Speedup | Notes |
|-----------|---------|-------|
| Sum 1M | 1.9x faster | SimdKernels multi-accumulator |
| MatMul 256 | 1.4x faster | SimdGemm BLIS tiled GEMM |
| Multiply 100K | 2.4x faster | SIMD path, small size avoids bandwidth limit |
| Sigmoid 1M | Tied | Hand-tuned polynomial approximation |
| MatMul 512 | Tied | SimdGemm matches libtorch |

## Allocation Hotspots

Many operations allocate full copies (4MB for 1M floats):
- Subtract, Divide, Exp, Log, Sqrt, Abs, Tanh, GELU, Mish, LeakyReLU
- Backward passes: 8MB each (input + grad copies)
- BatchNorm: 16MB
- Softmax: 6MB

Root cause: These ops go through non-in-place paths that call `.ToArray()` or create new Tensor<T>.

---

## GPU Benchmark Baseline

Date: 2026-03-09
Hardware: AMD gfx1012 (RDNA1), 11 CUs, 4GB VRAM, OpenCL backend
Runtime: .NET 10.0, Stopwatch-based (100 runs, 10 warmup)

### GPU Activation Performance (1M floats)

| Operation | Bandwidth (GB/s) | Status |
|-----------|------------------|--------|
| ReLU | 176 | OK |
| Sigmoid | 168 | OK |
| Tanh | 178 | OK |
| GELU | 128 | OK |
| **Softmax** | **1.67** | **P0 — serial, 105x slower than ReLU** |

### GPU Normalization Performance

| Operation | Config | GFLOPS | Status |
|-----------|--------|--------|--------|
| BatchNorm | 64ch 32x32 | 78-94 | OK |
| LayerNorm | 512x1024 | 19-48 | GAP |
| **GroupNorm** | 32groups | **6-12** | **P0 — no parallel reduction** |
| **InstanceNorm** | 64ch | **3-9** | **P0 — no parallel reduction** |
| RmsNorm | 512x1024 | ~48 | OK |

### GPU GEMM Performance

| Size | Time (ms) | GFLOPS | Status |
|------|-----------|--------|--------|
| 256x256 | 0.10 | 338 | NEEDS WORK |
| 512x512 | 0.15 | 1,786 | MODERATE |
| 1024x1024 | 0.86 | 2,506 | MODERATE |
| 2048x2048 | 5.61 | 3,061 | MODERATE |
| 4096x4096 | 60.74 | 2,263 | MODERATE |

Peak theoretical: ~5,000 GFLOPS (gfx1012 FP32). Achieving 50-60%.

### GPU Fused Operations (256x1024 * 1024x4096)

| Fusion | Time (ms) | GFLOPS | Status |
|--------|-----------|--------|--------|
| GEMM only | 6.60 | 326 | OK |
| GEMM+ReLU | 6.66 | 323 | OK |
| GEMM+GELU | 6.81 | 316 | OK |
| GEMM+Sigmoid | 6.73 | 319 | OK |
| GEMM+Tanh | 6.96 | 308 | OK |

### GPU Attention (batch=2, heads=8, headDim=64)

| SeqLen | FlashAttn (ms) | GFLOPS | Status |
|--------|----------------|--------|--------|
| 128 | 3.39 | 19.8 | OK |
| 256 | 12.28 | 21.9 | OK |
| 512 | 51.41 | 20.9 | OK |
| 1024 | 212.86 | 20.2 | OK |
| ScaledDotProduct | ERROR (-52) | - | BROKEN |

### GPU Convolution

| Config | GFLOPS | Status |
|--------|--------|--------|
| ResNet conv3x3 (64ch 56x56) | 559 | OK |
| 1x1 projection (256→64) | 50 | GAP |
| Backward input | 81 | GAP |
| Backward kernel | ERROR | BROKEN (missing key) |

### GPU Memory Transfer

| Metric | Value |
|--------|-------|
| Min operation overhead | 0.38 ms |
| Max operations/sec | 2,639 ops/s |
| 2048x2048 MatMul | 17.86 ms (984 GFLOPS) |
| Compute-bound ratio | 98% |

### GPU P0 Gaps

| Operation | Issue | Root Cause |
|-----------|-------|------------|
| Softmax | 105x slower than ReLU bandwidth | Serial reduction, no block-level parallelism |
| GroupNorm | 6-12 GFLOPS | No warp-shuffle parallel reduction |
| InstanceNorm | 3-9 GFLOPS | No warp-shuffle parallel reduction |
| ScaledDotProduct | ERROR -52 | Kernel enqueue failure |
| Conv backward kernel | ERROR | Missing dictionary key |
| 1x1 projection conv | 50 GFLOPS (11x below conv3x3) | Suboptimal kernel selection |
