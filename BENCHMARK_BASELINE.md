# CPU Benchmark Baseline

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
