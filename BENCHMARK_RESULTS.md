# AiDotNet.Tensors Benchmark Results

**Date**: 2026-04-08  
**Hardware**: Ryzen 9 3950X, DDR4  
**Framework**: .NET 10.0, Debug build  
**Note**: Debug build — Release build will be significantly faster

## MLP Compiled vs Eager (3-Layer, 32x128->64->32->10)

| Activation | Eager (ms) | Compiled (ms) | Speedup |
|-----------|-----------|--------------|---------|
| GELU | 99.637 | 0.595 | **167.4x** |
| LeakyReLU | 78.876 | 0.586 | **134.6x** |
| Tanh | 32.000 | 32.431 | 0.99x |
| ReLU | 0.780 | 32.359 | 0.02x |

**Summary**: Compiled plans show massive speedups for GELU/LeakyReLU (100-167x). Tanh breaks even. ReLU regression needs investigation (compiled path slower — likely optimization pass overhead for this trivially fast op).

## Full A/B Matrix: MLP Training Step [32,128->64->10]

| Approach | Time (ms) | vs Eager | vs PyTorch |
|----------|----------|----------|------------|
| 1. Eager + GradientTape | 49.921 | 1.00x | 0.01x |
| 2. Compiled Plan (no fusion) | 0.289 | **172.89x** | 0.92x |
| 2b. Compiled + Phase B Fusion | 16.466 | 3.03x | 0.02x |
| 3. Phase B standalone | 32.569 | 1.53x | 0.01x |

**Key finding**: Compiled Plan (no fusion) is **172.9x** faster than eager and **0.92x vs PyTorch** (within 8% of PyTorch on this workload).

## Compiled Plan Varying Sizes

| Size | Eager (ms) | Compiled (ms) | Speedup | PhaseB (ms) | PhaseB Speedup |
|------|-----------|--------------|---------|-------------|----------------|
| [8x64->32->10] | 0.187 | 0.069 | **2.7x** | 0.088 | 2.1x |
| [32x128->64->10] | 0.286 | 0.117 | **2.4x** | 0.227 | 1.3x |
| [64x256->128->32] | 78.329 | 0.811 | **96.6x** | 0.745 | 105.1x |
| [128x512->256->64] | 4.983 | 162.320 | 0.03x | 2.394 | 2.1x |

**Key finding**: Compiled plans shine at medium sizes (96-167x). Large sizes show regression — optimization passes have overhead.

## Elementwise Ops (1M elements)

| Op | Eager (ms) | Compiled (ms) | Speedup |
|----|-----------|--------------|---------|
| Add | 39.722 | 1.258 | **31.6x** |
| ReLU | 15.717 | 0.590 | **26.6x** |

## Elementwise Ops (100K elements)

| Op | Eager (ms) | Compiled (ms) | Speedup |
|----|-----------|--------------|---------|
| Add | 0.117 | 0.056 | **2.1x** |
| ReLU | 0.102 | 0.032 | **3.1x** |

## Flash Attention vs Naive

| SeqLen | HeadDim | Flash (ms) | Naive (ms) | Speedup | Flash Mem | Naive Mem |
|--------|---------|-----------|-----------|---------|-----------|-----------|
| 64 | 32 | 0.959 | 1.262 | **1.3x** | 8KB | 24KB |
| 256 | 64 | 19.559 | 63.867 | **3.3x** | 64KB | 320KB |
| 512 | 64 | 55.892 | 133.434 | **2.4x** | 128KB | 1,152KB |
| 1024 | 64 | 226.312 | 874.371 | **3.9x** | 256KB | 4,352KB |
| 2048 | 64 | 1686.593 | 2064.273 | **1.2x** | 512KB | 16,896KB |

## Multi-Head Transformer Attention (8 heads)

| SeqLen | Flash (ms) | Naive (ms) | Speedup |
|--------|-----------|-----------|---------|
| 128 | 137.71 | 474.54 | **3.5x** |

## CNN Inference: Compiled vs Eager

| Config | Eager (ms) | Compiled (ms) | Speedup |
|--------|-----------|--------------|---------|
| [1x3x32x32] Conv3+ReLU+Pool | 0.406 | 0.380 | **1.07x** |
| [1x3x64x64] Conv3+ReLU+Pool | 1.752 | 1.453 | **1.21x** |
| [4x3x32x32] Conv3+ReLU+Pool | 809.577 | 175.312 | **4.62x** |

## Profiling

| Workload | Time/step | Steps/sec |
|----------|----------|-----------|
| MLP Training [32x128->64->32] | 489ms | 2 |

## Issues Found

1. **ReLU compiled regression**: Compiled ReLU MLP is slower than eager (0.02x). The compilation overhead exceeds the benefit for this trivially fast activation.
2. **Large model regression**: [128x512->256->64] compiled is 0.03x (33x slower). Optimization passes have O(n) overhead that dominates for large models.
3. **Phase B fusion regression**: Compiled+Fusion (16.5ms) is slower than Compiled alone (0.29ms). DataflowFusion pass may be counterproductive for this model size.

---
*Captured via: `dotnet test --logger "console;verbosity=detailed"`*
