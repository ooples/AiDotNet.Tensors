// GPU-residency audit: parity tests for ops that were silently falling back to the
// CpuEngine base on a GPU device (causing host<->device ping-pong). Each new GPU
// override must produce bit-comparable results to the trusted CPU reference across a
// shape sweep. Runs only when a GPU is present; CPU-only hosts early-return (no-op pass).

#if !NETFRAMEWORK
#nullable disable

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using BM = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class GpuMissingKernelsParityTests : IDisposable
{
    private readonly CpuEngine _cpu = new CpuEngine();
    private readonly DirectGpuTensorEngine _gpu;
    private readonly bool _gpuReady;
    private readonly Exception _gpuInitException;

    public GpuMissingKernelsParityTests()
    {
        // GPU/CPU parity validates kernel LOGIC, so it must compare at MATCHED precision: force strict
        // fp32 on the GPU (TF32 off) before the backend initializes. TF32 stays the production default
        // (industry standard, ~5× fp32 throughput) and is unaffected here — but a TF32 GPU result vs a
        // true-fp32 CPU result legitimately differs by ~1e-3 relative (TF32's ~10-bit mantissa), which
        // would mask real logic bugs. PyTorch's own CUDA-vs-CPU correctness tests disable TF32 the same way.
        CudaDispatchPolicy.AllowTF32 = false;
        try
        {
            _gpu = new DirectGpuTensorEngine();
            _gpuReady = _gpu.IsGpuAvailable;
        }
        catch (Exception ex)
        {
            _gpuInitException = ex;
            _gpuReady = false;
        }
    }

    public void Dispose() => _gpu?.Dispose();

    private bool EnsureGpuReady()
    {
        if (_gpuReady) return true;
        if (string.Equals(Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS"), "1", StringComparison.Ordinal))
            throw new InvalidOperationException(
                "GPU tests were required (AIDOTNET_REQUIRE_GPU_TESTS=1) but DirectGpu init failed or no GPU is available.",
                _gpuInitException);
        return false;
    }

    // float32 GEMM at moderate K accumulates a few 1e-6 of abs error; 1e-3 cleanly
    // separates a correct result from a broken kernel.
    private const float Tol = 1e-3f;

    private static Tensor<float> Rand(int seed, params int[] shape)
    {
        int n = 1;
        foreach (int d in shape) n *= d;
        var rng = new Random(seed);
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return new Tensor<float>(data, shape);
    }

    private static void AssertMatch(Tensor<float> gpu, Tensor<float> cpu, string op, float tol = Tol)
    {
        Assert.Equal(cpu.Shape.ToArray(), gpu.Shape.ToArray());
        var sa = gpu.ToArray();
        var sb = cpu.ToArray();
        double m = 0;
        for (int i = 0; i < sa.Length; i++)
        {
            Assert.False(float.IsNaN(sa[i]) || float.IsInfinity(sa[i]), $"{op}: GPU non-finite {sa[i]} at {i}");
            double e = Math.Abs((double)sa[i] - sb[i]);
            if (e > m) m = e;
        }
        Assert.True(m < tol, $"{op}: GPU vs CPU max_abs_err {m:E3} exceeded {tol:E3}");
    }

    public static IEnumerable<object[]> AddMMSizes() => new List<object[]>
    {
        new object[] { 1, 1, 1 },
        new object[] { 4, 4, 4 },
        new object[] { 16, 32, 8 },
        new object[] { 32, 32, 32 },
        new object[] { 33, 17, 65 },
        new object[] { 64, 128, 32 },
        new object[] { 128, 256, 64 },
        new object[] { 1, 256, 256 },
        new object[] { 256, 1, 256 },
        new object[] { 200, 50, 120 },
    };

    [Theory]
    [MemberData(nameof(AddMMSizes))]
    public void TensorAddMM_DefaultAlphaBeta_GpuMatchesCpu(int m, int k, int n)
    {
        if (!EnsureGpuReady()) return;
        var input = Rand(1, m, n);
        var a = Rand(2, m, k);
        var b = Rand(3, k, n);

        var cpu = _cpu.TensorAddMM(input, a, b);
        var gpu = _gpu.TensorAddMM(input, a, b);
        AssertMatch(gpu, cpu, $"TensorAddMM[{m},{k},{n}]");
    }

    [Theory]
    [MemberData(nameof(AddMMSizes))]
    public void TensorAddMM_AlphaBeta_GpuMatchesCpu(int m, int k, int n)
    {
        if (!EnsureGpuReady()) return;
        var input = Rand(11, m, n);
        var a = Rand(12, m, k);
        var b = Rand(13, k, n);
        const float alpha = 0.75f, beta = -1.5f;

        var cpu = _cpu.TensorAddMM(input, a, b, alpha, beta);
        var gpu = _gpu.TensorAddMM(input, a, b, alpha, beta);
        AssertMatch(gpu, cpu, $"TensorAddMM_ab[{m},{k},{n}]");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(64)]
    [InlineData(257)]
    [InlineData(4096)]
    public void TensorVecDot_GpuMatchesCpu(int n)
    {
        if (!EnsureGpuReady()) return;
        var a = Rand(21, n);
        var b = Rand(22, n);

        float cpu = _cpu.TensorVecDot(a, b);
        float gpu = _gpu.TensorVecDot(a, b);
        // Relative tolerance: a length-4096 dot accumulates more abs error than the
        // 1e-3 elementwise bar, so scale by the magnitude of the result.
        double tol = 1e-3 * Math.Max(1.0, Math.Abs((double)cpu));
        Assert.False(float.IsNaN(gpu) || float.IsInfinity(gpu), $"VecDot[{n}] GPU non-finite: {gpu}");
        Assert.True(Math.Abs((double)gpu - cpu) < tol, $"VecDot[{n}]: GPU {gpu} vs CPU {cpu} (tol {tol:E3})");
    }

    [Theory]
    [MemberData(nameof(AddMMSizes))]
    public void TensorMatMul2DWithPrePackedB_GpuMatchesCpu(int m, int k, int n)
    {
        if (!EnsureGpuReady()) return;
        var a = Rand(31, m, k);
        var b = Rand(32, k, n);
        // Build a CPU managed-BLAS pre-packed handle for B ([k,n], row-major => ldb=n).
        WeightPackHandle packed = BM.PrePackB<float>(b.ToArray(), n, false, k, n);

        // Trusted reference is plain a@b; the pre-pack must not change the result.
        var cpu = _cpu.TensorMatMul(a, b);
        var gpu = _gpu.TensorMatMul2DWithPrePackedB(a, b, packed);
        AssertMatch(gpu, cpu, $"TensorMatMul2DWithPrePackedB[{m},{k},{n}]");
    }

    public static IEnumerable<object[]> ChainDims() => new List<object[]>
    {
        new object[] { new[] { 4, 4 } },                 // single product
        new object[] { new[] { 8, 16, 8 } },             // 2 products, narrow middle
        new object[] { new[] { 32, 4, 64, 8 } },         // 3 products, order matters
        new object[] { new[] { 5, 7, 3, 9, 4 } },        // 4 products, irregular dims
        new object[] { new[] { 64, 64, 64, 64 } },       // square chain
    };

    [Theory]
    [MemberData(nameof(ChainDims))]
    public void TensorMultiDot_GpuMatchesCpu(int[] dims)
    {
        if (!EnsureGpuReady()) return;
        int count = dims.Length - 1;
        var mats = new Tensor<float>[count];
        for (int i = 0; i < count; i++)
            mats[i] = Rand(40 + i, dims[i], dims[i + 1]);

        var cpu = _cpu.TensorMultiDot(mats);
        var gpu = _gpu.TensorMultiDot(mats);
        AssertMatch(gpu, cpu, $"TensorMultiDot[{string.Join("x", dims)}]");
    }

    // BatchNormInference's contract is rank-4 NCHW.
    public static IEnumerable<object[]> BnShapes() => new List<object[]>
    {
        new object[] { new[] { 2, 16, 3, 3 } },      // [N, C, H, W]
        new object[] { new[] { 1, 3, 8, 8 } },       // single image, 3 channels
        new object[] { new[] { 4, 8, 1, 1 } },       // 1x1 spatial
        new object[] { new[] { 2, 32, 5, 7 } },      // non-square spatial
    };

    [Theory]
    [MemberData(nameof(BnShapes))]
    public void BatchNormInference_GpuMatchesCpu(int[] shape)
    {
        if (!EnsureGpuReady()) return;
        int channels = shape[1];
        var x = Rand(50, shape);
        var gamma = Rand(51, channels);
        var beta = Rand(52, channels);
        var mean = Rand(53, channels);
        // variance must be positive.
        var varArr = new float[channels];
        var rng = new Random(54);
        for (int i = 0; i < channels; i++) varArr[i] = (float)(rng.NextDouble() * 2.0 + 0.1);
        var variance = new Tensor<float>(varArr, new[] { channels });
        const double eps = 1e-5;

        var cpu = _cpu.BatchNormInference(x, gamma, beta, mean, variance, eps);
        var gpu = _gpu.BatchNormInference(x, gamma, beta, mean, variance, eps);
        AssertMatch(gpu, cpu, $"BatchNormInference[{string.Join("x", shape)}]");
    }

    // Inner product over last axis: a[M,K], b[N,K] -> [M,N].
    public static IEnumerable<object[]> InnerSizes() => new List<object[]>
    {
        new object[] { 1, 1, 1 },
        new object[] { 4, 8, 3 },
        new object[] { 32, 64, 16 },
        new object[] { 128, 32, 256 },   // attention-ish Q·Kᵀ
        new object[] { 7, 5, 11 },
    };

    [Theory]
    [MemberData(nameof(InnerSizes))]
    public void TensorInner_GpuMatchesCpu(int mRows, int k, int nRows)
    {
        if (!EnsureGpuReady()) return;
        var a = Rand(61, mRows, k);
        var b = Rand(62, nRows, k);

        var cpu = _cpu.TensorInner(a, b);
        var gpu = _gpu.TensorInner(a, b);
        AssertMatch(gpu, cpu, $"TensorInner[{mRows},{k},{nRows}]");
    }

    // Higher-rank inner product (contract last axis), now handled on-GPU via reshape+GEMM.
    public static IEnumerable<object[]> InnerNdShapes() => new List<object[]>
    {
        new object[] { new[] { 3, 4 }, new[] { 2, 5, 4 } },        // 2-D ⋅ 3-D
        new object[] { new[] { 2, 3, 8 }, new[] { 4, 8 } },        // 3-D ⋅ 2-D
        new object[] { new[] { 2, 3, 6 }, new[] { 5, 7, 6 } },     // 3-D ⋅ 3-D
        new object[] { new[] { 4, 9 }, new[] { 9 } },              // 2-D ⋅ 1-D
        new object[] { new[] { 9 }, new[] { 3, 5, 9 } },           // 1-D ⋅ 3-D
    };

    [Theory]
    [MemberData(nameof(InnerNdShapes))]
    public void TensorInner_HigherRank_GpuMatchesCpu(int[] shapeA, int[] shapeB)
    {
        if (!EnsureGpuReady()) return;
        var a = Rand(71, shapeA);
        var b = Rand(72, shapeB);

        var cpu = _cpu.TensorInner(a, b);
        var gpu = _gpu.TensorInner(a, b);
        AssertMatch(gpu, cpu, $"TensorInnerND[{string.Join("x", shapeA)};{string.Join("x", shapeB)}]");
    }

    // General tensor contraction. Each case: shapeA, shapeB, axesA, axesB.
    public static IEnumerable<object[]> DotCases() => new List<object[]>
    {
        // standard matmul as a dot: [M,K]·[K,N] over (1),(0)
        new object[] { new[] { 6, 8 }, new[] { 8, 5 }, new[] { 1 }, new[] { 0 } },
        // 3-D ⋅ 2-D, contract a's last with b's first
        new object[] { new[] { 2, 3, 8 }, new[] { 8, 4 }, new[] { 2 }, new[] { 0 } },
        // permuted: contract a's MIDDLE axis (needs permute) with b's last
        new object[] { new[] { 4, 6, 5 }, new[] { 7, 6 }, new[] { 1 }, new[] { 1 } },
        // multi-axis contraction (na=2): [2,3,4]·[3,4,5] over (1,2),(0,1)
        new object[] { new[] { 2, 3, 4 }, new[] { 3, 4, 5 }, new[] { 1, 2 }, new[] { 0, 1 } },
        // multi-axis with permutation: [3,4,2]·[2,3,6] over (0,1),(1,?) ...
        new object[] { new[] { 3, 4, 2 }, new[] { 4, 3, 6 }, new[] { 0, 1 }, new[] { 1, 0 } },
    };

    [Theory]
    [MemberData(nameof(DotCases))]
    public void TensorDot_GpuMatchesCpu(int[] shapeA, int[] shapeB, int[] axesA, int[] axesB)
    {
        if (!EnsureGpuReady()) return;
        var a = Rand(81, shapeA);
        var b = Rand(82, shapeB);

        var cpu = _cpu.TensorDot(a, b, axesA, axesB);
        var gpu = _gpu.TensorDot(a, b, axesA, axesB);
        AssertMatch(gpu, cpu, $"TensorDot[{string.Join("x", shapeA)};{string.Join("x", shapeB)};a={string.Join(",", axesA)};b={string.Join(",", axesB)}]");
    }

    // ----- Category F: shape / view / layout -----

    public static IEnumerable<object[]> SwapAxesCases() => new List<object[]>
    {
        new object[] { new[] { 4, 6 }, 0, 1 },
        new object[] { new[] { 2, 3, 5 }, 0, 2 },
        new object[] { new[] { 2, 3, 5 }, 1, 2 },
        new object[] { new[] { 2, 3, 4, 5 }, 1, 3 },
        new object[] { new[] { 2, 3, 5 }, -1, 0 },     // negative axis
    };

    [Theory]
    [MemberData(nameof(SwapAxesCases))]
    public void TensorSwapAxes_GpuMatchesCpu(int[] shape, int axis1, int axis2)
    {
        if (!EnsureGpuReady()) return;
        var t = Rand(91, shape);
        var cpu = _cpu.TensorSwapAxes(t, axis1, axis2);
        var gpu = _gpu.TensorSwapAxes(t, axis1, axis2);
        AssertMatch(gpu, cpu, $"TensorSwapAxes[{string.Join("x", shape)};{axis1},{axis2}]");
    }

    public static IEnumerable<object[]> MoveDimCases() => new List<object[]>
    {
        new object[] { new[] { 2, 3, 5 }, 0, 2 },
        new object[] { new[] { 2, 3, 5 }, 2, 0 },
        new object[] { new[] { 2, 3, 4, 5 }, 1, 3 },
        new object[] { new[] { 2, 3, 4, 5 }, 3, 0 },
        new object[] { new[] { 4, 6 }, 0, 1 },
    };

    [Theory]
    [MemberData(nameof(MoveDimCases))]
    public void TensorMoveDim_GpuMatchesCpu(int[] shape, int source, int destination)
    {
        if (!EnsureGpuReady()) return;
        var t = Rand(92, shape);
        var cpu = _cpu.TensorMoveDim(t, source, destination);
        var gpu = _gpu.TensorMoveDim(t, source, destination);
        AssertMatch(gpu, cpu, $"TensorMoveDim[{string.Join("x", shape)};{source}->{destination}]");
    }

    private void AssertChunksMatch(Tensor<float>[] gpu, Tensor<float>[] cpu, string op)
    {
        Assert.Equal(cpu.Length, gpu.Length);
        for (int i = 0; i < cpu.Length; i++)
            AssertMatch(gpu[i], cpu[i], $"{op}#{i}");
    }

    public static IEnumerable<object[]> SplitCases() => new List<object[]>
    {
        new object[] { new[] { 8, 4 }, 2, 0 },
        new object[] { new[] { 4, 9 }, 3, 1 },
        new object[] { new[] { 2, 6, 5 }, 3, 1 },
        new object[] { new[] { 2, 3, 8 }, 4, 2 },
        new object[] { new[] { 12 }, 4, 0 },
    };

    [Theory]
    [MemberData(nameof(SplitCases))]
    public void TensorSplit_GpuMatchesCpu(int[] shape, int numSplits, int axis)
    {
        if (!EnsureGpuReady()) return;
        var t = Rand(94, shape);
        var cpu = _cpu.TensorSplit(t, numSplits, axis);
        var gpu = _gpu.TensorSplit(t, numSplits, axis);
        AssertChunksMatch(gpu, cpu, $"TensorSplit[{string.Join("x", shape)};{numSplits};ax{axis}]");
    }

    public static IEnumerable<object[]> TensorSplitIdxCases() => new List<object[]>
    {
        new object[] { new[] { 10, 4 }, new[] { 3, 7 }, 0 },
        new object[] { new[] { 4, 10 }, new[] { 2, 5, 8 }, 1 },
        new object[] { new[] { 2, 9, 5 }, new[] { 3 }, 1 },
    };

    [Theory]
    [MemberData(nameof(TensorSplitIdxCases))]
    public void TensorTensorSplit_Indices_GpuMatchesCpu(int[] shape, int[] indices, int dim)
    {
        if (!EnsureGpuReady()) return;
        var t = Rand(95, shape);
        var cpu = _cpu.TensorTensorSplit(t, indices, dim);
        var gpu = _gpu.TensorTensorSplit(t, indices, dim);
        AssertChunksMatch(gpu, cpu, $"TensorTensorSplit[{string.Join("x", shape)};idx={string.Join(",", indices)};dim{dim}]");
    }

    // ----- Category D -----

    [Theory]
    [InlineData(0.0f)]
    [InlineData(0.25f)]
    [InlineData(-0.5f)]
    public void TensorClampMin_GpuMatchesCpu(float min)
    {
        if (!EnsureGpuReady()) return;
        var t = Rand(96, 4, 16);
        var cpu = _cpu.TensorClampMin(t, min);
        var gpu = _gpu.TensorClampMin(t, min);
        AssertMatch(gpu, cpu, $"TensorClampMin[{min}]");
    }

    [Theory]
    [InlineData(0.0f)]
    [InlineData(0.25f)]
    [InlineData(-0.5f)]
    public void TensorClampMax_GpuMatchesCpu(float max)
    {
        if (!EnsureGpuReady()) return;
        var t = Rand(97, 4, 16);
        var cpu = _cpu.TensorClampMax(t, max);
        var gpu = _gpu.TensorClampMax(t, max);
        AssertMatch(gpu, cpu, $"TensorClampMax[{max}]");
    }

    // ----- Category C: gather -----

    public static IEnumerable<object[]> TakeCases() => new List<object[]>
    {
        new object[] { new[] { 5, 4 }, new[] { 0, 7, 19, 3 } },              // flat indices into [5,4]=20
        new object[] { new[] { 12 }, new[] { 11, 0, 5, 5, 2 } },             // 1-D, repeats
        new object[] { new[] { 3, 3, 3 }, new[] { 26, 0, 13 } },             // 3-D source
        new object[] { new[] { 8 }, new[] { 7 } },                          // single
    };

    [Theory]
    [MemberData(nameof(TakeCases))]
    public void TensorTake_GpuMatchesCpu(int[] shape, int[] flatIndices)
    {
        if (!EnsureGpuReady()) return;
        var t = Rand(98, shape);
        var idx = new Tensor<int>(flatIndices, new[] { flatIndices.Length });
        var cpu = _cpu.TensorTake(t, idx);
        var gpu = _gpu.TensorTake(t, idx);
        AssertMatch(gpu, cpu, $"TensorTake[{string.Join("x", shape)};n={flatIndices.Length}]");
    }

    // 1-D index_add (the case the backend ScatterAdd kernel covers). Includes [257] — the exact
    // shape the prior wiring diverged on — and heavy-collision cases that exercise atomicAdd.
    public static IEnumerable<object[]> ScatterAddCases() => new List<object[]>
    {
        new object[] { 8, new[] { 0, 3, 3, 7, 0 } },                         // collisions on 0 and 3
        new object[] { 257, null },                                         // the prior-failure shape (random idx)
        new object[] { 16, new[] { 5, 5, 5, 5, 5, 5 } },                    // all-same index (max collision)
        new object[] { 4, new[] { 0, 1, 2, 3 } },                          // bijective, no collision
        new object[] { 100, null },
    };

    [Theory]
    [MemberData(nameof(ScatterAddCases))]
    public void TensorScatterAdd_1D_GpuMatchesCpu(int n, int[] explicitIdx)
    {
        if (!EnsureGpuReady()) return;
        var dest = Rand(101, n);
        int[] idxArr = explicitIdx;
        if (idxArr == null)
        {
            // Random indices (with collisions) of length ~n.
            var rng = new Random(202);
            idxArr = new int[n];
            for (int i = 0; i < n; i++) idxArr[i] = rng.Next(n);
        }
        var indices = new Tensor<int>(idxArr, new[] { idxArr.Length });
        var updates = Rand(103, idxArr.Length);

        var cpu = _cpu.TensorScatterAdd(dest, indices, updates, 0);
        var gpu = _gpu.TensorScatterAdd(dest, indices, updates, 0);
        AssertMatch(gpu, cpu, $"TensorScatterAdd_1D[n={n};m={idxArr.Length}]");
    }

    [Theory]
    [MemberData(nameof(ScatterAddCases))]
    public void TensorIndexAdd_1D_GpuMatchesCpu(int n, int[] explicitIdx)
    {
        if (!EnsureGpuReady()) return;
        var t = Rand(104, n);
        int[] idxArr = explicitIdx;
        if (idxArr == null)
        {
            var rng = new Random(205);
            idxArr = new int[n];
            for (int i = 0; i < n; i++) idxArr[i] = rng.Next(n);
        }
        var indices = new Tensor<int>(idxArr, new[] { idxArr.Length });
        var source = Rand(106, idxArr.Length);

        var cpu = _cpu.TensorIndexAdd(t, 0, indices, source);
        var gpu = _gpu.TensorIndexAdd(t, 0, indices, source);
        AssertMatch(gpu, cpu, $"TensorIndexAdd_1D[n={n};m={idxArr.Length}]");
    }

    // repeat_interleave along the LAST axis (the case the backend kernel covers).
    public static IEnumerable<object[]> RepeatInterleaveCases() => new List<object[]>
    {
        new object[] { new[] { 6 }, 3 },
        new object[] { new[] { 4, 5 }, 2 },
        new object[] { new[] { 2, 3, 4 }, 4 },
        new object[] { new[] { 8 }, 1 },
    };

    [Theory]
    [MemberData(nameof(RepeatInterleaveCases))]
    public void TensorRepeatInterleave_LastDim_GpuMatchesCpu(int[] shape, int repeats)
    {
        if (!EnsureGpuReady()) return;
        var t = Rand(108, shape);
        int dim = shape.Length - 1;
        var cpu = _cpu.TensorRepeatInterleave(t, repeats, dim);
        var gpu = _gpu.TensorRepeatInterleave(t, repeats, dim);
        AssertMatch(gpu, cpu, $"TensorRepeatInterleave[{string.Join("x", shape)};r={repeats}]");
    }

    [Theory]
    [InlineData(4, 8, 12, 2)]
    [InlineData(6, 16, 8, 4)]
    public void FusedLinearMaxout_GpuMatchesCpu(int rows, int inF, int outFull, int numPieces)
    {
        if (!EnsureGpuReady()) return;
        var input = Rand(270, rows, inF);
        var weights = Rand(271, inF, outFull);   // outFull = units*numPieces
        var bias = Rand(272, outFull);
        AssertMatch(_gpu.FusedLinearMaxout(input, weights, bias, numPieces), _cpu.FusedLinearMaxout(input, weights, bias, numPieces), $"Maxout[{rows}x{inF};{outFull};p{numPieces}]");
    }

    [Theory]
    [InlineData(4, 6, 5, 8)]
    [InlineData(3, 8, 4, 6)]
    public void FusedHierarchicalSoftmax_GpuMatchesCpu(int rows, int d, int treeDepth, int numClasses)
    {
        if (!EnsureGpuReady()) return;
        var input = Rand(273, rows, d);
        var nodeWeights = Rand(274, treeDepth, d);
        AssertMatch(_gpu.FusedHierarchicalSoftmax(input, nodeWeights, numClasses), _cpu.FusedHierarchicalSoftmax(input, nodeWeights, numClasses), $"HSoftmax[{rows}x{d};td{treeDepth};nc{numClasses}]");
    }

    [Theory]
    [InlineData(1, 4, 8, 2)]
    [InlineData(2, 5, 12, 3)]
    [InlineData(2, 6, 16, 4)]
    public void Rwkv7SequenceForward_GpuMatchesCpu(int batch, int seq, int modelDim, int numHeads)
    {
        if (!EnsureGpuReady()) return;
        var r = Rand(260, batch, seq, modelDim);
        var k = Rand(261, batch, seq, modelDim);
        var v = Rand(262, batch, seq, modelDim);
        var a = Rand(263, batch, seq, modelDim);
        var b = Rand(264, batch, seq, modelDim);
        var cpu = _cpu.Rwkv7SequenceForward(r, k, v, a, b, numHeads);
        var gpu = _gpu.Rwkv7SequenceForward(r, k, v, a, b, numHeads);
        AssertMatch(gpu, cpu, $"RWKV7[b{batch};s{seq};d{modelDim};h{numHeads}]");
    }

    [Theory]
    [InlineData(2, 5, 4, 6, false)]
    [InlineData(2, 5, 4, 6, true)]
    [InlineData(3, 7, 8, 5, true)]
    public void LstmSequenceForward_GpuMatchesCpu(int batch, int seq, int inF, int hidden, bool returnSeq)
    {
        if (!EnsureGpuReady()) return;
        var input = Rand(250, batch, seq, inF);
        var wIh = Rand(251, 4 * hidden, inF);
        var wHh = Rand(252, 4 * hidden, hidden);
        var bIh = Rand(253, 4 * hidden);
        var bHh = Rand(254, 4 * hidden);
        var cpu = _cpu.LstmSequenceForward(input, null, null, wIh, wHh, bIh, bHh, returnSeq);
        var gpu = _gpu.LstmSequenceForward(input, null, null, wIh, wHh, bIh, bHh, returnSeq);
        AssertMatch(gpu, cpu, $"LSTM[b{batch};s{seq};in{inF};h{hidden};seq={returnSeq}]");
    }

    [Theory]
    [InlineData(1, 4, 8, 2)]
    [InlineData(2, 6, 12, 3)]
    [InlineData(1, 3, 16, 4)]
    public void MultiHeadAttentionForward_GpuMatchesCpu(int batch, int seq, int dModel, int numHeads)
    {
        if (!EnsureGpuReady()) return;
        var input = Rand(240, batch, seq, dModel);
        var qW = Rand(241, dModel, dModel);
        var kW = Rand(242, dModel, dModel);
        var vW = Rand(243, dModel, dModel);
        var oW = Rand(244, dModel, dModel);
        var cpu = _cpu.MultiHeadAttentionForward(input, qW, kW, vW, oW, numHeads, null);
        var gpu = _gpu.MultiHeadAttentionForward(input, qW, kW, vW, oW, numHeads, null);
        AssertMatch(gpu, cpu, $"MHA[b{batch};s{seq};d{dModel};h{numHeads}]");
    }

    [Theory]
    [InlineData("Sum")]
    [InlineData("Prod")]
    [InlineData("AMax")]
    [InlineData("AMin")]
    public void TensorScatterReduce_GpuMatchesCpu(string modeName)
    {
        if (!EnsureGpuReady()) return;
        var mode = (AiDotNet.Tensors.Engines.ScatterReduceMode)System.Enum.Parse(typeof(AiDotNet.Tensors.Engines.ScatterReduceMode), modeName);
        var t = Rand(300, 4, 5);
        var rng = new Random(301);
        var idxData = new int[20];
        for (int i = 0; i < 20; i++) idxData[i] = rng.Next(4);   // scatter along dim 0 (dstDim=4)
        var indices = new Tensor<int>(idxData, new[] { 4, 5 });
        var src = Rand(302, 4, 5);
        AssertMatch(_gpu.TensorScatterReduce(t, 0, indices, src, mode, true),
                    _cpu.TensorScatterReduce(t, 0, indices, src, mode, true), $"ScatterReduce[{modeName}]");
    }

    [Fact]
    public void TensorBlockDiag_GpuMatchesCpu()
    {
        if (!EnsureGpuReady()) return;
        var a = Rand(296, 2, 3);
        var b = Rand(297, 3, 2);
        var c = Rand(298, 1, 4);
        AssertMatch(_gpu.TensorBlockDiag(new[] { a, b, c }), _cpu.TensorBlockDiag(new[] { a, b, c }), "BlockDiag");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TensorIsIn_GpuMatchesCpu(bool invert)
    {
        if (!EnsureGpuReady()) return;
        var elements = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });
        var test = new Tensor<float>(new float[] { 5, 2, 9, 4 }, new[] { 4 });
        AssertBitMatch(_gpu.TensorIsIn(elements, test, invert), _cpu.TensorIsIn(elements, test, invert), "IsIn");
    }

    [Theory]
    [InlineData(new[] { 10 }, 0, 3, 2)]
    [InlineData(new[] { 4, 8 }, 1, 4, 2)]
    [InlineData(new[] { 3, 6, 5 }, 1, 3, 1)]
    public void TensorUnfold_GpuMatchesCpu(int[] shape, int dim, int size, int step)
    {
        if (!EnsureGpuReady()) return;
        var t = Rand(295, shape);
        AssertMatch(_gpu.TensorUnfold(t, dim, size, step), _cpu.TensorUnfold(t, dim, size, step), $"Unfold[{string.Join("x", shape)};d{dim};sz{size};st{step}]");
    }

    [Fact]
    public void TensorPut_MaskedScatter_GpuMatchesCpu()
    {
        if (!EnsureGpuReady()) return;
        var t = Rand(290, 4, 5);
        var idx = new Tensor<int>(new[] { 0, 7, 13, 19, 3 }, new[] { 5 });
        var src = Rand(291, 5);
        AssertMatch(_gpu.TensorPut(t, idx, src), _cpu.TensorPut(t, idx, src), "Put");

        var rng = new Random(292);
        var bits = new AiDotNet.Tensors.Bit[20];
        for (int i = 0; i < 20; i++) bits[i] = rng.Next(2) == 1;
        var mask = new Tensor<AiDotNet.Tensors.Bit>(bits, new[] { 4, 5 });
        var source = Rand(293, 20);
        AssertMatch(_gpu.TensorMaskedScatter(t, mask, source), _cpu.TensorMaskedScatter(t, mask, source), "MaskedScatter");
    }

    [Fact]
    public void TensorCountNonzero_NanMedian_GpuMatchesCpu()
    {
        if (!EnsureGpuReady()) return;
        var data = new float[] { 0f, 1f, 0f, -2f, 3f, 0f, 0f, 5f };
        var t = new Tensor<float>(data, new[] { 8 });
        Assert.Equal(_cpu.TensorCountNonzero(t), _gpu.TensorCountNonzero(t));
        var nanData = new float[] { 3f, float.NaN, 1f, 2f, float.NaN, 5f, 4f };
        var nt = new Tensor<float>(nanData, new[] { 7 });
        Assert.Equal((double)_cpu.TensorNanMedian(nt), (double)_gpu.TensorNanMedian(nt), 4);
    }

    [Theory]
    [InlineData(3, 4)]
    [InlineData(2, 3)]
    public void TensorCartesianProd_GpuMatchesCpu(int n0, int n1)
    {
        if (!EnsureGpuReady()) return;
        var a = Rand(280, n0);
        var b = Rand(281, n1);
        AssertMatch(_gpu.TensorCartesianProd(new[] { a, b }), _cpu.TensorCartesianProd(new[] { a, b }), "CartesianProd");
    }

    [Theory]
    [InlineData(10, 0f, 1f)]
    [InlineData(6, -2f, 4f)]
    public void TensorHistogram_GpuMatchesCpu(int bins, float mn, float mx)
    {
        if (!EnsureGpuReady()) return;
        var rng = new Random(282);
        var data = new float[400];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * (mx - mn) + mn);
        var t = new Tensor<float>(data, new[] { data.Length });
        Assert.Equal(_cpu.TensorHistogram(t, bins, mn, mx).ToArray(), _gpu.TensorHistogram(t, bins, mn, mx).ToArray());
    }

    [Fact]
    public void ArgsortToTakeAlongDim_ResidentChain_GpuMatchesCpu()
    {
        if (!EnsureGpuReady()) return;
        var x = DistinctRand(330, 4, 8);
        // Argsort -> gather along the last axis: indices stay GPU-resident through the chain and must
        // reproduce the sorted values.
        var gIdx = _gpu.TensorArgsort(x, -1, false);
        var gSorted = _gpu.TensorTakeAlongDim(x, gIdx, -1);
        var cIdx = _cpu.TensorArgsort(x, -1, false);
        var cSorted = _cpu.TensorTakeAlongDim(x, cIdx, -1);
        Assert.Equal(cIdx.ToArray(), gIdx.ToArray());
        AssertMatch(gSorted, cSorted, "argsort->takeAlongDim");
    }

    [Fact]
    public void PredicateToLogical_ResidentChain_GpuMatchesCpu()
    {
        if (!EnsureGpuReady()) return;
        var a = new Tensor<float>(new float[] { 1f, 2f, 3f, float.NaN, 5f, 0f }, new[] { 6 });
        var b = new Tensor<float>(new float[] { 1f, 0f, 3f, 4f, 0f, 0f }, new[] { 6 });
        // (a == b) & isfinite(a) | isnan(a)  — predicate results stay GPU-resident through the logical ops.
        var gpu = _gpu.TensorLogicalOr(_gpu.TensorLogicalAnd(_gpu.TensorEq(a, b), _gpu.TensorIsFinite(a)), _gpu.TensorIsNan(a));
        var cpu = _cpu.TensorLogicalOr(_cpu.TensorLogicalAnd(_cpu.TensorEq(a, b), _cpu.TensorIsFinite(a)), _cpu.TensorIsNan(a));
        AssertBitMatch(gpu, cpu, "predicate->logical chain");
    }

    [Fact]
    public void TensorLogical_GpuMatchesCpu_AndResidentChain()
    {
        if (!EnsureGpuReady()) return;
        AiDotNet.Tensors.Bit[] ad = { true, true, false, false, true, false };
        AiDotNet.Tensors.Bit[] bd = { true, false, true, false, false, true };
        var a = new Tensor<AiDotNet.Tensors.Bit>(ad, new[] { 6 });
        var b = new Tensor<AiDotNet.Tensors.Bit>(bd, new[] { 6 });
        AssertBitMatch(_gpu.TensorLogicalAnd(a, b), _cpu.TensorLogicalAnd(a, b), "And");
        AssertBitMatch(_gpu.TensorLogicalOr(a, b), _cpu.TensorLogicalOr(a, b), "Or");
        AssertBitMatch(_gpu.TensorLogicalXor(a, b), _cpu.TensorLogicalXor(a, b), "Xor");
        AssertBitMatch(_gpu.TensorLogicalNot(a), _cpu.TensorLogicalNot(a), "Not");
        // Boolean expression chain: (a & b) | !a  — the intermediate Bit masks stay GPU-resident.
        var chainG = _gpu.TensorLogicalOr(_gpu.TensorLogicalAnd(a, b), _gpu.TensorLogicalNot(a));
        var chainC = _cpu.TensorLogicalOr(_cpu.TensorLogicalAnd(a, b), _cpu.TensorLogicalNot(a));
        AssertBitMatch(chainG, chainC, "chain");
    }

    [Fact]
    public void GridSampleBackward_GpuMatchesCpu()
    {
        if (!EnsureGpuReady()) return;
        int N = 2, H = 4, W = 5, C = 3, outH = 3, outW = 4;   // NHWC
        var gradOut = Rand(340, N, outH, outW, C);
        var input = Rand(342, N, H, W, C);
        var grid = Rand(341, N, outH, outW, 2);               // normalized coords ~[-1,1]
        var mode = AiDotNet.Tensors.Engines.GridSampleMode.Bilinear;
        var pad = AiDotNet.Tensors.Engines.GridSamplePadding.Zeros;
        AssertMatch(_gpu.GridSampleBackwardInput(gradOut, grid, new[] { N, H, W, C }, mode, pad, false),
                    _cpu.GridSampleBackwardInput(gradOut, grid, new[] { N, H, W, C }, mode, pad, false), "GSBackwardInput");
        AssertMatch(_gpu.GridSampleBackwardGrid(gradOut, input, grid, mode, pad, false),
                    _cpu.GridSampleBackwardGrid(gradOut, input, grid, mode, pad, false), "GSBackwardGrid");
    }

    [Theory]
    [InlineData(0.5)]
    [InlineData(0.3)]
    [InlineData(0.7)]
    public void Nms_GpuMatchesCpu(double iou)
    {
        if (!EnsureGpuReady()) return;
        var boxes = new Tensor<float>(new float[]
        {
            0, 0, 10, 10,     1, 1, 11, 11,     20, 20, 30, 30,
            21, 21, 31, 31,   0, 0, 5, 5,       100, 100, 120, 120,
        }, new[] { 6, 4 });
        var scores = new Tensor<float>(new float[] { 0.9f, 0.8f, 0.95f, 0.7f, 0.6f, 0.85f }, new[] { 6 });
        Assert.Equal(_cpu.Nms(boxes, scores, iou).ToArray(), _gpu.Nms(boxes, scores, iou).ToArray());
    }

    [Fact]
    public void Resample_GpuMatchesCpu()
    {
        if (!EnsureGpuReady()) return;
        var wave = Rand(370, 64);
        AssertMatch(_gpu.Resample(wave, 4, 8), _cpu.Resample(wave, 4, 8), "Resample-up2x");
        AssertMatch(_gpu.Resample(wave, 8, 4), _cpu.Resample(wave, 8, 4), "Resample-down2x");
    }

    [Fact]
    public void PitchShift_GpuMatchesCpu()
    {
        if (!EnsureGpuReady()) return;
        // nSteps must keep TimeStretch's outFrames <= nFft or the CPU ISTFT indexes out of bounds
        // (pitchrate~2 overflows). 3 semitones -> pitchrate~1.19, outFrames floor(9/0.84)=10 <= 16.
        var wave = Rand(371, 256);
        // PitchShift is a composite STFT -> phase-vocoder time-stretch -> ISTFT -> resample pipeline. Its
        // component stages (TimeStretch, Resample) each pass the 1e-3 bar, but composing them through two
        // FFT round-trips + cross-frame phase accumulation legitimately compounds GPU-vs-CPU fp-ordering
        // differences to ~3e-3 (deterministic). Use an accumulation-appropriate tolerance for the pipeline
        // (cf. TensorVecDot, which already scales tolerance for accumulation depth) — a real kernel bug at
        // nFft=16 would diverge far more.
        AssertMatch(_gpu.PitchShift(wave, 16000, 3.0, 16, 4), _cpu.PitchShift(wave, 16000, 3.0, 16, 4), "PitchShift", 5e-3f);
    }

    [Fact]
    public void TimeStretch_GpuMatchesCpu()
    {
        if (!EnsureGpuReady()) return;
        int nFft = 16, hop = 4;
        var wave = Rand(360, 128);
        AssertMatch(_gpu.TimeStretch(wave, 1.5, nFft, hop), _cpu.TimeStretch(wave, 1.5, nFft, hop), "TimeStretch1.5");
        AssertMatch(_gpu.TimeStretch(wave, 0.8, nFft, hop), _cpu.TimeStretch(wave, 0.8, nFft, hop), "TimeStretch0.8");
    }

    [Fact]
    public void Spectrogram_GpuMatchesCpu()
    {
        if (!EnsureGpuReady()) return;
        int nFft = 16, hop = 4;
        var wave = Rand(350, 64);                 // rank-1 waveform
        AssertMatch(_gpu.Spectrogram(wave, nFft, hop, nFft), _cpu.Spectrogram(wave, nFft, hop, nFft), "Spectrogram1d");
        var waveB = Rand(351, 2, 48);             // batched waveform
        AssertMatch(_gpu.Spectrogram(waveB, nFft, hop, nFft), _cpu.Spectrogram(waveB, nFft, hop, nFft), "SpectrogramBatch");
    }

    [Fact]
    public void BatchedNms_GpuMatchesCpu()
    {
        if (!EnsureGpuReady()) return;
        var boxes = new Tensor<float>(new float[]
        {
            0, 0, 10, 10,     1, 1, 11, 11,     20, 20, 30, 30,
            21, 21, 31, 31,   0, 0, 5, 5,       2, 2, 12, 12,
        }, new[] { 6, 4 });
        var scores = new Tensor<float>(new float[] { 0.9f, 0.8f, 0.95f, 0.7f, 0.6f, 0.85f }, new[] { 6 });
        var classes = new Tensor<int>(new[] { 0, 0, 1, 1, 0, 1 }, new[] { 6 });
        Assert.Equal(_cpu.BatchedNms(boxes, scores, classes, 0.5).ToArray(), _gpu.BatchedNms(boxes, scores, classes, 0.5).ToArray());
    }

    [Fact]
    public void MasksToBoxes_GpuMatchesCpu()
    {
        if (!EnsureGpuReady()) return;
        // 3 masks [3,5,6]: one with a box, one empty, one full-edge.
        var data = new float[3 * 5 * 6];
        // mask 0: rectangle rows 1..3, cols 2..4
        for (int y = 1; y <= 3; y++) for (int x = 2; x <= 4; x++) data[0 * 30 + y * 6 + x] = 1f;
        // mask 1: empty (all zero)
        // mask 2: single pixel at (4,5)
        data[2 * 30 + 4 * 6 + 5] = 1f;
        var masks = new Tensor<float>(data, new[] { 3, 5, 6 });
        Assert.Equal(_cpu.MasksToBoxes(masks).ToArray(), _gpu.MasksToBoxes(masks).ToArray());
    }

    [Fact]
    public void TensorHistogramDD_GpuMatchesCpu()
    {
        if (!EnsureGpuReady()) return;
        var rng = new Random(320);
        var data = new float[200 * 2];
        for (int i = 0; i < data.Length; i++) data[i] = (float)rng.NextDouble();
        var samples = new Tensor<float>(data, new[] { 200, 2 });
        var bins = new[] { 4, 5 };
        var mins = new[] { 0f, 0f };
        var maxs = new[] { 1f, 1f };
        Assert.Equal(_cpu.TensorHistogramDD(samples, bins, mins, maxs).ToArray(),
                     _gpu.TensorHistogramDD(samples, bins, mins, maxs).ToArray());
    }

    [Fact]
    public void TensorNonzero_GpuMatchesCpu()
    {
        if (!EnsureGpuReady()) return;
        var data = new float[] { 0f, 1f, 0f, 0f, 2f, 0f, 3f, 0f, 0f, 4f, 5f, 0f };
        var t = new Tensor<float>(data, new[] { 3, 4 });
        Assert.Equal(_cpu.TensorNonzero(t).ToArray(), _gpu.TensorNonzero(t).ToArray());
    }

    [Fact]
    public void TensorUniqueConsecutive_GpuMatchesCpu()
    {
        if (!EnsureGpuReady()) return;
        var data = new float[] { 1f, 1f, 2f, 2f, 2f, 3f, 1f, 1f, 4f };
        var t = new Tensor<float>(data, new[] { 9 });
        AssertMatch(_gpu.TensorUniqueConsecutive(t), _cpu.TensorUniqueConsecutive(t), "UniqueConsecutive");
    }

    [Fact]
    public void TensorZeta_GpuMatchesCpu()
    {
        if (!EnsureGpuReady()) return;
        var x = new Tensor<float>(new float[] { 2f, 2.5f, 3f, 4f, 6f, 5f }, new[] { 6 });
        var q = new Tensor<float>(new float[] { 1f, 2f, 0.5f, 1.5f, 3f, 2f }, new[] { 6 });
        var ca = _cpu.TensorZeta(x, q).ToArray(); var ga = _gpu.TensorZeta(x, q).ToArray();
        for (int i = 0; i < ca.Length; i++)
            Assert.True(Math.Abs((double)ga[i] - ca[i]) < 1e-3 * Math.Max(1.0, Math.Abs((double)ca[i])), $"Zeta[{i}]: gpu {ga[i]} vs cpu {ca[i]}");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void TensorPolygamma_GpuMatchesCpu(int n)
    {
        if (!EnsureGpuReady()) return;
        var x = new Tensor<float>(new float[] { 1f, 1.5f, 2f, 3f, 5f, 8f }, new[] { 6 });
        var ca = _cpu.TensorPolygamma(n, x).ToArray(); var ga = _gpu.TensorPolygamma(n, x).ToArray();
        for (int i = 0; i < ca.Length; i++)
            Assert.True(Math.Abs((double)ga[i] - ca[i]) < 1e-3 * Math.Max(1.0, Math.Abs((double)ca[i])), $"Polygamma(n={n})[{i}]: gpu {ga[i]} vs cpu {ca[i]}");
    }

    [Fact]
    public void TensorMaskedSelect_GpuMatchesCpu()
    {
        if (!EnsureGpuReady()) return;
        var t = Rand(230, 4, 5);
        var rng = new Random(231);
        var bits = new AiDotNet.Tensors.Bit[20];
        for (int i = 0; i < 20; i++) bits[i] = rng.Next(2) == 1;
        var mask = new Tensor<AiDotNet.Tensors.Bit>(bits, new[] { 4, 5 });
        AssertMatch(_gpu.TensorMaskedSelect(t, mask), _cpu.TensorMaskedSelect(t, mask), "MaskedSelect");
    }

    [Fact]
    public void TensorUnique_GpuMatchesCpu()
    {
        if (!EnsureGpuReady()) return;
        var data = new float[] { 3f, 1f, 2f, 3f, 1f, 5f, 2f, 3f, 4f };
        var t = new Tensor<float>(data, new[] { data.Length });
        AssertMatch(_gpu.TensorUnique(t, true), _cpu.TensorUnique(t, true), "Unique");
    }

    // Distinct shuffled values so the sort permutation (and therefore indices) is unambiguous.
    private static Tensor<float> DistinctRand(int seed, params int[] shape)
    {
        int n = 1; foreach (var s in shape) n *= s;
        var vals = new float[n];
        for (int i = 0; i < n; i++) vals[i] = i - n / 2 + 0.25f;
        var rng = new Random(seed);
        for (int i = n - 1; i > 0; i--) { int j = rng.Next(i + 1); (vals[i], vals[j]) = (vals[j], vals[i]); }
        return new Tensor<float>(vals, shape);
    }

    [Theory]
    [InlineData(new[] { 16 }, false)]
    [InlineData(new[] { 16 }, true)]
    [InlineData(new[] { 5 }, false)]      // non power-of-two row
    [InlineData(new[] { 4, 7 }, false)]   // multi-row, last axis
    [InlineData(new[] { 3, 4, 6 }, true)]
    public void TensorSort_Argsort_GpuMatchesCpu(int[] shape, bool descending)
    {
        if (!EnsureGpuReady()) return;
        var t = DistinctRand(220, shape);
        var (gv, gi) = _gpu.TensorSort(t, -1, descending);
        var (cv, ci) = _cpu.TensorSort(t, -1, descending);
        AssertMatch(gv, cv, "Sort.values");
        Assert.Equal(ci.ToArray(), gi.ToArray());
        Assert.Equal(_cpu.TensorArgsort(t, -1, descending).ToArray(), _gpu.TensorArgsort(t, -1, descending).ToArray());
    }

    [Theory]
    [InlineData(15)]
    [InlineData(16)]
    [InlineData(100)]
    public void TensorMedian_Kthvalue_Mode_GpuMatchesCpu(int n)
    {
        if (!EnsureGpuReady()) return;
        var t = DistinctRand(221, n);
        Assert.Equal((double)_cpu.TensorMedian(t), (double)_gpu.TensorMedian(t), 4);
        int k = n / 3 + 1;
        var (cv, cidx) = _cpu.TensorKthvalue(t, k);
        var (gv, gidx) = _gpu.TensorKthvalue(t, k);
        Assert.Equal((double)cv, (double)gv, 4);
        Assert.Equal(cidx, gidx);
        // Mode: use a tensor with a clear most-frequent value.
        var modeData = new float[] { 1, 2, 2, 3, 3, 3, 4, 5 };
        var mt = new Tensor<float>(modeData, new[] { modeData.Length });
        var (cmv, cmc) = _cpu.TensorMode(mt);
        var (gmv, gmc) = _gpu.TensorMode(mt);
        Assert.Equal((double)cmv, (double)gmv, 4);
        Assert.Equal(cmc, gmc);
    }

    private static void AssertBitMatch(Tensor<AiDotNet.Tensors.Bit> gpu, Tensor<AiDotNet.Tensors.Bit> cpu, string op)
    {
        Assert.Equal(cpu.Shape.ToArray(), gpu.Shape.ToArray());
        var g = gpu.ToArray();
        var c = cpu.ToArray();
        for (int i = 0; i < c.Length; i++)
            Assert.True(g[i].Equals(c[i]), $"{op}: mismatch at {i} (gpu {g[i]} vs cpu {c[i]})");
    }

    [Fact]
    public void TensorEq_GpuMatchesCpu()
    {
        if (!EnsureGpuReady()) return;
        // Build a with some elements equal to b.
        var aData = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var bData = new float[] { 1, 0, 3, 0, 5, 6, 0, 8 };
        var a = new Tensor<float>(aData, new[] { 2, 4 });
        var b = new Tensor<float>(bData, new[] { 2, 4 });
        AssertBitMatch(_gpu.TensorEq(a, b), _cpu.TensorEq(a, b), "TensorEq");
    }

    [Theory]
    [InlineData("ij")]
    [InlineData("xy")]
    public void TensorMeshgrid_GpuMatchesCpu(string indexing)
    {
        if (!EnsureGpuReady()) return;
        var a = Rand(206, 3);
        var b = Rand(207, 4);
        var c = Rand(208, 2);
        var inputs = new[] { a, b, c };
        var gpu = _gpu.TensorMeshgrid(inputs, indexing);
        var cpu = _cpu.TensorMeshgrid(inputs, indexing);
        Assert.Equal(cpu.Length, gpu.Length);
        for (int k = 0; k < cpu.Length; k++)
            AssertMatch(gpu[k], cpu[k], $"Meshgrid[{indexing}][{k}]");
    }

    [Theory]
    [InlineData(10, 0f, 1f)]
    [InlineData(8, -2f, 2f)]
    [InlineData(5, 0f, 10f)]
    public void TensorHistc_GpuMatchesCpu(int bins, float mn, float mx)
    {
        if (!EnsureGpuReady()) return;
        var rng = new Random(205);
        var data = new float[500];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * (mx - mn) + mn) * 1.2f - 0.1f; // some out of range
        var t = new Tensor<float>(data, new[] { data.Length });
        AssertMatch(_gpu.TensorHistc(t, bins, mn, mx), _cpu.TensorHistc(t, bins, mn, mx), $"Histc[bins={bins};{mn}..{mx}]");
    }

    [Theory]
    [InlineData(5, 3, 2.0)]
    [InlineData(8, 4, 1.0)]
    [InlineData(4, 6, 3.0)]
    public void TensorPDist_GpuMatchesCpu(int n, int d, double p)
    {
        if (!EnsureGpuReady()) return;
        var x = Rand(204, n, d);
        AssertMatch(_gpu.TensorPDist(x, p), _cpu.TensorPDist(x, p), $"PDist[{n}x{d};p={p}]");
    }

    [Theory]
    [InlineData(4, 5, 3, 2.0)]
    [InlineData(6, 6, 8, 1.0)]
    [InlineData(3, 7, 4, 3.0)]
    public void TensorCDist_GpuMatchesCpu(int m, int n, int d, double p)
    {
        if (!EnsureGpuReady()) return;
        var x1 = Rand(202, m, d);
        var x2 = Rand(203, n, d);
        AssertMatch(_gpu.TensorCDist(x1, x2, p), _cpu.TensorCDist(x1, x2, p), $"CDist[{m}x{d},{n}x{d};p={p}]");
    }

    [Theory]
    [InlineData(new[] { 6, 4 }, 0, 1, 3)]
    [InlineData(new[] { 6, 4 }, 1, 0, 2)]
    [InlineData(new[] { 3, 5, 4 }, 1, 2, 2)]
    public void TensorSliceScatter_GpuMatchesCpu(int[] shape, int dim, int start, int length)
    {
        if (!EnsureGpuReady()) return;
        var t = Rand(210, shape);
        var srcShape = (int[])shape.Clone(); srcShape[dim] = length;
        var src = Rand(211, srcShape);
        AssertMatch(_gpu.TensorSliceScatter(t, src, dim, start, length), _cpu.TensorSliceScatter(t, src, dim, start, length),
            $"SliceScatter[{string.Join("x", shape)};dim={dim};{start}:{start + length}]");
    }

    [Theory]
    [InlineData(new[] { 6, 4 }, 0, 2)]
    [InlineData(new[] { 3, 5, 4 }, 1, 3)]
    public void TensorSelectScatter_GpuMatchesCpu(int[] shape, int dim, int index)
    {
        if (!EnsureGpuReady()) return;
        var t = Rand(212, shape);
        var srcShapeList = new List<int>();
        for (int i = 0; i < shape.Length; i++) if (i != dim) srcShapeList.Add(shape[i]);
        var src = Rand(213, srcShapeList.ToArray());
        AssertMatch(_gpu.TensorSelectScatter(t, src, dim, index), _cpu.TensorSelectScatter(t, src, dim, index),
            $"SelectScatter[{string.Join("x", shape)};dim={dim};idx={index}]");
    }

    [Theory]
    [InlineData(new[] { 5, 4 }, 0)]
    [InlineData(new[] { 5, 4 }, 1)]
    [InlineData(new[] { 3, 4, 5 }, 1)]
    public void TensorIndexCopy_GpuMatchesCpu(int[] shape, int axis)
    {
        if (!EnsureGpuReady()) return;
        var t = Rand(200, shape);
        // pick 2 unique indices along axis
        int dstAxis = shape[axis];
        var idx = new Tensor<int>(new[] { 0, Math.Min(2, dstAxis - 1) }, new[] { 2 });
        var srcShape = (int[])shape.Clone(); srcShape[axis] = 2;
        var src = Rand(201, srcShape);
        AssertMatch(_gpu.TensorIndexCopy(t, axis, idx, src), _cpu.TensorIndexCopy(t, axis, idx, src), $"IndexCopy[{string.Join("x", shape)};ax={axis}]");
        AssertMatch(_gpu.TensorIndexFill(t, axis, idx, 9.5f), _cpu.TensorIndexFill(t, axis, idx, 9.5f), $"IndexFill[{string.Join("x", shape)};ax={axis}]");
    }

    [Fact]
    public void TensorNextAfter_GpuMatchesCpu()
    {
        if (!EnsureGpuReady()) return;
        var a = new Tensor<float>(new float[] { 1f, 2f, -3f, 0f, 0f, 5f, 100f, -0.5f }, new[] { 8 });
        var b = new Tensor<float>(new float[] { 2f, 1f, -2f, 1f, -1f, 5f, 99f, -0.6f }, new[] { 8 });
        AssertMatch(_gpu.TensorNextAfter(a, b), _cpu.TensorNextAfter(a, b), "NextAfter");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TensorSearchSorted_Bucketize_GpuMatchesCpu(bool right)
    {
        if (!EnsureGpuReady()) return;
        var seq = new Tensor<float>(new float[] { 0f, 1f, 2.5f, 5f, 9f }, new[] { 5 });
        var vals = new Tensor<float>(new float[] { -1f, 0f, 1f, 2f, 2.5f, 7f, 9f, 12f }, new[] { 8 });
        var gSS = _gpu.TensorSearchSorted(seq, vals, right);
        var cSS = _cpu.TensorSearchSorted(seq, vals, right);
        Assert.Equal(cSS.ToArray(), gSS.ToArray());
        // Bucketize delegates to SearchSorted -> auto-on-GPU.
        var gB = _gpu.TensorBucketize(vals, seq, right);
        var cB = _cpu.TensorBucketize(vals, seq, right);
        Assert.Equal(cB.ToArray(), gB.ToArray());
    }

    [Theory]
    [InlineData(2, 3, 2, 2)]
    [InlineData(3, 1, 2, 4)]
    [InlineData(4, 4, 3, 3)]
    public void TensorKron_GpuMatchesCpu(int am, int an, int bp, int bq)
    {
        if (!EnsureGpuReady()) return;
        var a = Rand(196, am, an);
        var b = Rand(197, bp, bq);
        AssertMatch(_gpu.TensorKron(a, b), _cpu.TensorKron(a, b), $"Kron[{am}x{an} (x) {bp}x{bq}]");
    }

    [Theory]
    [InlineData(new[] { 3 }, -1)]
    [InlineData(new[] { 4, 3 }, 1)]
    [InlineData(new[] { 3, 5 }, 0)]
    [InlineData(new[] { 2, 3, 4 }, 1)]
    public void TensorCross_GpuMatchesCpu(int[] shape, int dim)
    {
        if (!EnsureGpuReady()) return;
        var a = Rand(192, shape);
        var b = Rand(193, shape);
        AssertMatch(_gpu.TensorCross(a, b, dim), _cpu.TensorCross(a, b, dim), $"Cross[{string.Join("x", shape)};dim={dim}]");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(17)]
    [InlineData(256)]
    public void TensorLdexp_GpuMatchesCpu(int n)
    {
        if (!EnsureGpuReady()) return;
        var x = Rand(194, n);
        var rng = new Random(195);
        var expData = new int[n];
        for (int i = 0; i < n; i++) expData[i] = rng.Next(-6, 7);
        var exp = new Tensor<int>(expData, new[] { n });
        AssertMatch(_gpu.TensorLdexp(x, exp), _cpu.TensorLdexp(x, exp), $"Ldexp[{n}]");
    }

    public static IEnumerable<object[]> TakeAlongDimCases() => new List<object[]>
    {
        new object[] { new[] { 4, 5 }, 1, 3 },     // gather along last axis
        new object[] { new[] { 4, 5 }, 0, 6 },     // gather along axis 0 (axisOut>axisIn)
        new object[] { new[] { 3, 4, 5 }, 2, 2 },  // 3-D, last axis
        new object[] { new[] { 3, 4, 5 }, 1, 7 },  // 3-D, middle axis
    };

    [Theory]
    [MemberData(nameof(TakeAlongDimCases))]
    public void TensorTakeAlongDim_GpuMatchesCpu(int[] shape, int dim, int axisOut)
    {
        if (!EnsureGpuReady()) return;
        var t = Rand(190, shape);
        int axisIn = shape[dim];
        var idxShape = (int[])shape.Clone();
        idxShape[dim] = axisOut;
        int idxLen = 1; foreach (var s in idxShape) idxLen *= s;
        var rng = new Random(191);
        var idxData = new int[idxLen];
        for (int i = 0; i < idxLen; i++) idxData[i] = rng.Next(axisIn);
        var indices = new Tensor<int>(idxData, idxShape);
        AssertMatch(_gpu.TensorTakeAlongDim(t, indices, dim), _cpu.TensorTakeAlongDim(t, indices, dim),
            $"TakeAlongDim[{string.Join("x", shape)};dim={dim};out={axisOut}]");
    }

    [Fact]
    public void TensorClampTensor_GpuMatchesCpu()
    {
        if (!EnsureGpuReady()) return;
        var t = Rand(180, 4, 8);
        var lo = Rand(181, 4, 8);
        var hi = new float[32];
        var loArr = lo.ToArray();
        for (int i = 0; i < 32; i++) hi[i] = loArr[i] + 0.5f; // ensure hi >= lo
        var hiT = new Tensor<float>(hi, new[] { 4, 8 });
        AssertMatch(_gpu.TensorClampTensor(t, lo, hiT), _cpu.TensorClampTensor(t, lo, hiT), "ClampTensor both");
        AssertMatch(_gpu.TensorClampTensor(t, lo, null), _cpu.TensorClampTensor(t, lo, null), "ClampTensor minonly");
        AssertMatch(_gpu.TensorClampTensor(t, null, hiT), _cpu.TensorClampTensor(t, null, hiT), "ClampTensor maxonly");
    }

    [Fact]
    public void TensorIsClose_AllClose_GpuMatchesCpu()
    {
        if (!EnsureGpuReady()) return;
        var a = new Tensor<float>(new float[] { 1f, 2f, 3f, float.NaN, 5f }, new[] { 5 });
        var bClose = new Tensor<float>(new float[] { 1.0000001f, 2f, 3.0000002f, float.NaN, 5f }, new[] { 5 });
        var bFar = new Tensor<float>(new float[] { 1.5f, 2f, 3f, 4f, 5f }, new[] { 5 });
        AssertBitMatch(_gpu.TensorIsClose(a, bClose, 1e-5f, 1e-8f), _cpu.TensorIsClose(a, bClose, 1e-5f, 1e-8f), "IsClose(close+nan)");
        AssertBitMatch(_gpu.TensorIsClose(a, bFar, 1e-5f, 1e-8f), _cpu.TensorIsClose(a, bFar, 1e-5f, 1e-8f), "IsClose(far)");
        var x = Rand(182, 64);
        var xCopy = new Tensor<float>((float[])x.ToArray().Clone(), new[] { 64 });
        Assert.Equal(_cpu.TensorAllClose(x, xCopy, 1e-5f, 1e-8f), _gpu.TensorAllClose(x, xCopy, 1e-5f, 1e-8f));
        Assert.Equal(_cpu.TensorAllClose(a, bFar, 1e-5f, 1e-8f), _gpu.TensorAllClose(a, bFar, 1e-5f, 1e-8f));
    }

    [Fact]
    public void TensorEqual_GpuMatchesCpu()
    {
        if (!EnsureGpuReady()) return;
        var a = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        var bSame = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        var bDiff = new Tensor<float>(new float[] { 1, 2, 3, 5 }, new[] { 2, 2 });
        Assert.Equal(_cpu.TensorEqual(a, bSame), _gpu.TensorEqual(a, bSame));
        Assert.Equal(_cpu.TensorEqual(a, bDiff), _gpu.TensorEqual(a, bDiff));
        Assert.True(_gpu.TensorEqual(a, bSame));
        Assert.False(_gpu.TensorEqual(a, bDiff));
    }

    [Fact]
    public void TensorIsNan_IsInf_IsFinite_GpuMatchesCpu()
    {
        if (!EnsureGpuReady()) return;
        var data = new float[] { 1f, float.NaN, 3f, float.PositiveInfinity, float.NegativeInfinity, 0f, -2.5f, float.NaN };
        var t = new Tensor<float>(data, new[] { 8 });
        AssertBitMatch(_gpu.TensorIsNan(t), _cpu.TensorIsNan(t), "TensorIsNan");
        AssertBitMatch(_gpu.TensorIsInf(t), _cpu.TensorIsInf(t), "TensorIsInf");
        AssertBitMatch(_gpu.TensorIsFinite(t), _cpu.TensorIsFinite(t), "TensorIsFinite");
    }

    [Theory]
    [InlineData(3f)]
    [InlineData(0f)]
    public void TensorEqScalar_GpuMatchesCpu(float scalar)
    {
        if (!EnsureGpuReady()) return;
        var a = new Tensor<float>(new float[] { 0, 3, 3, 1, 0, 3, 7, 3 }, new[] { 8 });
        AssertBitMatch(_gpu.TensorEqScalar(a, scalar), _cpu.TensorEqScalar(a, scalar), "TensorEqScalar");
    }

    [Theory]
    [InlineData(2, 2)]
    [InlineData(4, 4)]
    [InlineData(3, 5)]
    [InlineData(6, 3)]
    [InlineData(64, 64)]
    public void TensorTrace_GpuMatchesCpu(int rows, int cols)
    {
        if (!EnsureGpuReady()) return;
        var t = Rand(165, rows, cols);
        float cpu = _cpu.TensorTrace(t);
        float gpu = _gpu.TensorTrace(t);
        Assert.True(Math.Abs((double)gpu - cpu) < 1e-3, $"Trace[{rows}x{cols}]: gpu {gpu} vs cpu {cpu}");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(33)]
    [InlineData(1024)]
    [InlineData(5000)]
    public void TensorAminmax_GpuMatchesCpu(int n)
    {
        if (!EnsureGpuReady()) return;
        var t = Rand(160, n);
        var (cMin, cMax) = _cpu.TensorAminmax(t);
        var (gMin, gMax) = _gpu.TensorAminmax(t);
        Assert.True(Math.Abs((double)gMin - cMin) < 1e-4, $"Aminmax[{n}] min: gpu {gMin} vs cpu {cMin}");
        Assert.True(Math.Abs((double)gMax - cMax) < 1e-4, $"Aminmax[{n}] max: gpu {gMax} vs cpu {cMax}");
    }

    // Whole MLP forward (chain of FusedLinear layers) — now decomposed onto the GPU.
    public static IEnumerable<object[]> MlpCases() => new List<object[]>
    {
        new object[] { 4, new[] { 8, 16, 10 } },        // 1 hidden layer
        new object[] { 2, new[] { 6, 12, 12, 4 } },     // 2 hidden layers
        new object[] { 8, new[] { 32, 64 } },           // single layer
    };

    [Theory]
    [MemberData(nameof(MlpCases))]
    public void MlpForward_GpuMatchesCpu(int batch, int[] dims)
    {
        if (!EnsureGpuReady()) return;
        int layers = dims.Length - 1;
        var input = Rand(150, batch, dims[0]);
        var weights = new Tensor<float>[layers];
        var biases = new Tensor<float>?[layers];
        for (int i = 0; i < layers; i++)
        {
            weights[i] = Rand(151 + i, dims[i], dims[i + 1]);
            biases[i] = Rand(171 + i, dims[i + 1]);
        }
        var hid = AiDotNet.Tensors.Engines.FusedActivationType.ReLU;
        var outAct = AiDotNet.Tensors.Engines.FusedActivationType.None;

        var cpu = _cpu.MlpForward(input, weights, biases, hid, outAct);
        var gpu = _gpu.MlpForward(input, weights, biases, hid, outAct);
        AssertMatch(gpu, cpu, $"MlpForward[b{batch};{string.Join("x", dims)}]");
    }

    // General broadcast (size-1 → N at arbitrary axes, incl. rank expansion).
    public static IEnumerable<object[]> BroadcastCases() => new List<object[]>
    {
        new object[] { new[] { 1, 4 }, new[] { 3, 4 } },           // broadcast axis 0
        new object[] { new[] { 4, 1 }, new[] { 4, 5 } },           // broadcast axis 1
        new object[] { new[] { 1, 3, 1 }, new[] { 2, 3, 5 } },     // two broadcast axes
        new object[] { new[] { 4 }, new[] { 2, 3, 4 } },           // rank expansion + identical tail
        new object[] { new[] { 3, 1 }, new[] { 2, 3, 6 } },        // rank expansion + broadcast
    };

    [Theory]
    [MemberData(nameof(BroadcastCases))]
    public void TensorBroadcastTo_GpuMatchesCpu(int[] shape, int[] target)
    {
        if (!EnsureGpuReady()) return;
        var t = Rand(140, shape);
        var cpu = _cpu.TensorBroadcastTo(t, target);
        var gpu = _gpu.TensorBroadcastTo(t, target);
        AssertMatch(gpu, cpu, $"TensorBroadcastTo[{string.Join("x", shape)}->{string.Join("x", target)}]");
    }

    // Engine-fix regression: a deep INTERLEAVED chain of the view-returning TensorPermute and
    // TensorFlip (flip∘permute∘flip∘permute). Before the GetOrAllocateBuffer materialize-strided-views
    // fix, this diverged (~1.6) because a strided permute view of a deferred tensor read the source's
    // un-permuted cached buffer. Must now match CPU bit-for-bit.
    [Fact]
    public void InterleavedPermuteFlipChain_ViewPermute_GpuMatchesCpu()
    {
        if (!EnsureGpuReady()) return;
        var x = Rand(130, 4, 6);
        var perm = new[] { 1, 0 };

        var cpu = _cpu.TensorFlip(_cpu.TensorPermute(_cpu.TensorFlip(_cpu.TensorPermute(x, perm), new[] { 0 }), perm), new[] { 0 });
        var gpu = _gpu.TensorFlip(_gpu.TensorPermute(_gpu.TensorFlip(_gpu.TensorPermute(x, perm), new[] { 0 }), perm), new[] { 0 });
        AssertMatch(gpu, cpu, "InterleavedPermuteFlipChain(view)");
    }

    public static IEnumerable<object[]> Rot90Cases() => new List<object[]>
    {
        new object[] { new[] { 4, 6 }, 1, new[] { 0, 1 } },
        new object[] { new[] { 4, 6 }, 2, new[] { 0, 1 } },
        new object[] { new[] { 4, 6 }, 3, new[] { 0, 1 } },
        new object[] { new[] { 3, 4, 5 }, 1, new[] { 1, 2 } },
        new object[] { new[] { 3, 4, 5 }, 3, new[] { 0, 2 } },
        new object[] { new[] { 4, 4 }, -1, new[] { 0, 1 } },
    };

    [Theory]
    [MemberData(nameof(Rot90Cases))]
    public void TensorRot90_GpuMatchesCpu(int[] shape, int k, int[] axes)
    {
        if (!EnsureGpuReady()) return;
        var t = Rand(120, shape);
        var cpu = _cpu.TensorRot90(t, k, axes);
        var gpu = _gpu.TensorRot90(t, k, axes);
        AssertMatch(gpu, cpu, $"TensorRot90[{string.Join("x", shape)};k={k};axes={string.Join(",", axes)}]");
    }

    // cosine similarity along the last axis (the case the backend kernel covers).
    public static IEnumerable<object[]> CosSimShapes() => new List<object[]>
    {
        new object[] { new[] { 1, 8 } },
        new object[] { new[] { 4, 16 } },
        new object[] { new[] { 2, 3, 32 } },
        new object[] { new[] { 10, 64 } },
    };

    [Theory]
    [MemberData(nameof(CosSimShapes))]
    public void TensorCosineSimilarity_LastDim_GpuMatchesCpu(int[] shape)
    {
        if (!EnsureGpuReady()) return;
        var x1 = Rand(110, shape);
        var x2 = Rand(111, shape);
        var cpu = _cpu.TensorCosineSimilarity(x1, x2, -1);
        var gpu = _gpu.TensorCosineSimilarity(x1, x2, -1);
        AssertMatch(gpu, cpu, $"TensorCosineSimilarity[{string.Join("x", shape)}]");
    }
}
#endif
