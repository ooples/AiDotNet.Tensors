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

    private static void AssertMatch(Tensor<float> gpu, Tensor<float> cpu, string op)
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
        Assert.True(m < Tol, $"{op}: GPU vs CPU max_abs_err {m:E3} exceeded {Tol:E3}");
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
