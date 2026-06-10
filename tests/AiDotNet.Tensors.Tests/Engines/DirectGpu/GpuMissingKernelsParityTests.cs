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
