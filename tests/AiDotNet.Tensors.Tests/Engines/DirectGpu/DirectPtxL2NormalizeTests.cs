using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Tests.TestHelpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Focused coverage for the exact-shape FP32 fused row-L2-normalize direct-PTX
/// specialization (issue #843, reduction fusion). The emitter and shape-domain
/// assertions run without a GPU; the driver correctness assertion is skipped
/// unless a validated Ampere/GA10x device is present. The kernel folds a
/// sum-of-squares reduction and a broadcast rescale into a single register-
/// resident pass, so the per-row norm is never materialized. Disabled by default;
/// fails closed until three clean promotion runs clear the release gate.
/// </summary>
public class DirectPtxL2NormalizeTests
{
    [Fact]
    public void FusedL2NormalizeEmitter_IsRegisterResidentAndPointerOnly()
    {
        string ptx = PtxFusedRowL2NormalizeF32Kernel.EmitPtx(8, 6, 2048, 128);
        Assert.Contains(".maxntid 128, 1, 1", ptx);
        Assert.Contains("exact-shape rows=2048 columns=128 block=128", ptx);
        Assert.Contains("op=l2normalize", ptx);
        Assert.Equal(2, Count(ptx, "ld.param.u64"));
        Assert.Equal(1, Count(ptx, "ld.global.ca.v4.f32"));
        Assert.Equal(1, Count(ptx, "st.global.v4.f32"));
        Assert.Equal(4, Count(ptx, "fma.rn.f32"));
        Assert.Equal(5, Count(ptx, "shfl.sync.bfly.b32"));
        Assert.Equal(1, Count(ptx, "rsqrt.approx.f32"));
        // eps = 1e-12 baked as an IEEE-754 single-precision literal.
        string eps = "0f" + MathCompat.SingleToUInt32Bits(PtxFusedRowL2NormalizeF32Kernel.Epsilon)
            .ToString("X8", System.Globalization.CultureInfo.InvariantCulture);
        Assert.Contains(eps, ptx);
        Assert.Contains("ld.global.ca.v2.f32",
            PtxFusedRowL2NormalizeF32Kernel.EmitPtx(8, 6, 2048, 64));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("bar.sync", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void FusedL2NormalizeShapeDomain_IsClosedAndUnpromotedWithoutEvidence()
    {
        Assert.True(PtxFusedRowL2NormalizeF32Kernel.IsSupportedShape(256, 128));
        Assert.True(PtxFusedRowL2NormalizeF32Kernel.IsSupportedShape(2048, 64));
        Assert.True(PtxFusedRowL2NormalizeF32Kernel.IsSupportedShape(2048, 128));
        Assert.True(PtxFusedRowL2NormalizeF32Kernel.IsSupportedShape(8192, 128));
        Assert.False(PtxFusedRowL2NormalizeF32Kernel.IsSupportedShape(256, 64));
        Assert.False(PtxFusedRowL2NormalizeF32Kernel.IsSupportedShape(2048, 96));
        Assert.False(PtxFusedRowL2NormalizeF32Kernel.IsPromotedShape(2048, 128));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedRowL2NormalizeF32Kernel.EmitPtx(8, 6, 17, 128));
    }

    [SkippableTheory]
    [InlineData(256, 128)]
    [InlineData(2048, 64)]
    [InlineData(2048, 128)]
    [InlineData(8192, 128)]
    public void DriverOnlyFusedL2Normalize_MatchesReferenceAndHasZeroLocalBytes(int rows, int columns)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedRowReduction(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in L2-normalize specialization is measured on GA10x/SM86.");
        using var kernel = new PtxFusedRowL2NormalizeF32Kernel(runtime, rows, columns);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
        Assert.Equal(0, kernel.Audit.Function.StaticSharedBytes);
        Assert.True(kernel.Audit.ActiveBlocksPerMultiprocessor >= 3);
        Assert.Equal("row-l2-normalize-f32", kernel.Blueprint.Operation);
        Assert.Equal("none-no-materialized-norm-vector", kernel.Blueprint.Semantics["global-intermediates"]);

        int elements = rows * columns;
        var random = new Random(20260722);
        float[] input = Enumerable.Range(0, elements).Select(_ => (float)((random.NextDouble() * 2.0 - 1.0) * 4.0)).ToArray();
        var expected = new float[elements];
        for (int row = 0; row < rows; row++)
        {
            double sumSquares = 0;
            for (int col = 0; col < columns; col++)
            {
                double v = input[row * columns + col];
                sumSquares += v * v;
            }
            double invNorm = 1.0 / Math.Sqrt(sumSquares + 1e-12);
            for (int col = 0; col < columns; col++)
                expected[row * columns + col] = (float)(input[row * columns + col] * invNorm);
        }

        using var inputBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var outputBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        inputBuffer.Upload<float>(input);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(inputBuffer, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(outputBuffer, kernel.Blueprint.Tensors[1]));
        runtime.Synchronize();

        var actual = new float[elements];
        outputBuffer.Download<float>(actual);
        for (int i = 0; i < elements; i++)
        {
            float tolerance = 1e-4f * (MathF.Abs(expected[i]) + 1e-3f);
            Assert.True(MathF.Abs(actual[i] - expected[i]) <= tolerance,
                $"index {i}: actual {actual[i]:G9}, expected {expected[i]:G9}, tol {tolerance:G9}.");
        }
    }

    private static int Count(string text, string value)
    {
        int count = 0, offset = 0;
        while ((offset = text.IndexOf(value, offset, StringComparison.Ordinal)) >= 0)
        {
            count++;
            offset += value.Length;
        }
        return count;
    }
}
