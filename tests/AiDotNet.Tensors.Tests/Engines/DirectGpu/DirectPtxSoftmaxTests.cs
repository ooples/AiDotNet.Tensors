#if NET5_0_OR_GREATER
using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Focused coverage for the exact-shape FP32 direct-PTX row-softmax family
/// (issue #840). The emitter and manifest assertions run without a GPU; the
/// driver correctness assertion is skipped unless a validated Ampere device is
/// present. Every specialization stays behind the disabled-by-default,
/// fails-closed release gate until it clears three clean promotion runs.
/// </summary>
public class DirectPtxSoftmaxTests
{
    [Fact]
    public void FusedSoftmaxEmitter_IsRegisterResidentAndPointerOnly()
    {
        string ptx = PtxFusedSoftmaxF32Kernel.EmitPtx(8, 6, 2048, 128);
        Assert.Contains(".maxntid 512, 1, 1", ptx);
        Assert.Contains("exact-shape rows=2048 columns=128 block=512", ptx);
        Assert.Equal(2, Count(ptx, "ld.param.u64"));
        Assert.Equal(1, Count(ptx, "ld.global.ca.v4.f32"));
        Assert.Equal(1, Count(ptx, "st.global.v4.f32"));
        Assert.Contains("ld.global.ca.v2.f32",
            PtxFusedSoftmaxF32Kernel.EmitPtx(8, 6, 2048, 64));
        Assert.Equal(10, Count(ptx, "shfl.sync.bfly.b32"));
        Assert.Equal(4, Count(ptx, "ex2.approx.ftz.f32"));
        Assert.Equal(1, Count(ptx, "rcp.approx.ftz.f32"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("bar.sync", ptx, StringComparison.Ordinal);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedSoftmaxF32Kernel.EmitPtx(8, 6, 17, 128));

        string cta = PtxFusedSoftmaxF32Kernel.EmitPtx(8, 6, 256, 128, 128);
        Assert.Contains(".shared .align 16 .b8 scratch[16]", cta);
        Assert.Contains("mov.u64 %rd8, scratch;", cta);
        Assert.DoesNotContain("cvta.to.shared", cta, StringComparison.Ordinal);
        Assert.Equal(4, Count(cta, "bar.sync 0"));
        Assert.Equal(1, Count(cta, "ld.global.ca.f32"));
        Assert.Equal(1, Count(cta, "st.global.f32"));
        Assert.DoesNotContain(".local", cta, StringComparison.Ordinal);
    }

    [Fact]
    public void FusedSoftmaxShapeDomain_IsClosedAndUnpromotedWithoutEvidence()
    {
        Assert.True(PtxFusedSoftmaxF32Kernel.IsSupportedShape(256, 128));
        Assert.True(PtxFusedSoftmaxF32Kernel.IsSupportedShape(2048, 64));
        Assert.True(PtxFusedSoftmaxF32Kernel.IsSupportedShape(2048, 128));
        Assert.True(PtxFusedSoftmaxF32Kernel.IsSupportedShape(8192, 128));
        Assert.False(PtxFusedSoftmaxF32Kernel.IsSupportedShape(256, 64));
        Assert.False(PtxFusedSoftmaxF32Kernel.IsSupportedShape(2048, 96));
        Assert.False(PtxFusedSoftmaxF32Kernel.IsSupportedShape(8193, 128));
        Assert.False(PtxFusedSoftmaxF32Kernel.IsPromotedShape(2048, 128));
        Assert.Contains(".maxntid 256, 1, 1",
            PtxFusedSoftmaxF32Kernel.EmitPtx(8, 6, 2048, 128, 256));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedSoftmaxF32Kernel.EmitPtx(8, 6, 2048, 128, 192));
    }

    [Fact]
    public void SoftmaxCoverageManifest_AssignsEveryScopedApiExactlyOnce()
    {
        Assert.Equal(62, DirectPtxSoftmaxCoverageManifest.All.Count);
        string[] names = DirectPtxSoftmaxCoverageManifest.All
            .Select(cell => cell.Api).OrderBy(name => name, StringComparer.Ordinal).ToArray();
        Assert.Equal(names.Length, names.Distinct(StringComparer.Ordinal).Count());
        Assert.All(DirectPtxSoftmaxCoverageManifest.All, cell =>
        {
            Assert.False(string.IsNullOrWhiteSpace(cell.ExistingImplementation));
            Assert.False(string.IsNullOrWhiteSpace(cell.Semantics));
            Assert.False(string.IsNullOrWhiteSpace(cell.PhysicalLayout));
            Assert.False(string.IsNullOrWhiteSpace(cell.DTypes));
            Assert.False(string.IsNullOrWhiteSpace(cell.DirectPtxAssignment));
        });
        Assert.Equal(
            DirectPtxSoftmaxCoverageStatus.ExperimentalDirectPtx,
            DirectPtxSoftmaxCoverageManifest.Get("CudaBackend.Softmax").Status);
        Assert.Equal(
            DirectPtxSoftmaxCoverageStatus.ExistingBackend,
            DirectPtxSoftmaxCoverageManifest.Get(
                "PtxOnlineFusedAttention128x64Kernel.OnlineSoftmax").Status);
        Assert.Throws<System.Collections.Generic.KeyNotFoundException>(() =>
            DirectPtxSoftmaxCoverageManifest.Get("UnassignedSoftmaxApi"));

        string[] scopedTokens =
            ["Softmax", "Sparsemax", "LogSumExp", "MaskedFill", "TriangularMask"];
        string[] reflectedOperations = typeof(IEngine).GetMethods()
            .Concat(typeof(IDirectGpuBackend).GetMethods())
            .Select(method => method.Name)
            .Where(name => scopedTokens.Any(token =>
                name.Contains(token, StringComparison.Ordinal)))
            .Distinct(StringComparer.Ordinal)
            .ToArray();
        Assert.NotEmpty(reflectedOperations);
        Assert.All(reflectedOperations, operation =>
            Assert.True(
                DirectPtxSoftmaxCoverageManifest.All.Any(cell =>
                    cell.Api.Contains(operation, StringComparison.Ordinal)),
                $"Public/backend softmax-scope operation '{operation}' is unassigned."));
    }

    [SkippableTheory]
    [InlineData(PtxFusedSoftmaxF32Kernel.DefaultBlockThreads, 0)]
    [InlineData(128, 16)]
    public void DriverOnlyFusedSoftmax_MatchesReferenceAndHasZeroLocalBytes(
        int blockThreads,
        int expectedStaticSharedBytes)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(runtime.ArchitectureFamily == DirectPtxArchitectureFamily.Ampere,
            "The checked-in softmax specialization is validated on Ampere.");
        const int rows = 256, columns = 128;
        using var kernel = new PtxFusedSoftmaxF32Kernel(
            runtime, rows, columns, blockThreads);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
        Assert.Equal(expectedStaticSharedBytes, kernel.Audit.Function.StaticSharedBytes);
        Assert.True(kernel.Audit.ActiveBlocksPerMultiprocessor >= 3);
        Assert.Equal(64, kernel.Audit.PtxSha256.Length);
        Assert.Equal("softmax-forward-f32", kernel.Blueprint.Operation);
        Assert.Equal(2, kernel.Blueprint.Tensors.Count);
        Assert.Equal("zero-entire-allocation-view",
            kernel.Blueprint.Semantics["byte-offset"]);
        Assert.Equal("input-output-disjoint",
            kernel.Blueprint.Semantics["alias-policy"]);
        Assert.Equal("none-logical-equals-physical",
            kernel.Blueprint.Semantics["padding"]);

        int elements = rows * columns;
        using var input = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var output = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        var random = new Random(20260721);
        float[] values = Enumerable.Range(0, elements)
            .Select(_ => (random.NextSingle() * 2f - 1f) * 4f).ToArray();
        for (int column = 0; column < columns; column++)
        {
            values[column] = 0f;
            values[columns + column] = column == 17 ? 80f : -80f;
            values[2 * columns + column] = (column & 1) == 0 ? 100f : -100f;
            values[3 * columns + column] = 1_000f + (column % 7 - 3);
            values[5 * columns + column] = float.NegativeInfinity;
        }
        values[4 * columns + 11] = float.PositiveInfinity;
        values[6 * columns + 23] = float.NaN;
        var expected = new float[elements];
        for (int row = 0; row < rows; row++)
        {
            int rowBase = row * columns;
            double maximum = double.NegativeInfinity;
            for (int column = 0; column < columns; column++)
                maximum = Math.Max(maximum, values[rowBase + column]);
            double sum = 0;
            for (int column = 0; column < columns; column++)
                sum += Math.Exp(values[rowBase + column] - maximum);
            for (int column = 0; column < columns; column++)
                expected[rowBase + column] =
                    (float)(Math.Exp(values[rowBase + column] - maximum) / sum);
        }
        input.Upload<float>(values);
        Assert.Throws<ArgumentException>(() => kernel.Launch(
            DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[1])));
        using (var suffixAllocation = runtime.AllocateBytes(
            kernel.Blueprint.Tensors[0].RequiredBytes + (nuint)16))
        {
            DirectPtxTensorView suffix = DirectPtxTensorView.CreateOwned(
                suffixAllocation, kernel.Blueprint.Tensors[0], 16);
            Assert.Throws<ArgumentException>(() => kernel.Launch(
                suffix,
                DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[1])));
        }
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[1]));
        runtime.Synchronize();
        var actual = new float[elements];
        output.Download<float>(actual);
        AssertVectorClose(actual, expected, 5e-5f, $"softmax-{blockThreads}");
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

    private static void AssertVectorClose(
        float[] actual,
        float[] expected,
        float tolerance,
        string name)
    {
        Assert.Equal(expected.Length, actual.Length);
        float maximum = 0;
        int maximumIndex = -1;
        for (int i = 0; i < actual.Length; i++)
        {
            if (float.IsNaN(expected[i]))
            {
                Assert.True(float.IsNaN(actual[i]),
                    $"{name} expected NaN at {i}, actual {actual[i]:G9}.");
                continue;
            }
            Assert.True(float.IsFinite(actual[i]) == float.IsFinite(expected[i]),
                $"{name} finite/nonfinite mismatch at {i}: actual {actual[i]:G9}, " +
                $"expected {expected[i]:G9}.");
            float error = MathF.Abs(actual[i] - expected[i]);
            if (error > maximum) { maximum = error; maximumIndex = i; }
        }
        Assert.True(maximum <= tolerance,
            $"{name} max absolute error {maximum:G9} at {maximumIndex}; tolerance {tolerance:G9}.");
    }
}
#endif
