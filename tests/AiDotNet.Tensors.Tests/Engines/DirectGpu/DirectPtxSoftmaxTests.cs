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
        Skip.IfNot(DirectPtxArchitecture.HasValidatedRowSoftmax(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in softmax specialization is measured on GA10x/SM86.");
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

    [Theory]
    [InlineData(8, 6, true)]
    [InlineData(8, 0, false)]
    [InlineData(8, 7, false)]
    [InlineData(8, 9, false)]
    [InlineData(9, 0, false)]
    [InlineData(10, 0, false)]
    public void SoftmaxArchitectureMatrix_FailsClosedOutsideSm86(
        int major,
        int minor,
        bool expected)
    {
        Assert.Equal(expected,
            DirectPtxArchitecture.HasValidatedRowSoftmax(major, minor));
    }

    [SkippableFact]
    public void BackendSoftmax_PromotedGeometryDispatchesZeroAllocAndMatchesOracle()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previousEnabled = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.SoftmaxExperimentOverride;
        DirectPtxFeatureGate.TestOverride = true;
        DirectPtxFeatureGate.SoftmaxExperimentOverride = true;
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxSoftmaxEnabled, "Requires a GA10x/SM86 CUDA backend.");
            const int rows = 2048, columns = 128;
            float[] valuesHost = MakeRowValues(rows, columns, seed: 3);
            float[] expected = CpuRowSoftmax(valuesHost, rows, columns);
            using var input = backend.AllocateBuffer(valuesHost);
            using var output = backend.AllocateBuffer(rows * columns);

            Assert.True(backend.PrewarmDirectPtxSoftmax(rows, columns), backend.DirectPtxLastError);
            long before = backend.DirectPtxSoftmaxDispatchCount;
            Assert.True(backend.TryDirectPtxSoftmax(input, output, rows, columns),
                backend.DirectPtxLastError);
            backend.Synchronize();
            Assert.Equal(before + 1, backend.DirectPtxSoftmaxDispatchCount);
            AssertVectorClose(backend.DownloadBuffer(output), expected, 5e-5f, "direct-ptx softmax");

            // The resident dispatch path must not allocate managed memory.
            long allocBefore = GC.GetAllocatedBytesForCurrentThread();
            bool all = true;
            for (int i = 0; i < 32; i++)
                all &= backend.TryDirectPtxSoftmax(input, output, rows, columns);
            long allocated = GC.GetAllocatedBytesForCurrentThread() - allocBefore;
            backend.Synchronize();
            Assert.True(all, backend.DirectPtxLastError);
            Assert.Equal(0, allocated);

            // The public route selects the direct-PTX kernel and increments the counter.
            long publicBefore = backend.DirectPtxSoftmaxDispatchCount;
            backend.Softmax(input, output, rows, columns);
            backend.Synchronize();
            Assert.Equal(publicBefore + 1, backend.DirectPtxSoftmaxDispatchCount);
            AssertVectorClose(backend.DownloadBuffer(output), expected, 5e-5f, "public softmax");

            Assert.True(backend.TryGetDirectPtxSoftmaxAudit(
                rows, columns, out DirectPtxKernelAudit audit));
            Assert.Equal(0, audit.Function.LocalBytesPerThread);
            Assert.Equal(64, audit.PtxSha256.Length);
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previousEnabled;
            DirectPtxFeatureGate.SoftmaxExperimentOverride = previousExperiment;
        }
    }

    [SkippableFact]
    public void BackendSoftmax_UnsupportedShapeAndDisabledGateFallBackToExistingKernel()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previousEnabled = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.SoftmaxExperimentOverride;
        DirectPtxFeatureGate.TestOverride = true;
        DirectPtxFeatureGate.SoftmaxExperimentOverride = true;
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxSoftmaxEnabled, "Requires a GA10x/SM86 CUDA backend.");

            // An unsupported shape declines the direct-PTX route with an exact
            // reason, leaves the dispatch counter untouched, and the public
            // Softmax still returns a correct result through the existing kernel.
            const int rows = 96, columns = 96; // 96 % 32 == 0 but not a promoted bucket
            float[] valuesHost = MakeRowValues(rows, columns, seed: 7);
            float[] expected = CpuRowSoftmax(valuesHost, rows, columns);
            using var input = backend.AllocateBuffer(valuesHost);
            using var output = backend.AllocateBuffer(rows * columns);
            long before = backend.DirectPtxSoftmaxDispatchCount;
            Assert.False(backend.TryDirectPtxSoftmax(input, output, rows, columns));
            Assert.Equal("softmax-shape-not-implemented", backend.DirectPtxLastError);
            backend.Softmax(input, output, rows, columns);
            backend.Synchronize();
            Assert.Equal(before, backend.DirectPtxSoftmaxDispatchCount);
            AssertVectorClose(backend.DownloadBuffer(output), expected, 5e-5f,
                "unsupported-shape fallback");

            // A promoted shape with the feature disabled also falls back without
            // touching the direct-PTX dispatch counter.
            const int sr = 256, sc = 128;
            float[] sValues = MakeRowValues(sr, sc, seed: 11);
            float[] sExpected = CpuRowSoftmax(sValues, sr, sc);
            using var sInput = backend.AllocateBuffer(sValues);
            using var sOutput = backend.AllocateBuffer(sr * sc);
            DirectPtxFeatureGate.TestOverride = false;
            long disabledBefore = backend.DirectPtxSoftmaxDispatchCount;
            Assert.False(backend.TryDirectPtxSoftmax(sInput, sOutput, sr, sc));
            Assert.Equal("softmax-feature-disabled", backend.DirectPtxLastError);
            backend.Softmax(sInput, sOutput, sr, sc);
            backend.Synchronize();
            Assert.Equal(disabledBefore, backend.DirectPtxSoftmaxDispatchCount);
            AssertVectorClose(backend.DownloadBuffer(sOutput), sExpected, 5e-5f,
                "disabled-gate fallback");
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previousEnabled;
            DirectPtxFeatureGate.SoftmaxExperimentOverride = previousExperiment;
        }
    }

    [SkippableFact]
    public void BackendSoftmax_ThreeWay_CudaAndPtxBothMatchCpuOracle()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previousEnabled = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.SoftmaxExperimentOverride;
        DirectPtxFeatureGate.TestOverride = true;
        DirectPtxFeatureGate.SoftmaxExperimentOverride = true;
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxSoftmaxEnabled, "Requires a GA10x/SM86 CUDA backend.");
            const int rows = 2048, columns = 128;
            float[] valuesHost = MakeRowValues(rows, columns, seed: 21);
            float[] oracle = CpuRowSoftmax(valuesHost, rows, columns); // CPU fp64-accumulated reference
            using var input = backend.AllocateBuffer(valuesHost);
            using var cudaOut = backend.AllocateBuffer(rows * columns);
            using var ptxOut = backend.AllocateBuffer(rows * columns);

            // Leg 1 - the existing CUDA kernel (direct-PTX disabled). It must match
            // the CPU oracle and must NOT touch the direct-PTX dispatch counter.
            DirectPtxFeatureGate.TestOverride = false;
            long ptxBefore = backend.DirectPtxSoftmaxDispatchCount;
            backend.Softmax(input, cudaOut, rows, columns);
            backend.Synchronize();
            Assert.Equal(ptxBefore, backend.DirectPtxSoftmaxDispatchCount);
            float[] cuda = backend.DownloadBuffer(cudaOut);
            AssertVectorClose(cuda, oracle, 5e-5f, "CUDA vs CPU oracle");

            // Leg 2 - the direct-PTX kernel (gate on). It must match the CPU oracle
            // and the dispatch counter must advance (proving the PTX path fired).
            DirectPtxFeatureGate.TestOverride = true;
            Assert.True(backend.TryDirectPtxSoftmax(input, ptxOut, rows, columns),
                backend.DirectPtxLastError);
            backend.Synchronize();
            Assert.Equal(ptxBefore + 1, backend.DirectPtxSoftmaxDispatchCount);
            float[] ptx = backend.DownloadBuffer(ptxOut);
            AssertVectorClose(ptx, oracle, 5e-5f, "direct-PTX vs CPU oracle");

            // Leg 3 - the two GPU paths agree with each other (sum of both legs' tol).
            AssertVectorClose(ptx, cuda, 1e-4f, "direct-PTX vs CUDA");
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previousEnabled;
            DirectPtxFeatureGate.SoftmaxExperimentOverride = previousExperiment;
        }
    }

    private static float[] MakeRowValues(int rows, int columns, int seed)
    {
        var random = new Random(seed);
        var values = new float[rows * columns];
        for (int i = 0; i < values.Length; i++)
            values[i] = (random.NextSingle() * 2f - 1f) * 4f;
        return values;
    }

    private static float[] CpuRowSoftmax(float[] values, int rows, int columns)
    {
        var expected = new float[values.Length];
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
        return expected;
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
