#if NET5_0_OR_GREATER
using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Focused coverage for the exact-shape FP32 direct-PTX row-sum reduction family
/// (issue #843). The emitter and shape-domain assertions run without a GPU; the
/// driver correctness assertion is skipped unless a validated Ampere device is
/// present. The specialization stays disabled by default and fails closed until
/// three clean promotion runs clear the release gate.
/// </summary>
public class DirectPtxReductionTests
{
    [Fact]
    public void FusedRowSumEmitter_IsRegisterResidentAndPointerOnly()
    {
        string ptx = PtxFusedRowReduceF32Kernel.EmitPtx(8, 6, 2048, 128);
        Assert.Contains(".maxntid 128, 1, 1", ptx);
        Assert.Contains("exact-shape rows=2048 columns=128 block=128", ptx);
        Assert.Contains("op=sum", ptx);
        Assert.Equal(2, Count(ptx, "ld.param.u64"));
        Assert.Equal(1, Count(ptx, "ld.global.ca.v4.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.Equal(0, Count(ptx, "st.global.v4.f32"));
        Assert.Equal(5, Count(ptx, "shfl.sync.bfly.b32"));
        Assert.Contains("ld.global.ca.v2.f32",
            PtxFusedRowReduceF32Kernel.EmitPtx(8, 6, 2048, 64));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("bar.sync", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("ex2.approx", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void FusedRowSumShapeDomain_IsClosedAndUnpromotedWithoutEvidence()
    {
        Assert.True(PtxFusedRowReduceF32Kernel.IsSupportedShape(256, 128));
        Assert.True(PtxFusedRowReduceF32Kernel.IsSupportedShape(2048, 64));
        Assert.True(PtxFusedRowReduceF32Kernel.IsSupportedShape(2048, 128));
        Assert.True(PtxFusedRowReduceF32Kernel.IsSupportedShape(8192, 128));
        Assert.False(PtxFusedRowReduceF32Kernel.IsSupportedShape(256, 64));
        Assert.False(PtxFusedRowReduceF32Kernel.IsSupportedShape(2048, 96));
        Assert.False(PtxFusedRowReduceF32Kernel.IsSupportedShape(8193, 128));
        Assert.False(PtxFusedRowReduceF32Kernel.IsPromotedShape(2048, 128));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedRowReduceF32Kernel.EmitPtx(8, 6, 17, 128));
    }

    [SkippableTheory]
    [InlineData(256, 128)]
    [InlineData(2048, 64)]
    [InlineData(2048, 128)]
    [InlineData(8192, 128)]
    public void DriverOnlyFusedRowSum_MatchesReferenceAndHasZeroLocalBytes(
        int rows,
        int columns)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedRowReduction(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in row-sum specialization is measured on GA10x/SM86.");
        using var kernel = new PtxFusedRowReduceF32Kernel(runtime, rows, columns);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
        Assert.Equal(0, kernel.Audit.Function.StaticSharedBytes);
        Assert.True(kernel.Audit.ActiveBlocksPerMultiprocessor >= 3);
        Assert.Equal("row-sum-f32", kernel.Blueprint.Operation);
        Assert.Equal(2, kernel.Blueprint.Tensors.Count);
        Assert.Equal("none", kernel.Blueprint.Semantics["global-intermediates"]);

        int elements = rows * columns;
        using var input = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var output = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        var random = RandomHelper.CreateSeededRandom(20260722);
        float[] values = Enumerable.Range(0, elements)
            .Select(_ => (random.NextSingle() * 2f - 1f) * 4f).ToArray();
        Array.Clear(values, 0, columns);
        for (int column = 0; column < columns; column++)
            values[columns + column] = (column & 1) == 0 ? 4f : -4f;
        values[2 * columns + 7] = float.PositiveInfinity;
        values[3 * columns + 11] = float.NaN;
        values[4 * columns + 13] = float.PositiveInfinity;
        values[4 * columns + 17] = float.NegativeInfinity;
        var expected = new float[rows];
        for (int row = 0; row < rows; row++)
        {
            double sum = 0;
            for (int column = 0; column < columns; column++)
                sum += values[row * columns + column];
            expected[row] = (float)sum;
        }
        input.Upload<float>(values);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[1]));
        runtime.Synchronize();
        var actual = new float[rows];
        output.Download<float>(actual);
        for (int row = 0; row < rows; row++)
        {
            if (float.IsNaN(expected[row]))
            {
                Assert.True(float.IsNaN(actual[row]),
                    $"row {row}: expected NaN, actual {actual[row]:G9}.");
                continue;
            }
            if (float.IsInfinity(expected[row]))
            {
                Assert.Equal(expected[row], actual[row]);
                continue;
            }
            float tolerance = 1e-4f * (MathF.Abs(expected[row]) + 1f);
            Assert.True(MathF.Abs(actual[row] - expected[row]) <= tolerance,
                $"row {row}: actual {actual[row]:G9}, expected {expected[row]:G9}, tol {tolerance:G9}.");
        }

        using var suffixAllocation = runtime.AllocateBytes(
            kernel.Blueprint.Tensors[0].RequiredBytes + (nuint)16);
        DirectPtxTensorView suffix = DirectPtxTensorView.CreateOwned(
            suffixAllocation, kernel.Blueprint.Tensors[0], 16);
        Assert.Throws<ArgumentException>(() => kernel.Launch(
            suffix,
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[1])));
    }

    [Fact]
    public void ReductionCoverageManifest_AssignsEveryCellExactlyOnce()
    {
        Assert.NotEmpty(DirectPtxReductionCoverageManifest.All);
        string[] names = DirectPtxReductionCoverageManifest.All
            .Select(cell => cell.Api).OrderBy(name => name, StringComparer.Ordinal).ToArray();
        Assert.Equal(names.Length, names.Distinct(StringComparer.Ordinal).Count());
        Assert.All(DirectPtxReductionCoverageManifest.All, cell =>
        {
            Assert.False(string.IsNullOrWhiteSpace(cell.ExistingImplementation));
            Assert.False(string.IsNullOrWhiteSpace(cell.Semantics));
            Assert.False(string.IsNullOrWhiteSpace(cell.PhysicalLayout));
            Assert.False(string.IsNullOrWhiteSpace(cell.DTypes));
            Assert.False(string.IsNullOrWhiteSpace(cell.DirectPtxAssignment));
        });
        Assert.Equal(
            DirectPtxReductionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxReductionCoverageManifest.Get("CudaBackend.SumAxis").Status);
        Assert.Equal(
            DirectPtxReductionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxReductionCoverageManifest.Get("CudaBackend.NormalizeL2").Status);
        Assert.Equal(2, DirectPtxReductionCoverageManifest.All
            .Count(cell => cell.Status == DirectPtxReductionCoverageStatus.ExperimentalDirectPtx));
        Assert.All(DirectPtxReductionCoverageManifest.All,
            cell => Assert.NotEqual(
                DirectPtxReductionCoverageStatus.PromotedDirectPtx, cell.Status));
        Assert.Throws<System.Collections.Generic.KeyNotFoundException>(() =>
            DirectPtxReductionCoverageManifest.Get("UnassignedReductionApi"));
    }

    [Theory]
    [InlineData(8, 6, true)]
    [InlineData(8, 0, false)]
    [InlineData(8, 7, false)]
    [InlineData(8, 9, false)]
    [InlineData(9, 0, false)]
    [InlineData(10, 0, false)]
    public void ReductionArchitectureMatrix_FailsClosedOutsideSm86(
        int major,
        int minor,
        bool expected)
    {
        Assert.Equal(expected,
            DirectPtxArchitecture.HasValidatedRowReduction(major, minor));
    }

    [SkippableFact]
    public void BackendRowSum_PromotedGeometryDispatchesZeroAllocAndMatchesOracle()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previousEnabled = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.ReductionExperimentOverride;
        DirectPtxFeatureGate.TestOverride = true;
        DirectPtxFeatureGate.ReductionExperimentOverride = true;
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxReductionEnabled, "Requires a GA10x/SM86 CUDA backend.");
            const int rows = 2048, columns = 128;
            float[] valuesHost = MakeRowValues(rows, columns, seed: 5);
            float[] expected = CpuRowSum(valuesHost, rows, columns);
            using var input = backend.AllocateBuffer(valuesHost);
            using var output = backend.AllocateBuffer(rows);

            Assert.True(backend.PrewarmDirectPtxRowSum(rows, columns), backend.DirectPtxLastError);
            long before = backend.DirectPtxRowReduceDispatchCount;
            Assert.True(backend.TryDirectPtxRowSum(input, output, rows, columns),
                backend.DirectPtxLastError);
            backend.Synchronize();
            Assert.Equal(before + 1, backend.DirectPtxRowReduceDispatchCount);
            AssertRowSumsClose(backend.DownloadBuffer(output), expected, "direct-ptx row-sum");

            // The resident dispatch path must not allocate managed memory.
            long allocBefore = GC.GetAllocatedBytesForCurrentThread();
            bool all = true;
            for (int i = 0; i < 32; i++)
                all &= backend.TryDirectPtxRowSum(input, output, rows, columns);
            long allocated = GC.GetAllocatedBytesForCurrentThread() - allocBefore;
            backend.Synchronize();
            Assert.True(all, backend.DirectPtxLastError);
            Assert.Equal(0, allocated);

            // The public route selects the direct-PTX kernel and increments the counter.
            long publicBefore = backend.DirectPtxRowReduceDispatchCount;
            backend.SumAxis(input, output, rows, columns);
            backend.Synchronize();
            Assert.Equal(publicBefore + 1, backend.DirectPtxRowReduceDispatchCount);
            AssertRowSumsClose(backend.DownloadBuffer(output), expected, "public sum-axis");

            Assert.True(backend.TryGetDirectPtxRowReduceAudit(
                rows, columns, out DirectPtxKernelAudit audit));
            Assert.Equal(0, audit.Function.LocalBytesPerThread);
            Assert.Equal(0, audit.Function.StaticSharedBytes);
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previousEnabled;
            DirectPtxFeatureGate.ReductionExperimentOverride = previousExperiment;
        }
    }

    [SkippableFact]
    public void BackendRowSum_UnsupportedShapeAndDisabledGateFallBackToExistingKernel()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previousEnabled = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.ReductionExperimentOverride;
        DirectPtxFeatureGate.TestOverride = true;
        DirectPtxFeatureGate.ReductionExperimentOverride = true;
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxReductionEnabled, "Requires a GA10x/SM86 CUDA backend.");

            // An unsupported shape declines the direct-PTX route with an exact
            // reason, leaves the dispatch counter untouched, and the public
            // SumAxis still returns a correct result through the existing kernel.
            const int rows = 100, columns = 96; // 96 % 32 == 0 but not a promoted bucket
            float[] valuesHost = MakeRowValues(rows, columns, seed: 9);
            float[] expected = CpuRowSum(valuesHost, rows, columns);
            using var input = backend.AllocateBuffer(valuesHost);
            using var output = backend.AllocateBuffer(rows);
            long before = backend.DirectPtxRowReduceDispatchCount;
            Assert.False(backend.TryDirectPtxRowSum(input, output, rows, columns));
            Assert.Equal("reduction-shape-not-implemented", backend.DirectPtxLastError);
            backend.SumAxis(input, output, rows, columns);
            backend.Synchronize();
            Assert.Equal(before, backend.DirectPtxRowReduceDispatchCount);
            AssertRowSumsClose(backend.DownloadBuffer(output), expected, "unsupported-shape fallback");

            // A promoted shape with the feature disabled also falls back without
            // touching the direct-PTX dispatch counter.
            const int sr = 256, sc = 128;
            float[] sValues = MakeRowValues(sr, sc, seed: 13);
            float[] sExpected = CpuRowSum(sValues, sr, sc);
            using var sInput = backend.AllocateBuffer(sValues);
            using var sOutput = backend.AllocateBuffer(sr);
            DirectPtxFeatureGate.TestOverride = false;
            long disabledBefore = backend.DirectPtxRowReduceDispatchCount;
            Assert.False(backend.TryDirectPtxRowSum(sInput, sOutput, sr, sc));
            Assert.Equal("reduction-feature-disabled", backend.DirectPtxLastError);
            backend.SumAxis(sInput, sOutput, sr, sc);
            backend.Synchronize();
            Assert.Equal(disabledBefore, backend.DirectPtxRowReduceDispatchCount);
            AssertRowSumsClose(backend.DownloadBuffer(sOutput), sExpected, "disabled-gate fallback");
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previousEnabled;
            DirectPtxFeatureGate.ReductionExperimentOverride = previousExperiment;
        }
    }

    [SkippableFact]
    public void BackendRowSum_ThreeWay_CudaAndPtxBothMatchCpuOracle()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previousEnabled = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.ReductionExperimentOverride;
        DirectPtxFeatureGate.TestOverride = true;
        DirectPtxFeatureGate.ReductionExperimentOverride = true;
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxReductionEnabled, "Requires a GA10x/SM86 CUDA backend.");
            const int rows = 2048, columns = 128;
            float[] valuesHost = MakeRowValues(rows, columns, seed: 23);
            float[] oracle = CpuRowSum(valuesHost, rows, columns); // CPU fp64-accumulated reference
            using var input = backend.AllocateBuffer(valuesHost);
            using var cudaOut = backend.AllocateBuffer(rows);
            using var ptxOut = backend.AllocateBuffer(rows);

            // Leg 1 - the existing CUDA kernel (direct-PTX disabled). It must match
            // the CPU oracle and must NOT touch the direct-PTX dispatch counter.
            DirectPtxFeatureGate.TestOverride = false;
            long ptxBefore = backend.DirectPtxRowReduceDispatchCount;
            backend.SumAxis(input, cudaOut, rows, columns);
            backend.Synchronize();
            Assert.Equal(ptxBefore, backend.DirectPtxRowReduceDispatchCount);
            float[] cuda = backend.DownloadBuffer(cudaOut);
            AssertRowSumsClose(cuda, oracle, "CUDA vs CPU oracle");

            // Leg 2 - the direct-PTX kernel (gate on). It must match the CPU oracle
            // and the dispatch counter must advance (proving the PTX path fired).
            DirectPtxFeatureGate.TestOverride = true;
            Assert.True(backend.TryDirectPtxRowSum(input, ptxOut, rows, columns),
                backend.DirectPtxLastError);
            backend.Synchronize();
            Assert.Equal(ptxBefore + 1, backend.DirectPtxRowReduceDispatchCount);
            float[] ptx = backend.DownloadBuffer(ptxOut);
            AssertRowSumsClose(ptx, oracle, "direct-PTX vs CPU oracle");

            // Leg 3 - the two GPU paths agree with each other.
            AssertRowSumsClose(ptx, cuda, "direct-PTX vs CUDA");
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previousEnabled;
            DirectPtxFeatureGate.ReductionExperimentOverride = previousExperiment;
        }
    }

    private static float[] MakeRowValues(int rows, int columns, int seed)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var values = new float[rows * columns];
        for (int i = 0; i < values.Length; i++)
            values[i] = (random.NextSingle() * 2f - 1f) * 4f;
        return values;
    }

    private static float[] CpuRowSum(float[] values, int rows, int columns)
    {
        var expected = new float[rows];
        for (int row = 0; row < rows; row++)
        {
            double sum = 0;
            for (int column = 0; column < columns; column++)
                sum += values[row * columns + column];
            expected[row] = (float)sum;
        }
        return expected;
    }

    private static void AssertRowSumsClose(float[] actual, float[] expected, string name)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int row = 0; row < expected.Length; row++)
        {
            // Row-sum output is O(columns*magnitude); enforce relative closeness.
            float tolerance = 2e-4f * (MathF.Abs(expected[row]) + 1f);
            Assert.True(MathF.Abs(actual[row] - expected[row]) <= tolerance,
                $"{name} row {row}: actual {actual[row]:G9}, expected {expected[row]:G9}, tol {tolerance:G9}.");
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
#endif
