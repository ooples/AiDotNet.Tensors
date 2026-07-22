#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public class DirectPtxSolver4x4Tests
{
    [Fact]
    public void Emitters_ArePointerOnlyExactShapeAndWorkspaceFree()
    {
        string cholesky = PtxRegisterCholesky4x4F32Kernel.EmitPtx(8, 6, 1024);
        Assert.Contains(PtxRegisterCholesky4x4F32Kernel.EntryPoint, cholesky);
        Assert.Equal(3, Count(cholesky, ".param .u64"));
        Assert.DoesNotContain(".param .u32", cholesky, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", cholesky, StringComparison.Ordinal);
        Assert.DoesNotContain(".shared", cholesky, StringComparison.Ordinal);
        Assert.DoesNotContain("%n", cholesky, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", cholesky, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("@%p0 bra.uni", cholesky, StringComparison.Ordinal);

        foreach (DirectPtxSolver4x4Operation operation in Enum.GetValues<DirectPtxSolver4x4Operation>())
        {
            string ptx = PtxRegisterSolver4x4F32Kernel.EmitPtx(8, 6, operation, 1024);
            int expectedPointers = operation switch
            {
                DirectPtxSolver4x4Operation.SolveBackwardVector => 5,
                DirectPtxSolver4x4Operation.SvdReduced or
                DirectPtxSolver4x4Operation.LuSolveVector or
                DirectPtxSolver4x4Operation.LdlSolveVectorLower or
                DirectPtxSolver4x4Operation.GeneralSolveVector => 4,
                _ => 3
            };
            Assert.Equal(expectedPointers, Count(ptx, ".param .u64"));
            Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
            Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
            Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
            Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("@%p0 bra.uni", ptx, StringComparison.Ordinal);
            Assert.DoesNotContain("@!%p0 bra.uni", ptx, StringComparison.Ordinal);
            Assert.Contains("mul.wide.u32", ptx, StringComparison.Ordinal);
            Assert.Contains("st.global", ptx, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void ShapesAndArchitectures_AreExactAndUnpromoted()
    {
        Assert.True(DirectPtxArchitecture.IsCholesky4x4ExperimentArchitecture(8, 6));
        Assert.True(DirectPtxArchitecture.IsSolver4x4ExperimentArchitecture(8, 6));
        Assert.False(DirectPtxArchitecture.IsSolver4x4ExperimentArchitecture(8, 0));
        Assert.False(DirectPtxArchitecture.IsSolver4x4ExperimentArchitecture(8, 9));
        Assert.False(DirectPtxArchitecture.IsSolver4x4ExperimentArchitecture(9, 0));
        foreach (int batch in new[] { 1024, 4096, 16384, 65536 })
        {
            Assert.True(PtxRegisterCholesky4x4F32Kernel.IsSupportedBatchCount(batch));
            Assert.False(PtxRegisterCholesky4x4F32Kernel.IsPromotedShape(batch));
            Assert.True(PtxRegisterSolver4x4F32Kernel.IsSupportedBatchCount(batch));
            foreach (DirectPtxSolver4x4Operation operation in Enum.GetValues<DirectPtxSolver4x4Operation>())
                Assert.False(PtxRegisterSolver4x4F32Kernel.IsPromotedShape(operation, batch));
        }
        Assert.False(PtxRegisterSolver4x4F32Kernel.IsSupportedBatchCount(1));
        Assert.Equal(new[] { 64, 128, 256 }, DirectPtxSolver4x4Autotuner.Candidates.ToArray());
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            DirectPtxSolver4x4Autotuner.ValidateBlockThreads(32));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxRegisterSolver4x4F32Kernel.EmitPtx(8, 6, DirectPtxSolver4x4Operation.LuFactor, 1));
    }

    [Fact]
    public void SolverGateOverride_IsThreadLocal()
    {
        bool? previous = DirectPtxFeatureGate.Solver4x4ExperimentOverride;
        try
        {
            DirectPtxFeatureGate.Solver4x4ExperimentOverride = true;
            Assert.True(DirectPtxFeatureGate.IsLuFactor4x4Enabled);
            bool? observed = null;
            var thread = new Thread(() => observed = DirectPtxFeatureGate.Solver4x4ExperimentOverride);
            thread.Start();
            thread.Join();
            Assert.Null(observed);
            DirectPtxFeatureGate.Solver4x4ExperimentOverride = false;
            Assert.False(DirectPtxFeatureGate.IsSvd4x4Enabled);
        }
        finally
        {
            DirectPtxFeatureGate.Solver4x4ExperimentOverride = previous;
        }
    }

    [Fact]
    public void CoverageManifest_AssignsEverySolverFamilyCellExactlyOnce()
    {
        string[] critical =
        [
            "Linalg.CholeskyEx", "Linalg.LuFactor", "Linalg.LuSolve",
            "Linalg.LdlFactor", "Linalg.LdlSolve", "Linalg.QR(reduced)",
            "Linalg.QR(complete/r)", "Linalg.Eigh(upper)", "Linalg.Eigh(lower)",
            "Linalg.Svd", "Linalg.SvdLowRank", "Linalg.Solve", "Linalg.SolveEx",
            "Linalg.SolveTriangular", "Linalg.Lstsq", "Linalg.Cholesky backward",
            "Linalg.Solve backward",
            "IExtendedLinalgBackend.LinalgLuSolve", "IExtendedLinalgBackend.LinalgSvdReduced"
        ];
        string[] actual = DirectPtxSolverCoverageManifest.All.Select(cell => cell.Api).ToArray();
        Assert.Equal(actual.Length, actual.Distinct(StringComparer.Ordinal).Count());
        Assert.All(critical, api => Assert.Contains(
            DirectPtxSolverCoverageManifest.Get(api).Status,
            new[]
            {
                DirectPtxSolverCoverageStatus.ExperimentalDirectPtx,
                DirectPtxSolverCoverageStatus.CoveredByExperimentalPrimitive
            }));
        Assert.All(DirectPtxSolverCoverageManifest.All, cell =>
        {
            Assert.False(string.IsNullOrWhiteSpace(cell.ExistingImplementation));
            Assert.False(string.IsNullOrWhiteSpace(cell.DirectPtxAssignment));
        });
        Assert.Throws<KeyNotFoundException>(() => DirectPtxSolverCoverageManifest.Get("unassigned"));
    }

    [SkippableTheory]
    [InlineData("cholesky")]
    [InlineData("lu-factor")]
    [InlineData("qr")]
    [InlineData("eigh")]
    [InlineData("eigh-lower")]
    [InlineData("svd")]
    [InlineData("lu-solve")]
    [InlineData("ldl-factor")]
    [InlineData("ldl-solve")]
    [InlineData("solve")]
    [InlineData("tri-lower")]
    [InlineData("tri-upper")]
    [InlineData("chol-backward")]
    [InlineData("solve-backward")]
    public void DriverOnlySolverFamily_MatchesStructuralOracleAndResourceGate(string operationName)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.IsSolver4x4ExperimentArchitecture(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in solver family is admitted only on SM86.");

        const int batch = 1024;
        float[] matrix = RepeatMatrix(batch,
        [
            9f, 1f, 2f, 0.5f,
            1f, 8f, 0.25f, 1f,
            2f, 0.25f, 7f, 0.75f,
            0.5f, 1f, 0.75f, 6f
        ]);
        using var input = runtime.AllocateBytes((nuint)(matrix.Length * sizeof(float)));
        input.Upload<float>(matrix);

        if (operationName == "cholesky")
        {
            using var kernel = new PtxRegisterCholesky4x4F32Kernel(runtime, batch);
            using var output = runtime.AllocateBytes((nuint)(matrix.Length * sizeof(float)));
            using var info = runtime.AllocateBytes((nuint)(batch * sizeof(int)));
            kernel.Launch(
                DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(info, kernel.Blueprint.Tensors[2]));
            runtime.Synchronize();
            float[] factor = new float[matrix.Length]; output.Download<float>(factor);
            int[] status = new int[batch]; info.Download<int>(status);
            Assert.All(status, value => Assert.Equal(0, value));
            Assert.True(MaxCholeskyResidual(matrix, factor, 8) <= 2e-4f);
            AssertResource(kernel.Audit);
            return;
        }

        DirectPtxSolver4x4Operation operation = operationName switch
        {
            "lu-factor" => DirectPtxSolver4x4Operation.LuFactor,
            "qr" => DirectPtxSolver4x4Operation.QrReduced,
            "eigh" => DirectPtxSolver4x4Operation.EighUpper,
            "eigh-lower" => DirectPtxSolver4x4Operation.EighLower,
            "svd" => DirectPtxSolver4x4Operation.SvdReduced,
            "lu-solve" => DirectPtxSolver4x4Operation.LuSolveVector,
            "ldl-factor" => DirectPtxSolver4x4Operation.LdlFactorLower,
            "ldl-solve" => DirectPtxSolver4x4Operation.LdlSolveVectorLower,
            "solve" => DirectPtxSolver4x4Operation.GeneralSolveVector,
            "tri-lower" => DirectPtxSolver4x4Operation.TriangularSolveVectorLower,
            "tri-upper" => DirectPtxSolver4x4Operation.TriangularSolveVectorUpper,
            "chol-backward" => DirectPtxSolver4x4Operation.CholeskyBackwardLower,
            _ => DirectPtxSolver4x4Operation.SolveBackwardVector
        };
        using var solver = new PtxRegisterSolver4x4F32Kernel(runtime, operation, batch);
        using var second = runtime.AllocateBytes(solver.Blueprint.Tensors[1].RequiredBytes);
        using var third = runtime.AllocateBytes(solver.Blueprint.Tensors[2].RequiredBytes);
        second.Upload<float>(new float[checked((int)(solver.Blueprint.Tensors[1].RequiredBytes / sizeof(float)))]);
        third.Upload<float>(new float[checked((int)(solver.Blueprint.Tensors[2].RequiredBytes / sizeof(float)))]);
        if (solver.Blueprint.Tensors.Count == 3)
        {
            solver.Launch3(
                DirectPtxTensorView.CreateOwned(input, solver.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(second, solver.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(third, solver.Blueprint.Tensors[2]));
            runtime.Synchronize();
        }
        else if (operation == DirectPtxSolver4x4Operation.LuSolveVector)
        {
            float[] identity = RepeatMatrix(batch,
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
            int[] pivots = Enumerable.Range(0, batch).SelectMany(_ => new[] { 0, 1, 2, 3 }).ToArray();
            float[] rhs = Enumerable.Range(0, batch).SelectMany(_ => new[] { 1f, 2f, 3f, 4f }).ToArray();
            input.Upload<float>(identity); second.Upload<int>(pivots); third.Upload<float>(rhs);
            using var fourth = runtime.AllocateBytes(solver.Blueprint.Tensors[3].RequiredBytes);
            solver.Launch4(
                DirectPtxTensorView.CreateOwned(input, solver.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(second, solver.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(third, solver.Blueprint.Tensors[2]),
                DirectPtxTensorView.CreateOwned(fourth, solver.Blueprint.Tensors[3]));
            runtime.Synchronize();
            float[] actual = new float[rhs.Length]; fourth.Download<float>(actual);
            Assert.Equal(rhs, actual);
        }
        else if (solver.Blueprint.Tensors.Count == 4)
        {
            using var fourth = runtime.AllocateBytes(solver.Blueprint.Tensors[3].RequiredBytes);
            solver.Launch4(
                DirectPtxTensorView.CreateOwned(input, solver.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(second, solver.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(third, solver.Blueprint.Tensors[2]),
                DirectPtxTensorView.CreateOwned(fourth, solver.Blueprint.Tensors[3]));
            runtime.Synchronize();
        }
        else
        {
            using var fourth = runtime.AllocateBytes(solver.Blueprint.Tensors[3].RequiredBytes);
            using var fifth = runtime.AllocateBytes(solver.Blueprint.Tensors[4].RequiredBytes);
            solver.Launch5(
                DirectPtxTensorView.CreateOwned(input, solver.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(second, solver.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(third, solver.Blueprint.Tensors[2]),
                DirectPtxTensorView.CreateOwned(fourth, solver.Blueprint.Tensors[3]),
                DirectPtxTensorView.CreateOwned(fifth, solver.Blueprint.Tensors[4]));
            runtime.Synchronize();
        }
        AssertResource(solver.Audit);
    }

    private static void AssertResource(DirectPtxKernelAudit audit)
    {
        Assert.Equal(0, audit.Function.LocalBytesPerThread);
        Assert.Equal(0, audit.Function.StaticSharedBytes);
        Assert.True(audit.ActiveBlocksPerMultiprocessor >= 2);
    }

    private static int Count(string text, string value) =>
        (text.Length - text.Replace(value, string.Empty, StringComparison.Ordinal).Length) / value.Length;

    private static float[] RepeatMatrix(int count, float[] source)
    {
        var result = new float[count * 16];
        for (int i = 0; i < count; i++) Array.Copy(source, 0, result, i * 16, 16);
        return result;
    }

    private static float MaxCholeskyResidual(float[] a, float[] l, int matrices)
    {
        float maximum = 0;
        for (int batch = 0; batch < matrices; batch++)
        for (int row = 0; row < 4; row++)
        for (int col = 0; col < 4; col++)
        {
            float sum = 0;
            for (int k = 0; k < 4; k++) sum += l[batch * 16 + row * 4 + k] * l[batch * 16 + col * 4 + k];
            maximum = Math.Max(maximum, Math.Abs(sum - a[batch * 16 + row * 4 + col]));
        }
        return maximum;
    }
}
#endif
