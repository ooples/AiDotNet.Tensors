#if !NETFRAMEWORK
using System;
using System.Linq;
using System.Threading;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class DirectPtxSparseGraphTests
{
    [Fact]
    public void CompletionLedger_IsUniqueExplicitAndComplete()
    {
        Assert.True(DirectPtxSparseGraphCompletionLedger.All.Count >= 100);
        Assert.Equal(DirectPtxSparseGraphCompletionLedger.All.Count,
            DirectPtxSparseGraphCompletionLedger.All
                .Select(entry => entry.Operation).Distinct(StringComparer.Ordinal).Count());
        Assert.Equal(113, DirectPtxSparseGraphCompletionLedger.All.Count(entry =>
            entry.Status == DirectPtxSparseGraphCompletionStatus.ImplementedDirectPtx));
        Assert.True(DirectPtxSparseGraphCompletionLedger.IsComplete);
        DirectPtxSparseGraphCompletionLedger.RequireComplete();
    }

    [Fact]
    public void SparseEngineFamily_EmitsAllTwentyTwoDistinctExactModules()
    {
        DirectPtxSparseEngineOperation[] operations = Enum.GetValues<DirectPtxSparseEngineOperation>();
        Assert.Equal(22, operations.Length);
        Assert.Equal(22, operations.Select(PtxSparseEngineF32Kernel.GetEntryPoint)
            .Distinct(StringComparer.Ordinal).Count());

        foreach (DirectPtxSparseEngineOperation operation in operations)
        {
            string ptx = PtxSparseEngineF32Kernel.EmitPtx(8, 6, operation);
            DirectPtxKernelBlueprint blueprint = PtxSparseEngineF32Kernel.CreateBlueprint(
                DirectPtxArchitectureFamily.Ampere, operation);

            Assert.Contains(PtxSparseEngineF32Kernel.GetEntryPoint(operation), ptx);
            Assert.Equal(blueprint.Tensors.Count, Count(ptx, ".param .u64"));
            Assert.Contains("st.global", ptx, StringComparison.Ordinal);
            Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
            Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
            Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
            Assert.All(blueprint.Tensors, tensor =>
            {
                Assert.Equal(DirectPtxExtentMode.Exact, tensor.ExtentMode);
                Assert.Equal(16, tensor.AlignmentBytes);
            });
            Assert.Equal("device-pointers-only", blueprint.Semantics["parameter-abi"]);
            Assert.Equal("forbidden", blueprint.Semantics["dynamic-strides"]);
            Assert.Equal("host-pre-dispatch", blueprint.Semantics["coordinate-validation"]);
            Assert.Equal("experimental-hardware-evidence-pending", blueprint.Semantics["promotion"]);
            Assert.Equal("0", blueprint.Semantics["workspace-bytes"]);
            if (operation == DirectPtxSparseEngineOperation.SparseSampledAddMM)
                Assert.Equal("coo-unique", blueprint.Semantics["pattern"]);
            AssertPtxControlFlowClosed(ptx);
        }
    }

    [Fact]
    public void SparseEngineFamily_EmitsOperationSpecificMathAndReductionSemantics()
    {
        string spmv = PtxSparseEngineF32Kernel.EmitPtx(8, 6, DirectPtxSparseEngineOperation.SpMV);
        string spmm = PtxSparseEngineF32Kernel.EmitPtx(8, 6, DirectPtxSparseEngineOperation.SpMM);
        string coalesce = PtxSparseEngineF32Kernel.EmitPtx(8, 6, DirectPtxSparseEngineOperation.Coalesce);
        string softmax = PtxSparseEngineF32Kernel.EmitPtx(8, 6, DirectPtxSparseEngineOperation.SparseSoftmax);
        string logSoftmax = PtxSparseEngineF32Kernel.EmitPtx(8, 6, DirectPtxSparseEngineOperation.SparseLogSoftmax);

        Assert.Contains("SPMV_LOOP", spmv);
        Assert.Contains("fma.rn.f32", spmm);
        Assert.Contains("COALESCE_PRIOR", coalesce);
        Assert.Contains("COALESCE_SUM", coalesce);
        Assert.Contains("ex2.approx.f32", softmax);
        Assert.Contains("lg2.approx.f32", logSoftmax);
    }

    [Fact]
    public void SparseEngineFamily_HasARealHostFacingISparseEngineImplementation()
    {
        Assert.Contains(typeof(ISparseEngine), typeof(CudaPtxSparseEngine).GetInterfaces());
        string[] operations =
        [
            nameof(ISparseEngine.SpMV), nameof(ISparseEngine.SpMVTranspose), nameof(ISparseEngine.SpMM),
            nameof(ISparseEngine.SpSpMM), nameof(ISparseEngine.AddSparseDense), nameof(ISparseEngine.MultiplySparseDense),
            nameof(ISparseEngine.SparseGather), nameof(ISparseEngine.SparseScatter), nameof(ISparseEngine.SparseScatterAdd),
            nameof(ISparseEngine.SparseToDense), nameof(ISparseEngine.DenseToSparse), nameof(ISparseEngine.Coalesce),
            nameof(ISparseEngine.SparseTranspose), nameof(ISparseEngine.SparseMatMul),
            nameof(ISparseEngine.SparseMatMulPatternPreserving), nameof(ISparseEngine.SparseAddMM),
            nameof(ISparseEngine.SparseSampledAddMM), nameof(ISparseEngine.SparseSpGeMM),
            nameof(ISparseEngine.SparseSum), nameof(ISparseEngine.SparseMean), nameof(ISparseEngine.SparseSoftmax),
            nameof(ISparseEngine.SparseLogSoftmax)
        ];
        Assert.All(operations, operation => Assert.NotNull(typeof(CudaPtxSparseEngine).GetMethod(operation)));
    }

    [Fact]
    public void MeshPoolFamily_EmitsAllSixteenDistinctPointerOnlyModules()
    {
        DirectPtxMeshPoolOperation[] operations = Enum.GetValues<DirectPtxMeshPoolOperation>();
        Assert.Equal(16, operations.Length);
        Assert.Equal(16, operations.Select(PtxMeshPoolF32Kernel.GetEntryPoint)
            .Distinct(StringComparer.Ordinal).Count());
        Assert.Equal(operations.Select(PtxMeshPoolF32Kernel.GetKernelName).OrderBy(name => name),
            CudaMeshPoolKernels.GetKernelNames().OrderBy(name => name));

        foreach (DirectPtxMeshPoolOperation operation in operations)
        {
            string ptx = PtxMeshPoolF32Kernel.EmitPtx(8, 6, operation);
            DirectPtxKernelBlueprint blueprint = PtxMeshPoolF32Kernel.CreateBlueprint(
                DirectPtxArchitectureFamily.Ampere, operation);

            Assert.Contains(PtxMeshPoolF32Kernel.GetEntryPoint(operation), ptx);
            Assert.Equal(PtxMeshPoolF32Kernel.GetTensorCount(operation), blueprint.Tensors.Count);
            Assert.Equal(blueprint.Tensors.Count, Count(ptx, ".param .u64"));
            Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
            if (operation == DirectPtxMeshPoolOperation.SoftmaxScores)
                Assert.Contains(".shared .align 16 .b8 softmax_scratch[1024]", ptx);
            else
                Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
            Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
            Assert.All(blueprint.Tensors, tensor =>
            {
                Assert.Equal(DirectPtxExtentMode.Exact, tensor.ExtentMode);
                Assert.Equal(16, tensor.AlignmentBytes);
            });
            for (int i = 0; i < blueprint.Tensors.Count; i++)
                Assert.Equal((long)blueprint.Tensors[i].RequiredBytes,
                    PtxMeshPoolF32Kernel.GetRequiredBytes(operation, i));
            Assert.Equal("device-pointers-only", blueprint.Semantics["parameter-abi"]);
            Assert.Equal("experimental-hardware-evidence-pending", blueprint.Semantics["promotion"]);
            Assert.Equal("0", blueprint.Semantics["workspace-bytes"]);
        }
    }

    [Fact]
    public void MeshPoolFamily_HasPublicCudaProductionSurfaces()
    {
        string[] methods =
        [
            nameof(CudaBackend.MeshPoolComputeScores), nameof(CudaBackend.MeshPoolGather),
            nameof(CudaBackend.MeshPoolBackward), nameof(CudaBackend.MeshPoolImportanceBackward),
            nameof(CudaBackend.MeshPoolZeroGrad), nameof(CudaBackend.MeshPoolSoftmaxFindMax),
            nameof(CudaBackend.MeshPoolSoftmaxFinalMax), nameof(CudaBackend.MeshPoolSoftmaxExpSum),
            nameof(CudaBackend.MeshPoolSoftmaxFinalSum), nameof(CudaBackend.MeshPoolSoftmaxNormalize),
            nameof(CudaBackend.MeshPoolSoftmaxScores), nameof(CudaBackend.MeshPoolWeightedGather),
            nameof(CudaBackend.MeshPoolWeightedBackward), nameof(CudaBackend.MeshPoolScoresBackward)
        ];

        Assert.All(methods, method => Assert.NotNull(typeof(CudaBackend).GetMethod(method)));
    }

    [Fact]
    public void MeshPoolFamily_PreservesAtomicAndDeterministicReductionChoices()
    {
        string atomic = PtxMeshPoolF32Kernel.EmitPtx(
            8, 6, DirectPtxMeshPoolOperation.WeightedBackwardAtomic);
        string deterministic = PtxMeshPoolF32Kernel.EmitPtx(
            8, 6, DirectPtxMeshPoolOperation.WeightedBackwardDeterministic);

        Assert.Contains("atom.global.add.f32", atomic);
        Assert.DoesNotContain("atom.global", deterministic, StringComparison.Ordinal);
        Assert.Contains("BACKWARD_SCAN", deterministic);
        Assert.Contains("ld.global.f32", deterministic);
        Assert.Contains("st.global.f32", deterministic);
    }

    [Fact]
    public void MeshPoolSoftmax_BakesTemperatureAndUsesRegisterResidentOnlineStorage()
    {
        string fused = PtxMeshPoolF32Kernel.EmitPtx(
            8, 6, DirectPtxMeshPoolOperation.SoftmaxScores);
        string staged = PtxMeshPoolF32Kernel.EmitPtx(
            8, 6, DirectPtxMeshPoolOperation.SoftmaxExpSum);

        Assert.Contains("ex2.approx.f32", fused);
        Assert.Contains("ex2.approx.f32", staged);
        Assert.Contains("0f3fb8aa3b", fused);
        Assert.DoesNotContain("temperature", fused, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("stride", fused, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("1.0f-baked", PtxMeshPoolF32Kernel.CreateBlueprint(
            DirectPtxArchitectureFamily.Ampere,
            DirectPtxMeshPoolOperation.SoftmaxScores).Semantics["temperature"]);
    }

    [Fact]
    public void MeshPoolShapeMatrix_IsExactAndRejectsDynamicTemperature()
    {
        Assert.True(PtxMeshPoolF32Kernel.SupportsShape(
            DirectPtxMeshPoolOperation.ComputeScores, 16384, 4096, 64));
        Assert.True(PtxMeshPoolF32Kernel.SupportsShape(
            DirectPtxMeshPoolOperation.SoftmaxScores, 256, 4096, 64));
        Assert.False(PtxMeshPoolF32Kernel.SupportsShape(
            DirectPtxMeshPoolOperation.ComputeScores, 8192, 4096, 64));
        Assert.False(PtxMeshPoolF32Kernel.SupportsShape(
            DirectPtxMeshPoolOperation.ComputeScores, 16384, 2048, 64));
        Assert.False(PtxMeshPoolF32Kernel.SupportsShape(
            DirectPtxMeshPoolOperation.ComputeScores, 16384, 4096, 32));
        Assert.False(PtxMeshPoolF32Kernel.SupportsShape(
            DirectPtxMeshPoolOperation.ComputeScores, 16384, 4096, 64, 0.5f));
        Assert.False(PtxMeshPoolF32Kernel.SupportsShape(
            DirectPtxMeshPoolOperation.SoftmaxScores, 16384, 4096, 64));
    }

    [Fact]
    public void CsrSpmmEmitter_BakesExactShapeAndUsesRegisterResidentVec4Reduction()
    {
        string ptx = PtxFusedCsrSpmmVec4F32Kernel.EmitPtx(8, 6);

        Assert.Contains(PtxFusedCsrSpmmVec4F32Kernel.EntryPoint, ptx);
        Assert.Contains(".target sm_86", ptx);
        Assert.Contains("ld.global.v4.f32", ptx);
        Assert.Contains("st.global.v4.f32", ptx);
        Assert.Equal(4, Count(ptx, "fma.rn.f32"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("shape", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.Equal(5, Count(ptx, ".param .u64"));
    }

    [Fact]
    public void CsrSpmmBlueprint_DeclaresExactCsr32AndDensePhysicalAbi()
    {
        DirectPtxKernelBlueprint blueprint =
            PtxFusedCsrSpmmVec4F32Kernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere);

        Assert.Equal("csr-spmm-vec4-f32-v1-Ampere-m1024-k1024-n64-nnz16384-row-major", blueprint.Id);
        Assert.Equal(5, blueprint.Tensors.Count);
        Assert.All(blueprint.Tensors, tensor =>
        {
            Assert.Equal(DirectPtxExtentMode.Exact, tensor.ExtentMode);
            Assert.Equal(16, tensor.AlignmentBytes);
        });
        Assert.Equal(DirectPtxPhysicalLayout.CsrValues, blueprint.Tensors[0].Layout);
        Assert.Equal(DirectPtxPhysicalLayout.CsrColumnIndices, blueprint.Tensors[1].Layout);
        Assert.Equal(DirectPtxPhysicalLayout.CsrRowPointers, blueprint.Tensors[2].Layout);
        Assert.Equal(DirectPtxPhysicalType.Int32, blueprint.Tensors[1].PhysicalType);
        Assert.Equal(PtxFusedCsrSpmmVec4F32Kernel.NonZeros, blueprint.Tensors[0].LogicalExtent.ElementCount);
        Assert.Equal(PtxFusedCsrSpmmVec4F32Kernel.Rows + 1, blueprint.Tensors[2].LogicalExtent.ElementCount);
        Assert.Equal("0", blueprint.Semantics["workspace-bytes"]);
        Assert.Equal("0", blueprint.Semantics["intermediate-global-bytes"]);
    }

    [Fact]
    public void SddmmEmitter_BakesExactShapeAndUsesRegisterResidentUnrolledReduction()
    {
        string ptx = PtxFusedSddmmF32Kernel.EmitPtx(8, 6);

        Assert.Contains(PtxFusedSddmmF32Kernel.EntryPoint, ptx);
        Assert.Contains(".target sm_86", ptx);
        Assert.Equal(32, Count(ptx, "ld.global.v4.f32"));
        Assert.Equal(PtxFusedSddmmF32Kernel.Inner, Count(ptx, "fma.rn.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("bra", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("shape", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.Equal(5, Count(ptx, ".param .u64"));
    }

    [Fact]
    public void SddmmBlueprint_DeclaresExactCoo32AndDensePhysicalAbi()
    {
        DirectPtxKernelBlueprint blueprint =
            PtxFusedSddmmF32Kernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere);

        Assert.Equal("sddmm-f32-v1-Ampere-m1024-n1024-k64-nnz16384-row-major", blueprint.Id);
        Assert.Equal(5, blueprint.Tensors.Count);
        Assert.All(blueprint.Tensors, tensor =>
        {
            Assert.Equal(DirectPtxExtentMode.Exact, tensor.ExtentMode);
            Assert.Equal(16, tensor.AlignmentBytes);
        });
        Assert.Equal(DirectPtxPhysicalLayout.CooRowIndices, blueprint.Tensors[0].Layout);
        Assert.Equal(DirectPtxPhysicalLayout.CooColumnIndices, blueprint.Tensors[1].Layout);
        Assert.Equal(DirectPtxPhysicalLayout.RowMajor2D, blueprint.Tensors[2].Layout);
        Assert.Equal(DirectPtxPhysicalLayout.Vector, blueprint.Tensors[4].Layout);
        Assert.Equal("ascending-inner-index", blueprint.Semantics["reduction-order"]);
        Assert.Equal("0", blueprint.Semantics["workspace-bytes"]);
    }

    [Fact]
    public void SddmmShapeMatrix_IsExactAndNeverPromotedWithoutHardwareEvidence()
    {
        Assert.True(PtxFusedSddmmF32Kernel.SupportsShape(1024, 1024, 64, 16384));
        Assert.False(PtxFusedSddmmF32Kernel.SupportsShape(512, 1024, 64, 16384));
        Assert.False(PtxFusedSddmmF32Kernel.SupportsShape(1024, 1024, 32, 16384));
        Assert.False(PtxFusedSddmmF32Kernel.SupportsShape(1024, 1024, 64, 8192));
        Assert.False(PtxFusedSddmmF32Kernel.IsPromotedShape(1024, 1024, 64, 16384));
    }

    [Fact]
    public void CsrSpmmBiasEmitter_FusesBiasIntoRegisterResidentVec4Reduction()
    {
        string ptx = PtxFusedCsrSpmmBiasVec4F32Kernel.EmitPtx(8, 6);

        Assert.Contains(PtxFusedCsrSpmmBiasVec4F32Kernel.EntryPoint, ptx);
        Assert.Equal(2, Count(ptx, "ld.global.v4.f32"));
        Assert.Equal(4, Count(ptx, "fma.rn.f32"));
        Assert.Equal(1, Count(ptx, "st.global.v4.f32"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.Equal(6, Count(ptx, ".param .u64"));
    }

    [Fact]
    public void CsrSpmmBiasBlueprint_DeclaresExactBiasAndZeroWorkspace()
    {
        DirectPtxKernelBlueprint blueprint =
            PtxFusedCsrSpmmBiasVec4F32Kernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere);

        Assert.Equal(6, blueprint.Tensors.Count);
        Assert.Equal(DirectPtxPhysicalLayout.Vector, blueprint.Tensors[4].Layout);
        Assert.Equal(PtxFusedCsrSpmmBiasVec4F32Kernel.Columns,
            blueprint.Tensors[4].PhysicalExtent.ElementCount);
        Assert.All(blueprint.Tensors, tensor => Assert.Equal(DirectPtxExtentMode.Exact, tensor.ExtentMode));
        Assert.Equal("bias-then-stored-row-order", blueprint.Semantics["reduction-order"]);
        Assert.Equal("0", blueprint.Semantics["workspace-bytes"]);
        Assert.Equal("0", blueprint.Semantics["intermediate-global-bytes"]);
    }

    [Fact]
    public void CsrSpmmBiasReluEmitter_BakesBranchlessRegisterEpilogue()
    {
        string ptx = PtxFusedCsrSpmmBiasVec4F32Kernel.EmitPtx(8, 6, fuseRelu: true);
        DirectPtxKernelBlueprint blueprint = PtxFusedCsrSpmmBiasVec4F32Kernel.CreateBlueprint(
            DirectPtxArchitectureFamily.Ampere, fuseRelu: true);

        Assert.Contains(PtxFusedCsrSpmmBiasVec4F32Kernel.ReluEntryPoint, ptx);
        Assert.Equal(4, Count(ptx, "max.f32"));
        Assert.DoesNotContain(PtxFusedCsrSpmmBiasVec4F32Kernel.EntryPoint + "(", ptx);
        Assert.Equal("relu", blueprint.Semantics["epilogue"]);
        Assert.Equal("C=relu(A(csr)*B+bias)", blueprint.Semantics["equation"]);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void CsrSpmmF64Emitter_UsesTwoWideFp64RegisterReduction()
    {
        string ptx = PtxFusedCsrSpmmVec2F64Kernel.EmitPtx(8, 6);

        Assert.Contains(PtxFusedCsrSpmmVec2F64Kernel.EntryPoint, ptx);
        Assert.Contains("ld.global.v2.f64", ptx);
        Assert.Equal(2, Count(ptx, "fma.rn.f64"));
        Assert.Contains("st.global.v2.f64", ptx);
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.Equal(5, Count(ptx, ".param .u64"));
    }

    [Fact]
    public void CsrSpmmF64Blueprint_UsesEightBytePhysicalContract()
    {
        DirectPtxKernelBlueprint blueprint =
            PtxFusedCsrSpmmVec2F64Kernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere);

        Assert.Equal(DirectPtxPhysicalType.Float64, blueprint.Tensors[0].PhysicalType);
        Assert.Equal(8, blueprint.Tensors[0].ElementBytes);
        Assert.Equal((nuint)(PtxFusedCsrSpmmVec2F64Kernel.NonZeros * sizeof(double)),
            blueprint.Tensors[0].RequiredBytes);
        Assert.Equal("fp64-fma", blueprint.Semantics["accumulator"]);
        Assert.Equal("0", blueprint.Semantics["workspace-bytes"]);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void GraphGatherEmitter_BakesDirectionAndUsesOneVec4Transfer(bool gatherSource)
    {
        string ptx = PtxGraphEdgeGatherVec4F32Kernel.EmitPtx(8, 6, gatherSource);
        string entry = gatherSource
            ? PtxGraphEdgeGatherVec4F32Kernel.SourceEntryPoint
            : PtxGraphEdgeGatherVec4F32Kernel.TargetEntryPoint;
        DirectPtxKernelBlueprint blueprint = PtxGraphEdgeGatherVec4F32Kernel.CreateBlueprint(
            DirectPtxArchitectureFamily.Ampere, gatherSource);

        Assert.Contains(entry, ptx);
        Assert.Equal(1, Count(ptx, "ld.global.v4.f32"));
        Assert.Equal(1, Count(ptx, "st.global.v4.f32"));
        Assert.Equal(3, Count(ptx, ".param .u64"));
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("bra", ptx, StringComparison.Ordinal);
        Assert.Equal(gatherSource ? DirectPtxPhysicalLayout.GraphSourceIndices :
            DirectPtxPhysicalLayout.GraphTargetIndices, blueprint.Tensors[1].Layout);
        Assert.Equal(DirectPtxPhysicalLayout.EdgeMajor2D, blueprint.Tensors[2].Layout);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DeterministicGraphScatterEmitter_FusesInitializationAndWritesOnce(bool weighted)
    {
        string ptx = PtxGraphScatterAddDeterministicVec4F32Kernel.EmitPtx(8, 6, weighted);
        DirectPtxKernelBlueprint blueprint =
            PtxGraphScatterAddDeterministicVec4F32Kernel.CreateBlueprint(
                DirectPtxArchitectureFamily.Ampere, weighted);

        Assert.Contains(weighted
            ? PtxGraphScatterAddDeterministicVec4F32Kernel.WeightedEntryPoint
            : PtxGraphScatterAddDeterministicVec4F32Kernel.EntryPoint, ptx);
        Assert.Equal(1, Count(ptx, "st.global.v4.f32"));
        Assert.DoesNotContain("atom.", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.Equal(weighted ? 5 : 4, Count(ptx, ".param .u64"));
        Assert.Equal("ascending-edge-index", blueprint.Semantics["reduction-order"]);
        Assert.Equal("fused-zero-register-initialization", blueprint.Semantics["output-initialization"]);
        Assert.Equal("single-overwrite", blueprint.Semantics["output-write"]);
        Assert.Equal("0", blueprint.Semantics["workspace-bytes"]);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AtomicGraphScatterEmitter_BakesPointerOnlyAtomicAbi(bool weighted)
    {
        string ptx = PtxGraphScatterAddAtomicF32Kernel.EmitPtx(8, 6, weighted);
        DirectPtxKernelBlueprint blueprint = PtxGraphScatterAddAtomicF32Kernel.CreateBlueprint(
            DirectPtxArchitectureFamily.Ampere, weighted);

        Assert.Contains(weighted
            ? PtxGraphScatterAddAtomicF32Kernel.WeightedEntryPoint
            : PtxGraphScatterAddAtomicF32Kernel.EntryPoint, ptx);
        Assert.Equal(1, Count(ptx, "atom.global.add.f32"));
        Assert.DoesNotContain("st.global", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.Equal(weighted ? 5 : 4, Count(ptx, ".param .u64"));
        Assert.Equal("unordered-global-fp32-atomic", blueprint.Semantics["reduction-order"]);
        Assert.Equal("zero-initialized-by-direct-ptx-fill", blueprint.Semantics["output-precondition"]);
        Assert.Equal("0", blueprint.Semantics["workspace-bytes"]);
    }

    [Theory]
    [InlineData((int)DirectPtxSegmentReduction.Sum, 3)]
    [InlineData((int)DirectPtxSegmentReduction.Mean, 4)]
    [InlineData((int)DirectPtxSegmentReduction.Max, 3)]
    public void SegmentReduceEmitter_BakesReductionAndSingleStore(
        int reductionValue,
        int pointerParameters)
    {
        var reduction = (DirectPtxSegmentReduction)reductionValue;
        string ptx = PtxSegmentReduceDeterministicVec4F32Kernel.EmitPtx(8, 6, reduction);
        DirectPtxKernelBlueprint blueprint =
            PtxSegmentReduceDeterministicVec4F32Kernel.CreateBlueprint(
                DirectPtxArchitectureFamily.Ampere, reduction);

        Assert.Equal(pointerParameters, Count(ptx, ".param .u64"));
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("atom.", ptx, StringComparison.Ordinal);
        Assert.Equal(1, Count(ptx, "st.global.v4.f32"));
        Assert.Equal("ascending-item-index", blueprint.Semantics["reduction-order"]);
        Assert.Equal("single-overwrite", blueprint.Semantics["output-write"]);
        Assert.Equal("0", blueprint.Semantics["intermediate-global-bytes"]);
    }

    [Theory]
    [InlineData((int)DirectPtxCsrSegmentReduction.Max)]
    [InlineData((int)DirectPtxCsrSegmentReduction.Min)]
    [InlineData((int)DirectPtxCsrSegmentReduction.StdDev)]
    public void CsrSegmentReduceEmitter_UsesExactPointerOnlyLegacyCsrAbi(int reductionValue)
    {
        var reduction = (DirectPtxCsrSegmentReduction)reductionValue;
        string ptx = PtxCsrSegmentReduceVec4F32Kernel.EmitPtx(8, 6, reduction, 1e-8f);
        DirectPtxKernelBlueprint blueprint = PtxCsrSegmentReduceVec4F32Kernel.CreateBlueprint(
            DirectPtxArchitectureFamily.Ampere, reduction, 1e-8f);

        Assert.Equal(4, Count(ptx, ".param .u64"));
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.Equal(1, Count(ptx, "st.global.v4.f32"));
        Assert.Equal(DirectPtxPhysicalType.Float32, blueprint.Tensors[0].PhysicalType);
        Assert.Equal(DirectPtxPhysicalLayout.CsrFloatColumnIndices, blueprint.Tensors[0].Layout);
        Assert.Equal(DirectPtxPhysicalLayout.CsrFloatRowPointers, blueprint.Tensors[1].Layout);
        Assert.Equal("zero", blueprint.Semantics["empty-row"]);
        Assert.Equal("single-overwrite", blueprint.Semantics["output-write"]);
    }

    [Theory]
    [InlineData((int)DirectPtxCsrBackwardTarget.DenseB)]
    [InlineData((int)DirectPtxCsrBackwardTarget.Values)]
    public void CsrBackwardEmitter_UsesPointerOnlySingleWriteAbi(int targetValue)
    {
        var target = (DirectPtxCsrBackwardTarget)targetValue;
        string ptx = PtxCsrSpmmBackwardF32Kernel.EmitPtx(8, 6, target);
        DirectPtxKernelBlueprint blueprint = PtxCsrSpmmBackwardF32Kernel.CreateBlueprint(
            DirectPtxArchitectureFamily.Ampere, target);

        Assert.Equal(5, Count(ptx, ".param .u64"));
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("atom.", ptx, StringComparison.Ordinal);
        Assert.Equal(1, target == DirectPtxCsrBackwardTarget.DenseB
            ? Count(ptx, "st.global.v4.f32") : Count(ptx, "st.global.f32"));
        Assert.Equal("fused-register-initialization", blueprint.Semantics["output-initialization"]);
        Assert.Equal("single-overwrite", blueprint.Semantics["output-write"]);
        Assert.Equal("0", blueprint.Semantics["workspace-bytes"]);
    }

    [Theory]
    [InlineData((int)DirectPtxSparseUtility.FillZero, 1)]
    [InlineData((int)DirectPtxSparseUtility.FillNegativeInfinity, 1)]
    [InlineData((int)DirectPtxSparseUtility.DegreeNormalize, 3)]
    [InlineData((int)DirectPtxSparseUtility.SymmetricDegreeNormalize, 6)]
    public void SparseUtilityEmitter_BakesExactPointerOnlyAbi(int utilityValue, int pointerCount)
    {
        var utility = (DirectPtxSparseUtility)utilityValue;
        string ptx = PtxSparseUtilityF32Kernel.EmitPtx(8, 6, utility, 1e-8f);
        DirectPtxKernelBlueprint blueprint = PtxSparseUtilityF32Kernel.CreateBlueprint(
            DirectPtxArchitectureFamily.Ampere, utility, 1e-8f);

        Assert.Equal(pointerCount, Count(ptx, ".param .u64"));
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("0", blueprint.Semantics["workspace-bytes"]);
        Assert.All(blueprint.Tensors, tensor => Assert.Equal(DirectPtxExtentMode.Exact, tensor.ExtentMode));
    }

    [Theory]
    [InlineData((int)DirectPtxStructuredSparse2x4Operation.Enforce, 3)]
    [InlineData((int)DirectPtxStructuredSparse2x4Operation.Decompress, 3)]
    [InlineData((int)DirectPtxStructuredSparse2x4Operation.Gemm, 4)]
    [InlineData((int)DirectPtxStructuredSparse2x4Operation.GemmBiasRelu, 5)]
    [InlineData((int)DirectPtxStructuredSparse2x4Operation.MatMulBaseline, 4)]
    public void StructuredSparse2x4Emitter_BakesExactPointerOnlyAbi(
        int operationValue,
        int pointerCount)
    {
        var operation = (DirectPtxStructuredSparse2x4Operation)operationValue;
        string ptx = PtxStructuredSparse2x4F32Kernel.EmitPtx(8, 6, operation, 0.75f, 0.25f);
        DirectPtxKernelBlueprint blueprint = PtxStructuredSparse2x4F32Kernel.CreateBlueprint(
            DirectPtxArchitectureFamily.Ampere, operation, 0.75f, 0.25f);

        Assert.Equal(pointerCount, Count(ptx, ".param .u64"));
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("0", blueprint.Semantics["workspace-bytes"]);
        Assert.All(blueprint.Tensors, tensor => Assert.Equal(DirectPtxExtentMode.Exact, tensor.ExtentMode));
        Assert.Contains(blueprint.Tensors, tensor =>
            tensor.Layout == DirectPtxPhysicalLayout.StructuredSparse2x4Metadata &&
            tensor.PhysicalType == DirectPtxPhysicalType.UInt8);
    }

    [Theory]
    [InlineData((int)DirectPtxScalarScatterAddMode.Atomic, true, "preserved")]
    [InlineData((int)DirectPtxScalarScatterAddMode.DeterministicOverwrite, false, "discarded")]
    [InlineData((int)DirectPtxScalarScatterAddMode.DeterministicAccumulate, false, "preserved")]
    public void ScalarScatterAddEmitter_BakesReductionAndSeedSemantics(
        int modeValue,
        bool atomic,
        string seed)
    {
        var mode = (DirectPtxScalarScatterAddMode)modeValue;
        string ptx = PtxScatterAddScalarF32Kernel.EmitPtx(8, 6, mode);
        DirectPtxKernelBlueprint blueprint = PtxScatterAddScalarF32Kernel.CreateBlueprint(
            DirectPtxArchitectureFamily.Ampere, mode);

        Assert.Equal(3, Count(ptx, ".param .u64"));
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.Equal(atomic, ptx.Contains("atom.global.add.f32", StringComparison.Ordinal));
        Assert.Equal(seed, blueprint.Semantics["destination-seed"]);
        Assert.Equal("0", blueprint.Semantics["workspace-bytes"]);
        Assert.All(blueprint.Tensors, tensor => Assert.Equal(DirectPtxExtentMode.Exact, tensor.ExtentMode));
    }

    [Theory]
    [InlineData((int)DirectPtxScatterRowsOperation.AddAtomic, 3, true)]
    [InlineData((int)DirectPtxScatterRowsOperation.AddDeterministic, 3, false)]
    [InlineData((int)DirectPtxScatterRowsOperation.MeanAccumulateAtomic, 4, true)]
    [InlineData((int)DirectPtxScatterRowsOperation.MeanAccumulateDeterministic, 4, false)]
    [InlineData((int)DirectPtxScatterRowsOperation.MeanNormalize, 2, false)]
    public void ScatterRowsEmitter_BakesExactOperationAbi(
        int operationValue,
        int pointerCount,
        bool atomic)
    {
        var operation = (DirectPtxScatterRowsOperation)operationValue;
        string ptx = PtxScatterRowsF32Kernel.EmitPtx(8, 6, operation);
        DirectPtxKernelBlueprint blueprint = PtxScatterRowsF32Kernel.CreateBlueprint(
            DirectPtxArchitectureFamily.Ampere, operation);

        Assert.Equal(pointerCount, Count(ptx, ".param .u64"));
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.Equal(atomic, ptx.Contains("atom.global", StringComparison.Ordinal));
        Assert.Equal("0", blueprint.Semantics["workspace-bytes"]);
        Assert.All(blueprint.Tensors, tensor => Assert.Equal(DirectPtxExtentMode.Exact, tensor.ExtentMode));
    }

    [Fact]
    public void ScatterMaxRowsEmitter_BakesFirstTieAndEmptyGroupSemantics()
    {
        string ptx = PtxScatterMaxRowsF32Kernel.EmitPtx(8, 6);
        DirectPtxKernelBlueprint blueprint = PtxScatterMaxRowsF32Kernel.CreateBlueprint(
            DirectPtxArchitectureFamily.Ampere);

        Assert.Equal(4, Count(ptx, ".param .u64"));
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("atom.", ptx, StringComparison.Ordinal);
        Assert.Equal("lowest-source-row", blueprint.Semantics["tie-break"]);
        Assert.Equal("negative-infinity", blueprint.Semantics["empty-output"]);
        Assert.Equal("float-encoded-minus-one", blueprint.Semantics["empty-argmax"]);
        Assert.Equal("0", blueprint.Semantics["workspace-bytes"]);
    }

    [Theory]
    [InlineData((int)DirectPtxScatterBackwardRowsOperation.Add, 3)]
    [InlineData((int)DirectPtxScatterBackwardRowsOperation.Mean, 4)]
    public void ScatterBackwardRowsEmitter_BakesExactGatherAbi(int operationValue, int pointerCount)
    {
        var operation = (DirectPtxScatterBackwardRowsOperation)operationValue;
        string ptx = PtxScatterBackwardRowsF32Kernel.EmitPtx(8, 6, operation);
        DirectPtxKernelBlueprint blueprint = PtxScatterBackwardRowsF32Kernel.CreateBlueprint(
            DirectPtxArchitectureFamily.Ampere, operation);

        Assert.Equal(pointerCount, Count(ptx, ".param .u64"));
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("atom.", ptx, StringComparison.Ordinal);
        Assert.Equal("zero", blueprint.Semantics["invalid-index"]);
        Assert.Equal("0", blueprint.Semantics["workspace-bytes"]);
        Assert.All(blueprint.Tensors, tensor => Assert.Equal(DirectPtxExtentMode.Exact, tensor.ExtentMode));
    }

    [Theory]
    [InlineData((int)DirectPtxCapsuleRoutingOperation.WeightedSum, "ascending-input-capsule", 10240, 163840, 5120)]
    [InlineData((int)DirectPtxCapsuleRoutingOperation.Agreement, "ascending-dimension", 163840, 5120, 10240)]
    public void CapsuleRoutingEmitter_BakesExactPointerOnlyReductionAbi(
        int operationValue,
        string reductionOrder,
        int firstElements,
        int secondElements,
        int outputElements)
    {
        var operation = (DirectPtxCapsuleRoutingOperation)operationValue;
        string ptx = PtxCapsuleRoutingF32Kernel.EmitPtx(8, 6, operation);
        DirectPtxKernelBlueprint blueprint = PtxCapsuleRoutingF32Kernel.CreateBlueprint(
            DirectPtxArchitectureFamily.Ampere, operation);

        Assert.Equal(3, Count(ptx, ".param .u64"));
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.Equal(3, blueprint.Tensors.Count);
        Assert.All(blueprint.Tensors, tensor =>
        {
            Assert.Equal(DirectPtxExtentMode.Exact, tensor.ExtentMode);
            Assert.Equal(16, tensor.AlignmentBytes);
        });
        Assert.Equal(firstElements, blueprint.Tensors[0].PhysicalExtent.ElementCount);
        Assert.Equal(secondElements, blueprint.Tensors[1].PhysicalExtent.ElementCount);
        Assert.Equal(outputElements, blueprint.Tensors[2].PhysicalExtent.ElementCount);
        Assert.Equal(reductionOrder, blueprint.Semantics["reduction-order"]);
        Assert.Equal("single-overwrite", blueprint.Semantics["output-write"]);
        Assert.Equal("0", blueprint.Semantics["workspace-bytes"]);
        Assert.Equal("0", blueprint.Semantics["intermediate-global-bytes"]);
    }

    [Theory]
    [InlineData((int)DirectPtxCapsuleProjectionOperation.Predictions,
        (int)DirectPtxPhysicalLayout.CapsulePredictionWeights, 64)]
    [InlineData((int)DirectPtxCapsuleProjectionOperation.Transform,
        (int)DirectPtxPhysicalLayout.CapsuleTransformWeights, 640)]
    public void CapsuleProjectionEmitter_UnrollsExactLayoutSpecificReduction(
        int operationValue,
        int weightLayoutValue,
        int weightByteStride)
    {
        var operation = (DirectPtxCapsuleProjectionOperation)operationValue;
        string ptx = PtxCapsuleProjectionF32Kernel.EmitPtx(8, 6, operation);
        DirectPtxKernelBlueprint blueprint = PtxCapsuleProjectionF32Kernel.CreateBlueprint(
            DirectPtxArchitectureFamily.Ampere, operation);

        Assert.Equal(3, Count(ptx, ".param .u64"));
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(PtxCapsuleProjectionF32Kernel.InputDimension, Count(ptx, "fma.rn.f32"));
        Assert.Equal(PtxCapsuleProjectionF32Kernel.InputDimension * 2, Count(ptx, "ld.global.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.Contains($"[%rd6+{weightByteStride * 7}]", ptx, StringComparison.Ordinal);
        Assert.Equal((DirectPtxPhysicalLayout)weightLayoutValue, blueprint.Tensors[1].Layout);
        Assert.Equal(PtxCapsuleProjectionF32Kernel.InputElements,
            blueprint.Tensors[0].PhysicalExtent.ElementCount);
        Assert.Equal(PtxCapsuleProjectionF32Kernel.WeightElements,
            blueprint.Tensors[1].PhysicalExtent.ElementCount);
        Assert.Equal(PtxCapsuleProjectionF32Kernel.OutputElements,
            blueprint.Tensors[2].PhysicalExtent.ElementCount);
        Assert.All(blueprint.Tensors, tensor => Assert.Equal(DirectPtxExtentMode.Exact, tensor.ExtentMode));
        Assert.Equal("fully-unrolled-ascending-input-dimension", blueprint.Semantics["reduction-order"]);
        Assert.Equal("0", blueprint.Semantics["workspace-bytes"]);
    }

    [Theory]
    [InlineData((int)DirectPtxCapsuleSquashOperation.Forward, 2, 16, 16)]
    [InlineData((int)DirectPtxCapsuleSquashOperation.Backward, 3, 32, 16)]
    public void CapsuleSquashEmitter_RetainsCanonicalVectorAcrossFp64Reduction(
        int operationValue,
        int pointerCount,
        int loadCount,
        int storeCount)
    {
        var operation = (DirectPtxCapsuleSquashOperation)operationValue;
        string ptx = PtxCapsuleSquashF32Kernel.EmitPtx(8, 6, operation);
        DirectPtxKernelBlueprint blueprint = PtxCapsuleSquashF32Kernel.CreateBlueprint(
            DirectPtxArchitectureFamily.Ampere, operation);

        Assert.Equal(pointerCount, Count(ptx, ".param .u64"));
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("epsilon_ptr", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(loadCount, Count(ptx, "ld.global.f32"));
        Assert.Equal(storeCount, Count(ptx, "st.global.f32"));
        Assert.Contains("sqrt.rn.f64", ptx, StringComparison.Ordinal);
        Assert.Contains("cvt.f64.f32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.All(blueprint.Tensors, tensor =>
        {
            Assert.Equal(PtxCapsuleSquashF32Kernel.Elements, tensor.PhysicalExtent.ElementCount);
            Assert.Equal(DirectPtxExtentMode.Exact, tensor.ExtentMode);
        });
        Assert.Equal("fp64", blueprint.Semantics["reduction-precision"]);
        Assert.Equal("0", blueprint.Semantics["workspace-bytes"]);
        Assert.Equal("0", blueprint.Semantics["intermediate-global-bytes"]);
    }

    [Fact]
    public void CapsuleSquashShapeAndEpsilonAdmission_IsExact()
    {
        Assert.True(PtxCapsuleSquashF32Kernel.SupportsShape(320, 16, 1e-8f));
        Assert.False(PtxCapsuleSquashF32Kernel.SupportsShape(319, 16, 1e-8f));
        Assert.False(PtxCapsuleSquashF32Kernel.SupportsShape(320, 8, 1e-8f));
        Assert.False(PtxCapsuleSquashF32Kernel.SupportsShape(320, 16, 1e-6f));
    }

    [Theory]
    [InlineData((int)DirectPtxResidentScatterAuxOperation.MeanRowsWithCounts, 4, 2, "ascending-source-row")]
    [InlineData((int)DirectPtxResidentScatterAuxOperation.MaxBackwardRows, 3, 1, "ascending-source-row")]
    public void ResidentScatterAuxEmitter_BakesExactDeterministicAbi(
        int operationValue,
        int pointerCount,
        int storeCount,
        string reductionOrder)
    {
        var operation = (DirectPtxResidentScatterAuxOperation)operationValue;
        string ptx = PtxResidentScatterAuxF32Kernel.EmitPtx(8, 6, operation);
        DirectPtxKernelBlueprint blueprint = PtxResidentScatterAuxF32Kernel.CreateBlueprint(
            DirectPtxArchitectureFamily.Ampere, operation);

        Assert.Equal(pointerCount, Count(ptx, ".param .u64"));
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("atom.", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(storeCount, Count(ptx, "st.global.f32"));
        Assert.All(blueprint.Tensors, tensor => Assert.Equal(DirectPtxExtentMode.Exact, tensor.ExtentMode));
        Assert.Equal(reductionOrder, blueprint.Semantics["reduction-order"]);
        Assert.Equal("single-overwrite", blueprint.Semantics["output-write"]);
        Assert.Equal("0", blueprint.Semantics["workspace-bytes"]);
        Assert.Equal("0", blueprint.Semantics["intermediate-global-bytes"]);
    }

    [Theory]
    [InlineData((int)DirectPtxResidentScatterSoftmaxOperation.Forward, 3, true)]
    [InlineData((int)DirectPtxResidentScatterSoftmaxOperation.Backward, 4, false)]
    public void ResidentScatterSoftmaxEmitter_UsesOneGroupOwnedFusedLaunch(
        int operationValue,
        int pointerCount,
        bool hasExponent)
    {
        var operation = (DirectPtxResidentScatterSoftmaxOperation)operationValue;
        string ptx = PtxResidentScatterSoftmaxF32Kernel.EmitPtx(8, 6, operation);
        DirectPtxKernelBlueprint blueprint = PtxResidentScatterSoftmaxF32Kernel.CreateBlueprint(
            DirectPtxArchitectureFamily.Ampere, operation);

        Assert.Equal(pointerCount, Count(ptx, ".param .u64"));
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("atom.", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(hasExponent, ptx.Contains("ex2.approx.f32", StringComparison.Ordinal));
        Assert.Contains("one-thread-per-group-feature-plus-invalid-row-zeroing",
            blueprint.Semantics["ownership"], StringComparison.Ordinal);
        Assert.Equal("1", blueprint.Semantics["kernel-launches"]);
        Assert.Equal("0", blueprint.Semantics["workspace-bytes"]);
        Assert.Equal("0", blueprint.Semantics["intermediate-global-bytes"]);
        Assert.All(blueprint.Tensors, tensor => Assert.Equal(DirectPtxExtentMode.Exact, tensor.ExtentMode));
    }

    [Fact]
    public void UniformMeshLaplacianEmitter_BakesTriangleTopologyAndSingleWrite()
    {
        string ptx = PtxUniformMeshLaplacianF32Kernel.EmitPtx(8, 6);
        DirectPtxKernelBlueprint blueprint = PtxUniformMeshLaplacianF32Kernel.CreateBlueprint(
            DirectPtxArchitectureFamily.Ampere);

        Assert.Equal(2, Count(ptx, ".param .u64"));
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("atom.", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(3, Count(ptx, "ld.global.u32"));
        Assert.Equal(12, Count(ptx, "and.pred"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.Equal(PtxUniformMeshLaplacianF32Kernel.FaceIndices,
            blueprint.Tensors[0].PhysicalExtent.ElementCount);
        Assert.Equal(PtxUniformMeshLaplacianF32Kernel.OutputElements,
            blueprint.Tensors[1].PhysicalExtent.ElementCount);
        Assert.Equal("ascending-face", blueprint.Semantics["reduction-order"]);
        Assert.Equal("single-overwrite", blueprint.Semantics["output-write"]);
        Assert.Equal("0", blueprint.Semantics["workspace-bytes"]);
        Assert.All(blueprint.Tensors, tensor => Assert.Equal(DirectPtxExtentMode.Exact, tensor.ExtentMode));
    }

    [Theory]
    [InlineData((int)DirectPtxSparseOptimizerOperation.Sgd, 3, 0)]
    [InlineData((int)DirectPtxSparseOptimizerOperation.SgdMomentum, 4, 1)]
    [InlineData((int)DirectPtxSparseOptimizerOperation.Adam, 5, 2)]
    [InlineData((int)DirectPtxSparseOptimizerOperation.AdamW, 5, 2)]
    [InlineData((int)DirectPtxSparseOptimizerOperation.Rmsprop, 4, 1)]
    [InlineData((int)DirectPtxSparseOptimizerOperation.Adagrad, 4, 1)]
    [InlineData((int)DirectPtxSparseOptimizerOperation.Nag, 4, 1)]
    [InlineData((int)DirectPtxSparseOptimizerOperation.Adadelta, 5, 2)]
    [InlineData((int)DirectPtxSparseOptimizerOperation.Amsgrad, 6, 3)]
    [InlineData((int)DirectPtxSparseOptimizerOperation.Adamax, 5, 2)]
    [InlineData((int)DirectPtxSparseOptimizerOperation.Lion, 4, 1)]
    [InlineData((int)DirectPtxSparseOptimizerOperation.Nadam, 5, 2)]
    [InlineData((int)DirectPtxSparseOptimizerOperation.Ftrl, 5, 2)]
    [InlineData((int)DirectPtxSparseOptimizerOperation.ProximalL1, 3, 0)]
    public void SparseOptimizerEmitter_BakesPointerOnlyExactAbi(
        int operationValue,
        int pointerCount,
        int stateCount)
    {
        var operation = (DirectPtxSparseOptimizerOperation)operationValue;
        DirectPtxSparseOptimizerKey key = CreateSparseOptimizerKey(operation);
        string ptx = PtxSparseOptimizerF32Kernel.EmitPtx(8, 6, key);
        DirectPtxKernelBlueprint blueprint = PtxSparseOptimizerF32Kernel.CreateBlueprint(
            DirectPtxArchitectureFamily.Ampere, key);

        Assert.Equal(pointerCount, Count(ptx, ".param .u64"));
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("ld.global.b32", ptx, StringComparison.Ordinal);
        Assert.Contains("and.b32 %r5, %r3, 2139095040", ptx, StringComparison.Ordinal);
        Assert.Contains("cvt.rzi.s32.f32", ptx, StringComparison.Ordinal);
        Assert.Equal(3 + stateCount, blueprint.Tensors.Count);
        Assert.Equal(1 + stateCount, Count(ptx, "st.global.f32"));
        Assert.All(blueprint.Tensors, tensor =>
        {
            Assert.Equal(DirectPtxExtentMode.Exact, tensor.ExtentMode);
            Assert.Equal(16, tensor.AlignmentBytes);
        });
        Assert.Equal(PtxSparseOptimizerF32Kernel.ParameterElements,
            blueprint.Tensors[0].PhysicalExtent.ElementCount);
        Assert.Equal(PtxSparseOptimizerF32Kernel.NonZeros,
            blueprint.Tensors[1].PhysicalExtent.ElementCount);
        Assert.Equal(DirectPtxPhysicalLayout.SparseOptimizerFloatIndices,
            blueprint.Tensors[1].Layout);
        Assert.Equal("pointers-only", blueprint.Semantics["kernel-parameters"]);
        Assert.Equal("0", blueprint.Semantics["workspace-bytes"]);
        Assert.Equal("0", blueprint.Semantics["intermediate-global-bytes"]);
    }

    [Fact]
    public void SparseOptimizerAdmission_IsExactAndRejectsUnsafeScalarSpecializations()
    {
        Assert.True(PtxSparseOptimizerF32Kernel.SupportsShape(1_048_576, 16_384));
        Assert.False(PtxSparseOptimizerF32Kernel.SupportsShape(1_048_575, 16_384));
        Assert.False(PtxSparseOptimizerF32Kernel.SupportsShape(1_048_576, 8_192));
        Assert.True(PtxSparseOptimizerF32Kernel.SupportsConfiguration(
            CreateSparseOptimizerKey(DirectPtxSparseOptimizerOperation.Adam)));
        Assert.False(PtxSparseOptimizerF32Kernel.SupportsConfiguration(
            DirectPtxSparseOptimizerKey.Create(
                DirectPtxSparseOptimizerOperation.Adam,
                1e-3f, 1f, 0.999f, 1e-8f, 0, 1)));
        Assert.False(PtxSparseOptimizerF32Kernel.SupportsConfiguration(
            DirectPtxSparseOptimizerKey.Create(
                DirectPtxSparseOptimizerOperation.Ftrl,
                0, 0.1f, 0.1f, 1f)));
    }

    [Theory]
    [InlineData(0, "FSL_ACTIVATE")]
    [InlineData(1, "max.f32")]
    [InlineData(2, "0f3d372713")]
    [InlineData(3, "rcp.approx.f32")]
    [InlineData(4, "0fbf800000")]
    [InlineData(5, "selp.f32")]
    [InlineData(6, "rcp.approx.f32")]
    public void FusedSparseLinearEmitter_FusesEveryProductionActivation(
        int activation,
        string activationToken)
    {
        foreach (bool hasBias in new[] { false, true })
        {
            var key = new DirectPtxFusedSparseLinearKey(hasBias, activation);
            string ptx = PtxFusedSparseLinearF32Kernel.EmitPtx(8, 6, key);
            DirectPtxKernelBlueprint blueprint = PtxFusedSparseLinearF32Kernel.CreateBlueprint(
                DirectPtxArchitectureFamily.Ampere, key);

            Assert.Equal(hasBias ? 5 : 4, Count(ptx, ".param .u64"));
            Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
            Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
            Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
            Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
            Assert.Contains(activationToken, ptx, StringComparison.Ordinal);
            Assert.Contains("fma.rn.f32", ptx, StringComparison.Ordinal);
            Assert.Equal(1, Count(ptx, "st.global.f32"));
            Assert.Equal(DirectPtxPhysicalLayout.PackedCsrRowsAndColumns,
                blueprint.Tensors[1].Layout);
            Assert.Equal(PtxFusedSparseLinearF32Kernel.PackedCsrElements,
                blueprint.Tensors[1].PhysicalExtent.ElementCount);
            Assert.Equal("csr-dot-plus-optional-bias-plus-activation",
                blueprint.Semantics["fusion"]);
            Assert.Equal("single-overwrite", blueprint.Semantics["output-write"]);
            Assert.Equal("0", blueprint.Semantics["workspace-bytes"]);
            Assert.All(blueprint.Tensors, tensor =>
                Assert.Equal(DirectPtxExtentMode.Exact, tensor.ExtentMode));
        }
    }

    [Fact]
    public void FusedSparseLinearAdmission_IsExact()
    {
        Assert.True(PtxFusedSparseLinearF32Kernel.SupportsShape(32, 1024, 1024, 16_384));
        Assert.False(PtxFusedSparseLinearF32Kernel.SupportsShape(16, 1024, 1024, 16_384));
        Assert.False(PtxFusedSparseLinearF32Kernel.SupportsShape(32, 512, 1024, 16_384));
        Assert.False(PtxFusedSparseLinearF32Kernel.SupportsShape(32, 1024, 1024, 8_192));
        Assert.True(PtxFusedSparseLinearF32Kernel.SupportsActivation(6));
        Assert.False(PtxFusedSparseLinearF32Kernel.SupportsActivation(7));
    }

    [Fact]
    public void TensorGatherEmitter_UsesExactCoalescedPointerOnlyAbi()
    {
        string ptx = PtxTensorGatherRowsF32Kernel.EmitPtx(8, 6);
        DirectPtxKernelBlueprint blueprint = PtxTensorGatherRowsF32Kernel.CreateBlueprint(
            DirectPtxArchitectureFamily.Ampere);

        Assert.Equal(3, Count(ptx, ".param .u64"));
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(1, Count(ptx, "ld.global.v4.f32"));
        Assert.Equal(1, Count(ptx, "st.global.v4.f32"));
        Assert.Equal(DirectPtxPhysicalLayout.RowGatherIndices, blueprint.Tensors[1].Layout);
        Assert.Equal(PtxTensorGatherRowsF32Kernel.Indices,
            blueprint.Tensors[1].PhysicalExtent.ElementCount);
        Assert.Equal("one-coalesced-float4-per-thread", blueprint.Semantics["transfer"]);
        Assert.Equal("0", blueprint.Semantics["workspace-bytes"]);
        Assert.All(blueprint.Tensors, tensor =>
            Assert.Equal(DirectPtxExtentMode.Exact, tensor.ExtentMode));
    }

    [Theory]
    [InlineData((int)DirectPtxTensorScatterReduceMode.Sum, "add.rn.f32")]
    [InlineData((int)DirectPtxTensorScatterReduceMode.Product, "mul.rn.f32")]
    [InlineData((int)DirectPtxTensorScatterReduceMode.Maximum, "max.f32")]
    [InlineData((int)DirectPtxTensorScatterReduceMode.Minimum, "min.f32")]
    public void TensorScatterReduceEmitter_BakesCasReduction(
        int modeValue,
        string reductionInstruction)
    {
        var mode = (DirectPtxTensorScatterReduceMode)modeValue;
        string ptx = PtxTensorScatterReduceF32Kernel.EmitPtx(8, 6, mode);
        DirectPtxKernelBlueprint blueprint = PtxTensorScatterReduceF32Kernel.CreateBlueprint(
            DirectPtxArchitectureFamily.Ampere, mode);

        Assert.Equal(3, Count(ptx, ".param .u64"));
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("atom.global.cas.b32", ptx, StringComparison.Ordinal);
        Assert.Contains(reductionInstruction, ptx, StringComparison.Ordinal);
        Assert.Equal(DirectPtxPhysicalLayout.OuterDimensionInner, blueprint.Tensors[0].Layout);
        Assert.Equal(DirectPtxPhysicalLayout.ElementScatterIndices, blueprint.Tensors[2].Layout);
        Assert.Equal("preserved-include-self", blueprint.Semantics["output-seed"]);
        Assert.Equal("0", blueprint.Semantics["workspace-bytes"]);
        Assert.All(blueprint.Tensors, tensor =>
            Assert.Equal(DirectPtxExtentMode.Exact, tensor.ExtentMode));
    }

    [Fact]
    public void TensorIndexingAdmission_IsExact()
    {
        Assert.True(PtxTensorGatherRowsF32Kernel.SupportsShape(16_384, 64));
        Assert.False(PtxTensorGatherRowsF32Kernel.SupportsShape(8_192, 64));
        Assert.False(PtxTensorGatherRowsF32Kernel.SupportsShape(16_384, 32));
        Assert.True(PtxTensorScatterReduceF32Kernel.SupportsShape(32, 512, 1024, 64));
        Assert.False(PtxTensorScatterReduceF32Kernel.SupportsShape(16, 512, 1024, 64));
        Assert.False(PtxTensorScatterReduceF32Kernel.SupportsShape(32, 512, 512, 64));
    }

    [Theory]
    [InlineData((int)DirectPtxTensorScatterHighLevelOperation.Mean, 4)]
    [InlineData((int)DirectPtxTensorScatterHighLevelOperation.AddBackward, 3)]
    public void TensorScatterHighLevelEmitter_HasDistinctPointerOnlyAbi(
        int operationValue,
        int pointerCount)
    {
        var operation = (DirectPtxTensorScatterHighLevelOperation)operationValue;
        string ptx = PtxTensorScatterHighLevelF32Kernel.EmitPtx(8, 6, operation);
        DirectPtxKernelBlueprint blueprint =
            PtxTensorScatterHighLevelF32Kernel.CreateBlueprint(
                DirectPtxArchitectureFamily.Ampere, operation);

        Assert.Equal(pointerCount, Count(ptx, ".param .u64"));
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("single-overwrite", blueprint.Semantics["output-write"]);
        Assert.Equal("0", blueprint.Semantics["workspace-bytes"]);
        Assert.Equal("0", blueprint.Semantics["intermediate-global-bytes"]);
        Assert.All(blueprint.Tensors, tensor =>
            Assert.Equal(DirectPtxExtentMode.Exact, tensor.ExtentMode));
    }

    [Fact]
    public void TensorScatterMeanEmitter_WritesPublicInt32Counts()
    {
        string ptx = PtxTensorScatterHighLevelF32Kernel.EmitPtx(
            8, 6, DirectPtxTensorScatterHighLevelOperation.Mean);
        DirectPtxKernelBlueprint blueprint =
            PtxTensorScatterHighLevelF32Kernel.CreateBlueprint(
                DirectPtxArchitectureFamily.Ampere,
                DirectPtxTensorScatterHighLevelOperation.Mean);

        Assert.Contains("aidotnet_tensor_scatter_mean_f32", ptx, StringComparison.Ordinal);
        Assert.Contains("st.global.u32", ptx, StringComparison.Ordinal);
        Assert.Equal(DirectPtxPhysicalType.Int32, blueprint.Tensors[3].PhysicalType);
        Assert.Equal("int32", blueprint.Semantics["counts-type"]);
    }

    [Fact]
    public void TensorScatterAddBackwardEmitter_GathersOnceAndStoresOnce()
    {
        string ptx = PtxTensorScatterHighLevelF32Kernel.EmitPtx(
            8, 6, DirectPtxTensorScatterHighLevelOperation.AddBackward);
        DirectPtxKernelBlueprint blueprint =
            PtxTensorScatterHighLevelF32Kernel.CreateBlueprint(
                DirectPtxArchitectureFamily.Ampere,
                DirectPtxTensorScatterHighLevelOperation.AddBackward);

        Assert.Contains("aidotnet_tensor_scatter_add_backward_f32", ptx, StringComparison.Ordinal);
        Assert.Equal(1, Count(ptx, "ld.global.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.Equal("zero", blueprint.Semantics["invalid-index"]);
    }

    [Fact]
    public void TensorScatterHighLevelAdmission_IsExact()
    {
        Assert.True(PtxTensorScatterHighLevelF32Kernel.SupportsShape(16_384, 64, 1_024));
        Assert.False(PtxTensorScatterHighLevelF32Kernel.SupportsShape(8_192, 64, 1_024));
        Assert.False(PtxTensorScatterHighLevelF32Kernel.SupportsShape(16_384, 32, 1_024));
        Assert.False(PtxTensorScatterHighLevelF32Kernel.SupportsShape(16_384, 64, 512));
    }

    [Fact]
    public void NeuralScatterMaxEmitter_IsDeterministicAndWritesInt32Argmax()
    {
        string ptx = PtxNeuralScatterMaxF32Kernel.EmitPtx(8, 6);
        DirectPtxKernelBlueprint blueprint =
            PtxNeuralScatterMaxF32Kernel.CreateBlueprint(
                DirectPtxArchitectureFamily.Ampere);

        Assert.Contains(PtxNeuralScatterMaxF32Kernel.EntryPoint, ptx);
        Assert.Equal(4, Count(ptx, ".param .u64"));
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("atom.", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.Contains("st.global.s32", ptx, StringComparison.Ordinal);
        Assert.Equal(DirectPtxPhysicalType.Int32, blueprint.Tensors[3].PhysicalType);
        Assert.Equal("lowest-source-row", blueprint.Semantics["tie-break"]);
        Assert.Equal("single-overwrite", blueprint.Semantics["output-write"]);
        Assert.Equal("0", blueprint.Semantics["workspace-bytes"]);
        Assert.All(blueprint.Tensors, tensor =>
            Assert.Equal(DirectPtxExtentMode.Exact, tensor.ExtentMode));
    }

    [Fact]
    public void NeuralScatterMaxAdmission_IsExact()
    {
        Assert.True(PtxNeuralScatterMaxF32Kernel.SupportsShape(16_384, 64, 1_024));
        Assert.False(PtxNeuralScatterMaxF32Kernel.SupportsShape(8_192, 64, 1_024));
        Assert.False(PtxNeuralScatterMaxF32Kernel.SupportsShape(16_384, 32, 1_024));
        Assert.False(PtxNeuralScatterMaxF32Kernel.SupportsShape(16_384, 64, 512));
    }

    [Fact]
    public void StructuredSparseMmaSpEmitter_UsesEightRealTensorCoreInstructions()
    {
        string ptx = PtxStructuredSparse2x4MmaSpF32Kernel.EmitPtx(8, 6);
        DirectPtxKernelBlueprint blueprint =
            PtxStructuredSparse2x4MmaSpF32Kernel.CreateBlueprint(
                DirectPtxArchitectureFamily.Ampere);

        Assert.Contains(PtxStructuredSparse2x4MmaSpF32Kernel.EntryPoint, ptx);
        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Equal(8, Count(ptx,
            "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32"));
        Assert.Equal(4, Count(ptx, ".param .u64"));
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(" bra ", ptx, StringComparison.Ordinal);
        Assert.Equal("8", blueprint.Semantics["tensor-core-operations-per-warp"]);
        Assert.Equal("fp32-registers", blueprint.Semantics["accumulator"]);
        Assert.Equal("0", blueprint.Semantics["workspace-bytes"]);
        Assert.Equal("0", blueprint.Semantics["intermediate-global-bytes"]);
        Assert.All(blueprint.Tensors, tensor =>
            Assert.Equal(DirectPtxExtentMode.Exact, tensor.ExtentMode));
    }

    [Fact]
    public void StructuredSparseMmaSpAdmission_IsExact()
    {
        Assert.True(PtxStructuredSparse2x4MmaSpF32Kernel.SupportsShape(256, 64, 256));
        Assert.False(PtxStructuredSparse2x4MmaSpF32Kernel.SupportsShape(128, 64, 256));
        Assert.False(PtxStructuredSparse2x4MmaSpF32Kernel.SupportsShape(256, 32, 256));
        Assert.False(PtxStructuredSparse2x4MmaSpF32Kernel.SupportsShape(256, 64, 128));
    }

    [Theory]
    [InlineData(8, 6, true)]
    [InlineData(8, 0, false)]
    [InlineData(8, 7, false)]
    [InlineData(8, 9, false)]
    [InlineData(9, 0, false)]
    [InlineData(10, 0, false)]
    public void SparseGraphArchitectureMatrix_FailsClosedOutsideExactSm86(
        int major,
        int minor,
        bool expected) =>
        Assert.Equal(expected, DirectPtxArchitecture.HasValidatedSparseGraph(major, minor));

    [Fact]
    public void CsrSpmmShapeMatrix_IsExactAndNeverPromotedWithoutHardwareEvidence()
    {
        Assert.True(PtxFusedCsrSpmmVec4F32Kernel.SupportsShape(1024, 1024, 64, 16384));
        Assert.False(PtxFusedCsrSpmmVec4F32Kernel.SupportsShape(1024, 1024, 32, 16384));
        Assert.False(PtxFusedCsrSpmmVec4F32Kernel.SupportsShape(1024, 1024, 64, 8192));
        Assert.False(PtxFusedCsrSpmmVec4F32Kernel.IsPromotedShape(1024, 1024, 64, 16384));
    }

    [Fact]
    public void SparseGraphExperimentOverride_IsThreadIsolated()
    {
        bool? previous = DirectPtxFeatureGate.SparseGraphExperimentOverride;
        try
        {
            DirectPtxFeatureGate.SparseGraphExperimentOverride = null;
            bool environmentValue = DirectPtxFeatureGate.IsSparseGraphEnabled;
            DirectPtxFeatureGate.SparseGraphExperimentOverride = true;
            bool childValue = true;
            var thread = new Thread(() => childValue = DirectPtxFeatureGate.IsSparseGraphEnabled);
            thread.Start();
            thread.Join();
            Assert.True(DirectPtxFeatureGate.IsSparseGraphEnabled);
            Assert.Equal(environmentValue, childValue);
        }
        finally
        {
            DirectPtxFeatureGate.SparseGraphExperimentOverride = previous;
        }
    }

    [SkippableFact]
    public void DriverOnlyCsrSpmm_MatchesFp64OracleAndReportsNoDriverLocalMemory()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedSparseGraph(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in sparse/graph specialization admits only SM86.");
        using var kernel = new PtxFusedCsrSpmmVec4F32Kernel(runtime);

        int[] rowPointers = Enumerable.Range(0, PtxFusedCsrSpmmVec4F32Kernel.Rows + 1)
            .Select(row => row * 16).ToArray();
        int[] columns = new int[PtxFusedCsrSpmmVec4F32Kernel.NonZeros];
        float[] values = new float[columns.Length];
        for (int row = 0; row < PtxFusedCsrSpmmVec4F32Kernel.Rows; row++)
        for (int item = 0; item < 16; item++)
        {
            int index = row * 16 + item;
            columns[index] = (row * 17 + item * 13) & 1023;
            values[index] = (item - 7.5f) / 32f;
        }
        float[] dense = Enumerable.Range(0,
                PtxFusedCsrSpmmVec4F32Kernel.Inner * PtxFusedCsrSpmmVec4F32Kernel.Columns)
            .Select(index => ((index * 19) % 101 - 50) / 128f).ToArray();
        double[] expected = Oracle(rowPointers, columns, values, dense);

        using var valuesBuffer = runtime.AllocateBytes((nuint)(values.Length * sizeof(float)));
        using var columnsBuffer = runtime.AllocateBytes((nuint)(columns.Length * sizeof(int)));
        using var rowPointersBuffer = runtime.AllocateBytes((nuint)(rowPointers.Length * sizeof(int)));
        using var denseBuffer = runtime.AllocateBytes((nuint)(dense.Length * sizeof(float)));
        using var outputBuffer = runtime.AllocateBytes((nuint)(expected.Length * sizeof(float)));
        valuesBuffer.Upload<float>(values);
        columnsBuffer.Upload<int>(columns);
        rowPointersBuffer.Upload<int>(rowPointers);
        denseBuffer.Upload<float>(dense);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(valuesBuffer, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(columnsBuffer, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(rowPointersBuffer, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(denseBuffer, kernel.Blueprint.Tensors[3]),
            DirectPtxTensorView.CreateOwned(outputBuffer, kernel.Blueprint.Tensors[4]));
        runtime.Synchronize();
        var actual = new float[expected.Length];
        outputBuffer.Download<float>(actual);
        double maxError = actual.Select((value, index) => Math.Abs(value - expected[index])).Max();
        Assert.True(maxError <= 2e-5, $"maximum error {maxError:R}");
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
    }

    [SkippableFact]
    public void DriverOnlySddmm_MatchesFp64OracleAndReportsNoDriverLocalMemory()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedSparseGraph(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in sparse/graph specialization admits only SM86.");
        using var kernel = new PtxFusedSddmmF32Kernel(runtime);

        int[] rows = new int[PtxFusedSddmmF32Kernel.NonZeros];
        int[] columns = new int[rows.Length];
        for (int p = 0; p < rows.Length; p++)
        {
            rows[p] = (p * 17) & 1023;
            columns[p] = (p * 29 + 7) & 1023;
        }
        float[] x = Enumerable.Range(0, PtxFusedSddmmF32Kernel.Rows * PtxFusedSddmmF32Kernel.Inner)
            .Select(index => ((index * 19) % 101 - 50) / 128f).ToArray();
        float[] y = Enumerable.Range(0, PtxFusedSddmmF32Kernel.Columns * PtxFusedSddmmF32Kernel.Inner)
            .Select(index => ((index * 23) % 97 - 48) / 128f).ToArray();
        double[] expected = new double[rows.Length];
        for (int p = 0; p < rows.Length; p++)
        for (int k = 0; k < PtxFusedSddmmF32Kernel.Inner; k++)
            expected[p] += (double)x[rows[p] * PtxFusedSddmmF32Kernel.Inner + k] *
                y[columns[p] * PtxFusedSddmmF32Kernel.Inner + k];

        using var rowBuffer = runtime.AllocateBytes((nuint)(rows.Length * sizeof(int)));
        using var columnBuffer = runtime.AllocateBytes((nuint)(columns.Length * sizeof(int)));
        using var xBuffer = runtime.AllocateBytes((nuint)(x.Length * sizeof(float)));
        using var yBuffer = runtime.AllocateBytes((nuint)(y.Length * sizeof(float)));
        using var outputBuffer = runtime.AllocateBytes((nuint)(expected.Length * sizeof(float)));
        rowBuffer.Upload<int>(rows);
        columnBuffer.Upload<int>(columns);
        xBuffer.Upload<float>(x);
        yBuffer.Upload<float>(y);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(rowBuffer, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(columnBuffer, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(xBuffer, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(yBuffer, kernel.Blueprint.Tensors[3]),
            DirectPtxTensorView.CreateOwned(outputBuffer, kernel.Blueprint.Tensors[4]));
        runtime.Synchronize();
        var actual = new float[expected.Length];
        outputBuffer.Download<float>(actual);
        double maxError = actual.Select((value, index) => Math.Abs(value - expected[index])).Max();
        Assert.True(maxError <= 2e-5, $"maximum error {maxError:R}");
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
    }

    private static double[] Oracle(int[] rowPointers, int[] columns, float[] values, float[] dense)
    {
        var output = new double[PtxFusedCsrSpmmVec4F32Kernel.Rows * PtxFusedCsrSpmmVec4F32Kernel.Columns];
        for (int row = 0; row < PtxFusedCsrSpmmVec4F32Kernel.Rows; row++)
        for (int col = 0; col < PtxFusedCsrSpmmVec4F32Kernel.Columns; col++)
        {
            double sum = 0;
            for (int p = rowPointers[row]; p < rowPointers[row + 1]; p++)
                sum += (double)values[p] * dense[columns[p] * PtxFusedCsrSpmmVec4F32Kernel.Columns + col];
            output[row * PtxFusedCsrSpmmVec4F32Kernel.Columns + col] = sum;
        }
        return output;
    }

    private static DirectPtxSparseOptimizerKey CreateSparseOptimizerKey(
        DirectPtxSparseOptimizerOperation operation) => operation switch
    {
        DirectPtxSparseOptimizerOperation.Sgd =>
            DirectPtxSparseOptimizerKey.Create(operation, 1e-3f, 1e-4f),
        DirectPtxSparseOptimizerOperation.SgdMomentum or DirectPtxSparseOptimizerOperation.Nag =>
            DirectPtxSparseOptimizerKey.Create(operation, 1e-3f, 0.9f, 1e-4f),
        DirectPtxSparseOptimizerOperation.Adam or DirectPtxSparseOptimizerOperation.AdamW or
        DirectPtxSparseOptimizerOperation.Amsgrad or DirectPtxSparseOptimizerOperation.Adamax or
        DirectPtxSparseOptimizerOperation.Nadam =>
            DirectPtxSparseOptimizerKey.Create(
                operation, 1e-3f, 0.9f, 0.999f, 1e-8f, 1e-4f, 7),
        DirectPtxSparseOptimizerOperation.Rmsprop =>
            DirectPtxSparseOptimizerKey.Create(operation, 1e-3f, 0.9f, 1e-8f, 1e-4f),
        DirectPtxSparseOptimizerOperation.Adagrad =>
            DirectPtxSparseOptimizerKey.Create(operation, 1e-3f, 1e-8f, 1e-4f),
        DirectPtxSparseOptimizerOperation.Adadelta =>
            DirectPtxSparseOptimizerKey.Create(operation, 0.9f, 1e-8f, 1e-4f),
        DirectPtxSparseOptimizerOperation.Lion =>
            DirectPtxSparseOptimizerKey.Create(operation, 1e-3f, 0.9f, 0.99f, 1e-4f),
        DirectPtxSparseOptimizerOperation.Ftrl =>
            DirectPtxSparseOptimizerKey.Create(operation, 1e-2f, 0.1f, 0.1f, 1f),
        DirectPtxSparseOptimizerOperation.ProximalL1 =>
            DirectPtxSparseOptimizerKey.Create(operation, 1e-3f, 0.1f),
        _ => throw new ArgumentOutOfRangeException(nameof(operation))
    };

    private static int Count(string source, string value) =>
        source.Split(value, StringSplitOptions.None).Length - 1;

    private static void AssertPtxControlFlowClosed(string ptx)
    {
        Assert.Equal(Count(ptx, "{"), Count(ptx, "}"));
        string[] lines = ptx.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
        var labels = lines.Select(line => line.Trim())
            .Where(line => line.EndsWith(":", StringComparison.Ordinal))
            .Select(line => line[..^1]).ToHashSet(StringComparer.Ordinal);
        foreach (string line in lines)
        {
            int branch = line.IndexOf("bra ", StringComparison.Ordinal);
            if (branch < 0) continue;
            int start = branch + 4;
            int end = line.IndexOf(';', start);
            Assert.True(end > start, $"Malformed PTX branch: {line}");
            Assert.Contains(line[start..end].Trim(), labels);
        }
    }
}
#endif
