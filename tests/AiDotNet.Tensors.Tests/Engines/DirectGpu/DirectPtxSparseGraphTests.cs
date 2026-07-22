#if !NETFRAMEWORK
using System;
using System.Linq;
using System.Threading;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class DirectPtxSparseGraphTests
{
    [Fact]
    public void CompletionLedger_IsUniqueExplicitAndBlocksPrematurePr()
    {
        Assert.True(DirectPtxSparseGraphCompletionLedger.All.Count >= 100);
        Assert.Equal(DirectPtxSparseGraphCompletionLedger.All.Count,
            DirectPtxSparseGraphCompletionLedger.All
                .Select(entry => entry.Operation).Distinct(StringComparer.Ordinal).Count());
        Assert.Equal(45, DirectPtxSparseGraphCompletionLedger.All.Count(entry =>
            entry.Status == DirectPtxSparseGraphCompletionStatus.ImplementedDirectPtx));
        Assert.False(DirectPtxSparseGraphCompletionLedger.IsComplete);
        Assert.Throws<InvalidOperationException>(DirectPtxSparseGraphCompletionLedger.RequireComplete);
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

    private static int Count(string source, string value) =>
        source.Split(value, StringSplitOptions.None).Length - 1;
}
#endif
