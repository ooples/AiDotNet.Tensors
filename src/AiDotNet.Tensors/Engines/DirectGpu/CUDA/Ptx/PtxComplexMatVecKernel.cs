using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Batched complex matrix-vector product <c>y[b] = A * x[b]</c> for a shared <c>[dim,dim]</c> complex
/// matrix and per-batch complex vectors (issue #854), matching the NVRTC <c>complex_matvec</c> kernel.
/// One warp owns one (batch, row) output element, coalesces the <c>dim</c> columns, and reduces the
/// complex product <c>(mr+mi j)(xr+xi j) = (mr*xr - mi*xi) + (mr*xi + mi*xr) j</c>
/// with shuffles.
///
/// The matrix is stored once (<c>matReal</c>/<c>matImag</c>, <c>[dim,dim]</c> row-major) and reused
/// across the batch; vectors and outputs are split real/imag <c>[batch,dim]</c>. Shape (batch, dim)
/// is baked into the PTX, so the launch takes buffer pointers only. 256 threads/block (8 rows/block),
/// grid = (batch*dim)/8, required to divide evenly.
/// </summary>
internal sealed class PtxComplexMatVecKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const int WarpsPerBlock = BlockThreads / 32;
    internal const int MaxRows = 2048 * 4096;
    internal const int MaxDim = 4096;
    internal const string EntryPoint = "aidotnet_complex_matvec";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int BatchSize { get; }
    internal int Dim { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxComplexMatVecKernel(DirectPtxRuntime runtime, int batchSize, int dim)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in complex-matvec specialization is measured only on GA10x/SM86.");
        ValidateShape(batchSize, dim);
        BatchSize = batchSize;
        Dim = dim;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, batchSize, dim);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, batchSize, dim);
        var loaded = DirectPtxResourceInitialization.Complete(
            runtime.LoadModule(Ptx),
            module =>
            {
                IntPtr function = module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
                int activeBlocks = module.GetActiveBlocksPerMultiprocessor(function, BlockThreads);
                Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
                var audit = DirectPtxKernelAudit.Create(
                    Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks,
                    module);
                return (Function: function, Audit: audit);
            });
        _module = loaded.Resource;
        _function = loaded.Value.Function;
        Audit = loaded.Value.Audit;
    }

    internal unsafe void Launch(
        DirectPtxTensorView matReal, DirectPtxTensorView matImag,
        DirectPtxTensorView vecReal, DirectPtxTensorView vecImag,
        DirectPtxTensorView outReal, DirectPtxTensorView outImag)
    {
        DirectPtxAbi.Require(matReal, Blueprint.Tensors[0], nameof(matReal));
        DirectPtxAbi.Require(matImag, Blueprint.Tensors[1], nameof(matImag));
        DirectPtxAbi.Require(vecReal, Blueprint.Tensors[2], nameof(vecReal));
        DirectPtxAbi.Require(vecImag, Blueprint.Tensors[3], nameof(vecImag));
        DirectPtxAbi.Require(outReal, Blueprint.Tensors[4], nameof(outReal));
        DirectPtxAbi.Require(outImag, Blueprint.Tensors[5], nameof(outImag));

        IntPtr matRealPointer = matReal.Pointer;
        IntPtr matImagPointer = matImag.Pointer;
        IntPtr vecRealPointer = vecReal.Pointer;
        IntPtr vecImagPointer = vecImag.Pointer;
        IntPtr outRealPointer = outReal.Pointer;
        IntPtr outImagPointer = outImag.Pointer;
        void** arguments = stackalloc void*[6];
        arguments[0] = &matRealPointer;
        arguments[1] = &matImagPointer;
        arguments[2] = &vecRealPointer;
        arguments[3] = &vecImagPointer;
        arguments[4] = &outRealPointer;
        arguments[5] = &outImagPointer;
        uint grid = (uint)((BatchSize * Dim) / WarpsPerBlock);
        _module.Launch(_function, grid, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int batchSize, int dim)
    {
        ValidateShape(batchSize, dim);

        var ptx = new StringBuilder(4_500);
        DirectPtxPtxText.AppendModuleHeader(ptx, ccMajor, ccMinor, disableLoopUnrolling: true);
        ptx.AppendLine($"// complex-matvec batch={batchSize} dim={dim}; one output row per warp");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 mat_real_ptr,");
        ptx.AppendLine("    .param .u64 mat_imag_ptr,");
        ptx.AppendLine("    .param .u64 vec_real_ptr,");
        ptx.AppendLine("    .param .u64 vec_imag_ptr,");
        ptx.AppendLine("    .param .u64 out_real_ptr,");
        ptx.AppendLine("    .param .u64 out_imag_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<3>;");
        ptx.AppendLine("    .reg .b32 %r<14>;");
        ptx.AppendLine("    .reg .b64 %rd<24>;");
        ptx.AppendLine("    .reg .f32 %f<12>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [mat_real_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [mat_imag_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [vec_real_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [vec_imag_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd4, [out_real_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd5, [out_imag_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine("    shr.u32 %r2, %r0, 5;");                        // warp in block
        ptx.AppendLine("    and.b32 %r3, %r0, 31;");                       // lane
        ptx.AppendLine($"    mad.lo.u32 %r4, %r1, {WarpsPerBlock}, %r2;"); // idx = b*dim + row
        ptx.AppendLine($"    div.u32 %r5, %r4, {dim};");                   // b
        ptx.AppendLine($"    rem.u32 %r6, %r4, {dim};");                   // row
        ptx.AppendLine($"    mad.lo.u32 %r7, %r6, {dim}, %r3;");           // row*dim + lane
        ptx.AppendLine($"    mad.lo.u32 %r8, %r5, {dim}, %r3;");           // b*dim + lane
        ptx.AppendLine("    mul.wide.u32 %rd6, %r7, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd0, %rd6;");                   // &matReal[row*dim]
        ptx.AppendLine("    add.u64 %rd9, %rd1, %rd6;");                   // &matImag[row*dim]
        ptx.AppendLine("    mul.wide.u32 %rd7, %r8, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd2, %rd7;");                  // &vecReal[b*dim]
        ptx.AppendLine("    add.u64 %rd11, %rd3, %rd7;");                  // &vecImag[b*dim]
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");                   // sumReal
        ptx.AppendLine("    mov.f32 %f1, 0f00000000;");                   // sumImag
        ptx.AppendLine("    mov.u32 %r9, %r3;");                          // col = lane
        ptx.AppendLine("$CMV_COL_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r9, {dim};");
        ptx.AppendLine("    @%p0 bra $CMV_REDUCE;");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd8];");   // mr
        ptx.AppendLine("    ld.global.nc.f32 %f3, [%rd9];");   // mi
        ptx.AppendLine("    ld.global.nc.f32 %f4, [%rd10];");  // xr
        ptx.AppendLine("    ld.global.nc.f32 %f5, [%rd11];");  // xi
        ptx.AppendLine("    fma.rn.f32 %f0, %f2, %f4, %f0;");  // sumReal += mr*xr
        ptx.AppendLine("    neg.f32 %f6, %f3;");
        ptx.AppendLine("    fma.rn.f32 %f0, %f6, %f5, %f0;");  // sumReal += -mi*xi
        ptx.AppendLine("    fma.rn.f32 %f1, %f2, %f5, %f1;");  // sumImag += mr*xi
        ptx.AppendLine("    fma.rn.f32 %f1, %f3, %f4, %f1;");  // sumImag += mi*xr
        ptx.AppendLine("    add.u64 %rd8, %rd8, 128;");
        ptx.AppendLine("    add.u64 %rd9, %rd9, 128;");
        ptx.AppendLine("    add.u64 %rd10, %rd10, 128;");
        ptx.AppendLine("    add.u64 %rd11, %rd11, 128;");
        ptx.AppendLine("    add.u32 %r9, %r9, 32;");
        ptx.AppendLine("    bra $CMV_COL_LOOP;");
        ptx.AppendLine("$CMV_REDUCE:");
        DirectPtxPtxText.AppendWarpSum(ptx, "%f0", "%r11", "%r12", "%f9");
        DirectPtxPtxText.AppendWarpSum(ptx, "%f1", "%r11", "%r12", "%f9");
        ptx.AppendLine("    setp.ne.u32 %p1, %r3, 0;");
        ptx.AppendLine("    @%p1 bra $CMV_END;");
        ptx.AppendLine("    mul.wide.u32 %rd12, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd13, %rd4, %rd12;");
        ptx.AppendLine("    add.u64 %rd14, %rd5, %rd12;");
        ptx.AppendLine("    st.global.f32 [%rd13], %f0;");                // outReal[idx]
        ptx.AppendLine("    st.global.f32 [%rd14], %f1;");                // outImag[idx]
        ptx.AppendLine("$CMV_END:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int batchSize, int dim)
    {
        var matExtent = new DirectPtxExtent(dim * dim);
        var vecExtent = new DirectPtxExtent(batchSize * dim);
        return new DirectPtxKernelBlueprint(
            Operation: "complex-matvec",
            Version: 2,
            Architecture: architecture,
            Variant: $"fp32-b{batchSize}-d{dim}",
            Tensors:
            [
                new("matReal", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    matExtent, matExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("matImag", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    matExtent, matExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("vecReal", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    vecExtent, vecExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("vecImag", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    vecExtent, vecExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("outReal", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    vecExtent, vecExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact),
                new("outImag", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    vecExtent, vecExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: DirectPtxResourceBudget.FromDriverMeasurement(
                measuredRegistersPerThread: 24,
                maxStaticSharedBytes: 0,
                maxLocalBytesPerThread: 0,
                minBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "outReal[b,row]+j outImag[b,row] = sum_col (mat[row,col] * vec[b,col]) over complex",
                ["matrix-sharing"] = "single [dim,dim] complex matrix reused across the batch",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int batchSize, int dim)
    {
        if (batchSize <= 0 || dim <= 0 || dim > MaxDim) return false;
        long rows = (long)batchSize * dim;
        return rows > 0 && rows % WarpsPerBlock == 0 && rows <= MaxRows;
    }

    internal static bool IsPromotedShape(int batchSize, int dim) => false;

    private static void ValidateShape(int batchSize, int dim)
    {
        if (!IsSupportedShape(batchSize, dim))
            throw new ArgumentOutOfRangeException(
                nameof(batchSize),
                $"Complex matvec requires positive dims with dim<={MaxDim} and (batch*dim) a multiple of {WarpsPerBlock} up to {MaxRows}.");
    }

}
