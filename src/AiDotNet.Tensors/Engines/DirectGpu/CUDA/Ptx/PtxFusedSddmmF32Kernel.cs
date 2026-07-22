#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact-shape FP32 SDDMM specialization. One thread owns one sampled entry,
/// retains its dot-product accumulator in a register, and writes the result
/// directly without a workspace or intermediate global-memory round trip.
/// </summary>
internal sealed class PtxFusedSddmmF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_sddmm_f32_m1024_n1024_k64_nnz16384";
    internal const int Rows = 1024;
    internal const int Columns = 1024;
    internal const int Inner = 64;
    internal const int NonZeros = 16384;
    internal const int BlockThreads = 256;
    internal const int GridBlocks = NonZeros / BlockThreads;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }
    internal string JitInfoLog => _module.JitInfoLog;

    internal PtxFusedSddmmF32Kernel(DirectPtxRuntime runtime)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.HasValidatedSparseGraph(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"SDDMM has no SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} experimental specialization.");

        Blueprint = CreateBlueprint(runtime.ArchitectureFamily);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo,
            BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal static bool SupportsShape(int rows, int columns, int inner, int nonZeros) =>
        rows == Rows && columns == Columns && inner == Inner && nonZeros == NonZeros;

    internal static bool IsPromotedShape(int rows, int columns, int inner, int nonZeros) => false;

    internal unsafe void Launch(
        DirectPtxTensorView rowIndices,
        DirectPtxTensorView columnIndices,
        DirectPtxTensorView x,
        DirectPtxTensorView y,
        DirectPtxTensorView output)
    {
        Require(rowIndices, Blueprint.Tensors[0], nameof(rowIndices));
        Require(columnIndices, Blueprint.Tensors[1], nameof(columnIndices));
        Require(x, Blueprint.Tensors[2], nameof(x));
        Require(y, Blueprint.Tensors[3], nameof(y));
        Require(output, Blueprint.Tensors[4], nameof(output));
        if (Overlaps(output, rowIndices) || Overlaps(output, columnIndices) ||
            Overlaps(output, x) || Overlaps(output, y))
            throw new ArgumentException("SDDMM output must be disjoint from every input.", nameof(output));

        IntPtr rowPointer = rowIndices.Pointer;
        IntPtr columnPointer = columnIndices.Pointer;
        IntPtr xPointer = x.Pointer;
        IntPtr yPointer = y.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[5];
        arguments[0] = &rowPointer;
        arguments[1] = &columnPointer;
        arguments[2] = &xPointer;
        arguments[3] = &yPointer;
        arguments[4] = &outputPointer;
        _module.Launch(_function, GridBlocks, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture)
    {
        var pattern = new DirectPtxExtent(NonZeros);
        var x = new DirectPtxExtent(Rows, Inner);
        var y = new DirectPtxExtent(Columns, Inner);
        return new DirectPtxKernelBlueprint(
            Operation: "sddmm-f32",
            Version: 1,
            Architecture: architecture,
            Variant: "m1024-n1024-k64-nnz16384-row-major",
            Tensors:
            [
                new("row-indices", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.CooRowIndices,
                    pattern, pattern, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("column-indices", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.CooColumnIndices,
                    pattern, pattern, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("x", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    x, x, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("y", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    y, y, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    pattern, pattern, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(48, 0, 0, 4),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "output[p]=dot(x[row[p],:],y[column[p],:])",
                ["coo-index-base"] = "zero",
                ["coo-index-width"] = "int32",
                ["reduction-order"] = "ascending-inner-index",
                ["accumulator"] = "fp32-fma",
                ["dense-layout"] = "row-major-contiguous",
                ["intermediate-global-bytes"] = "0",
                ["workspace-bytes"] = "0"
            });
    }

    internal static string EmitPtx(int ccMajor, int ccMinor)
    {
        if (ccMajor <= 0 || ccMinor < 0) throw new ArgumentOutOfRangeException(nameof(ccMajor));
        var ptx = new StringBuilder(8192);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 row_indices_ptr,");
        ptx.AppendLine("    .param .u64 column_indices_ptr,");
        ptx.AppendLine("    .param .u64 x_ptr,");
        ptx.AppendLine("    .param .u64 y_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .b32 %r<8>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<10>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [row_indices_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [column_indices_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [x_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [y_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd4, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd0, %rd5;");
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd5;");
        ptx.AppendLine("    ld.global.u32 %r3, [%rd6];");
        ptx.AppendLine("    ld.global.u32 %r4, [%rd7];");
        ptx.AppendLine("    shl.b32 %r5, %r3, 8;");
        ptx.AppendLine("    shl.b32 %r6, %r4, 8;");
        ptx.AppendLine("    cvt.u64.u32 %rd8, %r5;");
        ptx.AppendLine("    cvt.u64.u32 %rd9, %r6;");
        ptx.AppendLine("    add.u64 %rd10, %rd2, %rd8;");
        ptx.AppendLine("    add.u64 %rd11, %rd3, %rd9;");
        ptx.AppendLine("    mov.f32 %f8, 0f00000000;");
        for (int byteOffset = 0; byteOffset < Inner * sizeof(float); byteOffset += 4 * sizeof(float))
        {
            ptx.AppendLine($"    ld.global.v4.f32 {{%f0, %f1, %f2, %f3}}, [%rd10+{byteOffset}];");
            ptx.AppendLine($"    ld.global.v4.f32 {{%f4, %f5, %f6, %f7}}, [%rd11+{byteOffset}];");
            ptx.AppendLine("    fma.rn.f32 %f8, %f0, %f4, %f8;");
            ptx.AppendLine("    fma.rn.f32 %f8, %f1, %f5, %f8;");
            ptx.AppendLine("    fma.rn.f32 %f8, %f2, %f6, %f8;");
            ptx.AppendLine("    fma.rn.f32 %f8, %f3, %f7, %f8;");
        }
        ptx.AppendLine("    add.u64 %rd12, %rd4, %rd5;");
        ptx.AppendLine("    st.global.f32 [%rd12], %f8;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent || view.ByteLength != contract.RequiredBytes ||
            view.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException($"{parameter} does not satisfy exact physical ABI '{contract.Name}'.", parameter);
    }

    private static bool Overlaps(DirectPtxTensorView left, DirectPtxTensorView right)
    {
        nuint leftStart = (nuint)left.Pointer;
        nuint rightStart = (nuint)right.Pointer;
        nuint leftEnd = checked(leftStart + left.ByteLength);
        nuint rightEnd = checked(rightStart + right.ByteLength);
        return leftStart < rightEnd && rightStart < leftEnd;
    }

    public void Dispose() => _module.Dispose();
}
#endif
