using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Componentwise octonion addition <c>out[i] = a[i] + b[i]</c> over contiguous 8-component
/// octonions (issue #854). One thread owns one octonion and adds its eight lanes — no shared
/// memory, no reduction, exact. 256 threads/block, grid = count/256 (positive multiple of 256).
/// Buffers are contiguous of length <c>8·count</c> floats.
/// </summary>
internal sealed class PtxOctonionAddKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const int Components = 8;
    internal const int MaxCount = 2048 * 4096;
    internal const string EntryPoint = "aidotnet_octonion_add";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Count { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxOctonionAddKernel(DirectPtxRuntime runtime, int count)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in octonion-add specialization is measured only on GA10x/SM86.");
        ValidateShape(count);
        Count = count;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, count);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, count);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(DirectPtxTensorView a, DirectPtxTensorView b, DirectPtxTensorView output)
    {
        Require(a, Blueprint.Tensors[0], nameof(a));
        Require(b, Blueprint.Tensors[1], nameof(b));
        Require(output, Blueprint.Tensors[2], nameof(output));

        IntPtr aPointer = a.Pointer;
        IntPtr bPointer = b.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &aPointer;
        arguments[1] = &bPointer;
        arguments[2] = &outputPointer;
        _module.Launch(_function, (uint)(Count / BlockThreads), 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int count)
    {
        ValidateShape(count);
        var ptx = new StringBuilder(4_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// octonion-add count={count}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 a_ptr,");
        ptx.AppendLine("    .param .u64 b_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .b32 %r<8>;");
        ptx.AppendLine("    .reg .b64 %rd<10>;");
        ptx.AppendLine("    .reg .f32 %f<8>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [a_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [b_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
        ptx.AppendLine($"    mul.wide.u32 %rd3, %r2, {Components * sizeof(float)};"); // octonion byte offset
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd3;");
        ptx.AppendLine("    add.u64 %rd6, %rd2, %rd3;");
        // Each octonion is 32 bytes at a 16-byte-aligned base (16-byte-aligned buffer + count*32),
        // so both 4-component halves are 16-byte aligned and a single v4.f32 moves four lanes.
        for (int half = 0; half < 2; half++)
        {
            int off = half * 4 * sizeof(float);
            ptx.AppendLine($"    ld.global.nc.v4.f32 {{%f0, %f1, %f2, %f3}}, [%rd4+{off}];");
            ptx.AppendLine($"    ld.global.nc.v4.f32 {{%f4, %f5, %f6, %f7}}, [%rd5+{off}];");
            ptx.AppendLine("    add.rn.f32 %f0, %f0, %f4;");
            ptx.AppendLine("    add.rn.f32 %f1, %f1, %f5;");
            ptx.AppendLine("    add.rn.f32 %f2, %f2, %f6;");
            ptx.AppendLine("    add.rn.f32 %f3, %f3, %f7;");
            ptx.AppendLine($"    st.global.v4.f32 [%rd6+{off}], {{%f0, %f1, %f2, %f3}};");
        }
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, int count)
    {
        var extent = new DirectPtxExtent(count, Components);
        return new DirectPtxKernelBlueprint(
            Operation: "octonion-add",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp32-count{count}",
            Tensors:
            [
                new("a", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("b", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 12,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "out[i][k] = a[i][k] + b[i][k] for k in 0..7",
                ["layout"] = "contiguous-8-component-octonions",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedCount(int count) =>
        count > 0 && count % BlockThreads == 0 && count <= MaxCount;

    internal static bool IsPromotedCount(int count) => false;

    private static void ValidateShape(int count)
    {
        if (!IsSupportedCount(count))
            throw new ArgumentOutOfRangeException(
                nameof(count),
                $"Octonion add supports a positive octonion count that is a multiple of {BlockThreads} up to {MaxCount}.");
    }

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent || view.ByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
    }
}
