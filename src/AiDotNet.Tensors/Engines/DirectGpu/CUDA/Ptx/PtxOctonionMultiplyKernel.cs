using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Octonion multiply <c>r = a * b</c> over contiguous 8-component octonions (issue #854),
/// using the explicit Cayley-Dickson multiplication table that matches the established NVRTC
/// kernel exactly. One thread owns one octonion: it loads both operands into registers and
/// emits each of the eight output components as a fully register-resident sum of eight signed
/// products — no shared memory, no reduction, no global intermediate.
///
/// 256 threads/block, grid = count/256 (positive multiple of 256). Buffers are contiguous of
/// length <c>8·count</c> floats.
/// </summary>
internal sealed class PtxOctonionMultiplyKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const int Components = 8;
    internal const int MaxCount = 2048 * 4096;
    internal const string EntryPoint = "aidotnet_octonion_multiply";

    // r[k] = sum_i Sign[k][i] * a[i] * b[BIndex[k][i]] — the explicit Cayley-Dickson table.
    private static readonly int[][] BIndex =
    [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 0, 3, 2, 5, 4, 7, 6],
        [2, 3, 0, 1, 6, 7, 4, 5],
        [3, 2, 1, 0, 7, 6, 5, 4],
        [4, 5, 6, 7, 0, 1, 2, 3],
        [5, 4, 7, 6, 1, 0, 3, 2],
        [6, 7, 4, 5, 2, 3, 0, 1],
        [7, 6, 5, 4, 3, 2, 1, 0]
    ];
    private static readonly int[][] Sign =
    [
        [1, -1, -1, -1, -1, -1, -1, -1],
        [1,  1,  1, -1,  1, -1, -1,  1],
        [1, -1,  1,  1,  1,  1, -1, -1],
        [1,  1, -1,  1,  1, -1,  1, -1],
        [1, -1, -1, -1,  1,  1,  1,  1],
        [1,  1, -1,  1, -1,  1, -1,  1],
        [1,  1,  1, -1, -1,  1,  1, -1],
        [1, -1,  1,  1, -1, -1,  1,  1]
    ];

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Count { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxOctonionMultiplyKernel(DirectPtxRuntime runtime, int count)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in octonion-multiply specialization is measured only on GA10x/SM86.");
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
        var ptx = new StringBuilder(8_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// octonion-multiply count={count}");
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
        ptx.AppendLine("    .reg .f32 %f<20>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [a_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [b_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
        ptx.AppendLine($"    mul.wide.u32 %rd3, %r2, {Components * sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");     // &a[oct]
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd3;");     // &b[oct]
        ptx.AppendLine("    add.u64 %rd6, %rd2, %rd3;");     // &out[oct]
        // a -> %f0..%f7, b -> %f8..%f15.
        for (int i = 0; i < Components; i++)
            ptx.AppendLine($"    ld.global.nc.f32 %f{i}, [%rd4+{i * sizeof(float)}];");
        for (int i = 0; i < Components; i++)
            ptx.AppendLine($"    ld.global.nc.f32 %f{8 + i}, [%rd5+{i * sizeof(float)}];");
        // Each output component: register-resident signed sum of eight products.
        for (int k = 0; k < Components; k++)
        {
            // First term (Sign[k][0] is always +1): acc = a[0] * b[BIndex[k][0]].
            ptx.AppendLine($"    mul.rn.f32 %f16, %f0, %f{8 + BIndex[k][0]};");
            for (int i = 1; i < Components; i++)
            {
                string bReg = $"%f{8 + BIndex[k][i]}";
                if (Sign[k][i] > 0)
                    ptx.AppendLine($"    fma.rn.f32 %f16, %f{i}, {bReg}, %f16;");
                else
                {
                    ptx.AppendLine($"    mul.rn.f32 %f17, %f{i}, {bReg};");
                    ptx.AppendLine("    sub.rn.f32 %f16, %f16, %f17;");
                }
            }
            ptx.AppendLine($"    st.global.f32 [%rd6+{k * sizeof(float)}], %f16;");
        }
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, int count)
    {
        var extent = new DirectPtxExtent(count, Components);
        return new DirectPtxKernelBlueprint(
            Operation: "octonion-multiply",
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
                MaxRegistersPerThread: 40,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "r = a * b via the explicit Cayley-Dickson octonion table",
                ["layout"] = "contiguous-8-component-octonions",
                ["register-resident"] = "both-operands-and-all-8-outputs",
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
                $"Octonion multiply supports a positive octonion count that is a multiple of {BlockThreads} up to {MaxCount}.");
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
