using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Fuses <c>masked = mask != 0 ? fill : input</c> with numerically stable row softmax for the
/// measured 1024-column family. A 64-thread block keeps four selected float4 packs per lane
/// live through both reductions, eliminating the standalone masked-fill output and every
/// subsequent read of that intermediate. This is an oracle candidate, not a promoted dispatch.
/// </summary>
internal sealed class PtxMaskedSoftmaxKernel : IDisposable
{
    internal const int BlockThreads = 64;
    internal const string EntryPoint = "aidotnet_masked_softmax_row";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int M { get; }
    internal int N { get; }
    internal float FillValue { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxMaskedSoftmaxKernel(DirectPtxRuntime runtime, int m, int n, float fillValue)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedSoftmax(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in masked-softmax candidate is measured only on GA10x/SM86.");
        ValidateShape(m, n);
        M = m;
        N = n;
        FillValue = fillValue;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, m, n, fillValue);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, m, n);
        var loaded = DirectPtxResourceInitialization.Complete(
            runtime.LoadModule(Ptx),
            module =>
            {
                IntPtr function = module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
                int activeBlocks = module.GetActiveBlocksPerMultiprocessor(function, BlockThreads);
                Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
                DirectPtxKernelAudit audit = DirectPtxKernelAudit.Create(
                    Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks,
                    module.JitInfoLog);
                return (Function: function, Audit: audit);
            });
        _module = loaded.Resource;
        _function = loaded.Value.Function;
        Audit = loaded.Value.Audit;
    }

    internal unsafe void Launch(
        DirectPtxTensorView input, DirectPtxTensorView mask, DirectPtxTensorView output)
    {
        PtxAbiGuard.Require(input, Blueprint.Tensors[0], nameof(input));
        PtxAbiGuard.Require(mask, Blueprint.Tensors[1], nameof(mask));
        PtxAbiGuard.Require(output, Blueprint.Tensors[2], nameof(output));

        IntPtr inputPointer = input.Pointer;
        IntPtr maskPointer = mask.Pointer;
        IntPtr outputPointer = output.Pointer;
        float fill = FillValue;
        void** arguments = stackalloc void*[4];
        arguments[0] = &inputPointer;
        arguments[1] = &maskPointer;
        arguments[2] = &outputPointer;
        arguments[3] = &fill;
        _module.Launch(_function, (uint)M, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int m, int n)
    {
        ValidateShape(m, n);
        int rowBytes = checked(n * sizeof(float));
        int reductionBytes = PtxRowReduce.SharedBytesFor(BlockThreads);
        const string Log2e = "0f3FB8AA3B";
        const string NegInf = "0fFF800000";

        var ptx = new StringBuilder(12_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// masked-softmax-row M={m} N={n}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 mask_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr,");
        ptx.AppendLine("    .param .f32 fill");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<12>;");
        ptx.AppendLine("    .reg .b64 %rd<20>;");
        ptx.AppendLine("    .reg .f32 %f<28>;");
        ptx.AppendLine($"    .shared .align 16 .b8 red[{reductionBytes * 2}];");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [mask_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
        ptx.AppendLine("    ld.param.f32 %f8, [fill];");
        ptx.AppendLine("    mov.u64 %rd5, red;");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mul.wide.u32 %rd6, %r1, {rowBytes};");
        ptx.AppendLine("    add.u64 %rd7, %rd0, %rd6;");
        ptx.AppendLine("    add.u64 %rd8, %rd1, %rd6;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd6;");
        ptx.AppendLine("    mul.wide.u32 %rd4, %r0, 16;");
        ptx.AppendLine("    mul.wide.u32 %rd13, %r0, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd5, %rd13;");

        ptx.AppendLine($"    mov.f32 %f0, {NegInf};");
        for (int pack = 0; pack < 4; pack++)
        {
            int[] values = VectorRegisters(pack);
            int columnBytes = pack * BlockThreads * 4 * sizeof(float);
            ptx.AppendLine($"    add.u64 %rd11, %rd4, {columnBytes};");
            ptx.AppendLine("    add.u64 %rd12, %rd7, %rd11;");
            ptx.AppendLine("    add.u64 %rd14, %rd8, %rd11;");
            ptx.AppendLine($"    ld.global.nc.v4.f32 " +
                $"{{%f{values[0]},%f{values[1]},%f{values[2]},%f{values[3]}}}, [%rd12];");
            ptx.AppendLine("    ld.global.nc.v4.f32 {%f24,%f25,%f26,%f27}, [%rd14];");
            for (int lane = 0; lane < 4; lane++)
            {
                ptx.AppendLine($"    setp.neu.f32 %p{lane}, %f{24 + lane}, 0f00000000;");
                ptx.AppendLine($"    selp.f32 %f{values[lane]}, %f8, %f{values[lane]}, %p{lane};");
                ptx.AppendLine($"    max.f32 %f0, %f0, %f{values[lane]};");
            }
        }
        PtxRowReduce.Emit(ptx, "max.f32", "%f0", BlockThreads);
        ptx.AppendLine("    ld.shared.f32 %f2, [%rd5];");

        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        for (int pack = 0; pack < 4; pack++)
        {
            foreach (int value in VectorRegisters(pack))
            {
                ptx.AppendLine($"    sub.rn.f32 %f{value}, %f{value}, %f2;");
                ptx.AppendLine($"    mul.rn.f32 %f{value}, %f{value}, {Log2e};");
                ptx.AppendLine($"    ex2.approx.f32 %f{value}, %f{value};");
                ptx.AppendLine($"    add.rn.f32 %f0, %f0, %f{value};");
            }
        }
        ptx.AppendLine($"    add.u64 %rd5, %rd5, {reductionBytes};");
        ptx.AppendLine("    add.u64 %rd10, %rd5, %rd13;");
        PtxRowReduce.Emit(ptx, "add.rn.f32", "%f0", BlockThreads);
        ptx.AppendLine("    ld.shared.f32 %f3, [%rd5];");
        ptx.AppendLine("    rcp.approx.f32 %f4, %f3;");

        for (int pack = 0; pack < 4; pack++)
        {
            int[] values = VectorRegisters(pack);
            int columnBytes = pack * BlockThreads * 4 * sizeof(float);
            ptx.AppendLine($"    add.u64 %rd11, %rd4, {columnBytes};");
            ptx.AppendLine("    add.u64 %rd14, %rd9, %rd11;");
            foreach (int value in values)
                ptx.AppendLine($"    mul.rn.f32 %f{value}, %f{value}, %f4;");
            ptx.AppendLine($"    st.global.v4.f32 [%rd14], " +
                $"{{%f{values[0]},%f{values[1]},%f{values[2]},%f{values[3]}}};");
        }
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static int[] VectorRegisters(int pack) => pack switch
    {
        0 => new[] { 1, 5, 6, 7 },
        1 => new[] { 12, 13, 14, 15 },
        2 => new[] { 16, 17, 18, 19 },
        3 => new[] { 20, 21, 22, 23 },
        _ => throw new ArgumentOutOfRangeException(nameof(pack))
    };

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int m, int n, float fillValue)
    {
        var extent = new DirectPtxExtent(m, n);
        return new DirectPtxKernelBlueprint(
            Operation: "masked-softmax-row",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp32-m{m}-n{n}-t64-v4x4-register",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("mask", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 64,
                MaxStaticSharedBytes: PtxRowReduce.SharedBytesFor(BlockThreads) * 2,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 16),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "softmax(mask != 0 ? fillValue : input, axis=last)",
                ["fill-value"] = fillValue.ToString("R", CultureInfo.InvariantCulture),
                ["stability"] = "row-max-subtracted-after-mask-selection",
                ["fusion"] = "masked-fill-plus-softmax-no-global-intermediate",
                ["block-threads"] = BlockThreads.ToString(CultureInfo.InvariantCulture),
                ["vector-width"] = "16 (four aligned float4 packs per lane)",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int m, int n) =>
        PtxRowShape.IsSupported(m, n) && n == 1024;

    internal static bool IsPromotedShape(int m, int n) => false;

    private static void ValidateShape(int m, int n)
    {
        if (!IsSupportedShape(m, n))
            throw new ArgumentOutOfRangeException(nameof(m),
                "Masked softmax currently supports the measured row family with N=1024.");
    }
}
