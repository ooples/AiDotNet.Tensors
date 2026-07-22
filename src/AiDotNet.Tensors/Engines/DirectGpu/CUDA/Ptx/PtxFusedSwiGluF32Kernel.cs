#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact contiguous FP32 SwiGLU specialization. Input rows are physically
/// [value | gate]; each thread streams four values and four gates into
/// registers, applies SiLU to the gate, and performs the sole output store.
/// </summary>
internal sealed class PtxFusedSwiGluF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_fused_swiglu_f32x4";
    internal const int BlockThreads = 256;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int OuterSize { get; }
    internal int HalfDimension { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedSwiGluF32Kernel(
        DirectPtxRuntime runtime,
        int outerSize,
        int halfDimension)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.HasValidatedGatedGlu(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in FP32 SwiGLU specialization is measured only on GA10x/SM86.");
        Validate(outerSize, halfDimension);
        OuterSize = outerSize;
        HalfDimension = halfDimension;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, outerSize, halfDimension);
        Ptx = EmitPtx(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            outerSize, halfDimension);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info,
            BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView input,
        DirectPtxTensorView output)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(output, Blueprint.Tensors[1], nameof(output));
        if (Overlaps(input, output))
            throw new ArgumentException("The SwiGLU output may not alias its split input tensor.");

        IntPtr inputPointer = input.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &inputPointer;
        arguments[1] = &outputPointer;
        uint vectorColumns = (uint)(HalfDimension / 4);
        _module.Launch(
            _function,
            (vectorColumns + BlockThreads - 1) / BlockThreads,
            (uint)OuterSize,
            1,
            BlockThreads,
            1,
            1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        int outerSize,
        int halfDimension)
    {
        Validate(outerSize, halfDimension);
        int vectorColumns = halfDimension / 4;
        int inputRowBytes = checked(2 * halfDimension * sizeof(float));
        int outputRowBytes = checked(halfDimension * sizeof(float));
        string negativeLog2E = FloatLiteral(-1.4426950408889634f);
        var ptx = new StringBuilder(6_144);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<8>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<20>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {vectorColumns};");
        ptx.AppendLine("    @%p0 bra SWIGLU_RETURN;");
        ptx.AppendLine("    mov.u32 %r3, %ctaid.y;");
        ptx.AppendLine($"    mul.wide.u32 %rd2, %r3, {inputRowBytes};");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r2, 16;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd2;");
        ptx.AppendLine("    add.u64 %rd4, %rd4, %rd3;");
        ptx.AppendLine($"    add.u64 %rd5, %rd4, {outputRowBytes};");
        ptx.AppendLine($"    mul.wide.u32 %rd6, %r3, {outputRowBytes};");
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd6;");
        ptx.AppendLine("    add.u64 %rd7, %rd7, %rd3;");
        ptx.AppendLine("    ld.global.nc.v4.f32 {%f0,%f1,%f2,%f3}, [%rd4];");
        ptx.AppendLine("    ld.global.nc.v4.f32 {%f4,%f5,%f6,%f7}, [%rd5];");
        for (int lane = 0; lane < 4; lane++)
        {
            int gate = 4 + lane;
            int temporary = 8 + lane;
            int result = 12 + lane;
            ptx.AppendLine($"    mul.rn.f32 %f{temporary}, %f{gate}, {negativeLog2E};");
            ptx.AppendLine($"    ex2.approx.f32 %f{temporary}, %f{temporary};");
            ptx.AppendLine($"    add.rn.f32 %f{temporary}, %f{temporary}, 0f3F800000;");
            ptx.AppendLine($"    rcp.approx.f32 %f{temporary}, %f{temporary};");
            ptx.AppendLine($"    mul.rn.f32 %f{temporary}, %f{gate}, %f{temporary};");
            ptx.AppendLine($"    mul.rn.f32 %f{result}, %f{lane}, %f{temporary};");
        }
        ptx.AppendLine("    st.global.v4.f32 [%rd7], {%f12,%f13,%f14,%f15};");
        ptx.AppendLine("SWIGLU_RETURN:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int outerSize,
        int halfDimension)
    {
        var inputExtent = new DirectPtxExtent(outerSize, checked(2 * halfDimension));
        var outputExtent = new DirectPtxExtent(outerSize, halfDimension);
        return new DirectPtxKernelBlueprint(
            Operation: "swiglu-forward-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"split-last-x4-o{outerSize}-d{halfDimension}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    inputExtent, inputExtent, 16, DirectPtxTensorAccess.Read,
                    DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    outputExtent, outputExtent, 16, DirectPtxTensorAccess.Write,
                    DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 24,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 6),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "value*gate*sigmoid(gate)",
                ["split"] = "input[row,0:D]=value; input[row,D:2D]=gate",
                ["mode"] = "inference-forward",
                ["input"] = "fp32",
                ["accumulator"] = "lane-private-fp32x4-register",
                ["output"] = "fp32",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int outerSize, int halfDimension) =>
        (outerSize, halfDimension) is
            (1, 4096) or (32, 4096) or (256, 4096) or (256, 11008);

    internal static bool IsPromotedShape(int outerSize, int halfDimension) => false;

    private static void Validate(int outerSize, int halfDimension)
    {
        if (!IsSupportedShape(outerSize, halfDimension))
            throw new ArgumentOutOfRangeException(nameof(outerSize),
                "The first SwiGLU family supports exact (outer,D) buckets " +
                "(1,4096), (32,4096), (256,4096), and (256,11008).");
        if ((halfDimension & 3) != 0)
            throw new ArgumentOutOfRangeException(nameof(halfDimension));
    }

    private static void Require(
        DirectPtxTensorView view,
        DirectPtxTensorContract contract,
        string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent || view.ByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
    }

    private static bool Overlaps(DirectPtxTensorView left, DirectPtxTensorView right)
    {
        nuint leftStart = (nuint)left.Pointer;
        nuint rightStart = (nuint)right.Pointer;
        nuint leftEnd = checked(leftStart + left.ByteLength);
        nuint rightEnd = checked(rightStart + right.ByteLength);
        return leftStart < rightEnd && rightStart < leftEnd;
    }

    private static string FloatLiteral(float value) =>
        "0f" + BitConverter.SingleToInt32Bits(value).ToString("X8", CultureInfo.InvariantCulture);
}
#endif
