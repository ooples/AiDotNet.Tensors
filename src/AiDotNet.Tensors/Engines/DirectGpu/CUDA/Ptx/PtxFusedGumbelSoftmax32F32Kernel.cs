#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// One-warp-per-row Gumbel-softmax for exactly 32 contiguous classes. Philox
/// noise, perturbation, max/sum reductions, and normalization stay in registers;
/// global memory sees only the logits read and final probability write.
/// </summary>
internal sealed class PtxFusedGumbelSoftmax32F32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_philox_gumbel_softmax32_f32";
    internal const int InnerSize = 32;
    internal const int ThreadsPerBlock = InnerSize;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int OuterSize { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedGumbelSoftmax32F32Kernel(DirectPtxRuntime runtime, int outerSize)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.HasExperimentalRngDropout(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The experimental Gumbel-softmax specialization targets SM86 only.");
        ValidateOuterSize(outerSize);
        OuterSize = outerSize;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, outerSize);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, outerSize);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, ThreadsPerBlock);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, ThreadsPerBlock, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info,
            ThreadsPerBlock, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView logits,
        DirectPtxTensorView output,
        ulong seed,
        ulong subsequence,
        ulong counterOffset,
        float temperature)
    {
        Require(logits, Blueprint.Tensors[0], nameof(logits));
        Require(output, Blueprint.Tensors[1], nameof(output));
        if (!float.IsFinite(temperature) || temperature <= 0f)
            throw new ArgumentOutOfRangeException(nameof(temperature));
        if (PtxFusedPhiloxDropoutF32Kernel.RangesOverlap(logits, output))
            throw new ArgumentException("Gumbel-softmax input and output may not alias.");

        IntPtr logitsPointer = logits.Pointer;
        IntPtr outputPointer = output.Pointer;
        float inverseTemperature = 1f / temperature;
        void** arguments = stackalloc void*[6];
        arguments[0] = &logitsPointer;
        arguments[1] = &outputPointer;
        arguments[2] = &seed;
        arguments[3] = &subsequence;
        arguments[4] = &counterOffset;
        arguments[5] = &inverseTemperature;
        _module.Launch(
            _function,
            checked((uint)OuterSize), 1, 1,
            ThreadsPerBlock, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static bool IsSupportedShape(int outerSize, int innerSize) =>
        innerSize == InnerSize && outerSize is 128 or 2_048 or 32_768;

    internal static string EmitPtx(int ccMajor, int ccMinor, int outerSize)
    {
        if (!DirectPtxArchitecture.HasExperimentalRngDropout(ccMajor, ccMinor))
            throw new PlatformNotSupportedException(
                "The experimental Gumbel-softmax emitter targets SM86 only.");
        ValidateOuterSize(outerSize);
        var ptx = new StringBuilder(9_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 logits_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr,");
        ptx.AppendLine("    .param .u64 seed,");
        ptx.AppendLine("    .param .u64 subsequence,");
        ptx.AppendLine("    .param .u64 counter_offset,");
        ptx.AppendLine("    .param .f32 inverse_temperature");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {ThreadsPerBlock}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<32>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<20>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [logits_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [output_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [seed];");
        ptx.AppendLine("    ld.param.u64 %rd3, [subsequence];");
        ptx.AppendLine("    ld.param.u64 %rd4, [counter_offset];");
        ptx.AppendLine("    ld.param.f32 %f0, [inverse_temperature];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine("    mad.lo.u32 %r2, %r1, 32, %r0;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd0, %rd5;");
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd5;");
        ptx.AppendLine("    ld.global.f32 %f1, [%rd6];");
        ptx.AppendLine("    mov.b64 {%r4,%r5}, %rd2;");
        ptx.AppendLine("    mov.b64 {%r6,%r7}, %rd3;");
        ptx.AppendLine("    mov.b64 {%r8,%r9}, %rd4;");
        ptx.AppendLine("    add.cc.u32 %r10, %r8, %r2;");
        ptx.AppendLine("    addc.u32 %r11, %r9, 0;");
        ptx.AppendLine("    mov.u32 %r12, %r6;");
        ptx.AppendLine("    mov.u32 %r13, %r7;");
        PtxFusedPhiloxDropoutF32Kernel.AppendPhiloxRounds(ptx);
        ptx.AppendLine("    cvt.rn.f32.u32 %f2, %r10;");
        ptx.AppendLine("    mul.rn.f32 %f2, %f2, 0f2F800000;");
        ptx.AppendLine("    max.f32 %f2, %f2, 0f2F800000;");
        ptx.AppendLine("    min.f32 %f2, %f2, 0f3F7FFFFF;");
        ptx.AppendLine("    lg2.approx.f32 %f3, %f2;");
        ptx.AppendLine("    mul.rn.f32 %f3, %f3, 0f3F317218;");
        ptx.AppendLine("    neg.f32 %f3, %f3;");
        ptx.AppendLine("    lg2.approx.f32 %f4, %f3;");
        ptx.AppendLine("    mul.rn.f32 %f4, %f4, 0f3F317218;");
        ptx.AppendLine("    neg.f32 %f4, %f4;");
        ptx.AppendLine("    add.rn.f32 %f5, %f1, %f4;");
        ptx.AppendLine("    mul.rn.f32 %f5, %f5, %f0;");
        ptx.AppendLine("    mov.f32 %f11, %f5;");
        AppendWarpReduce(ptx, "%f5", "max");
        ptx.AppendLine("    mov.b32 %r23, %f5;");
        ptx.AppendLine("    shfl.sync.idx.b32 %r24, %r23, 0, 0x1f, 0xffffffff;");
        ptx.AppendLine("    mov.b32 %f6, %r24;");
        ptx.AppendLine("    sub.rn.f32 %f7, %f11, %f6;");
        ptx.AppendLine("    mul.rn.f32 %f7, %f7, 0f3FB8AA3B;");
        ptx.AppendLine("    ex2.approx.f32 %f7, %f7;");
        ptx.AppendLine("    mov.f32 %f12, %f7;");
        AppendWarpReduce(ptx, "%f7", "add");
        ptx.AppendLine("    mov.b32 %r23, %f7;");
        ptx.AppendLine("    shfl.sync.idx.b32 %r24, %r23, 0, 0x1f, 0xffffffff;");
        ptx.AppendLine("    mov.b32 %f8, %r24;");
        ptx.AppendLine("    rcp.approx.f32 %f8, %f8;");
        ptx.AppendLine("    mul.rn.f32 %f9, %f12, %f8;");
        ptx.AppendLine("    st.global.f32 [%rd7], %f9;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        ptx.AppendLine($"// exact_shape=[{outerSize},32]; one_warp_per_row=1; no_tail_branch=1");
        return ptx.ToString();
    }

    private static void AppendWarpReduce(StringBuilder ptx, string value, string operation)
    {
        foreach (int offset in new[] { 16, 8, 4, 2, 1 })
        {
            ptx.AppendLine($"    mov.b32 %r23, {value};");
            ptx.AppendLine($"    shfl.sync.down.b32 %r24, %r23, {offset}, 0x1f, 0xffffffff;");
            ptx.AppendLine("    mov.b32 %f10, %r24;");
            ptx.AppendLine(operation == "max"
                ? $"    max.f32 {value}, {value}, %f10;"
                : $"    add.rn.f32 {value}, {value}, %f10;");
        }
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int outerSize)
    {
        var extent = new DirectPtxExtent(outerSize, InnerSize);
        return new DirectPtxKernelBlueprint(
            Operation: "philox-gumbel-softmax32-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"exact-o{outerSize}-i32-warp",
            Tensors:
            [
                new("logits", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 4, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("probabilities", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 4, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 64,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 16),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["rng-abi"] = "philox4x32-10-v1",
                ["counter"] = "counterOffset+global-element-index; subsequence in high64",
                ["word-mapping"] = "one Philox counter per element; word0 consumed",
                ["gumbel"] = "-ln(-ln(clamp(u,2^-32,nextafter(1,0))))",
                ["softmax"] = "warp max then warp sum; approximate PTX transcendental mode",
                ["global-reads"] = "one scalar logit per lane",
                ["global-writes"] = "one final scalar probability per lane",
                ["register-resident"] = "noise, perturbed logit, maximum, exponential, sum",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-and-tail-parameters"] = "none"
            });
    }

    private static void ValidateOuterSize(int outerSize)
    {
        if (!IsSupportedShape(outerSize, InnerSize))
            throw new ArgumentOutOfRangeException(
                nameof(outerSize), "Supported exact row counts are 128, 2048, and 32768 for 32 classes.");
    }

    private static void Require(
        DirectPtxTensorView view,
        DirectPtxTensorContract contract,
        string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent || view.ByteLength != contract.RequiredBytes ||
            view.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy exact physical ABI '{contract.Name}'.", parameter);
    }
}
#endif
