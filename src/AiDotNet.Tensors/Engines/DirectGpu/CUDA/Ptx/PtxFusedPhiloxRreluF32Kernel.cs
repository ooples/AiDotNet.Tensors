using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact-shape training RReLU. Philox slope generation and negative-side
/// multiplication are fused; only the public saved slopes and result reach
/// global memory.
/// </summary>
internal sealed class PtxFusedPhiloxRreluF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_philox_rrelu_f32";
    internal const int ThreadsPerBlock = 128;
    internal const int ValuesPerThread = 4;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int ElementCount { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedPhiloxRreluF32Kernel(DirectPtxRuntime runtime, int elementCount)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalRngDropout(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The experimental fused RReLU specialization targets SM86 only.");
        ValidateElementCount(elementCount);
        ElementCount = elementCount;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, elementCount);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, elementCount);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, ThreadsPerBlock);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, ThreadsPerBlock, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info,
            ThreadsPerBlock, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView input,
        DirectPtxTensorView noise,
        DirectPtxTensorView output,
        ulong seed,
        ulong subsequence,
        ulong counterOffset,
        float lower,
        float upper)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(noise, Blueprint.Tensors[1], nameof(noise));
        Require(output, Blueprint.Tensors[2], nameof(output));
        if (!PtxCompat.IsFinite(lower) || !PtxCompat.IsFinite(upper) || lower > upper)
            throw new ArgumentOutOfRangeException(nameof(lower));
        if (PtxFusedPhiloxDropoutF32Kernel.RangesOverlap(input, noise) ||
            PtxFusedPhiloxDropoutF32Kernel.RangesOverlap(input, output) ||
            PtxFusedPhiloxDropoutF32Kernel.RangesOverlap(noise, output))
            throw new ArgumentException("Fused RReLU tensors may not alias.");

        IntPtr inputPointer = input.Pointer;
        IntPtr noisePointer = noise.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[8];
        arguments[0] = &inputPointer;
        arguments[1] = &noisePointer;
        arguments[2] = &outputPointer;
        arguments[3] = &seed;
        arguments[4] = &subsequence;
        arguments[5] = &counterOffset;
        arguments[6] = &lower;
        arguments[7] = &upper;
        uint groups = checked((uint)(ElementCount / ValuesPerThread));
        _module.Launch(
            _function, groups / ThreadsPerBlock, 1, 1,
            ThreadsPerBlock, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static bool IsSupportedElementCount(int elementCount) =>
        elementCount is 4_096 or 65_536 or 1_048_576;

    internal static string EmitPtx(int ccMajor, int ccMinor, int elementCount)
    {
        if (!DirectPtxArchitecture.HasExperimentalRngDropout(ccMajor, ccMinor))
            throw new PlatformNotSupportedException(
                "The experimental fused RReLU emitter targets SM86 only.");
        ValidateElementCount(elementCount);
        var ptx = new StringBuilder(12_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 noise_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr,");
        ptx.AppendLine("    .param .u64 seed,");
        ptx.AppendLine("    .param .u64 subsequence,");
        ptx.AppendLine("    .param .u64 counter_offset,");
        ptx.AppendLine("    .param .f32 lower,");
        ptx.AppendLine("    .param .f32 upper");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {ThreadsPerBlock}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<32>;");
        ptx.AppendLine("    .reg .b64 %rd<14>;");
        ptx.AppendLine("    .reg .f32 %f<28>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [noise_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [seed];");
        ptx.AppendLine("    ld.param.u64 %rd4, [subsequence];");
        ptx.AppendLine("    ld.param.u64 %rd5, [counter_offset];");
        ptx.AppendLine("    ld.param.f32 %f8, [lower];");
        ptx.AppendLine("    ld.param.f32 %f9, [upper];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {ThreadsPerBlock}, %r0;");
        ptx.AppendLine("    shl.b32 %r3, %r2, 2;");
        ptx.AppendLine("    mov.b64 {%r4,%r5}, %rd3;");
        ptx.AppendLine("    mov.b64 {%r6,%r7}, %rd4;");
        ptx.AppendLine("    mov.b64 {%r8,%r9}, %rd5;");
        ptx.AppendLine("    add.cc.u32 %r10, %r8, %r2;");
        ptx.AppendLine("    addc.u32 %r11, %r9, 0;");
        ptx.AppendLine("    mov.u32 %r12, %r6;");
        ptx.AppendLine("    mov.u32 %r13, %r7;");
        PtxFusedPhiloxDropoutF32Kernel.AppendPhiloxRounds(ptx);
        for (int i = 0; i < 4; i++)
        {
            ptx.AppendLine($"    cvt.rn.f32.u32 %f{i}, %r{10 + i};");
            ptx.AppendLine($"    mul.rn.f32 %f{i}, %f{i}, 0f2F800000;");
            ptx.AppendLine($"    min.f32 %f{i}, %f{i}, 0f3F7FFFFF;");
        }
        ptx.AppendLine("    sub.rn.f32 %f10, %f9, %f8;");
        for (int i = 0; i < 4; i++)
            ptx.AppendLine($"    fma.rn.f32 %f{i}, %f{i}, %f10, %f8;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd0, %rd6;");
        ptx.AppendLine("    add.u64 %rd8, %rd1, %rd6;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd6;");
        ptx.AppendLine("    ld.global.v4.f32 {%f12,%f13,%f14,%f15}, [%rd7];");
        for (int i = 0; i < 4; i++)
        {
            ptx.AppendLine($"    setp.ge.f32 %p{i}, %f{12 + i}, 0f00000000;");
            ptx.AppendLine($"    selp.f32 %f{16 + i}, 0f3F800000, %f{i}, %p{i};");
            ptx.AppendLine($"    mul.rn.f32 %f{20 + i}, %f{12 + i}, %f{16 + i};");
        }
        ptx.AppendLine("    st.global.v4.f32 [%rd8], {%f0,%f1,%f2,%f3};");
        ptx.AppendLine("    st.global.v4.f32 [%rd9], {%f20,%f21,%f22,%f23};");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        ptx.AppendLine($"// exact_elements={elementCount}; fused_rng_consumer=1; no_tail_branch=1");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int elementCount)
    {
        var extent = new DirectPtxExtent(elementCount);
        return new DirectPtxKernelBlueprint(
            Operation: "philox-rrelu-training-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"exact-n{elementCount}-float4-t{ThreadsPerBlock}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("saved-noise", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 64,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 8),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["rng-abi"] = "philox4x32-10-v1",
                ["counter"] = "counterOffset+float4-group; explicit subsequence",
                ["rrelu"] = "x>=0 ? x : uniform(lower,upper)*x",
                ["global-reads"] = "one aligned fp32x4 input per thread",
                ["global-writes"] = "one aligned fp32x4 saved-noise and one output per thread",
                ["register-resident"] = "Philox state, four slopes, input, predicates, output",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-and-tail-parameters"] = "none"
            });
    }

    private static void ValidateElementCount(int elementCount)
    {
        if (!IsSupportedElementCount(elementCount))
            throw new ArgumentOutOfRangeException(
                nameof(elementCount), "Supported exact element counts are 4096, 65536, and 1048576.");
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
