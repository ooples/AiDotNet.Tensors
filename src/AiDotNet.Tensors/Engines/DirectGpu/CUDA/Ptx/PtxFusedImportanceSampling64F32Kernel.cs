using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact 64-coarse/64-fine NeRF importance sampler. One warp owns a ray,
/// cooperatively builds the 64-entry CDF once, and produces two stratified
/// samples per lane with a fixed-depth binary search. There are no
/// data-dependent branches or global intermediates.
/// </summary>
internal sealed class PtxFusedImportanceSampling64F32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_philox_importance_sampling64_f32";
    internal const int Samples = 64;
    internal const int ThreadsPerBlock = 32;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int NumRays { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedImportanceSampling64F32Kernel(DirectPtxRuntime runtime, int numRays)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalRngDropout(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The experimental importance-sampling specialization targets SM86 only.");
        ValidateNumRays(numRays);
        NumRays = numRays;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, numRays);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, numRays);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, ThreadsPerBlock);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, ThreadsPerBlock, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info,
            ThreadsPerBlock, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView tValues,
        DirectPtxTensorView weights,
        DirectPtxTensorView output,
        ulong seed,
        ulong subsequence,
        ulong counterOffset)
    {
        Require(tValues, Blueprint.Tensors[0], nameof(tValues));
        Require(weights, Blueprint.Tensors[1], nameof(weights));
        Require(output, Blueprint.Tensors[2], nameof(output));
        if (PtxFusedPhiloxDropoutF32Kernel.RangesOverlap(tValues, output) ||
            PtxFusedPhiloxDropoutF32Kernel.RangesOverlap(weights, output))
            throw new ArgumentException("Importance-sampling output may not alias either input.");

        IntPtr tPointer = tValues.Pointer;
        IntPtr weightPointer = weights.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[6];
        arguments[0] = &tPointer;
        arguments[1] = &weightPointer;
        arguments[2] = &outputPointer;
        arguments[3] = &seed;
        arguments[4] = &subsequence;
        arguments[5] = &counterOffset;
        _module.Launch(
            _function,
            checked((uint)NumRays), 1, 1,
            ThreadsPerBlock, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static bool IsSupportedShape(
        int numRays,
        int numCoarseSamples,
        int numFineSamples) =>
        numCoarseSamples == Samples && numFineSamples == Samples &&
        numRays is 64 or 1_024 or 16_384;

    internal static string EmitPtx(int ccMajor, int ccMinor, int numRays)
    {
        if (!DirectPtxArchitecture.HasExperimentalRngDropout(ccMajor, ccMinor))
            throw new PlatformNotSupportedException(
                "The experimental importance-sampling emitter targets SM86 only.");
        ValidateNumRays(numRays);
        var ptx = new StringBuilder(55_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 t_values_ptr,");
        ptx.AppendLine("    .param .u64 weights_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr,");
        ptx.AppendLine("    .param .u64 seed,");
        ptx.AppendLine("    .param .u64 subsequence,");
        ptx.AppendLine("    .param .u64 counter_offset");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {ThreadsPerBlock}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<8>;");
        ptx.AppendLine("    .reg .b32 %r<40>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<24>;");
        ptx.AppendLine("    .shared .align 16 .b8 t_shared[256];");
        ptx.AppendLine("    .shared .align 16 .b8 weights_shared[256];");
        ptx.AppendLine("    ld.param.u64 %rd0, [t_values_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [weights_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [seed];");
        ptx.AppendLine("    ld.param.u64 %rd4, [subsequence];");
        ptx.AppendLine("    ld.param.u64 %rd5, [counter_offset];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine("    mad.lo.u32 %r2, %r1, 64, %r0;");
        ptx.AppendLine("    add.u32 %r3, %r2, 32;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r2, 4;");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd0, %rd6;");
        ptx.AppendLine("    add.u64 %rd9, %rd0, %rd7;");
        ptx.AppendLine("    add.u64 %rd10, %rd1, %rd6;");
        ptx.AppendLine("    add.u64 %rd11, %rd1, %rd7;");
        ptx.AppendLine("    ld.global.f32 %f0, [%rd8];");
        ptx.AppendLine("    ld.global.f32 %f1, [%rd9];");
        ptx.AppendLine("    ld.global.f32 %f2, [%rd10];");
        ptx.AppendLine("    ld.global.f32 %f3, [%rd11];");
        ptx.AppendLine("    max.f32 %f2, %f2, 0f00000000;");
        ptx.AppendLine("    max.f32 %f3, %f3, 0f00000000;");
        ptx.AppendLine("    mul.wide.u32 %rd12, %r0, 4;");
        ptx.AppendLine("    add.u64 %rd13, %rd12, 128;");
        // PTX addresses permit register+immediate, but not symbol+register.
        // Materialize each shared symbol as an address before adding the
        // lane-owned byte offsets; the old symbol+%rd form failed ptxas.
        ptx.AppendLine("    mov.u64 %rd8, t_shared;");
        ptx.AppendLine("    mov.u64 %rd9, weights_shared;");
        ptx.AppendLine("    add.u64 %rd10, %rd8, %rd12;");
        ptx.AppendLine("    add.u64 %rd11, %rd8, %rd13;");
        ptx.AppendLine("    st.shared.f32 [%rd10], %f0;");
        ptx.AppendLine("    st.shared.f32 [%rd11], %f1;");
        ptx.AppendLine("    add.u64 %rd10, %rd9, %rd12;");
        ptx.AppendLine("    add.u64 %rd11, %rd9, %rd13;");
        AppendWarpInclusiveScan(ptx, "%f2");
        AppendWarpInclusiveScan(ptx, "%f3");
        ptx.AppendLine("    mov.b32 %r23, %f2;");
        ptx.AppendLine("    shfl.sync.idx.b32 %r24, %r23, 31, 0x1f, 0xffffffff;");
        ptx.AppendLine("    mov.b32 %f4, %r24;");
        ptx.AppendLine("    add.rn.f32 %f3, %f3, %f4;");
        ptx.AppendLine("    mov.b32 %r23, %f3;");
        ptx.AppendLine("    shfl.sync.idx.b32 %r24, %r23, 31, 0x1f, 0xffffffff;");
        ptx.AppendLine("    mov.b32 %f5, %r24;");
        ptx.AppendLine("    add.u64 %rd10, %rd9, %rd12;");
        ptx.AppendLine("    add.u64 %rd11, %rd9, %rd13;");
        ptx.AppendLine("    st.shared.f32 [%rd10], %f2;");
        ptx.AppendLine("    st.shared.f32 [%rd11], %f3;");
        ptx.AppendLine("    bar.sync 0;");

        AppendUniform(ptx, "%r2", "%f6");
        ptx.AppendLine("    cvt.rn.f32.u32 %f7, %r0;");
        ptx.AppendLine("    add.rn.f32 %f7, %f7, %f6;");
        ptx.AppendLine("    mul.rn.f32 %f7, %f7, 0f3C800000;");
        AppendCdfSample(ptx, "%f7", "%rd6");

        AppendUniform(ptx, "%r3", "%f6");
        ptx.AppendLine("    cvt.rn.f32.u32 %f7, %r0;");
        ptx.AppendLine("    add.rn.f32 %f7, %f7, 0f42000000;");
        ptx.AppendLine("    add.rn.f32 %f7, %f7, %f6;");
        ptx.AppendLine("    mul.rn.f32 %f7, %f7, 0f3C800000;");
        AppendCdfSample(ptx, "%f7", "%rd7");

        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        ptx.AppendLine($"// exact_shape=[{numRays},64,64]; one_warp_per_ray=1; no_tail_branch=1; cooperative_cdf=64; binary_search_steps=6");
        return ptx.ToString();
    }

    private static void AppendUniform(StringBuilder ptx, string globalIndex, string output)
    {
        ptx.AppendLine("    mov.b64 {%r4,%r5}, %rd3;");
        ptx.AppendLine("    mov.b64 {%r6,%r7}, %rd4;");
        ptx.AppendLine("    mov.b64 {%r8,%r9}, %rd5;");
        ptx.AppendLine($"    add.cc.u32 %r10, %r8, {globalIndex};");
        ptx.AppendLine("    addc.u32 %r11, %r9, 0;");
        ptx.AppendLine("    mov.u32 %r12, %r6;");
        ptx.AppendLine("    mov.u32 %r13, %r7;");
        PtxFusedPhiloxDropoutF32Kernel.AppendPhiloxRounds(ptx);
        ptx.AppendLine($"    cvt.rn.f32.u32 {output}, %r10;");
        ptx.AppendLine($"    mul.rn.f32 {output}, {output}, 0f2F800000;");
        ptx.AppendLine($"    min.f32 {output}, {output}, 0f3F7FFFFF;");
    }

    private static void AppendWarpInclusiveScan(StringBuilder ptx, string value)
    {
        foreach (int offset in new[] { 1, 2, 4, 8, 16 })
        {
            ptx.AppendLine($"    mov.b32 %r23, {value};");
            ptx.AppendLine($"    shfl.sync.up.b32 %r24, %r23, {offset}, 0, 0xffffffff;");
            ptx.AppendLine("    mov.b32 %f23, %r24;");
            ptx.AppendLine($"    setp.ge.u32 %p5, %r0, {offset};");
            ptx.AppendLine($"    @%p5 add.rn.f32 {value}, {value}, %f23;");
        }
    }

    private static void AppendCdfSample(StringBuilder ptx, string u, string byteOffset)
    {
        // Search the raw prefix sums. Multiplying u by the total once is
        // algebraically identical to dividing every weight by the total, and
        // turns the linear 64-bin/division chain into six fixed search steps.
        ptx.AppendLine($"    mul.rn.f32 %f22, {u}, %f5;");
        ptx.AppendLine("    mov.u32 %r25, 0;");
        foreach (int step in new[] { 32, 16, 8, 4, 2, 1 })
        {
            ptx.AppendLine($"    add.u32 %r26, %r25, {step};");
            ptx.AppendLine("    sub.u32 %r27, %r26, 1;");
            ptx.AppendLine("    mul.wide.u32 %rd10, %r27, 4;");
            ptx.AppendLine("    add.u64 %rd10, %rd9, %rd10;");
            ptx.AppendLine("    ld.shared.f32 %f11, [%rd10];");
            ptx.AppendLine("    setp.gt.f32 %p1, %f22, %f11;");
            ptx.AppendLine("    selp.u32 %r25, %r26, %r25, %p1;");
        }
        ptx.AppendLine("    min.u32 %r25, %r25, 63;");
        ptx.AppendLine("    ld.shared.f32 %f10, [t_shared];");
        ptx.AppendLine("    setp.gt.u32 %p2, %r25, 0;");
        ptx.AppendLine("    sub.u32 %r26, %r25, 1;");
        ptx.AppendLine("    mul.wide.u32 %rd10, %r26, 4;");
        ptx.AppendLine("    mul.wide.u32 %rd11, %r25, 4;");
        ptx.AppendLine("    add.u64 %rd12, %rd9, %rd10;");
        ptx.AppendLine("    add.u64 %rd13, %rd9, %rd11;");
        ptx.AppendLine("    mov.f32 %f12, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f13, 0f00000000;");
        ptx.AppendLine("    @%p2 ld.shared.f32 %f12, [%rd12];");
        ptx.AppendLine("    @%p2 ld.shared.f32 %f13, [%rd13];");
        ptx.AppendLine("    add.u64 %rd12, %rd8, %rd10;");
        ptx.AppendLine("    add.u64 %rd13, %rd8, %rd11;");
        ptx.AppendLine("    @%p2 ld.shared.f32 %f14, [%rd12];");
        ptx.AppendLine("    @%p2 ld.shared.f32 %f15, [%rd13];");
        ptx.AppendLine("    @%p2 mov.f32 %f10, %f14;");
        ptx.AppendLine("    sub.rn.f32 %f16, %f13, %f12;");
        ptx.AppendLine("    mul.rn.f32 %f17, %f5, 0f2EDBE6FF;");
        ptx.AppendLine("    setp.gt.f32 %p3, %f16, %f17;");
        ptx.AppendLine("    and.pred %p3, %p3, %p2;");
        ptx.AppendLine("    @%p3 sub.rn.f32 %f18, %f22, %f12;");
        ptx.AppendLine("    @%p3 div.rn.f32 %f18, %f18, %f16;");
        ptx.AppendLine("    @%p3 sub.rn.f32 %f19, %f15, %f14;");
        ptx.AppendLine("    @%p3 fma.rn.f32 %f10, %f18, %f19, %f14;");
        ptx.AppendLine("    ld.shared.f32 %f19, [t_shared];");
        ptx.AppendLine("    ld.shared.f32 %f20, [t_shared+252];");
        ptx.AppendLine("    sub.rn.f32 %f21, %f20, %f19;");
        ptx.AppendLine($"    fma.rn.f32 %f21, {u}, %f21, %f19;");
        ptx.AppendLine("    setp.le.f32 %p4, %f5, 0f2EDBE6FF;");
        ptx.AppendLine("    selp.f32 %f10, %f21, %f10, %p4;");
        ptx.AppendLine($"    add.u64 %rd14, %rd2, {byteOffset};");
        ptx.AppendLine("    st.global.f32 [%rd14], %f10;");
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int numRays)
    {
        var extent = new DirectPtxExtent(numRays, Samples);
        return new DirectPtxKernelBlueprint(
            Operation: "philox-importance-sampling64-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"exact-r{numRays}-c64-f64-warp",
            Tensors:
            [
                new("coarse-t", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 4, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("coarse-weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 4, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("fine-t", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 4, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 64,
                MaxStaticSharedBytes: 512,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 8),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["rng-abi"] = "philox4x32-10-v1",
                ["counter"] = "counterOffset+global-fine-output-index; subsequence in high64",
                ["stratification"] = "u=(sample+uniform)/64",
                ["weights"] = "negative and NaN weights clamp through max(weight,0); zero-sum uses uniform t-range",
                ["global-reads"] = "each coarse t and weight read once per ray",
                ["global-writes"] = "one final fine t-value per sample",
                ["shared-memory"] = "two aligned 64-float arrays; lane i owns banks i and i+32",
                ["register-resident"] = "Philox words, weight sum, CDF state, interpolation state",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-and-tail-parameters"] = "none"
            });
    }

    private static void ValidateNumRays(int numRays)
    {
        if (!IsSupportedShape(numRays, Samples, Samples))
            throw new ArgumentOutOfRangeException(
                nameof(numRays), "Supported exact ray counts are 64, 1024, and 16384 for 64/64 samples.");
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
