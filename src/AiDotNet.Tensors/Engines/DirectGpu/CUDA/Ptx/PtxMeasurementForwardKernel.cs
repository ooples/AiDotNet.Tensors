using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Combined quantum measurement-layer forward (issue #854), matching the NVRTC
/// <c>measurement_forward</c> kernel: from an interleaved complex state
/// <c>input[b, i] = (real, imag)</c>, produce the normalized probability distribution
/// <c>output[b, i] = |z_i|^2 / max(sum_i |z_i|^2, 1e-10)</c>. One warp owns one batch row; its lanes
/// stride-compute <c>|z|^2</c> into the output row, reduce the row total with warp shuffles, then
/// rescale the row in place without shared memory or block barriers. This fuses the <c>|z|^2</c>
/// evaluation and normalization that <see cref="PtxQuantumMeasurementKernel"/> and
/// <see cref="PtxNormalizeProbabilitiesKernel"/> expose separately.
///
/// Shape (batchSize, stateSize) is baked into the PTX, so the launch takes buffer pointers only.
/// Eight rows per 256-thread block; stateSize may exceed the warp width and either dimension may be
/// ragged.
/// </summary>
internal sealed class PtxMeasurementForwardKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const int MaxBatch = 2048 * 4096;
    internal const int MaxState = 1 << 22;
    internal const string EntryPoint = "aidotnet_measurement_forward";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int BatchSize { get; }
    internal int StateSize { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxMeasurementForwardKernel(DirectPtxRuntime runtime, int batchSize, int stateSize)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in measurement-forward specialization is measured only on GA10x/SM86.");
        ValidateShape(batchSize, stateSize);
        BatchSize = batchSize;
        StateSize = stateSize;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, batchSize, stateSize);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, batchSize, stateSize);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module);
    }

    internal unsafe void Launch(DirectPtxTensorView input, DirectPtxTensorView output)
    {
        DirectPtxAbi.Require(input, Blueprint.Tensors[0], nameof(input));
        DirectPtxAbi.Require(output, Blueprint.Tensors[1], nameof(output));

        IntPtr inputPointer = input.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &inputPointer;
        arguments[1] = &outputPointer;
        _module.Launch(
            _function, (uint)((BatchSize + (BlockThreads / 32) - 1) / (BlockThreads / 32)), 1, 1,
            BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();


    internal static string EmitPtx(int ccMajor, int ccMinor, int batchSize, int stateSize)
    {
        ValidateShape(batchSize, stateSize);
        string minSum = DirectPtxPtxText.Hex(1e-10f);

        var ptx = new StringBuilder(5_000);
        DirectPtxPtxText.AppendModuleHeader(ptx, ccMajor, ccMinor, disableLoopUnrolling: true);
        ptx.AppendLine($"// measurement-forward batch={batchSize} state={stateSize}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 in_ptr,");
        ptx.AppendLine("    .param .u64 out_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<3>;");
        ptx.AppendLine("    .reg .b32 %r<10>;");
        ptx.AppendLine("    .reg .b64 %rd<14>;");
        ptx.AppendLine("    .reg .f32 %f<10>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [in_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [out_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r0, %r1, {BlockThreads}, %r0;"); // global thread
        ptx.AppendLine("    shr.u32 %r2, %r0, 5;");                       // warp owns row
        ptx.AppendLine("    and.b32 %r3, %r0, 31;");                      // lane owns states
        ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {batchSize};");
        ptx.AppendLine("    @%p0 ret;");
        ptx.AppendLine($"    mul.wide.u32 %rd2, %r2, {stateSize * 2 * sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd3, %rd0, %rd2;");                   // &input[row]
        ptx.AppendLine($"    mul.wide.u32 %rd4, %r2, {stateSize * sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd4;");                   // &output[row]
        ptx.AppendLine("    mul.wide.u32 %rd6, %r3, 8;");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r3, 4;");

        // Pass 1: lane-strided magnitude writes and partial sum, then warp reduction and broadcast.
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r4, %r3;");
        ptx.AppendLine("    add.u64 %rd8, %rd3, %rd6;");
        ptx.AppendLine("    add.u64 %rd9, %rd5, %rd7;");
        ptx.AppendLine($"    setp.ge.u32 %p1, %r4, {stateSize};");
        ptx.AppendLine("    @%p1 bra $MF_SUM_END;");
        ptx.AppendLine("$MF_SUM:");
        ptx.AppendLine("    ld.global.nc.v2.f32 {%f1, %f2}, [%rd8];");
        ptx.AppendLine("    mul.rn.f32 %f3, %f1, %f1;");
        ptx.AppendLine("    fma.rn.f32 %f3, %f2, %f2, %f3;");
        ptx.AppendLine("    st.global.f32 [%rd9], %f3;");
        ptx.AppendLine("    add.rn.f32 %f0, %f0, %f3;");
        ptx.AppendLine("    add.u64 %rd8, %rd8, 256;");
        ptx.AppendLine("    add.u64 %rd9, %rd9, 128;");
        ptx.AppendLine("    add.u32 %r4, %r4, 32;");
        ptx.AppendLine($"    setp.lt.u32 %p1, %r4, {stateSize};");
        ptx.AppendLine("    @%p1 bra $MF_SUM;");
        ptx.AppendLine("$MF_SUM_END:");
        DirectPtxPtxText.AppendWarpSum(ptx, "%f0", "%r5", "%r6", "%f4");
        ptx.AppendLine("    mov.b32 %r5, %f0;");
        ptx.AppendLine("    shfl.sync.idx.b32 %r6, %r5, 0, 31, 0xffffffff;");
        ptx.AppendLine("    mov.b32 %f6, %r6;");
        ptx.AppendLine($"    max.f32 %f6, %f6, {minSum};");

        // Pass 2: lane-strided normalization.
        ptx.AppendLine("    mov.u32 %r4, %r3;");
        ptx.AppendLine("    add.u64 %rd9, %rd5, %rd7;");
        ptx.AppendLine($"    setp.ge.u32 %p2, %r4, {stateSize};");
        ptx.AppendLine("    @%p2 bra $MF_DIV_END;");
        ptx.AppendLine("$MF_DIV:");
        ptx.AppendLine("    ld.global.f32 %f7, [%rd9];");
        ptx.AppendLine("    div.rn.f32 %f7, %f7, %f6;");
        ptx.AppendLine("    st.global.f32 [%rd9], %f7;");
        ptx.AppendLine("    add.u64 %rd9, %rd9, 128;");
        ptx.AppendLine("    add.u32 %r4, %r4, 32;");
        ptx.AppendLine($"    setp.lt.u32 %p2, %r4, {stateSize};");
        ptx.AppendLine("    @%p2 bra $MF_DIV;");
        ptx.AppendLine("$MF_DIV_END:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, int batchSize, int stateSize)
    {
        var inExtent = new DirectPtxExtent(batchSize * stateSize * 2);
        var outExtent = new DirectPtxExtent(batchSize * stateSize);
        return new DirectPtxKernelBlueprint(
            Operation: "measurement-forward",
            Version: 2,
            Architecture: architecture,
            Variant: $"fp32-warp-v2-b{batchSize}-s{stateSize}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    inExtent, inExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    outExtent, outExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: DirectPtxResourceBudget.FromDriverMeasurement(
                measuredRegistersPerThread: 24,
                maxStaticSharedBytes: 0,
                maxLocalBytesPerThread: 0,
                minBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "output[b,i] = |input[b,i]|^2 / max(sum_i |input[b,i]|^2, 1e-10)",
                ["input-layout"] = "interleaved real/imag pairs per state",
                ["reduction"] = "one warp per batch row; shuffle sum; lane-strided state entries",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int batchSize, int stateSize)
    {
        if (batchSize <= 0 || stateSize <= 0) return false;
        return batchSize <= MaxBatch && stateSize <= MaxState;
    }

    internal static bool IsPromotedShape(int batchSize, int stateSize) => false;

    private static void ValidateShape(int batchSize, int stateSize)
    {
        if (!IsSupportedShape(batchSize, stateSize))
            throw new ArgumentOutOfRangeException(
                nameof(batchSize),
                $"Measurement forward requires positive batchSize<={MaxBatch} and stateSize<={MaxState}.");
    }

}
