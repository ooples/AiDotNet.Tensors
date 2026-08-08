using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// In-place probability normalization (issue #854), matching the NVRTC <c>normalize_probabilities</c>
/// kernel: for each batch row, divide every entry by the row sum (clamped to at least 1e-10). One
/// warp owns one row; its lanes stride-accumulate partial sums, reduce with warp shuffles, then
/// rescale their strided slices without shared memory or block barriers. This is the
/// quantum-measurement normalization pass, kept separate from the |amplitude|^2 evaluation exactly
/// as in the NVRTC split.
///
/// Shape (batchSize, stateSize) is baked into the PTX, so the launch takes the buffer pointer only.
/// Eight rows per 256-thread block; stateSize may exceed the warp width and either dimension may be
/// ragged.
/// </summary>
internal sealed class PtxNormalizeProbabilitiesKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const int MaxBatch = 2048 * 4096;
    internal const int MaxState = 1 << 22;
    internal const string EntryPoint = "aidotnet_normalize_probabilities";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int BatchSize { get; }
    internal int StateSize { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxNormalizeProbabilitiesKernel(DirectPtxRuntime runtime, int batchSize, int stateSize)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in normalize-probabilities specialization is measured only on GA10x/SM86.");
        ValidateShape(batchSize, stateSize);
        BatchSize = batchSize;
        StateSize = stateSize;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, batchSize, stateSize);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, batchSize, stateSize);
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

    internal unsafe void Launch(DirectPtxTensorView probabilities)
    {
        DirectPtxAbi.Require(probabilities, Blueprint.Tensors[0], nameof(probabilities));

        IntPtr probPointer = probabilities.Pointer;
        void** arguments = stackalloc void*[1];
        arguments[0] = &probPointer;
        _module.Launch(
            _function, (uint)((BatchSize + (BlockThreads / 32) - 1) / (BlockThreads / 32)), 1, 1,
            BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();


    internal static string EmitPtx(int ccMajor, int ccMinor, int batchSize, int stateSize)
    {
        ValidateShape(batchSize, stateSize);
        string minSum = DirectPtxPtxText.Hex(1e-10f);

        var ptx = new StringBuilder(4_500);
        DirectPtxPtxText.AppendModuleHeader(ptx, ccMajor, ccMinor, disableLoopUnrolling: true);
        ptx.AppendLine($"// normalize-probabilities batch={batchSize} state={stateSize}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 prob_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<3>;");
        ptx.AppendLine("    .reg .b32 %r<10>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<8>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [prob_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");                      // row (batch)
        ptx.AppendLine($"    mad.lo.u32 %r0, %r1, {BlockThreads}, %r0;"); // global thread
        ptx.AppendLine("    shr.u32 %r2, %r0, 5;");                       // warp owns row
        ptx.AppendLine("    and.b32 %r3, %r0, 31;");                      // lane owns state entries
        ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {batchSize};");
        ptx.AppendLine("    @%p0 ret;");
        ptx.AppendLine($"    mul.wide.u32 %rd2, %r2, {stateSize * sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd3, %rd0, %rd2;");                   // &probs[row]
        ptx.AppendLine("    mul.wide.u32 %rd4, %r3, 4;");

        // Pass 1: lane-strided partial sum, then a full-warp reduction and broadcast.
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r4, %r3;");
        ptx.AppendLine("    add.u64 %rd5, %rd3, %rd4;");
        ptx.AppendLine($"    setp.ge.u32 %p1, %r4, {stateSize};");
        ptx.AppendLine("    @%p1 bra $NP_SUM_END;");
        ptx.AppendLine("$NP_SUM:");
        ptx.AppendLine("    ld.global.f32 %f1, [%rd5];");
        ptx.AppendLine("    add.rn.f32 %f0, %f0, %f1;");
        ptx.AppendLine("    add.u64 %rd5, %rd5, 128;");
        ptx.AppendLine("    add.u32 %r4, %r4, 32;");
        ptx.AppendLine($"    setp.lt.u32 %p1, %r4, {stateSize};");
        ptx.AppendLine("    @%p1 bra $NP_SUM;");
        ptx.AppendLine("$NP_SUM_END:");
        DirectPtxPtxText.AppendWarpSum(ptx, "%f0", "%r5", "%r6", "%f2");
        ptx.AppendLine("    mov.b32 %r5, %f0;");
        ptx.AppendLine("    shfl.sync.idx.b32 %r6, %r5, 0, 31, 0xffffffff;");
        ptx.AppendLine("    mov.b32 %f4, %r6;");
        ptx.AppendLine($"    max.f32 %f4, %f4, {minSum};");

        // Pass 2: lane-strided divide (probs[i] /= totalSum).
        ptx.AppendLine("    mov.u32 %r4, %r3;");
        ptx.AppendLine("    add.u64 %rd5, %rd3, %rd4;");
        ptx.AppendLine($"    setp.ge.u32 %p2, %r4, {stateSize};");
        ptx.AppendLine("    @%p2 bra $NP_DIV_END;");
        ptx.AppendLine("$NP_DIV:");
        ptx.AppendLine("    ld.global.f32 %f5, [%rd5];");
        ptx.AppendLine("    div.rn.f32 %f5, %f5, %f4;");
        ptx.AppendLine("    st.global.f32 [%rd5], %f5;");
        ptx.AppendLine("    add.u64 %rd5, %rd5, 128;");
        ptx.AppendLine("    add.u32 %r4, %r4, 32;");
        ptx.AppendLine($"    setp.lt.u32 %p2, %r4, {stateSize};");
        ptx.AppendLine("    @%p2 bra $NP_DIV;");
        ptx.AppendLine("$NP_DIV_END:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, int batchSize, int stateSize)
    {
        var extent = new DirectPtxExtent(batchSize * stateSize);
        return new DirectPtxKernelBlueprint(
            Operation: "normalize-probabilities",
            Version: 2,
            Architecture: architecture,
            Variant: $"fp32-warp-b{batchSize}-s{stateSize}",
            Tensors:
            [
                new("probabilities", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.ReadWrite, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: DirectPtxResourceBudget.FromDriverMeasurement(
                measuredRegistersPerThread: 24,
                maxStaticSharedBytes: 0,
                maxLocalBytesPerThread: 0,
                minBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "probabilities[b,i] /= max(sum_i probabilities[b,i], 1e-10)",
                ["reduction"] = "one warp per batch row; shuffle sum; lane-strided state entries",
                ["global-intermediates"] = "in-place on probabilities",
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
                $"Normalize probabilities requires positive batchSize<={MaxBatch} and stateSize<={MaxState}.");
    }

}
