using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// In-place probability normalization (issue #854), matching the NVRTC <c>normalize_probabilities</c>
/// kernel: for each batch row, divide every entry by the row sum (clamped to at least 1e-10). One
/// block owns one row; the 256 threads stride-accumulate a partial sum into shared memory, tree-reduce
/// it to the row total, then each thread rescales its strided slice — matching the NVRTC shared-memory
/// reduction. This is the quantum-measurement normalization pass, kept separate from the |amplitude|^2
/// evaluation exactly as in the NVRTC split.
///
/// Shape (batchSize, stateSize) is baked into the PTX, so the launch takes the buffer pointer only.
/// One block per batch row (grid = batchSize), 256 threads/block; stateSize may exceed the block width.
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
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(DirectPtxTensorView probabilities)
    {
        Require(probabilities, Blueprint.Tensors[0], nameof(probabilities));

        IntPtr probPointer = probabilities.Pointer;
        void** arguments = stackalloc void*[1];
        arguments[0] = &probPointer;
        _module.Launch(_function, (uint)BatchSize, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    private static string Hex(float value) => "0f" + BitConverter.ToInt32(BitConverter.GetBytes(value), 0).ToString("X8");

    internal static string EmitPtx(int ccMajor, int ccMinor, int batchSize, int stateSize)
    {
        ValidateShape(batchSize, stateSize);
        string minSum = Hex(1e-10f);

        var ptx = new StringBuilder(4_500);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// normalize-probabilities batch={batchSize} state={stateSize}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 prob_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<10>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<8>;");
        ptx.AppendLine($"    .shared .align 16 .b8 red[{BlockThreads * sizeof(float)}];");
        ptx.AppendLine("    ld.param.u64 %rd0, [prob_ptr];");
        ptx.AppendLine("    mov.u64 %rd1, red;");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");                      // row (batch)
        ptx.AppendLine($"    mul.wide.u32 %rd2, %r1, {stateSize * sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd3, %rd0, %rd2;");                   // &probs[row]
        ptx.AppendLine("    mul.wide.u32 %rd4, %r0, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd4;");                   // &red[tid]

        // Pass 1: strided partial sum (i = tid; i < stateSize; i += 256)
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");                   // sum
        ptx.AppendLine("    mov.u32 %r2, %r0;");                          // i = tid
        ptx.AppendLine("$NP_SUM:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {stateSize};");
        ptx.AppendLine("    @%p0 bra $NP_SUM_END;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd3, %rd6;");
        ptx.AppendLine("    ld.global.f32 %f1, [%rd7];");
        ptx.AppendLine("    add.rn.f32 %f0, %f0, %f1;");
        ptx.AppendLine($"    add.u32 %r2, %r2, {BlockThreads};");
        ptx.AppendLine("    bra.uni $NP_SUM;");
        ptx.AppendLine("$NP_SUM_END:");
        ptx.AppendLine("    st.shared.f32 [%rd5], %f0;");
        ptx.AppendLine("    bar.sync 0;");

        // Tree reduction over 256 lanes (strides 128..1).
        foreach (int stride in new[] { 128, 64, 32, 16, 8, 4, 2, 1 })
        {
            ptx.AppendLine($"    setp.lt.u32 %p1, %r0, {stride};");
            ptx.AppendLine("    @%p1 ld.shared.f32 %f2, [%rd5];");
            ptx.AppendLine($"    @%p1 ld.shared.f32 %f3, [%rd5+{stride * sizeof(float)}];");
            ptx.AppendLine("    @%p1 add.rn.f32 %f2, %f2, %f3;");
            ptx.AppendLine("    @%p1 st.shared.f32 [%rd5], %f2;");
            ptx.AppendLine("    bar.sync 0;");
        }
        ptx.AppendLine("    ld.shared.f32 %f4, [%rd1];");                 // totalSum = red[0]
        ptx.AppendLine($"    max.f32 %f4, %f4, {minSum};");              // clamp to >= 1e-10

        // Pass 2: strided divide (probs[i] /= totalSum)
        ptx.AppendLine("    mov.u32 %r2, %r0;");                          // i = tid
        ptx.AppendLine("$NP_DIV:");
        ptx.AppendLine($"    setp.ge.u32 %p2, %r2, {stateSize};");
        ptx.AppendLine("    @%p2 bra $NP_DIV_END;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd3, %rd6;");
        ptx.AppendLine("    ld.global.f32 %f5, [%rd7];");
        ptx.AppendLine("    div.rn.f32 %f5, %f5, %f4;");
        ptx.AppendLine("    st.global.f32 [%rd7], %f5;");
        ptx.AppendLine($"    add.u32 %r2, %r2, {BlockThreads};");
        ptx.AppendLine("    bra.uni $NP_DIV;");
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
            Version: 1,
            Architecture: architecture,
            Variant: $"fp32-b{batchSize}-s{stateSize}",
            Tensors:
            [
                new("probabilities", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.ReadWrite, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 20,
                MaxStaticSharedBytes: BlockThreads * sizeof(float),
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "probabilities[b,i] /= max(sum_i probabilities[b,i], 1e-10)",
                ["reduction"] = "one block per batch row; 256-lane shared-memory tree reduction",
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

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent || view.ByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
    }
}
