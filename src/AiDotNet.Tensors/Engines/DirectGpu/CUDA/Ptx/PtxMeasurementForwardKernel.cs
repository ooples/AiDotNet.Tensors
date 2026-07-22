using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Combined quantum measurement-layer forward (issue #854), matching the NVRTC
/// <c>measurement_forward</c> kernel: from an interleaved complex state
/// <c>input[b, i] = (real, imag)</c>, produce the normalized probability distribution
/// <c>output[b, i] = |z_i|^2 / max(sum_i |z_i|^2, 1e-10)</c>. One block owns one batch row; the 256
/// threads stride-compute <c>|z|^2</c> into the output row and a shared partial sum, tree-reduce the
/// row total, then rescale the row in place — matching the NVRTC shared-memory reduction. This fuses
/// the <c>|z|^2</c> evaluation and normalization that <see cref="PtxQuantumMeasurementKernel"/> and
/// <see cref="PtxNormalizeProbabilitiesKernel"/> expose separately.
///
/// Shape (batchSize, stateSize) is baked into the PTX, so the launch takes buffer pointers only.
/// One block per batch row (grid = batchSize), 256 threads/block; stateSize may exceed the block width.
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
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(DirectPtxTensorView input, DirectPtxTensorView output)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(output, Blueprint.Tensors[1], nameof(output));

        IntPtr inputPointer = input.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &inputPointer;
        arguments[1] = &outputPointer;
        _module.Launch(_function, (uint)BatchSize, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    private static string Hex(float value) => "0f" + BitConverter.ToInt32(BitConverter.GetBytes(value), 0).ToString("X8");

    internal static string EmitPtx(int ccMajor, int ccMinor, int batchSize, int stateSize)
    {
        ValidateShape(batchSize, stateSize);
        string minSum = Hex(1e-10f);

        var ptx = new StringBuilder(5_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// measurement-forward batch={batchSize} state={stateSize}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 in_ptr,");
        ptx.AppendLine("    .param .u64 out_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<10>;");
        ptx.AppendLine("    .reg .b64 %rd<18>;");
        ptx.AppendLine("    .reg .f32 %f<10>;");
        ptx.AppendLine($"    .shared .align 16 .b8 red[{BlockThreads * sizeof(float)}];");
        ptx.AppendLine("    ld.param.u64 %rd0, [in_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [out_ptr];");
        ptx.AppendLine("    mov.u64 %rd2, red;");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");                      // row (batch)
        ptx.AppendLine($"    mul.wide.u32 %rd3, %r1, {stateSize * 2 * sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");                   // &input[row] (interleaved)
        ptx.AppendLine($"    mul.wide.u32 %rd5, %r1, {stateSize * sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd5;");                   // &output[row]
        ptx.AppendLine("    mul.wide.u32 %rd7, %r0, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd2, %rd7;");                   // &red[tid]

        // Pass 1: out[i] = |z_i|^2 ; localSum += out[i]   (i = tid; i < stateSize; i += 256)
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");                   // localSum
        ptx.AppendLine("    mov.u32 %r2, %r0;");                          // i = tid
        ptx.AppendLine("$MF_SUM:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {stateSize};");
        ptx.AppendLine("    @%p0 bra $MF_SUM_END;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r2, 8;");                 // i*2 complex elems
        ptx.AppendLine("    add.u64 %rd10, %rd4, %rd9;");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd10];");            // real
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd10+4];");          // imag
        ptx.AppendLine("    mul.rn.f32 %f3, %f1, %f1;");
        ptx.AppendLine("    fma.rn.f32 %f3, %f2, %f2, %f3;");            // mag = re^2 + im^2
        ptx.AppendLine("    mul.wide.u32 %rd11, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd12, %rd6, %rd11;");
        ptx.AppendLine("    st.global.f32 [%rd12], %f3;");              // out[i] = mag
        ptx.AppendLine("    add.rn.f32 %f0, %f0, %f3;");
        ptx.AppendLine($"    add.u32 %r2, %r2, {BlockThreads};");
        ptx.AppendLine("    bra.uni $MF_SUM;");
        ptx.AppendLine("$MF_SUM_END:");
        ptx.AppendLine("    st.shared.f32 [%rd8], %f0;");
        ptx.AppendLine("    bar.sync 0;");

        // Tree reduction over 256 lanes (strides 128..1).
        foreach (int stride in new[] { 128, 64, 32, 16, 8, 4, 2, 1 })
        {
            ptx.AppendLine($"    setp.lt.u32 %p1, %r0, {stride};");
            ptx.AppendLine("    @%p1 ld.shared.f32 %f4, [%rd8];");
            ptx.AppendLine($"    @%p1 ld.shared.f32 %f5, [%rd8+{stride * sizeof(float)}];");
            ptx.AppendLine("    @%p1 add.rn.f32 %f4, %f4, %f5;");
            ptx.AppendLine("    @%p1 st.shared.f32 [%rd8], %f4;");
            ptx.AppendLine("    bar.sync 0;");
        }
        ptx.AppendLine("    ld.shared.f32 %f6, [%rd2];");                 // totalSum = red[0]
        ptx.AppendLine($"    max.f32 %f6, %f6, {minSum};");              // clamp to >= 1e-10

        // Pass 2: out[i] /= totalSum
        ptx.AppendLine("    mov.u32 %r2, %r0;");                          // i = tid
        ptx.AppendLine("$MF_DIV:");
        ptx.AppendLine($"    setp.ge.u32 %p2, %r2, {stateSize};");
        ptx.AppendLine("    @%p2 bra $MF_DIV_END;");
        ptx.AppendLine("    mul.wide.u32 %rd11, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd12, %rd6, %rd11;");
        ptx.AppendLine("    ld.global.f32 %f7, [%rd12];");
        ptx.AppendLine("    div.rn.f32 %f7, %f7, %f6;");
        ptx.AppendLine("    st.global.f32 [%rd12], %f7;");
        ptx.AppendLine($"    add.u32 %r2, %r2, {BlockThreads};");
        ptx.AppendLine("    bra.uni $MF_DIV;");
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
            Version: 1,
            Architecture: architecture,
            Variant: $"fp32-b{batchSize}-s{stateSize}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    inExtent, inExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    outExtent, outExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 20,
                MaxStaticSharedBytes: BlockThreads * sizeof(float),
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "output[b,i] = |input[b,i]|^2 / max(sum_i |input[b,i]|^2, 1e-10)",
                ["input-layout"] = "interleaved real/imag pairs per state",
                ["reduction"] = "one block per batch row; 256-lane shared-memory tree reduction",
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

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent || view.ByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
    }
}
