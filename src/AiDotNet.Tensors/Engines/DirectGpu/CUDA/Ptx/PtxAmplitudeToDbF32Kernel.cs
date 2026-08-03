using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Amplitude-to-decibel conversion for issue #850, matching the NVRTC <c>audio_amplitude_to_db</c> kernel:
/// <c>db = 20*log10(max(input, minAmp))</c>, optionally floored at a precomputed <c>topDbFloor</c>. This is
/// the log step of a log-mel / log-magnitude spectrogram. One thread owns one element. PTX has no log10
/// primitive, so the value is <c>lg2.approx(v) * (20*log10(2))</c>; the specialization is therefore covered
/// by a TOLERANCE-based parity spec, not bit-exact. <c>minAmp</c> and <c>topDbFloor</c> are per-launch
/// <c>.param .f32</c>; the <c>clipTopDb</c> flag is baked so the floor is emitted only when requested.
/// <c>length</c> is baked; the launch rounds up and a single guard drops the tail lanes. Two pointers plus
/// two f32 scalars reach the launch ABI.
///
/// The specialization stays disabled by default and fails closed until three clean promotion runs clear
/// the release gate.
/// </summary>
internal sealed class PtxAmplitudeToDbF32Kernel : IDisposable
{
    internal const int DefaultBlockThreads = 256;
    internal const string EntryPoint = "aidotnet_amplitude_to_db_f32";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Length { get; }
    internal bool ClipTopDb { get; }
    internal int BlockThreads { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxAmplitudeToDbF32Kernel(
        DirectPtxRuntime runtime, int length, bool clipTopDb, int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedComplexUnary(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in amplitude-to-db specialization is admitted only on SM86.");
        Validate(length);
        ValidateBlockThreads(blockThreads);
        Length = length;
        ClipTopDb = clipTopDb;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, length, clipTopDb, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, length, clipTopDb, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(DirectPtxTensorView input, DirectPtxTensorView output, float minAmp, float topDbFloor)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(output, Blueprint.Tensors[1], nameof(output));

        IntPtr inputPointer = input.Pointer, outputPointer = output.Pointer;
        float minAmpArg = minAmp, topDbFloorArg = topDbFloor;
        void** arguments = stackalloc void*[4];
        arguments[0] = &inputPointer;
        arguments[1] = &outputPointer;
        arguments[2] = &minAmpArg;
        arguments[3] = &topDbFloorArg;
        _module.Launch(
            _function,
            (uint)((Length + BlockThreads - 1) / BlockThreads), 1, 1,
            checked((uint)BlockThreads), 1, 1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    private static string Hex(float value) => "0f" + BitConverter.ToInt32(BitConverter.GetBytes(value), 0).ToString("X8");

    internal static string EmitPtx(int ccMajor, int ccMinor, int length, bool clipTopDb, int blockThreads = DefaultBlockThreads)
    {
        Validate(length);
        ValidateBlockThreads(blockThreads);
        string twentyLog10Of2 = Hex((float)(20.0 * Math.Log10(2.0)));

        var ptx = new StringBuilder(2_048);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// exact-shape length={length} block={blockThreads} clipTopDb={(clipTopDb ? 1 : 0)} op=amplitude-to-db");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr,");
        ptx.AppendLine("    .param .f32 min_amp,");
        ptx.AppendLine("    .param .f32 top_db_floor");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<3>;");
        ptx.AppendLine("    .reg .b64 %rd<8>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [output_ptr];");
        ptx.AppendLine("    ld.param.f32 %f2, [min_amp];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {blockThreads}, %r0;");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {length};");
        ptx.AppendLine("    @%p0 bra $A2DB_RET;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd4];");
        ptx.AppendLine("    max.f32 %f0, %f0, %f2;");             // max(input, minAmp)
        ptx.AppendLine("    lg2.approx.f32 %f0, %f0;");
        ptx.AppendLine($"    mul.rn.f32 %f0, %f0, {twentyLog10Of2};");   // 20*log10(v)
        if (clipTopDb)
        {
            ptx.AppendLine("    ld.param.f32 %f3, [top_db_floor];");
            ptx.AppendLine("    max.f32 %f0, %f0, %f3;");          // floor at topDbFloor
        }
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd3;");
        ptx.AppendLine("    st.global.f32 [%rd5], %f0;");
        ptx.AppendLine("$A2DB_RET:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int length, bool clipTopDb, int blockThreads)
    {
        var extent = new DirectPtxExtent(length);
        return new DirectPtxKernelBlueprint(
            Operation: "amplitude-to-db-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"b{blockThreads}-n{length}-clip{(clipTopDb ? 1 : 0)}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 16,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "db = 20*log10(max(input, minAmp)); floored at topDbFloor when clipTopDb",
                ["mode"] = "inference-forward-amplitude-to-db",
                ["arithmetic"] = "lg2.approx scaled by 20*log10(2); tolerance-based parity, not bit-exact",
                ["scalars"] = "minAmp and topDbFloor are per-launch .param .f32; clipTopDb is baked",
                ["bounds-check"] = "single guard drops lanes past the element count",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    internal static bool IsSupportedShape(int length) => length >= 1 && length <= (1 << 26);

    internal static bool IsPromotedShape(int length) => false;

    private static void Validate(int length)
    {
        if (!IsSupportedShape(length))
            throw new ArgumentOutOfRangeException(nameof(length),
                "The amplitude-to-db family supports lengths in [1, 2^26].");
    }

    private static void ValidateBlockThreads(int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512))
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Amplitude-to-db block threads must be 128, 256, or 512.");
    }

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent ||
            view.ByteLength != contract.RequiredBytes ||
            view.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
    }
}
