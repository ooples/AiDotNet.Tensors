using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Batched contiguous circular roll for issue #850, backing the <c>Fft.FftShift</c> / <c>Fft.IFftShift</c>
/// variants (and any per-axis fft roll): <c>output[b, i] = input[b, (i - shift) mod dim]</c> along the
/// canonical last axis, flattened over the leading batch. <c>fftshift</c> rolls by <c>dim/2</c> (floor) and
/// <c>ifftshift</c> by <c>dim - dim/2</c> (ceil); the caller passes the physical roll amount, so the same
/// kernel serves both directions and both the real and complex-interleaved axes (the engine passes a
/// pair-aligned shift for the interleaved case). It is pure bit-exact data movement - each output reads one
/// input element with a single wrap-around subtraction, so NaN payloads and signed zeros pass through
/// unchanged. <c>dim</c>, <c>shift</c>, and <c>batch</c> are baked; the launch rounds up and a single guard
/// drops the tail lanes. Two pointers reach the launch ABI.
///
/// The specialization stays disabled by default and fails closed until three clean promotion runs clear
/// the release gate.
/// </summary>
internal sealed class PtxFftRollF32Kernel : IDisposable
{
    internal const int DefaultBlockThreads = 256;
    internal const string EntryPoint = "aidotnet_fft_roll_f32";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Dim { get; }
    internal int Shift { get; }
    internal int Batch { get; }
    internal int BlockThreads { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFftRollF32Kernel(
        DirectPtxRuntime runtime, int dim, int shift, int batch, int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedComplexMultiply(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in fft-roll specialization is admitted only on SM86.");
        Validate(dim, shift, batch);
        ValidateBlockThreads(blockThreads);
        Dim = dim;
        Shift = shift;
        Batch = batch;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, dim, shift, batch, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, dim, shift, batch, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    /// <summary>The fftshift roll amount for a physical axis length <paramref name="dim"/> (floor(dim/2)).</summary>
    internal static int FftShiftAmount(int dim) => dim / 2;

    /// <summary>The ifftshift roll amount for a physical axis length <paramref name="dim"/> (ceil(dim/2)).</summary>
    internal static int IFftShiftAmount(int dim) => dim - dim / 2;

    internal unsafe void Launch(DirectPtxTensorView input, DirectPtxTensorView output)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(output, Blueprint.Tensors[1], nameof(output));

        IntPtr inputPointer = input.Pointer, outputPointer = output.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &inputPointer;
        arguments[1] = &outputPointer;
        int total = Batch * Dim;
        _module.Launch(
            _function,
            (uint)((total + BlockThreads - 1) / BlockThreads), 1, 1,
            checked((uint)BlockThreads), 1, 1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int dim, int shift, int batch, int blockThreads = DefaultBlockThreads)
    {
        Validate(dim, shift, batch);
        ValidateBlockThreads(blockThreads);
        int total = checked(batch * dim);
        int forward = dim - shift;   // output[i] = input[(i + (dim - shift)) mod dim]

        var ptx = new StringBuilder(2_048);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// exact-shape dim={dim} shift={shift} batch={batch} block={blockThreads} op=fft-roll");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<3>;");
        ptx.AppendLine("    .reg .b32 %r<8>;");
        ptx.AppendLine("    .reg .b64 %rd<10>;");
        ptx.AppendLine("    .reg .f32 %f<2>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {blockThreads}, %r0;");   // idx
        ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {total};");
        ptx.AppendLine("    @%p0 bra $ROLL_RET;");
        ptx.AppendLine($"    rem.u32 %r3, %r2, {dim};");                   // i (position in row)
        ptx.AppendLine("    sub.u32 %r4, %r2, %r3;");                      // base = idx - i = b*dim
        ptx.AppendLine($"    add.u32 %r5, %r3, {forward};");              // i + (dim - shift)
        ptx.AppendLine($"    setp.ge.u32 %p1, %r5, {dim};");
        ptx.AppendLine($"    @%p1 sub.u32 %r5, %r5, {dim};");            // mod dim (single wrap)
        ptx.AppendLine("    add.u32 %r6, %r4, %r5;");                     // srcElem = base + src
        ptx.AppendLine("    mul.wide.u32 %rd2, %r6, 4;");
        ptx.AppendLine("    add.u64 %rd3, %rd0, %rd2;");
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd3];");
        ptx.AppendLine("    mul.wide.u32 %rd4, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd4;");
        ptx.AppendLine("    st.global.f32 [%rd5], %f0;");
        ptx.AppendLine("$ROLL_RET:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int dim, int shift, int batch, int blockThreads)
    {
        var extent = new DirectPtxExtent(checked(batch * dim));
        return new DirectPtxKernelBlueprint(
            Operation: "fft-roll-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"b{blockThreads}-dim{dim}-shift{shift}-batch{batch}",
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
                ["formula"] = "output[b,i] = input[b, (i - shift) mod dim]",
                ["mode"] = "inference-forward-fft-roll",
                ["arithmetic"] = "none - pure bit-exact data movement; NaN payloads and signed zeros preserved",
                ["shift"] = "fftshift=floor(dim/2), ifftshift=ceil(dim/2); caller passes physical roll amount",
                ["bounds-check"] = "single guard drops lanes past batch*dim",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    internal static bool IsSupportedShape(int dim, int shift, int batch) =>
        dim >= 2 && batch >= 1 && shift >= 1 && shift < dim &&
        (long)batch * dim <= (1L << 26);

    internal static bool IsPromotedShape(int dim, int shift, int batch) => false;

    private static void Validate(int dim, int shift, int batch)
    {
        if (!IsSupportedShape(dim, shift, batch))
            throw new ArgumentOutOfRangeException(nameof(dim),
                "The fft-roll family requires dim>=2, batch>=1, 1<=shift<dim, and batch*dim<=2^26.");
    }

    private static void ValidateBlockThreads(int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512))
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Fft-roll block threads must be 128, 256, or 512.");
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
