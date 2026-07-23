using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// ISTFT window normalization for issue #850, matching the NVRTC <c>parity210_istft_normalize</c> kernel:
/// divide the overlap-added reconstruction in place by its window-sum-of-squares, guarded against tiny
/// denominators - <c>result[i] = windowSum[i] &gt; 1e-8 ? result[i] / windowSum[i] : result[i]</c>. One
/// thread owns one sample; the division is a <c>div.rn</c> matching the reference and the guard is a
/// predicated <c>selp</c>, so the spec is bit-exact. This is the final stage of the fused-spectrum ISTFT
/// path. <c>total</c> is baked; the launch rounds up and a single guard drops the tail lanes. Two pointers
/// reach the launch ABI.
///
/// The specialization stays disabled by default and fails closed until three clean promotion runs clear
/// the release gate.
/// </summary>
internal sealed class PtxIstftNormalizeF32Kernel : IDisposable
{
    internal const int DefaultBlockThreads = 256;
    internal const string EntryPoint = "aidotnet_istft_normalize_f32";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Total { get; }
    internal int BlockThreads { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxIstftNormalizeF32Kernel(DirectPtxRuntime runtime, int total, int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedComplexUnary(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in ISTFT normalize specialization is admitted only on SM86.");
        Validate(total);
        ValidateBlockThreads(blockThreads);
        Total = total;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, total, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, total, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(DirectPtxTensorView result, DirectPtxTensorView windowSum)
    {
        Require(result, Blueprint.Tensors[0], nameof(result));
        Require(windowSum, Blueprint.Tensors[1], nameof(windowSum));

        IntPtr resultPointer = result.Pointer, windowSumPointer = windowSum.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &resultPointer;
        arguments[1] = &windowSumPointer;
        _module.Launch(
            _function,
            (uint)((Total + BlockThreads - 1) / BlockThreads), 1, 1,
            checked((uint)BlockThreads), 1, 1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    private static string Hex(float value) => "0f" + BitConverter.ToInt32(BitConverter.GetBytes(value), 0).ToString("X8");

    internal static string EmitPtx(int ccMajor, int ccMinor, int total, int blockThreads = DefaultBlockThreads)
    {
        Validate(total);
        ValidateBlockThreads(blockThreads);
        string tiny = Hex(1e-8f);

        var ptx = new StringBuilder(1_792);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// exact-shape total={total} block={blockThreads} op=istft-normalize");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 result_ptr,");
        ptx.AppendLine("    .param .u64 windowsum_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<3>;");
        ptx.AppendLine("    .reg .b32 %r<3>;");
        ptx.AppendLine("    .reg .b64 %rd<8>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [result_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [windowsum_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {blockThreads}, %r0;");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {total};");
        ptx.AppendLine("    @%p0 bra $ISN_RET;");
        ptx.AppendLine("    mul.wide.u32 %rd2, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd3, %rd0, %rd2;");
        ptx.AppendLine("    add.u64 %rd4, %rd1, %rd2;");
        ptx.AppendLine("    ld.global.f32 %f0, [%rd3];");        // result
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd4];");     // windowSum
        ptx.AppendLine("    div.rn.f32 %f2, %f0, %f1;");         // result / windowSum
        ptx.AppendLine($"    setp.gt.f32 %p1, %f1, {tiny};");
        ptx.AppendLine("    selp.f32 %f0, %f2, %f0, %p1;");      // ws > 1e-8 ? divided : original
        ptx.AppendLine("    st.global.f32 [%rd3], %f0;");
        ptx.AppendLine("$ISN_RET:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int total, int blockThreads)
    {
        var extent = new DirectPtxExtent(total);
        return new DirectPtxKernelBlueprint(
            Operation: "istft-normalize-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"b{blockThreads}-n{total}",
            Tensors:
            [
                new("result", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.ReadWrite, DirectPtxExtentMode.Exact),
                new("windowSum", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 16,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "result[i] = windowSum[i] > 1e-8 ? result[i] / windowSum[i] : result[i]",
                ["mode"] = "inference-forward-istft-normalize",
                ["arithmetic"] = "div.rn guarded by a predicated selp; bit-exact against the reference",
                ["bounds-check"] = "single guard drops lanes past total",
                ["global-intermediates"] = "in-place on result",
                ["temporary-device-allocation"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    internal static bool IsSupportedShape(int total) => total >= 1 && total <= (1 << 26);

    internal static bool IsPromotedShape(int total) => false;

    private static void Validate(int total)
    {
        if (!IsSupportedShape(total))
            throw new ArgumentOutOfRangeException(nameof(total),
                "The ISTFT normalize family supports totals in [1, 2^26].");
    }

    private static void ValidateBlockThreads(int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512))
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "ISTFT normalize block threads must be 128, 256, or 512.");
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
