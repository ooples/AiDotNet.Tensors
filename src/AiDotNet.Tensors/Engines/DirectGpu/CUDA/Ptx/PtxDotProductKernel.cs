using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// FP32 dot product <c>output[0] = sum_k a[k] * b[k]</c>. A single 256-thread block
/// (8 warps): each thread accumulates a strided slice of K in a register, then a
/// butterfly-shuffle warp reduction, an 8-partial shared reduction, and a final
/// warp reduction leave the scalar in lane 0, which writes it. Register-resident
/// apart from 32 bytes of shared partials. K must be a multiple of 256.
/// </summary>
internal sealed class PtxDotProductKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const string EntryPoint = "aidotnet_dot_product";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int K { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxDotProductKernel(DirectPtxRuntime runtime, int k)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedFusedLinear(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in dot-product specialization is measured only on GA10x/SM86.");
        ValidateShape(k);
        K = k;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, k);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, k);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module);
    }

    internal unsafe void Launch(DirectPtxTensorView a, DirectPtxTensorView b, DirectPtxTensorView output)
    {
        Require(a, Blueprint.Tensors[0], nameof(a));
        Require(b, Blueprint.Tensors[1], nameof(b));
        Require(output, Blueprint.Tensors[2], nameof(output));
        if (Overlaps(output, a) || Overlaps(output, b))
            throw new ArgumentException("Dot-product output may not alias a or b.");

        IntPtr aPointer = a.Pointer;
        IntPtr bPointer = b.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &aPointer;
        arguments[1] = &bPointer;
        arguments[2] = &outputPointer;
        _module.Launch(_function, 1, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int k)
    {
        ValidateShape(k);
        var ptx = new StringBuilder(4_096);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// dot product K={k}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 a_ptr,");
        ptx.AppendLine("    .param .u64 b_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<12>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<8>;");
        ptx.AppendLine("    .shared .align 4 .b8 partial[32];");
        ptx.AppendLine("    ld.param.u64 %rd0, [a_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [b_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
        ptx.AppendLine("    mov.u64 %rd3, partial;");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    and.b32 %r1, %r0, 31;");                           // lane
        ptx.AppendLine("    shr.u32 %r2, %r0, 5;");                            // warpId
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");                        // acc
        ptx.AppendLine("    mov.u32 %r3, %r0;");                               // kk = tid
        ptx.AppendLine("DOT_LOOP:");
        ptx.AppendLine("    mul.wide.u32 %rd4, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd0, %rd4;");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd5];");                   // a[kk]
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd4;");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd6];");                   // b[kk]
        ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        ptx.AppendLine($"    add.u32 %r3, %r3, {BlockThreads};");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r3, {k};");
        ptx.AppendLine("    @%p0 bra.uni DOT_LOOP;");
        EmitWarpReduce(ptx);                                                   // reduce %f0 within warp
        ptx.AppendLine("    setp.eq.u32 %p1, %r1, 0;");                        // lane 0
        ptx.AppendLine("    mul.lo.u32 %r4, %r2, 4;");
        ptx.AppendLine("    cvt.u64.u32 %rd7, %r4;");
        ptx.AppendLine("    add.u64 %rd8, %rd3, %rd7;");
        ptx.AppendLine("    @%p1 st.shared.f32 [%rd8], %f0;");                 // partial[warpId]
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine("    setp.ne.u32 %p2, %r2, 0;");                        // not warp 0
        ptx.AppendLine("    @%p2 bra DOT_DONE;");
        ptx.AppendLine("    setp.lt.u32 %p1, %r1, 8;");                        // first 8 lanes
        ptx.AppendLine("    mul.lo.u32 %r5, %r1, 4;");
        ptx.AppendLine("    cvt.u64.u32 %rd9, %r5;");
        ptx.AppendLine("    add.u64 %rd10, %rd3, %rd9;");
        ptx.AppendLine("    @%p1 ld.shared.f32 %f0, [%rd10];");
        ptx.AppendLine("    @!%p1 mov.f32 %f0, 0f00000000;");
        EmitWarpReduce(ptx);                                                   // reduce the 8 partials
        ptx.AppendLine("    setp.eq.u32 %p1, %r0, 0;");                        // thread 0
        ptx.AppendLine("    @%p1 st.global.f32 [%rd2], %f0;");
        ptx.AppendLine("DOT_DONE:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    // ISA-correct butterfly warp reduction of %f0 (reinterpret through %r10/%r11).
    private static void EmitWarpReduce(StringBuilder ptx)
    {
        foreach (int delta in new[] { 16, 8, 4, 2, 1 })
        {
            ptx.AppendLine("    mov.b32 %r10, %f0;");
            ptx.AppendLine($"    shfl.sync.bfly.b32 %r11, %r10, {delta}, 31, 0xffffffff;");
            ptx.AppendLine("    mov.b32 %f3, %r11;");
            ptx.AppendLine("    add.rn.f32 %f0, %f0, %f3;");
        }
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, int k)
    {
        var vec = new DirectPtxExtent(k);
        var scalar = new DirectPtxExtent(1);
        return new DirectPtxKernelBlueprint(
            Operation: "dot-product",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp32-k{k}",
            Tensors:
            [
                new("a", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    vec, vec, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("b", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    vec, vec, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    scalar, scalar, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 24,
                MaxStaticSharedBytes: 32,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "output[0] = sum_k a[k] * b[k]",
                ["reduction"] = "warp-shuffle-then-8-partial-shared-then-warp",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int k) =>
        k > 0 && k % BlockThreads == 0 && k is 256 or 512 or 1024 or 2048 or 4096;

    internal static bool IsPromotedShape(int k) => false;

    private static void ValidateShape(int k)
    {
        if (!IsSupportedShape(k))
            throw new ArgumentOutOfRangeException(
                nameof(k), "Dot product supports K in {256,512,1024,2048,4096} (multiples of 256).");
    }

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent || view.ByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
    }

    private static bool Overlaps(DirectPtxTensorView left, DirectPtxTensorView right)
    {
        nuint leftStart = PtxCompat.ToNuint(left.Pointer);
        nuint rightStart = PtxCompat.ToNuint(right.Pointer);
        nuint leftEnd = checked(leftStart + left.ByteLength);
        nuint rightEnd = checked(rightStart + right.ByteLength);
        return leftStart < rightEnd && rightStart < leftEnd;
    }
}
