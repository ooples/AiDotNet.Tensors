using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Fused per-row L2 normalization for issue #850, matching the NVRTC <c>normalize_rows_fused</c> kernel:
/// <c>output[r, c] = input[r, c] / ||input[r, :]||_2</c> (rows with zero norm pass through as zero). One
/// block of 256 threads owns one row: the threads grid-stride over the columns accumulating the sum of
/// squares into static shared memory, tree-reduce it under barriers, take <c>rsqrt.approx</c> of the total,
/// then grid-stride again writing the scaled output. The reciprocal-norm uses <c>rsqrt.approx</c> (as the
/// reference <c>rsqrtf</c> does), so the spec is TOLERANCE-based. <c>rows</c> and <c>cols</c> are baked; the
/// launch uses one 256-thread block per row. Two pointers reach the launch ABI.
///
/// The specialization stays disabled by default and fails closed until three clean promotion runs clear
/// the release gate.
/// </summary>
internal sealed class PtxNormalizeRowsFusedF32Kernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const string EntryPoint = "aidotnet_normalize_rows_fused_f32";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Rows { get; }
    internal int Cols { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxNormalizeRowsFusedF32Kernel(DirectPtxRuntime runtime, int rows, int cols)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedComplexUnary(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in normalize-rows-fused specialization is admitted only on SM86.");
        Validate(rows, cols);
        Rows = rows;
        Cols = cols;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, rows, cols);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, rows, cols);
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

        IntPtr inputPointer = input.Pointer, outputPointer = output.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &inputPointer;
        arguments[1] = &outputPointer;
        _module.Launch(
            _function,
            checked((uint)Rows), 1, 1,
            BlockThreads, 1, 1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int rows, int cols)
    {
        Validate(rows, cols);

        var ptx = new StringBuilder(3_584);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// exact-shape rows={rows} cols={cols} block={BlockThreads} op=normalize-rows-fused");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine($"    .shared .align 4 .b32 sdata[{BlockThreads}];");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<6>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");                      // row
        ptx.AppendLine($"    mul.lo.u32 %r2, %r1, {cols};");             // rowOff
        ptx.AppendLine("    mov.u32 %r3, sdata;");                         // shared base
        ptx.AppendLine("    shl.b32 %r4, %r0, 2;");
        ptx.AppendLine("    add.u32 %r5, %r3, %r4;");                     // &sdata[tid]
        // Phase 1: sum of squares, grid-stride over cols
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");                    // local
        ptx.AppendLine("    mov.u32 %r6, %r0;");                           // c = tid
        ptx.AppendLine("$NR_P1:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r6, {cols};");
        ptx.AppendLine("    @%p0 bra $NR_P1END;");
        ptx.AppendLine("    add.u32 %r7, %r2, %r6;");
        ptx.AppendLine("    mul.wide.u32 %rd2, %r7, 4;");
        ptx.AppendLine("    add.u64 %rd3, %rd0, %rd2;");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd3];");
        ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f1, %f0;");             // local += v*v
        ptx.AppendLine($"    add.u32 %r6, %r6, {BlockThreads};");
        ptx.AppendLine("    bra $NR_P1;");
        ptx.AppendLine("$NR_P1END:");
        ptx.AppendLine("    st.shared.f32 [%r5], %f0;");
        ptx.AppendLine("    bar.sync 0;");
        // Tree reduction: for (s = 128; s > 0; s >>= 1)
        ptx.AppendLine("    mov.u32 %r8, 128;");
        ptx.AppendLine("$NR_RED:");
        ptx.AppendLine("    setp.eq.u32 %p1, %r8, 0;");
        ptx.AppendLine("    @%p1 bra $NR_REDEND;");
        ptx.AppendLine("    setp.lt.u32 %p2, %r0, %r8;");
        ptx.AppendLine("    @%p2 shl.b32 %r9, %r8, 2;");
        ptx.AppendLine("    @%p2 add.u32 %r10, %r5, %r9;");    // &sdata[tid+s]
        ptx.AppendLine("    @%p2 ld.shared.f32 %f2, [%r5];");
        ptx.AppendLine("    @%p2 ld.shared.f32 %f3, [%r10];");
        ptx.AppendLine("    @%p2 add.rn.f32 %f2, %f2, %f3;");
        ptx.AppendLine("    @%p2 st.shared.f32 [%r5], %f2;");
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine("    shr.u32 %r8, %r8, 1;");
        ptx.AppendLine("    bra $NR_RED;");
        ptx.AppendLine("$NR_REDEND:");
        ptx.AppendLine("    ld.shared.f32 %f4, [%r3];");                  // sdata[0] = sumSq
        ptx.AppendLine("    setp.gt.f32 %p3, %f4, 0f00000000;");
        ptx.AppendLine("    rsqrt.approx.f32 %f5, %f4;");
        ptx.AppendLine("    selp.f32 %f5, %f5, 0f00000000, %p3;");        // invNorm
        // Phase 2: write normalized output, grid-stride over cols
        ptx.AppendLine("    mov.u32 %r6, %r0;");                           // c = tid
        ptx.AppendLine("$NR_P2:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r6, {cols};");
        ptx.AppendLine("    @%p0 bra $NR_RET;");
        ptx.AppendLine("    add.u32 %r7, %r2, %r6;");
        ptx.AppendLine("    mul.wide.u32 %rd4, %r7, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd0, %rd4;");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd5];");
        ptx.AppendLine("    mul.rn.f32 %f1, %f1, %f5;");
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd4;");
        ptx.AppendLine("    st.global.f32 [%rd6], %f1;");
        ptx.AppendLine($"    add.u32 %r6, %r6, {BlockThreads};");
        ptx.AppendLine("    bra $NR_P2;");
        ptx.AppendLine("$NR_RET:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int rows, int cols)
    {
        var extent = new DirectPtxExtent(checked(rows * cols));
        return new DirectPtxKernelBlueprint(
            Operation: "normalize-rows-fused-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"b{BlockThreads}-r{rows}-c{cols}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 24,
                MaxStaticSharedBytes: BlockThreads * 4,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "output[r,c] = input[r,c] * rsqrt(sum_c input[r,c]^2); zero-norm rows -> zero",
                ["mode"] = "inference-forward-normalize-rows-fused",
                ["arithmetic"] = "fma sum-of-squares, shared-memory tree reduction, rsqrt.approx; tolerance-based parity",
                ["shared-memory"] = "256 partial sums (static), one 256-thread block per row",
                ["reduction"] = "log2(256)=8-step tree reduction under bar.sync barriers",
                ["bounds-check"] = "grid covers exactly the rows; grid-stride loops cover the columns",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    internal static bool IsSupportedShape(int rows, int cols) =>
        rows >= 1 && cols >= 1 && (long)rows * cols <= (1L << 26);

    internal static bool IsPromotedShape(int rows, int cols) => false;

    private static void Validate(int rows, int cols)
    {
        if (!IsSupportedShape(rows, cols))
            throw new ArgumentOutOfRangeException(nameof(cols),
                "The normalize-rows-fused family requires rows>=1, cols>=1, and rows*cols<=2^26.");
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
