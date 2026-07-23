using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxDenseVectorOperation
{
    Dot,
    Outer
}

/// <summary>
/// Exact-shape pointer-only PTX for the vector-product cells in issue #836.
/// Dot keeps partial sums in registers and performs one shared-memory block
/// reduction; outer writes each independent product exactly once.
/// </summary>
internal sealed class PtxDenseVectorKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const int WarpCount = BlockThreads / 32;
    internal const string DotEntryPoint = "aidotnet_dense_dot";
    internal const string OuterEntryPoint = "aidotnet_dense_outer";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal DirectPtxDenseVectorOperation Operation { get; }
    internal int M { get; }
    internal int N { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxDenseVectorKernel(
        DirectPtxRuntime runtime,
        DirectPtxDenseVectorOperation operation,
        int m,
        int n = 1)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedFusedLinear(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The dense-vector PTX specializations are measured only on GA10x/SM86.");
        Validate(operation, m, n);

        Operation = operation;
        M = m;
        N = n;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, operation, m, n);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, operation, m, n);
        _module = runtime.LoadModule(Ptx);
        string entry = operation == DirectPtxDenseVectorOperation.Dot
            ? DotEntryPoint : OuterEntryPoint;
        _function = _module.GetFunction(entry, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(entry, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info,
            BlockThreads, activeBlocks, _module);
    }

    internal unsafe void Launch(
        DirectPtxTensorView left,
        DirectPtxTensorView right,
        DirectPtxTensorView output)
    {
        Require(left, Blueprint.Tensors[0], nameof(left));
        Require(right, Blueprint.Tensors[1], nameof(right));
        Require(output, Blueprint.Tensors[2], nameof(output));
        if (Overlaps(output, left) || Overlaps(output, right))
            throw new ArgumentException("Dense-vector output may not alias an input.");

        IntPtr leftPointer = left.Pointer;
        IntPtr rightPointer = right.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &leftPointer;
        arguments[1] = &rightPointer;
        arguments[2] = &outputPointer;
        long workItems = Operation == DirectPtxDenseVectorOperation.Outer && (N & 3) == 0
            ? (long)M * (N / 4)
            : (long)M * N;
        uint grid = Operation == DirectPtxDenseVectorOperation.Dot
            ? 1u
            : checked((uint)((workItems + BlockThreads - 1) / BlockThreads));
        _module.Launch(
            _function, grid, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        DirectPtxDenseVectorOperation operation,
        int m,
        int n = 1)
    {
        Validate(operation, m, n);
        var ptx = new StringBuilder(8_192);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        if (operation == DirectPtxDenseVectorOperation.Dot)
            EmitDot(ptx, m);
        else
            EmitOuter(ptx, m, n);
        return ptx.ToString();
    }

    private static void EmitOuter(StringBuilder ptx, int m, int n)
    {
        if ((n & 3) == 0)
        {
            EmitVectorizedOuter(ptx, m, n);
            return;
        }

        int total = checked(m * n);
        ptx.AppendLine($"// exact outer product M={m} N={n}; pointer-only ABI");
        EmitHeader(ptx, OuterEntryPoint);
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<8>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        EmitPointers(ptx);
        ptx.AppendLine("    mov.u32 %r0, %ctaid.x;");
        ptx.AppendLine("    mov.u32 %r1, %tid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r0, {BlockThreads}, %r1;");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {total};");
        ptx.AppendLine("    @%p0 bra.uni OUTER_DONE;");
        ptx.AppendLine($"    div.u32 %r3, %r2, {n};");
        ptx.AppendLine($"    rem.u32 %r4, %r2, {n};");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd4];");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd5;");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd6];");
        ptx.AppendLine("    mul.rn.f32 %f2, %f0, %f1;");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd2, %rd7;");
        ptx.AppendLine("    st.global.f32 [%rd8], %f2;");
        ptx.AppendLine("OUTER_DONE:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
    }

    private static void EmitVectorizedOuter(StringBuilder ptx, int m, int n)
    {
        int vectorsPerRow = n / 4;
        int totalVectors = checked(m * vectorsPerRow);
        ptx.AppendLine($"// exact outer product M={m} N={n}; aligned float4 streaming specialization");
        EmitHeader(ptx, OuterEntryPoint);
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<8>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<10>;");
        EmitPointers(ptx);
        ptx.AppendLine("    mov.u32 %r0, %ctaid.x;");
        ptx.AppendLine("    mov.u32 %r1, %tid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r0, {BlockThreads}, %r1;");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {totalVectors};");
        ptx.AppendLine("    @%p0 bra.uni OUTER_DONE;");
        if (IsPowerOfTwo(vectorsPerRow))
        {
            ptx.AppendLine($"    shr.u32 %r3, %r2, {Log2(vectorsPerRow)};");
            ptx.AppendLine($"    and.b32 %r4, %r2, {vectorsPerRow - 1};");
        }
        else
        {
            ptx.AppendLine($"    div.u32 %r3, %r2, {vectorsPerRow};");
            ptx.AppendLine($"    rem.u32 %r4, %r2, {vectorsPerRow};");
        }
        ptx.AppendLine("    mul.wide.u32 %rd3, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd4];");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r4, 16;");
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd5;");
        ptx.AppendLine("    ld.global.nc.v4.f32 {%f1,%f2,%f3,%f4}, [%rd6];");
        ptx.AppendLine("    mul.rn.f32 %f5, %f0, %f1;");
        ptx.AppendLine("    mul.rn.f32 %f6, %f0, %f2;");
        ptx.AppendLine("    mul.rn.f32 %f7, %f0, %f3;");
        ptx.AppendLine("    mul.rn.f32 %f8, %f0, %f4;");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r2, 16;");
        ptx.AppendLine("    add.u64 %rd8, %rd2, %rd7;");
        ptx.AppendLine("    st.global.v4.f32 [%rd8], {%f5,%f6,%f7,%f8};");
        ptx.AppendLine("OUTER_DONE:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
    }

    private static void EmitDot(StringBuilder ptx, int length)
    {
        bool vectorized = (length & 3) == 0;
        ptx.AppendLine($"// exact dot product K={length}; {(vectorized ? "aligned float4 streaming; " : string.Empty)}one register/shared reduction, one global store");
        EmitHeader(ptx, DotEntryPoint);
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<10>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine($"    .reg .f32 %f<{(vectorized ? 13 : 6)}>;");
        ptx.AppendLine($"    .shared .align 16 .b8 partial[{WarpCount * sizeof(float)}];");
        EmitPointers(ptx);
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine(vectorized
            ? "    shl.b32 %r1, %r0, 2;"
            : "    mov.u32 %r1, %r0;");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        if (vectorized)
        {
            ptx.AppendLine("    mov.f32 %f6, 0f00000000;");
            ptx.AppendLine("    mov.f32 %f7, 0f00000000;");
            ptx.AppendLine("    mov.f32 %f8, 0f00000000;");
        }
        ptx.AppendLine("DOT_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r1, {length};");
        ptx.AppendLine("    @%p0 bra.uni DOT_REDUCE;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r1, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd3;");
        if (vectorized)
        {
            ptx.AppendLine("    ld.global.nc.v4.f32 {%f1,%f2,%f3,%f4}, [%rd4];");
            ptx.AppendLine("    ld.global.nc.v4.f32 {%f9,%f10,%f11,%f12}, [%rd5];");
            ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f9, %f0;");
            ptx.AppendLine("    fma.rn.f32 %f6, %f2, %f10, %f6;");
            ptx.AppendLine("    fma.rn.f32 %f7, %f3, %f11, %f7;");
            ptx.AppendLine("    fma.rn.f32 %f8, %f4, %f12, %f8;");
            ptx.AppendLine($"    add.u32 %r1, %r1, {BlockThreads * 4};");
        }
        else
        {
            ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd4];");
            ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd5];");
            ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
            ptx.AppendLine($"    add.u32 %r1, %r1, {BlockThreads};");
        }
        ptx.AppendLine("    bra.uni DOT_LOOP;");
        ptx.AppendLine("DOT_REDUCE:");
        if (vectorized)
        {
            ptx.AppendLine("    add.rn.f32 %f0, %f0, %f6;");
            ptx.AppendLine("    add.rn.f32 %f7, %f7, %f8;");
            ptx.AppendLine("    add.rn.f32 %f0, %f0, %f7;");
        }
        ptx.AppendLine("    mov.u64 %rd6, partial;");
        ptx.AppendLine("    and.b32 %r3, %r0, 31;");
        ptx.AppendLine("    shr.u32 %r4, %r0, 5;");
        EmitWarpReduction(ptx, "%f0", "%f3", "%r5", "%r6");
        ptx.AppendLine("    setp.ne.u32 %p1, %r3, 0;");
        ptx.AppendLine("    @%p1 bra.uni DOT_WARP_PUBLISHED;");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd6, %rd7;");
        ptx.AppendLine("    st.shared.f32 [%rd8], %f0;");
        ptx.AppendLine("DOT_WARP_PUBLISHED:");
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine("    setp.ne.u32 %p2, %r4, 0;");
        ptx.AppendLine("    @%p2 bra.uni DOT_DONE;");
        ptx.AppendLine($"    setp.lt.u32 %p3, %r3, {WarpCount};");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd6, %rd9;");
        ptx.AppendLine("    @%p3 ld.shared.f32 %f0, [%rd10];");
        EmitWarpReduction(ptx, "%f0", "%f3", "%r5", "%r6");
        ptx.AppendLine("    setp.ne.u32 %p2, %r3, 0;");
        ptx.AppendLine("    @%p2 bra.uni DOT_DONE;");
        ptx.AppendLine("    st.global.f32 [%rd2], %f0;");
        ptx.AppendLine("DOT_DONE:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
    }

    private static void EmitWarpReduction(
        StringBuilder ptx,
        string value,
        string shuffled,
        string valueBits,
        string shuffledBits)
    {
        foreach (int offset in new[] { 16, 8, 4, 2, 1 })
        {
            ptx.AppendLine($"    mov.b32 {valueBits}, {value};");
            ptx.AppendLine(
                $"    shfl.sync.down.b32 {shuffledBits}, {valueBits}, {offset}, 31, 0xffffffff;");
            ptx.AppendLine($"    mov.b32 {shuffled}, {shuffledBits};");
            ptx.AppendLine($"    add.rn.f32 {value}, {value}, {shuffled};");
        }
    }

    private static void EmitHeader(StringBuilder ptx, string entry)
    {
        ptx.AppendLine($".visible .entry {entry}(");
        ptx.AppendLine("    .param .u64 left_ptr,");
        ptx.AppendLine("    .param .u64 right_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
    }

    private static void EmitPointers(StringBuilder ptx)
    {
        ptx.AppendLine("    ld.param.u64 %rd0, [left_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [right_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        DirectPtxDenseVectorOperation operation,
        int m,
        int n)
    {
        bool dot = operation == DirectPtxDenseVectorOperation.Dot;
        bool vectorizedDot = dot && (m & 3) == 0;
        bool vectorizedOuter = !dot && (n & 3) == 0;
        var left = new DirectPtxExtent(m);
        var right = new DirectPtxExtent(dot ? m : n);
        var output = dot ? new DirectPtxExtent(1) : new DirectPtxExtent(m, n);
        return new DirectPtxKernelBlueprint(
            Operation: dot ? "dense-dot" : "dense-outer",
            Version: vectorizedDot || vectorizedOuter ? 2 : 1,
            Architecture: architecture,
            Variant: dot ? $"{(vectorizedDot ? "fp32x4" : "fp32")}-k{m}" :
                $"{(vectorizedOuter ? "fp32x4" : "fp32")}-m{m}-n{n}",
            Tensors:
            [
                new("left", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    left, left, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("right", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    right, right, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32,
                    dot ? DirectPtxPhysicalLayout.Vector : DirectPtxPhysicalLayout.RowMajor2D,
                    output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 32,
                MaxStaticSharedBytes: dot ? WarpCount * sizeof(float) : 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: dot ? 1 : 4),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = dot ? "sum(left[K] * right[K])" : "left[M] outer right[N]",
                ["dtype"] = "fp32",
                ["shape-parameters"] = "none",
                ["stride-parameters"] = "none",
                ["temporary-device-allocation"] = "none",
                ["global-intermediates"] = "none",
                ["memory-pipeline"] = vectorizedOuter
                    ? "aligned float4 load/multiply/store; one output write"
                    : vectorizedDot
                        ? "aligned float4 loads; register/shared reduction; one output write"
                        : "scalar register path; one output write"
            });
    }

    private static bool IsPowerOfTwo(int value) => value > 0 && (value & (value - 1)) == 0;

    private static int Log2(int value)
    {
        int result = 0;
        while ((value >>= 1) != 0) result++;
        return result;
    }

    private static void Validate(DirectPtxDenseVectorOperation operation, int m, int n)
    {
        if (!Enum.IsDefined(typeof(DirectPtxDenseVectorOperation), operation))
            throw new ArgumentOutOfRangeException(nameof(operation));
        if (m <= 0 || m > 1_048_576)
            throw new ArgumentOutOfRangeException(nameof(m));
        if (operation == DirectPtxDenseVectorOperation.Outer && (n <= 0 || n > 65_536))
            throw new ArgumentOutOfRangeException(nameof(n));
        _ = checked(m * n);
    }

    private static void Require(
        DirectPtxTensorView view,
        DirectPtxTensorContract contract,
        string parameter)
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
