using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxBatchedVectorOperation
{
    Dot,
    Outer
}

/// <summary>
/// Shape-baked pointer-only PTX for batched dot and batched outer products.
/// Batch and logical dimensions are compile-time constants; dispatch performs
/// no shape, layout, or runtime-stride interpretation.
/// </summary>
internal sealed class PtxBatchedVectorKernel : IDisposable
{
    internal const int BaselineBlockThreads = 256;
    internal const int WarpBlockThreads = 32;
    internal const int VectorBlockThreads = 32;
    internal const int WarpCount = BaselineBlockThreads / 32;
    internal const string DotEntryPoint = "aidotnet_batched_dot";
    internal const string OuterEntryPoint = "aidotnet_batched_outer";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal DirectPtxBatchedVectorOperation Operation { get; }
    internal int Batch { get; }
    internal int M { get; }
    internal int N { get; }
    internal int BlockThreads { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxBatchedVectorKernel(
        DirectPtxRuntime runtime,
        DirectPtxBatchedVectorOperation operation,
        int batch,
        int m,
        int n = 1)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedFusedLinear(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The batched-vector PTX specializations are measured only on GA10x/SM86.");
        Validate(operation, batch, m, n);

        Operation = operation;
        Batch = batch;
        M = m;
        N = n;
        BlockThreads = GetBlockThreads(operation, batch, m);
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, operation, batch, m, n);
        Ptx = EmitPtx(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            operation, batch, m, n);
        _module = runtime.LoadModule(Ptx);
        string entry = operation == DirectPtxBatchedVectorOperation.Dot
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
            throw new ArgumentException("Batched-vector output may not alias an input.");

        IntPtr leftPointer = left.Pointer;
        IntPtr rightPointer = right.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &leftPointer;
        arguments[1] = &rightPointer;
        arguments[2] = &outputPointer;
        uint grid = Operation == DirectPtxBatchedVectorOperation.Dot
            ? checked((uint)Batch)
            : checked((uint)(((long)Batch * M * N + BaselineBlockThreads - 1) /
                BaselineBlockThreads));
        _module.Launch(
            _function, grid, 1, 1, checked((uint)BlockThreads), 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        DirectPtxBatchedVectorOperation operation,
        int batch,
        int m,
        int n = 1)
    {
        Validate(operation, batch, m, n);
        var ptx = new StringBuilder(8_192);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        if (operation == DirectPtxBatchedVectorOperation.Dot)
            EmitDot(ptx, batch, m, GetBlockThreads(operation, batch, m));
        else
            EmitOuter(ptx, batch, m, n);
        return ptx.ToString();
    }

    private static void EmitOuter(StringBuilder ptx, int batch, int m, int n)
    {
        int matrixElements = checked(m * n);
        int total = checked(batch * matrixElements);
        ptx.AppendLine($"// exact batched outer B={batch} M={m} N={n}; pointer-only ABI");
        EmitHeader(ptx, OuterEntryPoint);
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<12>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        EmitPointers(ptx);
        ptx.AppendLine("    mov.u32 %r0, %ctaid.x;");
        ptx.AppendLine("    mov.u32 %r1, %tid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r0, {BaselineBlockThreads}, %r1;");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {total};");
        ptx.AppendLine("    @%p0 bra.uni BATCH_OUTER_DONE;");
        ptx.AppendLine($"    div.u32 %r3, %r2, {matrixElements};");
        ptx.AppendLine($"    rem.u32 %r4, %r2, {matrixElements};");
        ptx.AppendLine($"    div.u32 %r5, %r4, {n};");
        ptx.AppendLine($"    rem.u32 %r6, %r4, {n};");
        ptx.AppendLine($"    mad.lo.u32 %r7, %r3, {m}, %r5;");
        ptx.AppendLine($"    mad.lo.u32 %r8, %r3, {n}, %r6;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r7, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd4];");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r8, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd5;");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd6];");
        ptx.AppendLine("    mul.rn.f32 %f2, %f0, %f1;");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd2, %rd7;");
        ptx.AppendLine("    st.global.f32 [%rd8], %f2;");
        ptx.AppendLine("BATCH_OUTER_DONE:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
    }

    private static void EmitDot(StringBuilder ptx, int batch, int dimension, int blockThreads)
    {
        if (IsFusedBatchDot(DirectPtxBatchedVectorOperation.Dot, batch, dimension))
        {
            EmitFusedBatchDot(ptx, batch, dimension, blockThreads);
            return;
        }

        if (IsWarpPerBatchDot(DirectPtxBatchedVectorOperation.Dot, batch, dimension))
        {
            EmitWarpPerBatchDot(ptx, batch, dimension, blockThreads);
            return;
        }

        int batchBytes = checked(dimension * sizeof(float));
        ptx.AppendLine($"// exact batched dot B={batch} D={dimension}; one block and one store per batch");
        EmitHeader(ptx, DotEntryPoint, blockThreads);
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<12>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<6>;");
        ptx.AppendLine($"    .shared .align 16 .b8 partial[{WarpCount * sizeof(float)}];");
        EmitPointers(ptx);
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mul.wide.u32 %rd3, %r1, {batchBytes};");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd3;");
        ptx.AppendLine("    mov.u32 %r2, %r0;");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("BATCH_DOT_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {dimension};");
        ptx.AppendLine("    @%p0 bra.uni BATCH_DOT_REDUCE;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd4, %rd6;");
        ptx.AppendLine("    add.u64 %rd8, %rd5, %rd6;");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd7];");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd8];");
        ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        ptx.AppendLine($"    add.u32 %r2, %r2, {blockThreads};");
        ptx.AppendLine("    bra.uni BATCH_DOT_LOOP;");
        ptx.AppendLine("BATCH_DOT_REDUCE:");
        ptx.AppendLine("    mov.u64 %rd9, partial;");
        ptx.AppendLine("    and.b32 %r3, %r0, 31;");
        ptx.AppendLine("    shr.u32 %r4, %r0, 5;");
        EmitWarpReduction(ptx, "%f0", "%f3", "%r5", "%r6");
        ptx.AppendLine("    setp.ne.u32 %p1, %r3, 0;");
        ptx.AppendLine("    @%p1 bra.uni BATCH_DOT_WARP_PUBLISHED;");
        ptx.AppendLine("    mul.wide.u32 %rd10, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd11, %rd9, %rd10;");
        ptx.AppendLine("    st.shared.f32 [%rd11], %f0;");
        ptx.AppendLine("BATCH_DOT_WARP_PUBLISHED:");
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine("    setp.ne.u32 %p2, %r4, 0;");
        ptx.AppendLine("    @%p2 bra.uni BATCH_DOT_DONE;");
        ptx.AppendLine($"    setp.lt.u32 %p3, %r3, {WarpCount};");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    mul.wide.u32 %rd12, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd13, %rd9, %rd12;");
        ptx.AppendLine("    @%p3 ld.shared.f32 %f0, [%rd13];");
        EmitWarpReduction(ptx, "%f0", "%f3", "%r5", "%r6");
        ptx.AppendLine("    setp.ne.u32 %p2, %r3, 0;");
        ptx.AppendLine("    @%p2 bra.uni BATCH_DOT_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd14, %r1, 4;");
        ptx.AppendLine("    add.u64 %rd15, %rd2, %rd14;");
        ptx.AppendLine("    st.global.f32 [%rd15], %f0;");
        ptx.AppendLine("BATCH_DOT_DONE:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
    }

    private static void EmitFusedBatchDot(
        StringBuilder ptx,
        int batch,
        int dimension,
        int blockThreads)
    {
        int batchBytes = checked(dimension * sizeof(float));
        ptx.AppendLine($"// exact batched dot B={batch} D={dimension}; one unrolled register-only warp per batch");
        EmitHeader(ptx, DotEntryPoint, blockThreads);
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<8>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<14>;");
        EmitPointers(ptx);
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mul.wide.u32 %rd3, %r1, {batchBytes};");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd3;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r0, 16;");
        ptx.AppendLine("    add.u64 %rd7, %rd4, %rd6;");
        ptx.AppendLine("    add.u64 %rd8, %rd5, %rd6;");
        ptx.AppendLine("    mov.f32 %f8, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f10, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f11, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f12, 0f00000000;");
        foreach (int offset in new[] { 0, 512, 1_024, 1_536 })
        {
            string suffix = offset == 0 ? string.Empty : $"+{offset}";
            ptx.AppendLine($"    ld.global.v4.f32 {{%f0,%f1,%f2,%f3}}, [%rd7{suffix}];");
            ptx.AppendLine($"    ld.global.v4.f32 {{%f4,%f5,%f6,%f7}}, [%rd8{suffix}];");
            ptx.AppendLine("    fma.rn.f32 %f8, %f0, %f4, %f8;");
            ptx.AppendLine("    fma.rn.f32 %f10, %f1, %f5, %f10;");
            ptx.AppendLine("    fma.rn.f32 %f11, %f2, %f6, %f11;");
            ptx.AppendLine("    fma.rn.f32 %f12, %f3, %f7, %f12;");
        }
        ptx.AppendLine("    add.rn.f32 %f8, %f8, %f10;");
        ptx.AppendLine("    add.rn.f32 %f11, %f11, %f12;");
        ptx.AppendLine("    add.rn.f32 %f8, %f8, %f11;");
        EmitWarpReduction(ptx, "%f8", "%f13", "%r2", "%r3");
        ptx.AppendLine("    setp.ne.u32 %p0, %r0, 0;");
        ptx.AppendLine("    @%p0 bra.uni BATCH_DOT_VECTOR_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r1, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd2, %rd9;");
        ptx.AppendLine("    st.global.f32 [%rd10], %f8;");
        ptx.AppendLine("BATCH_DOT_VECTOR_DONE:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
    }

    private static void EmitWarpPerBatchDot(
        StringBuilder ptx,
        int batch,
        int dimension,
        int blockThreads)
    {
        int batchBytes = checked(dimension * sizeof(float));
        ptx.AppendLine($"// exact batched dot B={batch} D={dimension}; one warp block per batch, aligned float4 streaming, register-only reduction");
        EmitHeader(ptx, DotEntryPoint, blockThreads);
        ptx.AppendLine("    .reg .pred %p<3>;");
        ptx.AppendLine("    .reg .b32 %r<10>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<15>;");
        EmitPointers(ptx);
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %r0;");
        ptx.AppendLine("    mov.u32 %r2, %ctaid.x;");
        ptx.AppendLine($"    mul.wide.u32 %rd3, %r2, {batchBytes};");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd3;");
        ptx.AppendLine("    shl.b32 %r3, %r1, 2;");
        ptx.AppendLine("    mov.f32 %f9, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f10, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f11, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f12, 0f00000000;");
        ptx.AppendLine("BATCH_DOT_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r3, {dimension};");
        ptx.AppendLine("    @%p0 bra.uni BATCH_DOT_REDUCE;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd4, %rd6;");
        ptx.AppendLine("    add.u64 %rd8, %rd5, %rd6;");
        ptx.AppendLine("    ld.global.nc.v4.f32 {%f1,%f2,%f3,%f4}, [%rd7];");
        ptx.AppendLine("    ld.global.nc.v4.f32 {%f5,%f6,%f7,%f8}, [%rd8];");
        ptx.AppendLine("    fma.rn.f32 %f9, %f1, %f5, %f9;");
        ptx.AppendLine("    fma.rn.f32 %f10, %f2, %f6, %f10;");
        ptx.AppendLine("    fma.rn.f32 %f11, %f3, %f7, %f11;");
        ptx.AppendLine("    fma.rn.f32 %f12, %f4, %f8, %f12;");
        ptx.AppendLine("    add.u32 %r3, %r3, 128;");
        ptx.AppendLine("    bra.uni BATCH_DOT_LOOP;");
        ptx.AppendLine("BATCH_DOT_REDUCE:");
        ptx.AppendLine("    add.rn.f32 %f9, %f9, %f10;");
        ptx.AppendLine("    add.rn.f32 %f11, %f11, %f12;");
        ptx.AppendLine("    add.rn.f32 %f9, %f9, %f11;");
        EmitWarpReduction(ptx, "%f9", "%f13", "%r4", "%r5");
        ptx.AppendLine("    setp.ne.u32 %p1, %r1, 0;");
        ptx.AppendLine("    @%p1 bra.uni BATCH_DOT_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd2, %rd9;");
        ptx.AppendLine("    st.global.f32 [%rd10], %f9;");
        ptx.AppendLine("BATCH_DOT_DONE:");
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

    private static void EmitHeader(
        StringBuilder ptx,
        string entry,
        int blockThreads = BaselineBlockThreads)
    {
        ptx.AppendLine($".visible .entry {entry}(");
        ptx.AppendLine("    .param .u64 left_ptr,");
        ptx.AppendLine("    .param .u64 right_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
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
        DirectPtxBatchedVectorOperation operation,
        int batch,
        int m,
        int n)
    {
        bool dot = operation == DirectPtxBatchedVectorOperation.Dot;
        bool fusedBatch = IsFusedBatchDot(operation, batch, m);
        bool warpPerBatch = IsWarpPerBatchDot(operation, batch, m);
        var left = dot ? new DirectPtxExtent(batch, m) : new DirectPtxExtent(batch, m);
        var right = dot ? new DirectPtxExtent(batch, m) : new DirectPtxExtent(batch, n);
        var output = dot ? new DirectPtxExtent(batch) : new DirectPtxExtent(batch, m, n);
        return new DirectPtxKernelBlueprint(
            Operation: dot ? "batched-dense-dot" : "batched-dense-outer",
            Version: fusedBatch ? 6 : warpPerBatch ? 4 : 1,
            Architecture: architecture,
            Variant: dot
                ? $"{(fusedBatch ? "unrolled-warp-per-batch-fp32x4" : warpPerBatch ? "warp-block-per-batch-fp32x4" : "fp32")}-b{batch}-d{m}"
                : $"fp32-b{batch}-m{m}-n{n}",
            Tensors:
            [
                new("left", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    left, left, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("right", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    right, right, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32,
                    dot ? DirectPtxPhysicalLayout.Vector : DirectPtxPhysicalLayout.RowMajor3D,
                    output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: fusedBatch ? 48 : 32,
                MaxStaticSharedBytes: dot && !warpPerBatch ? WarpCount * sizeof(float) : 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: fusedBatch ? 10 : warpPerBatch ? 12 : dot ? 1 : 4),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = dot
                    ? "output[b]=sum(left[b,d]*right[b,d])"
                    : "output[b,i,j]=left[b,i]*right[b,j]",
                ["dtype"] = "fp32",
                ["batch-and-shape-parameters"] = "none",
                ["stride-parameters"] = "none",
                ["temporary-device-allocation"] = "none",
                ["global-intermediates"] = "none",
                ["reduction-pipeline"] = warpPerBatch
                    ? fusedBatch
                        ? "four aligned float4 pairs per lane; one warp per batch; register-only reduction; one output store"
                        : "aligned float4 loads; one 32-thread CTA per batch; register-only shuffles; one output store"
                    : dot ? "block reduction through shared warp partials" : "none"
            });
    }

    private static bool IsWarpPerBatchDot(
        DirectPtxBatchedVectorOperation operation,
        int batch,
        int dimension) =>
        operation == DirectPtxBatchedVectorOperation.Dot &&
        batch <= 65_535 && (dimension & 31) == 0;

    private static bool IsFusedBatchDot(
        DirectPtxBatchedVectorOperation operation,
        int batch,
        int dimension) =>
        operation == DirectPtxBatchedVectorOperation.Dot && batch == 4 && dimension == 512;

    private static int GetBlockThreads(
        DirectPtxBatchedVectorOperation operation,
        int batch,
        int dimension) => IsFusedBatchDot(operation, batch, dimension)
            ? VectorBlockThreads
            : IsWarpPerBatchDot(operation, batch, dimension) ? WarpBlockThreads
            : BaselineBlockThreads;

    private static void Validate(
        DirectPtxBatchedVectorOperation operation,
        int batch,
        int m,
        int n)
    {
        if (!Enum.IsDefined(typeof(DirectPtxBatchedVectorOperation), operation))
            throw new ArgumentOutOfRangeException(nameof(operation));
        if (batch <= 0 || batch > 65_535)
            throw new ArgumentOutOfRangeException(nameof(batch));
        if (m <= 0 || m > 1_048_576)
            throw new ArgumentOutOfRangeException(nameof(m));
        if (operation == DirectPtxBatchedVectorOperation.Outer && (n <= 0 || n > 65_536))
            throw new ArgumentOutOfRangeException(nameof(n));
        _ = checked(batch * checked(m * n));
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
