using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Batched cosine similarity (issue #854), matching the NVRTC <c>cosine_similarity</c> kernel:
/// <c>output[b] = dot(a[b], b[b]) / (||a[b]|| * ||b[b]|| + 1e-8)</c>. One warp owns one batch row,
/// loads its contiguous feature axis in coalesced lane-strided chunks, and reduces the dot product
/// and both squared norms with warp shuffles — no shared memory or block barriers.
///
/// Shape (batchSize, dim) is baked into the PTX, so the launch takes buffer pointers only.
/// 256 threads/block (8 rows/block), grid = batchSize/8 for supported exact shapes.
/// </summary>
internal sealed class PtxCosineSimilarityKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const int WarpsPerBlock = BlockThreads / 32;
    internal const int MaxBatch = 2048 * 4096;
    internal const int MaxDim = 4096;
    internal const string EntryPoint = "aidotnet_cosine_similarity";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int BatchSize { get; }
    internal int Dim { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxCosineSimilarityKernel(DirectPtxRuntime runtime, int batchSize, int dim)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in cosine-similarity specialization is measured only on GA10x/SM86.");
        ValidateShape(batchSize, dim);
        BatchSize = batchSize;
        Dim = dim;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, batchSize, dim);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, batchSize, dim);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module);
    }

    internal unsafe void Launch(DirectPtxTensorView a, DirectPtxTensorView b, DirectPtxTensorView output)
    {
        DirectPtxAbi.Require(a, Blueprint.Tensors[0], nameof(a));
        DirectPtxAbi.Require(b, Blueprint.Tensors[1], nameof(b));
        DirectPtxAbi.Require(output, Blueprint.Tensors[2], nameof(output));

        IntPtr aPointer = a.Pointer;
        IntPtr bPointer = b.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &aPointer;
        arguments[1] = &bPointer;
        arguments[2] = &outputPointer;
        _module.Launch(_function, (uint)(BatchSize / WarpsPerBlock), 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();


    internal static string EmitPtx(int ccMajor, int ccMinor, int batchSize, int dim)
    {
        ValidateShape(batchSize, dim);
        string eps = DirectPtxPtxText.Hex(1e-8f);

        var ptx = new StringBuilder(3_500);
        DirectPtxPtxText.AppendModuleHeader(ptx, ccMajor, ccMinor, disableLoopUnrolling: true);
        ptx.AppendLine($"// cosine-similarity batch={batchSize} dim={dim}; one row per warp");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 a_ptr,");
        ptx.AppendLine("    .param .u64 b_ptr,");
        ptx.AppendLine("    .param .u64 out_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<3>;");
        ptx.AppendLine("    .reg .b32 %r<12>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<12>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [a_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [b_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [out_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine("    shr.u32 %r2, %r0, 5;");                       // warp in block
        ptx.AppendLine("    and.b32 %r3, %r0, 31;");                      // lane
        ptx.AppendLine($"    mad.lo.u32 %r4, %r1, {WarpsPerBlock}, %r2;"); // batch
        ptx.AppendLine($"    mul.lo.u32 %r5, %r4, {dim};");               // batch*dim
        ptx.AppendLine("    add.u32 %r5, %r5, %r3;");                     // + lane
        ptx.AppendLine("    mul.wide.u32 %rd3, %r5, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd0, %rd3;");                   // &a[batch,0]
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd3;");                   // &b[batch,0]
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");                   // dot
        ptx.AppendLine("    mov.f32 %f1, 0f00000000;");                   // norm_a
        ptx.AppendLine("    mov.f32 %f2, 0f00000000;");                   // norm_b
        ptx.AppendLine("    mov.u32 %r6, %r3;");                          // i = lane
        ptx.AppendLine("$COS_DIM_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r6, {dim};");
        ptx.AppendLine("    @%p0 bra $COS_REDUCE;");
        ptx.AppendLine("    ld.global.nc.f32 %f3, [%rd6];");             // ai
        ptx.AppendLine("    ld.global.nc.f32 %f4, [%rd7];");             // bi
        ptx.AppendLine("    fma.rn.f32 %f0, %f3, %f4, %f0;");            // dot += ai*bi
        ptx.AppendLine("    fma.rn.f32 %f1, %f3, %f3, %f1;");            // norm_a += ai*ai
        ptx.AppendLine("    fma.rn.f32 %f2, %f4, %f4, %f2;");            // norm_b += bi*bi
        ptx.AppendLine("    add.u64 %rd6, %rd6, 128;");
        ptx.AppendLine("    add.u64 %rd7, %rd7, 128;");
        ptx.AppendLine("    add.u32 %r6, %r6, 32;");
        ptx.AppendLine("    bra $COS_DIM_LOOP;");
        ptx.AppendLine("$COS_REDUCE:");
        EmitWarpReduce(ptx, "%f0");
        EmitWarpReduce(ptx, "%f1");
        EmitWarpReduce(ptx, "%f2");
        ptx.AppendLine("    setp.ne.u32 %p1, %r3, 0;");
        ptx.AppendLine("    @%p1 bra $COS_END;");
        ptx.AppendLine("    sqrt.rn.f32 %f5, %f1;");                     // ||a||
        ptx.AppendLine("    sqrt.rn.f32 %f6, %f2;");                     // ||b||
        ptx.AppendLine("    mul.rn.f32 %f7, %f5, %f6;");                 // ||a||*||b||
        ptx.AppendLine($"    add.rn.f32 %f7, %f7, {eps};");             // + 1e-8
        ptx.AppendLine("    div.rn.f32 %f8, %f0, %f7;");                 // dot / denom
        ptx.AppendLine("    mul.wide.u32 %rd8, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd8;");
        ptx.AppendLine("    st.global.f32 [%rd9], %f8;");
        ptx.AppendLine("$COS_END:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static void EmitWarpReduce(StringBuilder ptx, string partial)
    {
        foreach (int offset in new[] { 16, 8, 4, 2, 1 })
        {
            ptx.AppendLine($"    mov.b32 %r8, {partial};");
            ptx.AppendLine($"    shfl.sync.down.b32 %r9, %r8, {offset}, 31, 0xffffffff;");
            ptx.AppendLine("    mov.b32 %f9, %r9;");
            ptx.AppendLine($"    add.rn.f32 {partial}, {partial}, %f9;");
        }
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, int batchSize, int dim)
    {
        var vecExtent = new DirectPtxExtent(batchSize * dim);
        var outExtent = new DirectPtxExtent(batchSize);
        return new DirectPtxKernelBlueprint(
            Operation: "cosine-similarity",
            Version: 2,
            Architecture: architecture,
            Variant: $"fp32-b{batchSize}-d{dim}",
            Tensors:
            [
                new("a", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    vecExtent, vecExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("b", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    vecExtent, vecExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    outExtent, outExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: DirectPtxResourceBudget.FromDriverMeasurement(
                measuredRegistersPerThread: 24,
                maxStaticSharedBytes: 0,
                maxLocalBytesPerThread: 0,
                minBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "output[b] = dot(a[b],b[b]) / (||a[b]|| * ||b[b]|| + 1e-8)",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int batchSize, int dim)
    {
        if (batchSize <= 0 || dim <= 0 || dim > MaxDim) return false;
        return batchSize % WarpsPerBlock == 0 && batchSize <= MaxBatch;
    }

    internal static bool IsPromotedShape(int batchSize, int dim) => false;

    private static void ValidateShape(int batchSize, int dim)
    {
        if (!IsSupportedShape(batchSize, dim))
            throw new ArgumentOutOfRangeException(
                nameof(batchSize),
                $"Cosine similarity requires positive dims with dim<={MaxDim} and batchSize a multiple of {WarpsPerBlock} up to {MaxBatch}.");
    }

}
