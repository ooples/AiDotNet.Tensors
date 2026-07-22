using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Batched cosine similarity (issue #854), matching the NVRTC <c>cosine_similarity</c> kernel:
/// <c>output[b] = dot(a[b], b[b]) / (||a[b]|| * ||b[b]|| + 1e-8)</c>. One thread owns one batch row
/// and walks the feature axis <c>i = 0..dim-1</c> serially in registers, accumulating the dot product
/// and both squared norms — no shared memory, no reduction.
///
/// Shape (batchSize, dim) is baked into the PTX, so the launch takes buffer pointers only.
/// 256 threads/block, grid = batchSize/256 (a positive multiple of 256), so there is no divergent
/// bounds guard.
/// </summary>
internal sealed class PtxCosineSimilarityKernel : IDisposable
{
    internal const int BlockThreads = 256;
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
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(DirectPtxTensorView a, DirectPtxTensorView b, DirectPtxTensorView output)
    {
        Require(a, Blueprint.Tensors[0], nameof(a));
        Require(b, Blueprint.Tensors[1], nameof(b));
        Require(output, Blueprint.Tensors[2], nameof(output));

        IntPtr aPointer = a.Pointer;
        IntPtr bPointer = b.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &aPointer;
        arguments[1] = &bPointer;
        arguments[2] = &outputPointer;
        _module.Launch(_function, (uint)(BatchSize / BlockThreads), 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    private static string Hex(float value) => "0f" + BitConverter.ToInt32(BitConverter.GetBytes(value), 0).ToString("X8");

    internal static string EmitPtx(int ccMajor, int ccMinor, int batchSize, int dim)
    {
        ValidateShape(batchSize, dim);
        string eps = Hex(1e-8f);

        var ptx = new StringBuilder(3_500);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// cosine-similarity batch={batchSize} dim={dim}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 a_ptr,");
        ptx.AppendLine("    .param .u64 b_ptr,");
        ptx.AppendLine("    .param .u64 out_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<8>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<10>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [a_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [b_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [out_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");   // batch
        ptx.AppendLine($"    mul.lo.u32 %r3, %r2, {dim};");               // batch*dim
        ptx.AppendLine("    mul.wide.u32 %rd3, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd0, %rd3;");                   // &a[batch,0]
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd3;");                   // &b[batch,0]
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");                   // dot
        ptx.AppendLine("    mov.f32 %f1, 0f00000000;");                   // norm_a
        ptx.AppendLine("    mov.f32 %f2, 0f00000000;");                   // norm_b
        ptx.AppendLine("    mov.u32 %r4, 0;");                            // i = 0
        ptx.AppendLine("$COS_DIM_LOOP:");
        ptx.AppendLine("    ld.global.nc.f32 %f3, [%rd6];");             // ai
        ptx.AppendLine("    ld.global.nc.f32 %f4, [%rd7];");             // bi
        ptx.AppendLine("    fma.rn.f32 %f0, %f3, %f4, %f0;");            // dot += ai*bi
        ptx.AppendLine("    fma.rn.f32 %f1, %f3, %f3, %f1;");            // norm_a += ai*ai
        ptx.AppendLine("    fma.rn.f32 %f2, %f4, %f4, %f2;");            // norm_b += bi*bi
        ptx.AppendLine("    add.u64 %rd6, %rd6, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd7, 4;");
        ptx.AppendLine("    add.u32 %r4, %r4, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r4, {dim};");
        ptx.AppendLine("    @%p0 bra $COS_DIM_LOOP;");
        ptx.AppendLine("    sqrt.rn.f32 %f5, %f1;");                     // ||a||
        ptx.AppendLine("    sqrt.rn.f32 %f6, %f2;");                     // ||b||
        ptx.AppendLine("    mul.rn.f32 %f7, %f5, %f6;");                 // ||a||*||b||
        ptx.AppendLine($"    add.rn.f32 %f7, %f7, {eps};");             // + 1e-8
        ptx.AppendLine("    div.rn.f32 %f8, %f0, %f7;");                 // dot / denom
        ptx.AppendLine("    mul.wide.u32 %rd8, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd8;");
        ptx.AppendLine("    st.global.f32 [%rd9], %f8;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, int batchSize, int dim)
    {
        var vecExtent = new DirectPtxExtent(batchSize * dim);
        var outExtent = new DirectPtxExtent(batchSize);
        return new DirectPtxKernelBlueprint(
            Operation: "cosine-similarity",
            Version: 1,
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
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 20,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
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
        return batchSize % BlockThreads == 0 && batchSize <= MaxBatch;
    }

    internal static bool IsPromotedShape(int batchSize, int dim) => false;

    private static void ValidateShape(int batchSize, int dim)
    {
        if (!IsSupportedShape(batchSize, dim))
            throw new ArgumentOutOfRangeException(
                nameof(batchSize),
                $"Cosine similarity requires positive dims with dim<={MaxDim} and batchSize a multiple of {BlockThreads} up to {MaxBatch}.");
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
