using System;
using System.Collections.Generic;
using System.Text;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// IVF nearest-centroid assignment (issue #854), matching the NVRTC <c>ann_ivf_assign</c> kernel:
/// for each vector, scan all centroids and record the index of the best one — <c>argmin</c> for
/// squared-L2, <c>argmax</c> for inner product — with ties resolving to the lowest index (strict
/// improvement replaces during an ascending scan). One thread owns one vector, walks the feature axis
/// per centroid serially in registers, and writes an int32 assignment — no shared memory, no reduction.
/// The metric is baked into the PTX.
///
/// Shape (numVectors, numCentroids, dim) and the metric are baked in, so the launch takes buffer
/// pointers only. 256 threads/block, grid = numVectors/256 (a positive multiple of 256).
/// </summary>
internal sealed class PtxAnnIvfAssignKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const int MaxVectors = 2048 * 4096;
    internal const int MaxDim = 4096;
    internal const string EntryPoint = "aidotnet_ann_ivf_assign";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal AnnMetric Metric { get; }
    internal int NumVectors { get; }
    internal int NumCentroids { get; }
    internal int Dim { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxAnnIvfAssignKernel(DirectPtxRuntime runtime, AnnMetric metric, int numVectors, int numCentroids, int dim)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in ann-ivf-assign specialization is measured only on GA10x/SM86.");
        ValidateShape(numVectors, numCentroids, dim);
        Metric = metric;
        NumVectors = numVectors;
        NumCentroids = numCentroids;
        Dim = dim;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, metric, numVectors, numCentroids, dim);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, metric, numVectors, numCentroids, dim);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(DirectPtxTensorView vectors, DirectPtxTensorView centroids, DirectPtxTensorView assignments)
    {
        Require(vectors, Blueprint.Tensors[0], nameof(vectors));
        Require(centroids, Blueprint.Tensors[1], nameof(centroids));
        Require(assignments, Blueprint.Tensors[2], nameof(assignments));

        IntPtr vectorsPointer = vectors.Pointer;
        IntPtr centroidsPointer = centroids.Pointer;
        IntPtr assignmentsPointer = assignments.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &vectorsPointer;
        arguments[1] = &centroidsPointer;
        arguments[2] = &assignmentsPointer;
        _module.Launch(_function, (uint)(NumVectors / BlockThreads), 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, AnnMetric metric, int numVectors, int numCentroids, int dim)
    {
        ValidateShape(numVectors, numCentroids, dim);
        bool ip = metric == AnnMetric.InnerProduct;
        // Best score init: -inf for argmax (IP), +inf for argmin (L2).
        string bestInit = ip ? "0fFF800000" : "0f7F800000";

        var ptx = new StringBuilder(4_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// ann-ivf-assign metric={metric} vectors={numVectors} centroids={numCentroids} dim={dim}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 vec_ptr,");
        ptx.AppendLine("    .param .u64 cen_ptr,");
        ptx.AppendLine("    .param .u64 asg_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<6>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [vec_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [cen_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [asg_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");   // i (vector)
        ptx.AppendLine($"    mul.lo.u32 %r3, %r2, {dim};");               // vOff (elems)
        ptx.AppendLine("    mul.wide.u32 %rd3, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");                   // &vectors[i]
        ptx.AppendLine("    mov.u64 %rd5, %rd1;");                        // centroid walker (starts at c=0)
        ptx.AppendLine($"    mov.f32 %f0, {bestInit};");                 // bestScore
        ptx.AppendLine("    mov.u32 %r4, 0;");                            // best = 0
        ptx.AppendLine("    mov.u32 %r5, 0;");                            // c = 0
        ptx.AppendLine("$IVF_C_LOOP:");
        // score = metric(vectors[i], centroids[c]) over dim
        ptx.AppendLine("    mov.u64 %rd6, %rd4;");                        // vec walker
        ptx.AppendLine("    mov.u64 %rd7, %rd5;");                        // cen walker
        ptx.AppendLine("    mov.f32 %f1, 0f00000000;");                  // score
        ptx.AppendLine("    mov.u32 %r6, 0;");                            // k = 0
        ptx.AppendLine("$IVF_K_LOOP:");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd6];");
        ptx.AppendLine("    ld.global.nc.f32 %f3, [%rd7];");
        if (ip)
        {
            ptx.AppendLine("    fma.rn.f32 %f1, %f2, %f3, %f1;");        // score += a*b
        }
        else
        {
            ptx.AppendLine("    sub.rn.f32 %f4, %f2, %f3;");
            ptx.AppendLine("    fma.rn.f32 %f1, %f4, %f4, %f1;");        // score += d*d
        }
        ptx.AppendLine("    add.u64 %rd6, %rd6, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd7, 4;");
        ptx.AppendLine("    add.u32 %r6, %r6, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r6, {dim};");
        ptx.AppendLine("    @%p0 bra $IVF_K_LOOP;");
        // better = (ip ? score > bestScore : score < bestScore); strict -> ties keep lowest index
        ptx.AppendLine(ip ? "    setp.gt.f32 %p1, %f1, %f0;" : "    setp.lt.f32 %p1, %f1, %f0;");
        ptx.AppendLine("    @%p1 mov.f32 %f0, %f1;");                    // bestScore = score
        ptx.AppendLine("    @%p1 mov.u32 %r4, %r5;");                    // best = c
        // advance centroid base by dim, c++
        ptx.AppendLine($"    add.u64 %rd5, %rd5, {dim * sizeof(float)};");
        ptx.AppendLine("    add.u32 %r5, %r5, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p2, %r5, {numCentroids};");
        ptx.AppendLine("    @%p2 bra $IVF_C_LOOP;");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd8;");
        ptx.AppendLine("    st.global.u32 [%rd9], %r4;");                // assignments[i] = best (int32)
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, AnnMetric metric, int numVectors, int numCentroids, int dim)
    {
        var vExtent = new DirectPtxExtent(numVectors * dim);
        var cExtent = new DirectPtxExtent(numCentroids * dim);
        var aExtent = new DirectPtxExtent(numVectors);
        return new DirectPtxKernelBlueprint(
            Operation: "ann-ivf-assign",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp32-{metric}-v{numVectors}-c{numCentroids}-d{dim}",
            Tensors:
            [
                new("vectors", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    vExtent, vExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("centroids", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    cExtent, cExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("assignments", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.Vector,
                    aExtent, aExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 24,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = metric == AnnMetric.InnerProduct
                    ? "assignments[i] = argmax_c sum_k vectors[i,k]*centroids[c,k] (ties -> lowest c)"
                    : "assignments[i] = argmin_c sum_k (vectors[i,k]-centroids[c,k])^2 (ties -> lowest c)",
                ["metric"] = metric.ToString(),
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int numVectors, int numCentroids, int dim)
    {
        if (numVectors <= 0 || numCentroids <= 0 || dim <= 0 || dim > MaxDim) return false;
        return numVectors % BlockThreads == 0 && numVectors <= MaxVectors;
    }

    internal static bool IsPromotedShape(int numVectors, int numCentroids, int dim) => false;

    private static void ValidateShape(int numVectors, int numCentroids, int dim)
    {
        if (!IsSupportedShape(numVectors, numCentroids, dim))
            throw new ArgumentOutOfRangeException(
                nameof(numVectors),
                $"ANN IVF assign requires positive dims with dim<={MaxDim} and numVectors a multiple of {BlockThreads} up to {MaxVectors}.");
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
