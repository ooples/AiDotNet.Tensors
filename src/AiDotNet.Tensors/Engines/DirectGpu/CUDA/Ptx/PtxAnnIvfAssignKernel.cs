using System;
using System.Collections.Generic;
using System.Text;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// IVF nearest-centroid assignment (issue #854), matching the NVRTC <c>ann_ivf_assign</c> kernel:
/// for each vector, scan all centroids and record the index of the best one — <c>argmin</c> for
/// squared-L2, <c>argmax</c> for inner product — with ties resolving to the lowest index (strict
/// improvement replaces during an ascending scan). One warp owns one vector, coalesces the feature
/// axis for each centroid, and reduces the score with shuffles. Lane zero preserves the ordered
/// centroid scan and writes the int32 assignment. The metric is baked into the PTX.
///
/// Shape (numVectors, numCentroids, dim) and the metric are baked in, so the launch takes buffer
/// pointers only. 256 threads/block (8 vectors/block), grid = numVectors/8.
/// </summary>
internal sealed class PtxAnnIvfAssignKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const int WarpsPerBlock = BlockThreads / 32;
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
        var loaded = DirectPtxResourceInitialization.Complete(
            runtime.LoadModule(Ptx),
            module =>
            {
                IntPtr function = module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
                int activeBlocks = module.GetActiveBlocksPerMultiprocessor(function, BlockThreads);
                Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
                var audit = DirectPtxKernelAudit.Create(
                    Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks,
                    module);
                return (Function: function, Audit: audit);
            });
        _module = loaded.Resource;
        _function = loaded.Value.Function;
        Audit = loaded.Value.Audit;
    }

    internal unsafe void Launch(DirectPtxTensorView vectors, DirectPtxTensorView centroids, DirectPtxTensorView assignments)
    {
        DirectPtxAbi.Require(vectors, Blueprint.Tensors[0], nameof(vectors));
        DirectPtxAbi.Require(centroids, Blueprint.Tensors[1], nameof(centroids));
        DirectPtxAbi.Require(assignments, Blueprint.Tensors[2], nameof(assignments));

        IntPtr vectorsPointer = vectors.Pointer;
        IntPtr centroidsPointer = centroids.Pointer;
        IntPtr assignmentsPointer = assignments.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &vectorsPointer;
        arguments[1] = &centroidsPointer;
        arguments[2] = &assignmentsPointer;
        _module.Launch(_function, (uint)(NumVectors / WarpsPerBlock), 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, AnnMetric metric, int numVectors, int numCentroids, int dim)
    {
        ValidateShape(numVectors, numCentroids, dim);
        bool ip = metric == AnnMetric.InnerProduct;
        // Best score init: -inf for argmax (IP), +inf for argmin (L2).
        string bestInit = ip ? "0fFF800000" : "0f7F800000";

        var ptx = new StringBuilder(4_000);
        DirectPtxPtxText.AppendModuleHeader(ptx, ccMajor, ccMinor, disableLoopUnrolling: true);
        ptx.AppendLine($"// ann-ivf-assign metric={metric} vectors={numVectors} centroids={numCentroids} dim={dim}; one vector per warp");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 vec_ptr,");
        ptx.AppendLine("    .param .u64 cen_ptr,");
        ptx.AppendLine("    .param .u64 asg_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<5>;");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<6>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [vec_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [cen_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [asg_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine("    shr.u32 %r2, %r0, 5;");                        // warp in block
        ptx.AppendLine("    and.b32 %r3, %r0, 31;");                       // lane
        ptx.AppendLine($"    mad.lo.u32 %r4, %r1, {WarpsPerBlock}, %r2;"); // vector
        ptx.AppendLine($"    mad.lo.u32 %r8, %r4, {dim}, %r3;");           // vector*dim + lane
        ptx.AppendLine("    mul.wide.u32 %rd3, %r8, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");                   // &vectors[i,lane]
        ptx.AppendLine("    mul.wide.u32 %rd10, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd10;");                  // &centroids[0,lane]
        ptx.AppendLine($"    mov.f32 %f0, {bestInit};");                 // bestScore
        ptx.AppendLine("    mov.u32 %r5, 0;");                            // best = 0
        ptx.AppendLine("    mov.u32 %r6, 0;");                            // c = 0
        ptx.AppendLine("$IVF_C_LOOP:");
        // score = metric(vectors[i], centroids[c]) over dim
        ptx.AppendLine("    mov.u64 %rd6, %rd4;");                        // vec walker
        ptx.AppendLine("    mov.u64 %rd7, %rd5;");                        // cen walker
        ptx.AppendLine("    mov.f32 %f1, 0f00000000;");                  // score
        ptx.AppendLine("    mov.u32 %r7, %r3;");                          // k = lane
        ptx.AppendLine("$IVF_K_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r7, {dim};");
        ptx.AppendLine("    @%p0 bra $IVF_REDUCE;");
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
        ptx.AppendLine("    add.u64 %rd6, %rd6, 128;");
        ptx.AppendLine("    add.u64 %rd7, %rd7, 128;");
        ptx.AppendLine("    add.u32 %r7, %r7, 32;");
        ptx.AppendLine("    bra $IVF_K_LOOP;");
        ptx.AppendLine("$IVF_REDUCE:");
        DirectPtxPtxText.AppendWarpSum(ptx, "%f1", "%r12", "%r13", "%f5");
        ptx.AppendLine("    setp.ne.u32 %p1, %r3, 0;");
        ptx.AppendLine("    @%p1 bra $IVF_NEXT_C;");
        // better = (ip ? score > bestScore : score < bestScore); strict -> ties keep lowest index
        ptx.AppendLine(ip ? "    setp.gt.f32 %p2, %f1, %f0;" : "    setp.lt.f32 %p2, %f1, %f0;");
        ptx.AppendLine("    @%p2 mov.f32 %f0, %f1;");                    // bestScore = score
        ptx.AppendLine("    @%p2 mov.u32 %r5, %r6;");                    // best = c
        // advance centroid base by dim, c++
        ptx.AppendLine("$IVF_NEXT_C:");
        ptx.AppendLine($"    add.u64 %rd5, %rd5, {dim * sizeof(float)};");
        ptx.AppendLine("    add.u32 %r6, %r6, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p3, %r6, {numCentroids};");
        ptx.AppendLine("    @%p3 bra $IVF_C_LOOP;");
        ptx.AppendLine("    @%p1 bra $IVF_END;");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd8;");
        ptx.AppendLine("    st.global.u32 [%rd9], %r5;");                // assignments[i] = best (int32)
        ptx.AppendLine("$IVF_END:");
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
            Version: 2,
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
            ResourceBudget: DirectPtxResourceBudget.FromDriverMeasurement(
                measuredRegistersPerThread: 32,
                maxStaticSharedBytes: 0,
                maxLocalBytesPerThread: 0,
                minBlocksPerMultiprocessor: 1),
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
        return numVectors % WarpsPerBlock == 0 && numVectors <= MaxVectors;
    }

    internal static bool IsPromotedShape(int numVectors, int numCentroids, int dim) => false;

    private static void ValidateShape(int numVectors, int numCentroids, int dim)
    {
        if (!IsSupportedShape(numVectors, numCentroids, dim))
            throw new ArgumentOutOfRangeException(
                nameof(numVectors),
                $"ANN IVF assign requires positive dims with dim<={MaxDim} and numVectors a multiple of {WarpsPerBlock} up to {MaxVectors}.");
    }

}
