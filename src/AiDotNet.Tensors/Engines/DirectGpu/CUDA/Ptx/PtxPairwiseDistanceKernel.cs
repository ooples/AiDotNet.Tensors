using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Pairwise distance between two point sets: <c>output[i,j] = ||a[i] - b[j]||</c> (L2) or
/// <c>||a[i] - b[j]||^2</c> (squared), matching the NVRTC <c>pairwise_distance</c> /
/// <c>pairwise_distance_squared</c> kernels (issue #854). One thread owns one (i,j) output pair
/// and walks the <c>dim</c> feature axis serially in registers — no shared memory, no reduction.
/// The squared variant simply skips the closing <c>sqrt.rn.f32</c>.
///
/// Shape (M, N, dim) is baked into the PTX, so the launch takes buffer pointers only.
/// 256 threads/block, grid = (M*N)/256, required to divide evenly (no divergent bounds guard).
/// </summary>
internal sealed class PtxPairwiseDistanceKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const int MaxPairs = 2048 * 4096;
    internal const int MaxDim = 4096;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int M { get; }
    internal int N { get; }
    internal int Dim { get; }
    internal bool Squared { get; }
    internal string EntryPoint { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal static string EntryPointFor(bool squared) =>
        squared ? "aidotnet_pairwise_distance_sq" : "aidotnet_pairwise_distance_l2";

    internal PtxPairwiseDistanceKernel(DirectPtxRuntime runtime, int m, int n, int dim, bool squared)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in pairwise-distance specialization is measured only on GA10x/SM86.");
        ValidateShape(m, n, dim);
        M = m;
        N = n;
        Dim = dim;
        Squared = squared;
        EntryPoint = EntryPointFor(squared);
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, m, n, dim, squared);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, m, n, dim, squared);
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
        uint grid = (uint)((M * N) / BlockThreads);
        _module.Launch(_function, grid, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int m, int n, int dim, bool squared)
    {
        ValidateShape(m, n, dim);
        string entry = EntryPointFor(squared);

        var ptx = new StringBuilder(4_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// pairwise-distance M={m} N={n} dim={dim} squared={squared}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {entry}(");
        ptx.AppendLine("    .param .u64 a_ptr,");
        ptx.AppendLine("    .param .u64 b_ptr,");
        ptx.AppendLine("    .param .u64 out_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<10>;");
        ptx.AppendLine("    .reg .b64 %rd<14>;");
        ptx.AppendLine("    .reg .f32 %f<6>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [a_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [b_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [out_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");   // idx = i*N + j
        ptx.AppendLine($"    div.u32 %r3, %r2, {n};");                     // i
        ptx.AppendLine($"    rem.u32 %r4, %r2, {n};");                     // j
        ptx.AppendLine($"    mul.lo.u32 %r5, %r3, {dim};");               // i*dim
        ptx.AppendLine($"    mul.lo.u32 %r6, %r4, {dim};");               // j*dim
        ptx.AppendLine("    mul.wide.u32 %rd4, %r5, 4;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r6, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd0, %rd4;");                   // &a[i*dim]
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd5;");                   // &b[j*dim]
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");                   // distSq
        ptx.AppendLine("    mov.u32 %r7, 0;");                            // d = 0
        ptx.AppendLine("$PD_DIM_LOOP:");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd6];");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd7];");
        ptx.AppendLine("    sub.rn.f32 %f3, %f1, %f2;");                  // diff
        ptx.AppendLine("    fma.rn.f32 %f0, %f3, %f3, %f0;");             // distSq += diff*diff
        ptx.AppendLine("    add.u64 %rd6, %rd6, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd7, 4;");
        ptx.AppendLine("    add.u32 %r7, %r7, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r7, {dim};");
        ptx.AppendLine("    @%p0 bra $PD_DIM_LOOP;");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd8;");
        if (squared)
        {
            ptx.AppendLine("    st.global.f32 [%rd9], %f0;");             // output = distSq
        }
        else
        {
            ptx.AppendLine("    sqrt.rn.f32 %f4, %f0;");                  // output = sqrt(distSq)
            ptx.AppendLine("    st.global.f32 [%rd9], %f4;");
        }
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int m, int n, int dim, bool squared)
    {
        var aExtent = new DirectPtxExtent(m * dim);
        var bExtent = new DirectPtxExtent(n * dim);
        var outExtent = new DirectPtxExtent(m * n);
        return new DirectPtxKernelBlueprint(
            Operation: squared ? "pairwise-distance-squared" : "pairwise-distance",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp32-m{m}-n{n}-d{dim}",
            Tensors:
            [
                new("a", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    aExtent, aExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("b", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    bExtent, bExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
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
                ["formula"] = squared
                    ? "output[i,j] = sum_d (a[i,d]-b[j,d])^2"
                    : "output[i,j] = sqrt(sum_d (a[i,d]-b[j,d])^2)",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int m, int n, int dim)
    {
        if (m <= 0 || n <= 0 || dim <= 0 || dim > MaxDim) return false;
        long pairs = (long)m * n;
        return pairs > 0 && pairs % BlockThreads == 0 && pairs <= MaxPairs;
    }

    internal static bool IsPromotedShape(int m, int n, int dim) => false;

    private static void ValidateShape(int m, int n, int dim)
    {
        if (!IsSupportedShape(m, n, dim))
            throw new ArgumentOutOfRangeException(
                nameof(m),
                $"Pairwise distance requires positive dims with dim<={MaxDim} and (M*N) a multiple of {BlockThreads} up to {MaxPairs}.");
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
