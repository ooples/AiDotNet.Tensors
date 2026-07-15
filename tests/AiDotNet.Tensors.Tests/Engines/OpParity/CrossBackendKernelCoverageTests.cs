#if !NETFRAMEWORK
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.OpParity;

/// <summary>
/// Cross-backend kernel-source coverage guard (#775). Only the OpenCL backend is runtime-validated
/// on this machine; CUDA/HIP/Metal/Vulkan/WebGPU are source-validated (they compile, but their
/// kernels are never executed here). A source-only backend can silently FAIL to mirror a kernel the
/// OpenCL reference has — nothing else in the suite catches that. This test reflects every backend's
/// static <c>GetKernelNames()</c> registries and flags kernels present in OpenCL but absent from a
/// mirror backend.
///
/// CUDA and HIP mirror OpenCL's string-named <c>GetKernelNames()</c> convention (identical snake_case
/// kernel names), so the set difference is meaningful and ASSERTED against a regression floor that
/// ratchets DOWN only: a newly-added OpenCL compute kernel with no CUDA/HIP twin raises the diff
/// above the floor and fails the build. The non-zero baselines cover kernels that are legitimately
/// OpenCL-specific (e.g. CLBlast packing) plus the current known mirror gaps — drive them down by
/// adding the kernel to the backend, never by raising the floor.
///
/// Metal/Vulkan/WebGPU register shaders through non-name-based mechanisms (Metal functions / SPIR-V /
/// WGSL pipelines), so a raw name diff is not a faithful coverage signal there; their enumerable
/// coverage is REPORTED for visibility but not asserted.
/// </summary>
public sealed class CrossBackendKernelCoverageTests
{
    // OpenCL kernels not mirrored by the backend's GetKernelNames() registries. Regression floor,
    // ratchets DOWN only. Measured baselines — see gpu-cross-backend-coverage.md for the itemised list.
    // Baselines measured on the current tree. The raw diff is a SUPERSET of true functional gaps: it
    // also includes (a) OpenCL-specific plumbing (CLBlast GEMM/transpose variants CUDA covers via
    // cuBLAS) and (b) naming variance (OpenCL `swish_forward` vs CUDA `swish`). Confirmed genuine gaps
    // in the baseline — kernels added to OpenCL during #775 with NO CUDA/HIP equivalent under any name —
    // include spiral_conv[_backward_*], trilinear_interpolate[_backward], conv_transpose3d[_backward_*],
    // adaptive_max_pool2d and the scatter_*_rows / *_backward_rows GNN family. Ratchet DOWN as kernels
    // are mirrored; the gate's job is to stop the diff GROWING (a new OpenCL kernel with no twin fails).
    private const int CudaMissingFloor = 156;
    private const int HipMissingFloor = 156;

    [Fact]
    public void MirrorBackends_CoverEveryOpenClComputeKernel()
    {
        var ocl = KernelNames("OpenCL");
        var cuda = KernelNames("CUDA");
        var hip = KernelNames("HIP");
        var metal = KernelNames("Metal");
        var vulkan = KernelNames("Vulkan");
        var webgpu = KernelNames("WebGpu");

        var cudaMissing = ocl.Except(cuda).OrderBy(x => x, StringComparer.Ordinal).ToArray();
        var hipMissing = ocl.Except(hip).OrderBy(x => x, StringComparer.Ordinal).ToArray();

        var sb = new StringBuilder();
        sb.AppendLine("# Cross-backend kernel-name coverage vs OpenCL (#775)");
        sb.AppendLine($"OpenCL kernels: {ocl.Count}");
        sb.AppendLine($"CUDA: {cuda.Count}  HIP: {hip.Count}  Metal: {metal.Count}  " +
            $"Vulkan: {vulkan.Count}  WebGpu: {webgpu.Count}");
        sb.AppendLine();
        sb.AppendLine("Asserted mirrors (string-named GetKernelNames, comparable to OpenCL):");
        sb.AppendLine($"## CUDA missing {cudaMissing.Length} OpenCL kernel name(s)");
        foreach (var n in cudaMissing) sb.AppendLine($"  [ ] {n}");
        sb.AppendLine();
        sb.AppendLine($"## HIP missing {hipMissing.Length} OpenCL kernel name(s)");
        foreach (var n in hipMissing) sb.AppendLine($"  [ ] {n}");
        sb.AppendLine();
        sb.AppendLine("Reported-only backends (non-name-based shader registration — not asserted):");
        sb.AppendLine($"  Metal enumerable kernels: {metal.Count}");
        sb.AppendLine($"  Vulkan enumerable kernels: {vulkan.Count}");
        sb.AppendLine($"  WebGpu enumerable kernels: {webgpu.Count}");
        WriteRaw("gpu-cross-backend-coverage.md", sb.ToString());

        Assert.True(cudaMissing.Length <= CudaMissingFloor,
            $"CUDA fails to mirror {cudaMissing.Length} OpenCL kernels (floor {CudaMissingFloor}). " +
            $"A newly-added OpenCL kernel has no CUDA twin. Add it to the CUDA backend; do not raise the " +
            $"floor. See gpu-cross-backend-coverage.md. First few: {string.Join(", ", cudaMissing.Take(8))}");
        Assert.True(hipMissing.Length <= HipMissingFloor,
            $"HIP fails to mirror {hipMissing.Length} OpenCL kernels (floor {HipMissingFloor}). " +
            $"A newly-added OpenCL kernel has no HIP twin. Add it to the HIP backend; do not raise the " +
            $"floor. See gpu-cross-backend-coverage.md. First few: {string.Join(", ", hipMissing.Take(8))}");
    }

    /// <summary>Union of every <c>public static string[] GetKernelNames()</c> registry declared under
    /// the <c>.DirectGpu.&lt;segment&gt;</c> namespace of the product assembly.</summary>
    private static SortedSet<string> KernelNames(string backendSegment)
    {
        var set = new SortedSet<string>(StringComparer.Ordinal);
        var asm = typeof(AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend).Assembly;
        string marker = $".DirectGpu.{backendSegment}";
        foreach (var t in asm.GetTypes())
        {
            var ns = t.Namespace;
            if (ns is null || ns.IndexOf(marker, StringComparison.Ordinal) < 0) continue;
            var m = t.GetMethod("GetKernelNames", BindingFlags.Public | BindingFlags.Static,
                binder: null, types: Type.EmptyTypes, modifiers: null);
            if (m is null || m.ReturnType != typeof(string[])) continue;
            try
            {
                if (m.Invoke(null, null) is string[] arr)
                    foreach (var n in arr)
                        if (!string.IsNullOrWhiteSpace(n)) set.Add(n);
            }
            catch { /* a registry that needs a device/context — skip, best effort */ }
        }
        return set;
    }

    private static void WriteRaw(string file, string content)
    {
        try
        {
            var dir = Environment.GetEnvironmentVariable("AIDOTNET_OPPARITY_REPORT_DIR")
                      ?? Path.Combine(Path.GetTempPath(), "aidotnet-opparity");
            Directory.CreateDirectory(dir);
            File.WriteAllText(Path.Combine(dir, file), content);
        }
        catch { /* best-effort */ }
    }
}
#endif
