// Copyright (c) CrowdTrainer LLC. All rights reserved.
// The two kernels around the single-GEMM transposed-convolution forward of CUDA/CudaBackend.ConvTransposeGemm.cs (the
// batch regroup and the overlap-add gather of KernelSources/CudaConvolution.tail.c), written once in
// WinogradF23KernelTemplate's body conventions and emitted as OpenCL C, HIP C++, Vulkan GLSL and Metal Shading Language.
// WGSL keeps its own copy (WebGpu/WebGpuConvTransposeGemmKernels.cs).
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The regroup <c>xt[c, b*L + l] = x[b, c, l]</c> of one batch chunk (so the batch joins the GEMM's N) and the
/// overlap-add <c>y[b, c, oh, ow] = sum over the taps (kh, kw) whose input position ih = (oh + pH - kh) / sH,
/// iw = (ow + pW - kw) / sW is whole and in range of col[(c*kH + kh)*kW + kw, (b*inH + ih)*inW + iw]</c>, one thread per
/// output (a gather: no atomics, every output written, the taps summed in CUDA's order).
/// </summary>
/// <remarks>
/// Both are elementwise over a 1-D grid of 256-wide groups. The chunk's place in the full tensors rides in an int offset
/// (GLSL has no pointer arithmetic), so the host never needs a sub-buffer view.
/// </remarks>
internal static class ConvTransposeGemmKernelTemplate
{
    internal const string Regroup = "convt_gemm_regroup";
    internal const string OverlapAdd = "convt_gemm_overlap_add";

    /// <summary>Every kernel's group size.</summary>
    internal const int GroupSize = 256;

    private const string RegroupBody = @"
    int total = nb * C * L;
    if (gid < total)
    {
        int l = gid % L;
        int r = gid / L;
        int c = r % C;
        int b = r / C;
        xt[c * (nb * L) + b * L + l] = src[srcOffset + gid];
    }
";

    // Only the taps congruent to (oh + PH) modulo the stride contribute, so (oh + PH - kh) is an exact multiple of the
    // stride and the division is exact even where it is negative.
    private const string OverlapAddBody = @"
    int total = nb * OC * OH * OW;
    if (gid < total)
    {
        int ow = gid % OW;
        int r = gid / OW;
        int oh = r % OH;
        r = r / OH;
        int c = r % OC;
        int b = r / OC;
        int n = nb * IH * IW;
        float sum = 0.0f;
        for (int kh = (oh + PH) % SH; kh < KH; kh += SH)
        {
            int ih = (oh + PH - kh) / SH;
            if (ih >= 0 && ih < IH)
            {
                for (int kw = (ow + PW) % SW; kw < KW; kw += SW)
                {
                    int iw = (ow + PW - kw) / SW;
                    if (iw >= 0 && iw < IW) sum += col[((c * KH + kh) * KW + kw) * n + (b * IH + ih) * IW + iw];
                }
            }
        }
        dst[dstOffset + gid] = sum;
    }
";

    internal static readonly string[] RegroupInts = { "nb", "C", "L", "srcOffset" };

    internal static readonly string[] OverlapAddInts =
        { "nb", "OC", "OH", "OW", "KH", "KW", "SH", "SW", "PH", "PW", "IH", "IW", "dstOffset" };

    /// <summary>Both kernels, in this template's conventions (binding 1 written, binding 0 read).</summary>
    internal static readonly WinogradF23KernelTemplate.Spec[] Kernels =
    {
        new(Regroup, new[] { "src", "xt" }, RegroupInts, RegroupBody),
        new(OverlapAdd, new[] { "col", "dst" }, OverlapAddInts, OverlapAddBody),
    };

    internal static string[] GetKernelNames() => new[] { Regroup, OverlapAdd };

    /// <summary>Both kernels in one OpenCL C, HIP C++ or Metal Shading Language module.</summary>
    internal static string GetModuleSource(WinogradKernelDialect dialect)
        => WinogradF23KernelTemplate.EmitModule(Kernels, dialect);

    /// <summary>One kernel as a Vulkan compute shader (bindings in buffer order, the ints as push constants).</summary>
    internal static string GetGlslSource(string name)
    {
        foreach (var spec in Kernels)
            if (spec.Name == name) return WinogradF23KernelTemplate.EmitGlsl(spec);
        throw new KeyNotFoundException(name);
    }
}
