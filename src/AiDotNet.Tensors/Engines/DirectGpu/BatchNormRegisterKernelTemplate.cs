// Copyright (c) AiDotNet. All rights reserved.
// The register-resident training BatchNorm of CudaNormalizationKernels (bn_fwd_reg / bn_bwd_reg, #1376, #1489), written
// once in WinogradF23KernelTemplate's body conventions and emitted as OpenCL C, HIP C++, Vulkan GLSL and Metal Shading
// Language. WGSL keeps its own copy (WebGpu/WebGpuBatchNormRegisterKernels.cs).
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Training BatchNorm for channel slices of at most <see cref="MaxElements"/> values: one 256-wide group per channel holds
/// the channel's whole slice in registers, so the forward reads x once and the backward reads x and dY once each. Each
/// kernel takes a mode word, so one compiled kernel covers every fused form of the CUDA family:
/// <list type="bullet">
/// <item>forward: bit 0 ReLU, bit 1 a residual added before it;</item>
/// <item>backward: bit 0 dY masked where the ReLU output is not positive, bit 1 the masked dY also written as the
/// residual's gradient, bit 2 the mask rebuilt from x instead of read (the forward's own value, same fma), bit 3 a second
/// output gradient summed into dY as it loads (gradOutput + gradOutput2).</item>
/// </list>
/// The per-channel parameters ride in packed buffers (see <see cref="PortableBatchNorm"/>) so the backward binds eight
/// storage buffers, WebGPU's guaranteed limit; epsilon and 1/count ride in the int list as float bits.
/// </summary>
internal static class BatchNormRegisterKernelTemplate
{
    /// <summary>The group size: every kernel's reduction assumes exactly 256 threads.</summary>
    internal const int GroupSize = 256;

    /// <summary>Values per thread of the compiled variants; the largest bounds the channel slice.</summary>
    internal static readonly int[] PerThread = { 4, 8, 16, 32 };

    /// <summary>The largest channel slice (batch x spatial) the kernels take: 256 threads x 32 values.</summary>
    internal const int MaxElements = GroupSize * 32;

    internal static string ForwardName(int perThread) => "bn_reg_fwd_k" + perThread;

    internal static string BackwardName(int perThread) => "bn_reg_bwd_k" + perThread;

    /// <summary>The smallest variant whose 256 x k values cover <paramref name="count"/>, or 0 when none does.</summary>
    internal static int VariantFor(long count)
    {
        foreach (int k in PerThread)
            if ((long)k * GroupSize >= count) return k;
        return 0;
    }

    private static readonly string[] Ints = { "batch", "channels", "spatial", "epsBits", "invCountBits", "mode" };

    // The group's sum of `value` into float `result` (every thread), through sm[256]: a fixed tree, so one device sums
    // a channel the same way every time.
    private static string Reduce(string value, string result) => $@"
    sm[tid] = {value};
    BARRIER();
    for (int st = 128; st > 0; st >>= 1)
    {{
        if (tid < st) sm[tid] += sm[tid + st];
        BARRIER();
    }}
    float {result} = sm[0];
    BARRIER();
";

    private const string Prologue = @"
    int tid = lid, c = grp;
    int n = batch * spatial;
    float eps = IBITS(epsBits), invCount = IBITS(invCountBits);
";

    private static string ForwardBody(int k) => (Prologue + @"
    float v[KPT];
    float s1 = 0.0f;
    for (int k = 0; k < KPT; k++)
    {
        int e = tid + k * 256;
        v[k] = 0.0f;
        if (e < n)
        {
            int b = e / spatial;
            v[k] = xin[(b * channels + c) * spatial + (e - b * spatial)];
            s1 += v[k];
        }
    }
" + Reduce("s1", "total") + @"
    float mean = total * invCount;
    float s2 = 0.0f;
    for (int k = 0; k < KPT; k++)
        if (tid + k * 256 < n)
        {
            float d = v[k] - mean;
            s2 += d * d;
        }
" + Reduce("s2", "sq") + @"
    float varc = sq * invCount;
    float inv = RSQRT(varc + eps);
    if (tid == 0) { stats[c] = mean; stats[channels + c] = varc; }
    float g = prm[c], bt = prm[channels + c];
    for (int k = 0; k < KPT; k++)
    {
        int e = tid + k * 256;
        if (e < n)
        {
            int b = e / spatial;
            int idx = (b * channels + c) * spatial + (e - b * spatial);
            float y = FMA(g, (v[k] - mean) * inv, bt);
            if ((mode & 2) != 0) y += res[idx];
            if ((mode & 1) != 0) y = FMAX(y, 0.0f);
            yout[idx] = y;
        }
    }
").Replace("KPT", k.ToString(System.Globalization.CultureInfo.InvariantCulture));

    private static string BackwardBody(int k) => (Prologue + @"
    float g = prm[c], bt = prm[channels + c], mean = prm[2 * channels + c];
    float inv = RSQRT(prm[3 * channels + c] + eps);
    float xv[KPT];
    float dv[KPT];
    float sdy = 0.0f, sdyx = 0.0f;
    for (int k = 0; k < KPT; k++)
    {
        int e = tid + k * 256;
        xv[k] = 0.0f;
        dv[k] = 0.0f;
        if (e < n)
        {
            int b = e / spatial;
            int idx = (b * channels + c) * spatial + (e - b * spatial);
            float x = xin[idx];
            float d = dyin[idx];
            if ((mode & 8) != 0) d = d + dy2[idx];
            if ((mode & 4) != 0) { if (!(FMA(g, (x - mean) * inv, bt) > 0.0f)) d = 0.0f; }
            else if ((mode & 1) != 0) { if (!(msk[idx] > 0.0f)) d = 0.0f; }
            if ((mode & 2) != 0) dres[idx] = d;
            xv[k] = x;
            dv[k] = d;
            sdy += d;
            sdyx += d * ((x - mean) * inv);
        }
    }
" + Reduce("sdy", "sumDy") + Reduce("sdyx", "sumDyXhat") + @"
    if (tid == 0) { dprm[c] = sumDyXhat; dprm[channels + c] = sumDy; }
    float sumDxhat = g * sumDy, sumDxhatXhat = g * sumDyXhat;
    for (int k = 0; k < KPT; k++)
    {
        int e = tid + k * 256;
        if (e < n)
        {
            int b = e / spatial;
            int idx = (b * channels + c) * spatial + (e - b * spatial);
            dxout[idx] = inv * (dv[k] * g - invCount * (sumDxhat + ((xv[k] - mean) * inv) * sumDxhatXhat));
        }
    }
").Replace("KPT", k.ToString(System.Globalization.CultureInfo.InvariantCulture));

    /// <summary>
    /// Forward buffers: x, residual (x when absent), y (written), prm = [gamma C][beta C], stats (written) = [mean C][var C].
    /// Backward buffers: dY, mask source (the ReLU output, or x when unused), the residual's gradient (written; a
    /// placeholder when absent), x, prm = [gamma C][beta C][mean C][var C], dX (written), dprm (written) =
    /// [dGamma C][dBeta C], the second output gradient (dY when absent).
    /// </summary>
    internal static readonly WinogradF23KernelTemplate.Spec[] Kernels = BuildKernels();

    private static WinogradF23KernelTemplate.Spec[] BuildKernels()
    {
        var specs = new List<WinogradF23KernelTemplate.Spec>();
        foreach (int k in PerThread)
        {
            specs.Add(new WinogradF23KernelTemplate.Spec(ForwardName(k), new[] { "xin", "res", "yout", "prm", "stats" }, Ints,
                ForwardBody(k), GroupSize, new[] { 2, 4 }));
            specs.Add(new WinogradF23KernelTemplate.Spec(BackwardName(k),
                new[] { "dyin", "msk", "dres", "xin", "prm", "dxout", "dprm", "dy2" }, Ints, BackwardBody(k), GroupSize,
                new[] { 2, 5, 6 }));
        }
        return specs.ToArray();
    }

    internal static string[] GetKernelNames()
    {
        var names = new string[Kernels.Length];
        for (int i = 0; i < Kernels.Length; i++) names[i] = Kernels[i].Name;
        return names;
    }

    internal static string GetModuleSource(WinogradKernelDialect dialect) => WinogradF23KernelTemplate.EmitModule(Kernels, dialect);

    internal static string GetGlslSource(string name)
    {
        foreach (var spec in Kernels)
            if (spec.Name == name) return WinogradF23KernelTemplate.EmitGlsl(spec);
        throw new System.ArgumentException($"No BatchNorm kernel named '{name}'.", nameof(name));
    }
}
