// Copyright (c) CrowdTrainer LLC. All rights reserved.
// The per-element and per-signal kernels of IDataTransformBackend (Box-Muller and threshold transforms of host-drawn
// uniforms, the lfilter recursion, row gather / scatter, integral rounding), written once in WinogradF23KernelTemplate's
// body conventions and emitted as OpenCL C, HIP C++ (which CUDA's NVRTC compiles too), Vulkan GLSL and Metal Shading
// Language. WGSL keeps its own copy (WebGpu/WebGpuDataTransformKernels.cs).
using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The kernels behind <see cref="IDataTransformBackend"/>, every one over 256-wide groups of float buffers:
/// <list type="bullet">
/// <item><see cref="BoxMuller"/>: one thread per element turns the host-drawn pair, carried as <c>(r, u2)</c> with
/// <c>r = sqrt(-2 ln u1)</c> taken on the host in double, into <c>mean + std * r cos(2 pi u2)</c> (or sin), optionally
/// clamped.</item>
/// <item><see cref="Threshold"/>: one thread per element compares the host-drawn double uniform, carried as a float and
/// its signed residual, with a float threshold exactly, and writes one of two values.</item>
/// <item><see cref="LinearFilter"/>: one thread per signal runs scipy's direct form II transposed recursion over time,
/// the state in a global workspace, with the optional forecast guard.</item>
/// <item><see cref="GatherRows"/> / <see cref="ScatterRows"/>: one thread per element copies a table row element to or
/// from the packed rows; a negative id (out of range, or for the scatter a repeated id that is not the last) is
/// skipped.</item>
/// <item><see cref="RoundIntegral"/>: one thread per element rounds to an integer under a <see cref="RoundingMode"/>,
/// clamps to the target's range, NaN as 0. Every step is exact in float.</item>
/// </list>
/// Integer ids ride in the float buffers exactly (below 2^24). Float parameters ride in the int list as their bits.
/// Every kernel's last int, <c>first</c>, offsets the group index, so a grid larger than one dispatch's 65535 groups runs
/// as several launches.
/// </summary>
internal static class DataTransformKernelTemplate
{
    /// <summary>Threads per group.</summary>
    internal const int GroupSize = 256;

    /// <summary>The largest table row id the float-carried ids represent exactly.</summary>
    internal const int MaxExactId = 1 << 24;

    internal const string BoxMuller = "dt_box_muller";
    internal const string Threshold = "dt_uniform_threshold";
    internal const string LinearFilter = "dt_linear_filter";
    internal const string GatherRows = "dt_gather_rows";
    internal const string ScatterRows = "dt_scatter_rows";
    internal const string RoundIntegral = "dt_round_integral";

    // uv holds (r, u2) per element, r = sqrt(-2 ln u1) from the host's double u1 (a float u1 loses every u1 within 2^-25
    // of 1, whose radius is up to 2.4e-4); 6.2831855f is 2 pi rounded to float.
    private static readonly string BoxMullerBody = KernelSourceResources.Read("DataTransform.dt_box_muller.c");

    // uv holds (a, r) per element: a = (float)u, r = (float)(u - a). Against a float t, u < t exactly when a < t or
    // (a == t and r < 0), because a is u rounded to nearest and every float between them is a itself.
    private static readonly string ThresholdBody = KernelSourceResources.Read("DataTransform.dt_uniform_threshold.c");

    // One signal per thread: z (the K - 1 states) lives in the workspace; coefficients are divided by a[0] as they are
    // read (the same quotients scipy's normalization stores). NaN is tested on the bits (WGSL may assume no NaN).
    private static readonly string LinearFilterBody = KernelSourceResources.Read("DataTransform.dt_linear_filter.c");

    private static readonly string GatherRowsBody = KernelSourceResources.Read("DataTransform.dt_gather_rows.c");

    private static readonly string ScatterRowsBody = KernelSourceResources.Read("DataTransform.dt_scatter_rows.c");

    // t = trunc(v) and f = v - t are exact; the modes decide from f alone (ties: |f| == 0.5). An odd t: t / 2 is not an
    // integer (t / 2 is exact). Infinities keep t (f is NaN, so no comparison fires); NaN becomes 0.
    private static readonly string RoundIntegralBody = KernelSourceResources.Read("DataTransform.dt_round_integral.c");

    /// <summary>
    /// Buffers in binding order (the written ones listed in each spec) and int parameters:
    /// Box-Muller uv [2N], out [N] (written) | N, branch, meanBits, stdBits, clampOn, lowBits, highBits, first;
    /// threshold uv [2N], out [N] (written), thresholds [N] (or any buffer when scalar) | N, above, perElement, thrBits,
    /// onBits, offBits, first;
    /// linear filter x [S, L], y [S, L] (written), b [NB], a [NA], zi [S, K-1] (or any buffer), workspace [S, K-1]
    /// (written) | S, L, NB, NA, hasZi, guardOn, guardBits, first;
    /// gather table [V, D], out [K, D] (written), ids [K] | K, D, first;
    /// scatter source [K, D], table [V, D] (written in place), ids [K] | K, D, first;
    /// round x [N], y [N] (written) | N, mode, lowBits, highBits, first.
    /// </summary>
    private static readonly WinogradF23KernelTemplate.Spec[] Generic =
    {
        new(BoxMuller, new[] { "uv", "ot" }, new[] { "N", "branch", "meanBits", "stdBits", "clampOn", "lowBits", "highBits", "first" },
            BoxMullerBody, 0, new[] { 1 }),
        new(Threshold, new[] { "uv", "ot", "th" }, new[] { "N", "above", "perElement", "thrBits", "onBits", "offBits", "first" },
            ThresholdBody, 0, new[] { 1 }),
        new(LinearFilter, new[] { "xs", "ys", "bc", "ac", "zi", "zs" },
            new[] { "S", "L", "NB", "NA", "hasZi", "guardOn", "guardBits", "first" }, LinearFilterBody, 0, new[] { 1, 5 }),
        new(GatherRows, new[] { "tb", "ot", "ids" }, new[] { "K", "D", "first" }, GatherRowsBody, 0, new[] { 1 }),
        new(ScatterRows, new[] { "sr", "tb", "ids" }, new[] { "K", "D", "first" }, ScatterRowsBody, 0, new[] { 1 }),
        new(RoundIntegral, new[] { "xs", "ys" }, new[] { "N", "mode", "lowBits", "highBits", "first" }, RoundIntegralBody, 0,
            new[] { 1 }),
    };

    /// <summary>The kernels' specs, in binding conventions shared by every dialect.</summary>
    internal static IReadOnlyList<WinogradF23KernelTemplate.Spec> Kernels => Generic;

    /// <summary>Every kernel name.</summary>
    internal static string[] GetKernelNames() => new[] { BoxMuller, Threshold, LinearFilter, GatherRows, ScatterRows, RoundIntegral };

    // This family's spellings on top of the template's: log / sqrt / sin / cos / trunc (HIP and CUDA take the float
    // functions by name) and a float's bits as an int.
    private static string Spell(string body, WinogradKernelDialect dialect)
    {
        bool hip = dialect == WinogradKernelDialect.Hip;
        return body
            .Replace("LOGF(", hip ? "logf(" : "log(")
            .Replace("SQRTF(", hip ? "sqrtf(" : "sqrt(")
            .Replace("SINF(", hip ? "sinf(" : "sin(")
            .Replace("COSF(", hip ? "cosf(" : "cos(")
            .Replace("TRUNCF(", hip ? "truncf(" : "trunc(")
            .Replace("FBITS(", dialect switch
            {
                WinogradKernelDialect.OpenCl => "as_int(",
                WinogradKernelDialect.Glsl => "floatBitsToInt(",
                WinogradKernelDialect.Metal => "as_type<int>(",
                _ => "__float_as_int(",
            });
    }

    private static WinogradF23KernelTemplate.Spec ForDialect(WinogradF23KernelTemplate.Spec spec, WinogradKernelDialect dialect)
        => new(spec.Name, spec.Buffers, spec.Ints, Spell(spec.Body, dialect), spec.SharedFloats, spec.Written);

    /// <summary>Every kernel in one OpenCL C, HIP C++ (also compiled by CUDA's NVRTC) or Metal module.</summary>
    internal static string GetModuleSource(WinogradKernelDialect dialect)
    {
        var specs = new List<WinogradF23KernelTemplate.Spec>();
        foreach (var spec in Generic) specs.Add(ForDialect(spec, dialect));
        return WinogradF23KernelTemplate.EmitModule(specs, dialect);
    }

    /// <summary>The standalone GLSL compute shader for one kernel (Vulkan compiles one pipeline per kernel).</summary>
    internal static string GetGlslSource(string name)
    {
        foreach (var spec in Generic)
            if (spec.Name == name)
                return WinogradF23KernelTemplate.EmitGlsl(ForDialect(spec, WinogradKernelDialect.Glsl));
        throw new ArgumentException($"No data transform kernel named '{name}'.", nameof(name));
    }
}
