// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System.Linq;
using System.Text.RegularExpressions;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.WebGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The WGSL translation keeps double-float arithmetic exact under a compiler that treats float arithmetic as real
/// arithmetic (Mesa folded (a + b) - a to b, so every extracted rounding error read 0), and keeps NaN nonzero under one
/// that assumes no NaN. The numerics themselves are checked on Dawn by the parity suites; these pin the translation.
/// </summary>
public sealed class CudaDialectToWgslExactnessTests
{
    private static readonly System.Collections.Generic.Dictionary<string, string> Modules =
        CudaDialectToWgsl.Translate(TorchParityF32Kernels.GetSource());

    [Fact]
    public void Every_module_ends_its_uniform_struct_with_the_zero_field()
    {
        foreach (var entry in Modules)
        {
            string[] fields = Regex.Match(entry.Value, @"struct P \{ ([^}]*) \};").Groups[1].Value.Split(new[] { ", " }, System.StringSplitOptions.None);
            string last = fields[fields.Length - 1];
            Assert.True(last == CudaDialectToWgsl.ZeroField + ": u32", $"{entry.Key}: struct P ends with {last}");
        }
    }

    [Fact]
    public void The_exactness_helpers_are_opaque_in_WGSL()
    {
        foreach (string helper in CudaDialectToWgsl.ExactHelpers)
        {
            string module = Modules.Values.First(m => m.Contains("fn " + helper + "("));
            Assert.Contains($"fn {helper}(v: f32) -> f32 {{ return bitcast<f32>(bitcast<u32>(v) ^ p.{CudaDialectToWgsl.ZeroField}); }}", module);
        }
    }

    [Fact]
    public void The_double_float_steps_round_through_the_exactness_helper()
    {
        // The sums and splits whose rounding error is extracted afterwards: each one is materialized.
        string source = TorchParityF32Kernels.GetSource();
        Assert.Contains("float c = tp_exact(4097.0f * a);", source);
        Assert.Contains("float s = tp_exact(a.x + b.x);", source);
        Assert.Contains("float bb = tp_exact(s - a.x);", source);
        Assert.Contains("float s = mo_exact(hi + y);", source);
        Assert.Contains("float bb = mo_exact(s - hi);", source);
    }

    [Fact]
    public void Nonzero_tests_read_the_bits_not_a_float_comparison()
    {
        // v != 0.0f is false for NaN under a no-NaN assumption WGSL permits (lavapipe makes it).
        string nonzero = Modules["tp_nonzero"];
        Assert.DoesNotContain("v != 0.0", nonzero);
        Assert.Contains("tp_iszero(v)", nonzero);
        Assert.DoesNotContain("x[i] == 0.0", Modules["tp_nonzero_static_scatter"]);
    }
}
