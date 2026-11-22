// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Under the default (type-preserving) precision policy the GPU engine must never compute a double op through a float
/// kernel. This sweeps every <see cref="IEngine"/> op that maps one or two tensors to a tensor or a scalar, or writes into its tensor operands (up to three), with operands whose
/// low bits a float cannot hold, and compares the GPU engine with the CPU engine: a narrowed op is off by about 1e-8,
/// five orders above the bound.
/// </summary>
[Collection("DirectGpuSerial")]
public class GpuDoublePrecisionSweepTests
{
    private const double Bound = 1e-11;
    private readonly ITestOutputHelper _output;

    public GpuDoublePrecisionSweepTests(ITestOutputHelper output) => _output = output;

    // Ops whose result is not a function of the operands alone.
    private static readonly string[] NonDeterministic = { "Random", "Dropout", "Noise", "Gumbel", "Sample", "Shuffle", "Bernoulli", "Multinomial" };

    private static Tensor<double> Operand(int side, int seed)
    {
        var values = new double[side * side];
        for (int i = 0; i < values.Length; i++)
            values[i] = 0.25 + 0.5 * ((i * 37 + seed * 101) % 1009) / 1009.0 + (i % 997 + 1) * 1e-11;
        return new Tensor<double>(values, new[] { side, side });
    }

    private static IEnumerable<MethodInfo> TensorToTensorOps()
    {
        foreach (var method in typeof(IEngine).GetMethods().OrderBy(m => m.Name, StringComparer.Ordinal))
        {
            if (!method.IsGenericMethodDefinition || method.GetGenericArguments().Length != 1) continue;
            if (NonDeterministic.Any(word => method.Name.IndexOf(word, StringComparison.OrdinalIgnoreCase) >= 0)) continue;
            MethodInfo closed;
            try
            {
                closed = method.MakeGenericMethod(typeof(double));
            }
            catch (ArgumentException)
            {
                continue; // a constraint double does not satisfy
            }

            var parameters = closed.GetParameters();
            bool inPlace = closed.ReturnType == typeof(void);
            if (!inPlace && closed.ReturnType != typeof(Tensor<double>) && closed.ReturnType != typeof(double)) continue;
            if (parameters.Length < 1 || parameters.Length > (inPlace ? 3 : 2)) continue;
            if (parameters.Any(p => p.ParameterType != typeof(Tensor<double>))) continue;
            yield return closed;
        }
    }

    private static double[]? Run(MethodInfo op, IEngine engine, int side)
    {
        var arguments = new object[op.GetParameters().Length];
        for (int i = 0; i < arguments.Length; i++) arguments[i] = Operand(side, i + 1);
        try
        {
            return op.Invoke(engine, arguments) switch
            {
                Tensor<double> result => result.Contiguous().AsSpan().ToArray(),
                double scalar => new[] { scalar },
                // An in-place or "into" op: whichever operand it wrote, the operands afterwards are its result.
                null when op.ReturnType == typeof(void) => arguments.SelectMany(a => ((Tensor<double>)a).Contiguous().AsSpan().ToArray()).ToArray(),
                _ => null,
            };
        }
        catch (TargetInvocationException)
        {
            return null; // the op does not accept these operands (shape, domain); nothing to compare
        }
    }

    [SkippableTheory]
    [InlineData(48)]
    [InlineData(512)]
    public void NoDoubleOp_IsComputedThroughAFloatKernel(int side)
    {
        Skip.If(AiDotNetEngine.Current is not DirectGpuTensorEngine, "No GPU engine on this host.");
        var gpu = AiDotNetEngine.Current;
        var cpu = (IEngine)new CpuEngine();
        var narrowed = new List<string>();
        int compared = 0;
        foreach (var op in TensorToTensorOps())
        {
            var expected = Run(op, cpu, side);
            if (expected is null) continue;
            var actual = Run(op, gpu, side);
            if (actual is null || actual.Length != expected.Length)
            {
                narrowed.Add($"{op.Name}: the GPU engine returned {(actual is null ? "nothing" : actual.Length + " elements")}, the CPU engine {expected.Length} elements");
                continue;
            }

            compared++;
            double scale = 0.0, worst = 0.0;
            for (int i = 0; i < expected.Length; i++)
            {
                if (double.IsNaN(expected[i]) && double.IsNaN(actual[i])) continue;
                if (expected[i].Equals(actual[i])) continue;
                if (!double.IsInfinity(expected[i])) scale = Math.Max(scale, Math.Abs(expected[i]));
                double difference = Math.Abs(expected[i] - actual[i]);
                worst = Math.Max(worst, double.IsNaN(difference) ? double.PositiveInfinity : difference);
            }

            if (worst > Bound * Math.Max(scale, 1e-300)) narrowed.Add($"{op.Name}: off by {worst / Math.Max(scale, 1e-300):E1} of the result's scale");
        }

        _output.WriteLine($"Compared {compared} ops at {side}x{side}.");
        Assert.True(compared >= 40, $"Only {compared} ops were compared; the sweep is not reaching the engine surface.");
        Assert.True(narrowed.Count == 0, $"{narrowed.Count} double op(s) lost precision on the GPU engine:\n  " + string.Join("\n  ", narrowed));
    }
}
