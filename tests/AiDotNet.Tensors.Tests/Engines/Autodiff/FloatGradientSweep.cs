using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Runs every op <see cref="DifferentiableOpsGradCheckSweep"/> can drive in <c>float</c> on a given engine and checks
/// (1) its tape gradient against the CPU's and (2) its compiled training gradient against its own tape.
/// </summary>
/// <remarks>
/// Float matters on its own: the CPU has float-only fused paths (softmax cross-entropy), GPU kernels are float, and a
/// saved state boxed as <c>T</c> but read as <c>double</c> passes every double test and throws in every float compiled
/// step (binary cross-entropy did, #1680).
/// </remarks>
internal static class FloatGradientSweep
{
    internal const double Tolerance = 2e-3;

    internal sealed class Result
    {
        public int TapeOk, CompiledOk, NoGradient;
        public readonly List<string> TapeFailures = new(), CompiledFailures = new(), Skipped = new(), Refused = new();

        public IReadOnlyList<string> AllFailures => TapeFailures.Select(f => "tape: " + f)
            .Concat(CompiledFailures.Select(f => "compiled: " + f)).ToList();

        public void Report(ITestOutputHelper output, string engineName)
        {
            output.WriteLine($"{engineName} tape == CPU tape       : {TapeOk}");
            output.WriteLine($"{engineName} tape FAILED            : {TapeFailures.Count}");
            output.WriteLine($"{engineName} compiled == its tape   : {CompiledOk}");
            output.WriteLine($"{engineName} compiled FAILED        : {CompiledFailures.Count}");
            output.WriteLine($"no CPU tape gradient     : {NoGradient}");
            output.WriteLine($"skipped                  : {Skipped.Count}");
            output.WriteLine($"refused by design        : {Refused.Count}");
            foreach (var f in Refused.OrderBy(x => x)) output.WriteLine("REFUSED   " + f);
            foreach (var f in TapeFailures.OrderBy(x => x)) output.WriteLine("TAPE      " + f);
            foreach (var f in CompiledFailures.OrderBy(x => x)) output.WriteLine("COMPILED  " + f);
            foreach (var s in Skipped.OrderBy(x => x)) output.WriteLine("skip      " + s);
        }
    }

    internal static Result Run(IEngine engine, ITestOutputHelper output)
    {
        var cpu = new CpuEngine();
        bool checkTape = engine is not CpuEngine || engine is DirectGpuTensorEngine;
        var r = new Result();
        var prior = AiDotNetEngine.Current;
        try
        {
            foreach (var (name, doubleMethod, doubleArgs) in new DifferentiableOpsGradCheckSweep(output).DrivableOps(r.Skipped))
            {
                MethodInfo m;
                object[] args;
                try
                {
                    m = doubleMethod.GetGenericMethodDefinition().MakeGenericMethod(typeof(float));
                    args = ToFloat(doubleArgs, m.GetParameters());
                }
                catch (Exception ex)
                {
                    r.Skipped.Add($"{name}: no float form ({DifferentiableOpsGradCheckSweep.Inner(ex).Message})");
                    continue;
                }

                string id = $"{name}|{DifferentiableOpsGradCheckSweep.ParamFingerprint(m)}";
                var inputs = TensorInputs(args);
                Dictionary<Tensor<float>, float[]> cpuGrads;
                try { cpuGrads = TapeGradients(cpu, m, args, inputs); }
                catch (Exception ex)
                {
                    r.Skipped.Add($"{id}: CPU float tape threw {DifferentiableOpsGradCheckSweep.Inner(ex).GetType().Name}");
                    continue;
                }

                if (cpuGrads.Count == 0) { r.NoGradient++; continue; }
                var parameters = cpuGrads.Keys.ToArray();

                Dictionary<Tensor<float>, float[]> tapeGrads = cpuGrads;
                if (checkTape)
                {
                    try
                    {
                        var engineGrads = TapeGradients(engine, m, args, inputs);
                        tapeGrads = engineGrads;
                        var bad = Compare(parameters, cpuGrads, p => engineGrads.TryGetValue(p, out var g) ? g : null,
                            "CPU tape", "tape");
                        if (bad is not null)
                        {
                            // Float rounding differs between kernels; the exact reference is the CPU tape in double. An
                            // ill-conditioned gradient (a phase near a small-magnitude bin) misses it in float on the CPU
                            // too; there the engine must be as accurate as the CPU's float, within a factor of two.
                            var exact = DoubleReference(doubleMethod, doubleArgs, inputs, parameters);
                            if (exact is not null)
                            {
                                double engineErr = MaxRelativeError(parameters, exact, p => engineGrads.TryGetValue(p, out var g) ? g : null);
                                double cpuErr = MaxRelativeError(parameters, exact, p => cpuGrads[p]);
                                if (engineErr <= Math.Max(Tolerance, 2 * cpuErr))
                                    bad = null;
                                else
                                    bad += $" (max relative error vs CPU double: engine {engineErr:G3}, CPU float {cpuErr:G3}; "
                                        + Compare(parameters, exact, p => engineGrads.TryGetValue(p, out var g) ? g : null, "double", "engine")
                                        + ")";
                            }
                        }

                        if (bad is null) r.TapeOk++;
                        else r.TapeFailures.Add($"{id}: {bad}");
                    }
                    catch (Exception ex)
                    {
                        var inner = DifferentiableOpsGradCheckSweep.Inner(ex);
                        r.TapeFailures.Add($"{id}: tape threw {inner.GetType().Name}: {inner.Message}");
                        continue;
                    }
                }

                ICompiledTrainingPlan<float>? plan = null;
                try
                {
                    AiDotNetEngine.Current = engine;
                    using (var scope = GraphMode.EnableTraining(parameters))
                    {
                        var traced = (Tensor<float>)m.Invoke(engine, DifferentiableOpsGradCheckSweep.CopyArgs(args))!;
                        engine.ReduceSum(traced, null);
                        plan = scope.CompileTraining(parameters);
                    }

                    plan.Step();
                    var compiled = new Dictionary<Tensor<float>, float[]>();
                    for (int i = 0; i < parameters.Length; i++)
                        compiled[parameters[i]] = plan.Gradients[i].GetFlattenedData().ToArray();
                    var bad = Compare(parameters, tapeGrads, p => compiled[p], "tape", "compiled");
                    if (bad is null) r.CompiledOk++;
                    else r.CompiledFailures.Add($"{id}: {bad}");
                }
                catch (Exception ex) when (DifferentiableOpsGradCheckSweep.IsRefusedByDesign(DifferentiableOpsGradCheckSweep.Inner(ex)))
                {
                    r.Refused.Add($"{id}: {DifferentiableOpsGradCheckSweep.Inner(ex).Message}");
                }
                catch (Exception ex)
                {
                    var inner = DifferentiableOpsGradCheckSweep.Inner(ex);
                    r.CompiledFailures.Add($"{id}: compiled step threw {inner.GetType().Name}: {inner.Message}");
                }
                finally
                {
                    plan?.Dispose();
                    AiDotNetEngine.Current = prior;
                }
            }
        }
        finally
        {
            AiDotNetEngine.Current = prior;
        }

        return r;
    }

    private static Dictionary<Tensor<float>, float[]> TapeGradients(
        IEngine engine, MethodInfo m, object[] args, Tensor<float>[] inputs)
    {
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            using var tape = new GradientTape<float>();
            var output = (Tensor<float>)m.Invoke(engine, DifferentiableOpsGradCheckSweep.CopyArgs(args))!;
            var grads = tape.ComputeGradients(engine.ReduceSum(output, null), inputs);
            return inputs.Where(x => grads.TryGetValue(x, out var g) && g is not null)
                .ToDictionary(x => x, x => grads[x].GetFlattenedData().ToArray());
        }
        finally
        {
            AiDotNetEngine.Current = prior;
        }
    }

    // The CPU double tape's gradients, keyed by the float inputs they correspond to (same operand order).
    private static Dictionary<Tensor<float>, float[]>? DoubleReference(
        MethodInfo doubleMethod, object[] doubleArgs, Tensor<float>[] floatInputs, Tensor<float>[] parameters)
    {
        try
        {
            var cpu = new CpuEngine();
            var doubleInputs = doubleArgs.SelectMany(a => a switch
            {
                Tensor<double> t => new[] { t },
                Tensor<double>[] arr => arr,
                IEnumerable<Tensor<double>> seq => seq.ToArray(),
                _ => Array.Empty<Tensor<double>>(),
            }).Distinct().ToArray();
            if (doubleInputs.Length != floatInputs.Length) return null;
            using var tape = new GradientTape<double>();
            var output = (Tensor<double>)doubleMethod.Invoke(cpu, DifferentiableOpsGradCheckSweep.CopyArgs(doubleArgs))!;
            var grads = tape.ComputeGradients(cpu.ReduceSum(output, null), doubleInputs);
            var result = new Dictionary<Tensor<float>, float[]>();
            for (int i = 0; i < floatInputs.Length; i++)
                if (grads.TryGetValue(doubleInputs[i], out var g) && g is not null)
                    result[floatInputs[i]] = g.GetFlattenedData().Select(v => (float)v).ToArray();
            return parameters.All(result.ContainsKey) ? result : null;
        }
        catch
        {
            return null;
        }
    }

    private static string? Compare(Tensor<float>[] parameters, Dictionary<Tensor<float>, float[]> expected,
        Func<Tensor<float>, float[]?> actual, string expectedName, string actualName)
    {
        for (int i = 0; i < parameters.Length; i++)
        {
            var want = expected[parameters[i]];
            var got = actual(parameters[i]);
            if (got is null) return $"input {i}: {expectedName} has a gradient, {actualName} has none";
            if (got.Length != want.Length) return $"input {i}: length {got.Length} vs {want.Length}";
            for (int k = 0; k < want.Length; k++)
            {
                if (float.IsNaN(want[k]) && float.IsNaN(got[k])) continue;
                double denom = Math.Max(1.0, Math.Abs(want[k]));
                if (!(Math.Abs(got[k] - want[k]) / denom <= Tolerance))
                    return $"input {i}[{k}]: {expectedName} {want[k]:G6} {actualName} {got[k]:G6}";
            }
        }

        return null;
    }

    // The largest |got - want| / max(1, |want|) over every gradient element; infinite when a gradient is missing.
    private static double MaxRelativeError(Tensor<float>[] parameters, Dictionary<Tensor<float>, float[]> expected,
        Func<Tensor<float>, float[]?> actual)
    {
        double worst = 0;
        foreach (var p in parameters)
        {
            var want = expected[p];
            var got = actual(p);
            if (got is null || got.Length != want.Length) return double.PositiveInfinity;
            for (int k = 0; k < want.Length; k++)
            {
                if (float.IsNaN(want[k]) && float.IsNaN(got[k])) continue;
                double err = Math.Abs(got[k] - want[k]) / Math.Max(1.0, Math.Abs(want[k]));
                if (double.IsNaN(err)) return double.PositiveInfinity;
                worst = Math.Max(worst, err);
            }
        }

        return worst;
    }

    private static Tensor<float>[] TensorInputs(object[] args) => args.SelectMany(a => a switch
    {
        Tensor<float> t => new[] { t },
        Tensor<float>[] arr => arr,
        IEnumerable<Tensor<float>> seq => seq.ToArray(),
        _ => Array.Empty<Tensor<float>>(),
    }).Distinct().ToArray();

    // The sweep's double arguments as float arguments; a tensor passed twice stays one tensor.
    private static object[] ToFloat(object[] args, ParameterInfo[] parameters)
    {
        var map = new Dictionary<Tensor<double>, Tensor<float>>();
        Tensor<float> Convert(Tensor<double> t)
        {
            if (!map.TryGetValue(t, out var f))
                map[t] = f = new Tensor<float>(t.GetFlattenedData().Select(v => (float)v).ToArray(), t.Shape.ToArray());
            return f;
        }

        var result = new object[args.Length];
        for (int i = 0; i < args.Length; i++)
        {
            var target = parameters[i].ParameterType;
            if (target.IsByRef) target = target.GetElementType()!;
            result[i] = args[i] switch
            {
                Tensor<double> t => Convert(t),
                Tensor<double>[] arr => arr.Select(Convert).ToArray(),
                IEnumerable<Tensor<double>> seq => seq.Select(Convert).ToArray(),
                double d when target == typeof(float) => (float)d,
                double[] ds when target == typeof(float[]) => ds.Select(v => (float)v).ToArray(),
                _ => args[i],
            };
        }

        return result;
    }
}
