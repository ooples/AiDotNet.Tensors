using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Export;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.Engines.Autodiff;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Export;

/// <summary>Exercises the generated tracing decorator and typed dispatcher with real, independently evaluated ops.</summary>
[Collection("EngineCurrentGlobalState")]
public class ProgramOperationReplaySweepTests : IDisposable
{
    private readonly IEngine _previous = AiDotNetEngine.Current;
    private readonly CpuEngine _engine = new();
    private readonly ITestOutputHelper _output;

    public ProgramOperationReplaySweepTests(ITestOutputHelper output)
    {
        _output = output;
        AiDotNetEngine.Current = _engine;
    }

    public void Dispose() => AiDotNetEngine.Current = _previous;

    private static IEnumerable<Tensor<double>> Inputs(object arg) => arg switch
    {
        Tensor<double> t => new[] { t },
        IEnumerable<Tensor<double>> ts => ts,
        _ => Array.Empty<Tensor<double>>()
    };

    private static Tensor<double> Copy(Tensor<double> tensor, bool changed = false) =>
        new(tensor.ToArray().Select(x => changed ? x * 0.9 : x).ToArray(), (int[])tensor._shape.Clone());

    private static void Equal(Tensor<double> expected, Tensor<double> actual)
    {
        Assert.Equal(expected._shape, actual._shape);
        for (int i = 0; i < expected.Length; i++)
        {
            double a = expected[i], b = actual[i];
            Assert.True(a == b || (double.IsNaN(a) && double.IsNaN(b)) ||
                (IsFinite(a) && IsFinite(b) && Math.Abs(a - b) <= 1e-8 * Math.Max(1, Math.Abs(a))),
                $"element {i}: eager {a:G17}, replay {b:G17}");
        }
    }

    // double.IsFinite and ReferenceEqualityComparer are .NET Core only; the test project also builds net471 and net462.
    private static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);

    private sealed class ReferenceEqualityComparerLocal : IEqualityComparer<object>
    {
        public static readonly ReferenceEqualityComparerLocal Instance = new();
        public new bool Equals(object? x, object? y) => ReferenceEquals(x, y);
        public int GetHashCode(object obj) => System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(obj);
    }

    [Fact]
    public void DeterministicDrivableOps_TraceReplayAndSerializedReplayMatchEagerOnNewValues()
    {
        var harness = new DifferentiableOpsGradCheckSweep(_output);
        var unavailable = new List<string>();
        var failures = new List<string>();
        int tested = 0;
        foreach (var (name, method, args) in harness.DrivableOps(unavailable, includeDelegators: true))
        {
            // Random sampling has no deterministic eager oracle. Dedicated dropout/RReLU tests cover seeded capture.
            if (name is "TensorDropout" or "TensorRandomLike" or "TensorRReLU" or "RReLU") continue;
            var originals = args.SelectMany(Inputs).Distinct(ReferenceEqualityComparerLocal.Instance).Cast<Tensor<double>>().ToArray();
            if (originals.Length == 0) continue;
            Tensor<double>[] Forward(Tensor<double>[] xs)
            {
                Tensor<double> Substitute(Tensor<double> tensor) => xs[Array.FindIndex(originals, x => ReferenceEquals(x, tensor))];
                object SubstituteArg(object arg) => arg switch
                {
                    Tensor<double> t => Substitute(t),
                    Tensor<double>[] ts => ts.Select(Substitute).ToArray(),
                    List<Tensor<double>> ts => ts.Select(Substitute).ToList(),
                    IEnumerable<Tensor<double>> ts => ts.Select(Substitute).ToArray(),
                    _ => arg
                };
                return new[] { (Tensor<double>)method.Invoke(AiDotNetEngine.Current, args.Select(SubstituteArg).ToArray())! };
            }
            try
            {
                var examples = originals.Select(x => Copy(x)).ToArray();
                // Writable host index arrays and Bit/complex tensors are outside the closed wire format.
                // Check that these fail explicitly rather than silently freezing their values into a trace.
                bool nonPortable = name is "MaxPool2DWithIndices" or "MaxPool3DWithIndices"
                    or "NativeSpectralFilter" or "NativeSpectralFilterBatch"
                    || (name == "ReduceMax" && method.GetParameters().Any(p => p.ParameterType.IsByRef))
                    || args.Any(x => x is Tensor<Bit>);
                if (nonPortable)
                {
                    var rejection = Record.Exception(() => TensorProgram.Trace<double>(Forward, examples,
                        new TraceOptions { InferDynamicDimensions = false }));
                    Assert.NotNull(rejection);
                    Assert.Contains("has no portable form", rejection.GetBaseException().Message);
                    continue;
                }
                // Reversing packed ROI coordinates moves a coordinate into the batch-id column. Use valid
                // explicit inputs for these structured tensors, and compare both replay paths below instead.
                bool structuredRois = name is "RoIAlign" or "RoIPool" or "PsRoIAlign" or "PsRoIPool";
                var program = TensorProgram.Trace<double>(Forward, examples,
                    new TraceOptions { InferDynamicDimensions = false, Verify = !structuredRois });
                Assert.NotEmpty(program.Nodes);
                var changed = originals.Select(x => Copy(x, changed: true)).ToArray();
                var expected = Copy(Forward(changed.Select(x => Copy(x)).ToArray())[0]);
                Equal(expected, program.Run(changed.Select(x => Copy(x)).ToArray())[0]);
                using var stream = new MemoryStream();
                new ProgramPackage(program).Save(stream);
                stream.Position = 0;
                var loaded = ProgramPackage.Load(stream).Program;
                Equal(expected, loaded.Run(changed.Select(x => Copy(x)).ToArray())[0]);
                tested++;
            }
            catch (Exception ex)
            {
                failures.Add($"{name}|{DifferentiableOpsGradCheckSweep.ParamFingerprint(method)}: {ex.GetBaseException().Message}");
            }
        }
        _output.WriteLine($"Trace/replay/serialization checked {tested} operations. Argument harness omissions: {string.Join("; ", unavailable)}");
        Assert.True(tested >= 450, $"Only {tested} operations were exercised.");
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }
}
