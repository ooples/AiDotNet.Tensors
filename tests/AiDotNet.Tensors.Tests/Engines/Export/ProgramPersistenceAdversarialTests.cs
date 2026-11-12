using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Export;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Export;

[Collection("CompilationGlobalState")]
public class ProgramPersistenceAdversarialTests : IDisposable
{
    private readonly IEngine _prior = AiDotNetEngine.Current;
    public ProgramPersistenceAdversarialTests() => AiDotNetEngine.Current = new CpuEngine();
    public void Dispose() => AiDotNetEngine.Current = _prior;

    private static Tensor<float> T(float value) => new(new[] { value }, new[] { 1 });

    private static TensorProgram Weighted(out Tensor<float> weight)
    {
        var w = T(2); weight = w;
        return TensorProgram.Trace(x => AiDotNetEngine.Current.TensorMultiply(x, w), T(3),
            new TraceOptions { Parameters = new object[] { w } });
    }

    private static TensorProgram Reload(TensorProgram program, bool includeValues = true)
    {
        using var stream = new MemoryStream();
        new ProgramPackage(program).Save(stream, includeParameterValues: includeValues);
        stream.Position = 0;
        return ProgramPackage.Load(stream).Program;
    }

    [Fact]
    public void Save_PreservesUpdatedLiveWeightsWithoutReplayingFirst()
    {
        var program = Weighted(out var w);
        w[0] = 7;
        Assert.Equal(21f, Reload(program).Run(T(3)).ToArray()[0]);
    }

    [Fact]
    public void Save_PreservesReboundWeightsEvenWhenThePackageAlreadyHadValues()
    {
        var program = Reload(Weighted(out _));
        program.BindParameters(new[] { T(9) });
        Assert.Equal(27f, Reload(program).Run(T(3)).ToArray()[0]);
    }

    [Fact]
    public void ShapeOnlyBinding_IsAlsoAvailableAtAnotherReplayDtype()
    {
        var program = Reload(Weighted(out _), includeValues: false);
        program.BindParameters(new[] { T(4) });
        var x = new Tensor<double>(new[] { 3.0 }, new[] { 1 });
        Assert.Equal(12.0, program.Run(x).ToArray()[0]);
    }

    [Fact]
    public void SetStored_ReplacesTheLiveParameterAndItsCachedReplay()
    {
        var program = Weighted(out _);
        Assert.Equal(6f, program.Run(T(3)).ToArray()[0]);
        program.SetStored(program.ParameterIds.Single(), T(8));
        Assert.Equal(24f, program.Run(T(3)).ToArray()[0]);
        Assert.Equal(24f, Reload(program).Run(T(3)).ToArray()[0]);
    }

    [Fact]
    public void Reader_RejectsUseBeforeDefinition()
    {
        var program = Weighted(out _);
        var node = program.Nodes[0];
        var arg = node.Args.First(a => a.Kind == ProgramArgKind.Value);
        ((int[])arg.Values)[0] = node.Outputs[0];
        Assert.Throws<ProgramFormatException>(() =>
            ProgramSerializer.ReadProgram(ProgramSerializer.WriteProgram(program), program.Stored));
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void Trace_RejectsUnrecordedNonFiniteOutput(float replacement)
    {
        Tensor<float> F(Tensor<float> x)
        {
            var result = AiDotNetEngine.Current.TensorMultiplyScalar(x, 2f);
            result[0] = replacement;
            return result;
        }
        Assert.Throws<TraceException>(() => TensorProgram.Trace(F, T(3),
            new TraceOptions { InferDynamicDimensions = false }));
    }

    [Fact]
    public void PlanVerification_ChecksEachElementAndNonFiniteValues()
    {
        static Tensor<float> V(params float[] data) => new(data, new[] { data.Length });
        Assert.False(ProgramPackage.Agrees(V(1e9f, 0f), V(1e9f, 1000f)));
        Assert.True(ProgramPackage.Agrees(V(float.NaN, float.PositiveInfinity, 1f), V(float.NaN, float.PositiveInfinity, 1f)));
        Assert.False(ProgramPackage.Agrees(V(float.PositiveInfinity, 1f), V(float.NegativeInfinity, 1f)));
        Assert.False(ProgramPackage.Agrees(V(float.NaN, 1f), V(float.NaN, 2f)));
    }

    [Fact]
    public void EagerPackageExecution_ReturnsIndependentOwnedOutputs()
    {
        var program = TensorProgram.Trace<float>(xs => new[] { xs[0], xs[0] }, new[] { T(3) });
        var package = new ProgramPackage(program);
        var input = T(5);
        var result = package.Execute(AiDotNetEngine.Current, new[] { input });
        Assert.NotSame(input, result[0]);
        Assert.NotSame(result[0], result[1]);
        result[0][0] = 99;
        Assert.Equal(5f, input.ToArray()[0]);
        Assert.Equal(5f, result[1].ToArray()[0]);
    }

    [Fact]
    public void Reader_RejectsAnUndefinedProgramOutput()
    {
        var program = Weighted(out _);
        var values = (List<ProgramValue>)program.Values;
        int id = values.Count;
        values.Add(new ProgramValue(id, ProgramValueRole.Intermediate, "$T", new[] { 1 }, "Tensor"));
        ((int[])program.Outputs)[0] = id;
        Assert.Throws<ProgramFormatException>(() =>
            ProgramSerializer.ReadProgram(ProgramSerializer.WriteProgram(program), program.Stored));
    }
}
