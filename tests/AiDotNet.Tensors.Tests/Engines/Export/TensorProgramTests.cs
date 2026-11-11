using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Export;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Export;

[Collection("CompilationGlobalState")]
public class TensorProgramTests : IDisposable
{
    private readonly IEngine _prior = AiDotNetEngine.Current;
    public TensorProgramTests() { AiDotNetEngine.Current = new CpuEngine(); }
    public void Dispose() { AiDotNetEngine.Current = _prior; }

    private static Tensor<float> Random(int seed, params int[] shape)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        var d = t.GetDataArray();
        for (int i = 0; i < d.Length; i++) d[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static void AssertClose(Tensor<float> expected, Tensor<float> actual, float tol = 1e-5f)
    {
        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        var e = expected.GetDataArray();
        var a = actual.GetDataArray();
        for (int i = 0; i < e.Length; i++) Assert.True(Math.Abs(e[i] - a[i]) <= tol, $"element {i}: {a[i]} vs {e[i]}");
    }

    [Fact]
    public void Trace_RecordsOuterEngineCallsOnly_AndReplaysExactly()
    {
        var w = Random(1, 4, 3);
        var b = Random(2, 3);
        Tensor<float> Mlp(Tensor<float> x)
        {
            var e = AiDotNetEngine.Current;
            return e.Sigmoid(e.TensorAdd(e.TensorMatMul(x, w), b));
        }

        var x0 = Random(3, 5, 4);
        var program = TensorProgram.Trace(Mlp, x0);

        Assert.Equal(new[] { "TensorMatMul", "TensorAdd", "Sigmoid" }, program.Nodes.Select(n => n.Op.Name).ToArray());
        Assert.Equal(2, program.Values.Count(v => v.Role == ProgramValueRole.Constant));
        AssertClose(Mlp(x0), program.Run(x0));
    }

    [Fact]
    public void Trace_InfersSymbolicBatch_FromShapeDependentArguments()
    {
        Tensor<float> Flatten(Tensor<float> x)
        {
            var e = AiDotNetEngine.Current;
            return e.Reshape(e.TensorMultiplyScalar(x, 2f), new[] { x.Shape[0], x.Shape[1] * x.Shape[2] });
        }

        var program = TensorProgram.Trace(Flatten, Random(4, 2, 3, 4));
        Assert.Single(program.Symbols);
        Assert.Equal(0, program.InputDimensions[0][0].Symbol);

        var bigger = Random(5, 7, 3, 4);
        AssertClose(Flatten(bigger), program.Run(bigger));
        Assert.Throws<ArgumentException>(() => program.Run(Random(6, 7, 3, 5)));
    }

    [Fact]
    public void Run_AtAnotherElementType_ConvertsConstantsAndScalars()
    {
        var w = Random(7, 3, 2);
        var program = TensorProgram.Trace(x => AiDotNetEngine.Current.TensorMultiplyScalar(AiDotNetEngine.Current.TensorMatMul(x, w), 0.5f), Random(8, 4, 3));

        var xd = new Tensor<double>(new[] { 4, 3 });
        var xdData = xd.GetDataArray();
        for (int i = 0; i < xdData.Length; i++) xdData[i] = i * 0.1;
        var yd = program.Run(xd);

        var wd = new Tensor<double>(w.GetDataArray().Select(v => (double)v).ToArray(), new[] { 3, 2 });
        var expected = new CpuEngine().TensorMultiplyScalar(new CpuEngine().TensorMatMul(xd, wd), 0.5);
        Assert.Equal(expected.GetDataArray(), yd.GetDataArray());
    }

    [Fact]
    public void Replay_IsTrainable_ParametersReceiveGradients()
    {
        var w = Random(9, 3, 2);
        var program = TensorProgram.Trace(x => AiDotNetEngine.Current.TensorMatMul(x, w), Random(10, 4, 3),
            new TraceOptions { Parameters = new object[] { w } });
        var p = Assert.Single(program.GetParameters<float>());
        Assert.Same(w, p); // the live parameter is reused at the traced element type

        var x = Random(11, 4, 3);
        using var tape = new GradientTape<float>();
        var y = program.Run(x);
        var loss = AiDotNetEngine.Current.ReduceSum(y);
        var grads = tape.ComputeGradients(loss, new[] { p });
        Assert.True(grads.ContainsKey(p));
    }

    [Fact]
    public void Trace_RejectsWorkDoneOutsideTheEngine()
    {
        Tensor<float> Sneaky(Tensor<float> x)
        {
            var y = AiDotNetEngine.Current.TensorMultiplyScalar(x, 1f);
            y.GetDataArray()[0] = 42f; // not an engine op: a program cannot record it
            return y;
        }

        var ex = Assert.Throws<TraceException>(() => TensorProgram.Trace(Sneaky, Random(12, 2, 2)));
        Assert.Contains("outside the engine", ex.Message);
    }

    [Fact]
    public void TracingIsThreadLocal()
    {
        var program = TensorProgram.Trace(x =>
        {
            // Another thread computing while this one traces must not appear in the program.
            var other = new System.Threading.Thread(() => AiDotNetEngine.Current.TensorAdd(Random(1, 2), Random(2, 2)));
            other.Start();
            other.Join();
            return AiDotNetEngine.Current.TensorNegate(x);
        }, Random(13, 3));
        Assert.Equal("TensorNegate", Assert.Single(program.Nodes).Op.Name);
    }
}
