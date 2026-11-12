using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Export;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Export;

/// <summary>Replay and tracing edge cases: size-dependent constants, in-place ops, two-type ops, enums, threads.</summary>
[Collection("CompilationGlobalState")]
public class ProgramReplayHardeningTests : IDisposable
{
    private readonly IEngine _prior = AiDotNetEngine.Current;
    public ProgramReplayHardeningTests() { AiDotNetEngine.Current = new CpuEngine(); }
    public void Dispose() { AiDotNetEngine.Current = _prior; }

    private static Tensor<float> Random(int seed, params int[] shape) => ProgramPackagePlanTests.Random(seed, shape);

    [Fact]
    public void Trace_KeepsABatchStatic_WhenACapturedConstantDependsOnIt()
    {
        // x + 1/batch, with the 1/batch held in a captured tensor rather than an argument.
        static Tensor<float> F(Tensor<float> x) =>
            AiDotNetEngine.Current.TensorAdd(x, new Tensor<float>(Enumerable.Repeat(1f / x.Shape[0], 4).ToArray(), new[] { 4 }));

        var program = TensorProgram.Trace(F, Random(1, 4, 4));
        Assert.Equal(-1, program.InputDimensions[0][0].Symbol);
        Assert.Contains(program.Diagnostics, d => d.Contains("captured constant"));
        Assert.Throws<ArgumentException>(() => program.Run(Random(2, 7, 4)));
        ProgramPackagePlanTests.Close(F(Random(3, 4, 4)).GetDataArray(), program.Run(Random(3, 4, 4)).GetDataArray());
    }

    [Fact]
    public void Trace_StillGeneralizesABatch_WhenConstantsDoNotDependOnIt()
    {
        var bias = Random(5, 4);
        Tensor<float> F(Tensor<float> x) => AiDotNetEngine.Current.TensorAdd(x, bias);
        var program = TensorProgram.Trace(F, Random(1, 4, 4));
        Assert.Equal(0, program.InputDimensions[0][0].Symbol);
        ProgramPackagePlanTests.Close(F(Random(2, 7, 4)).GetDataArray(), program.Run(Random(2, 7, 4)).GetDataArray());
    }

    [Fact]
    public void Run_DoesNotMutateCachedConstants_ThroughInPlaceOps()
    {
        var accumulator = Random(5, 3, 4);
        var original = accumulator.GetDataArray().ToArray();
        Tensor<float> F(Tensor<float> x)
        {
            var e = AiDotNetEngine.Current;
            e.TensorAddInPlace(accumulator, x);
            return e.TensorMultiplyScalar(accumulator, 2f);
        }
        var program = TensorProgram.Trace(F, Random(1, 3, 4), new TraceOptions { Verify = false, InferDynamicDimensions = false });
        var x = Random(2, 3, 4);
        var expected = Enumerable.Range(0, original.Length).Select(i => (original[i] + x.GetDataArray()[i]) * 2f).ToArray();

        var first = program.Run(x).GetDataArray();
        var second = program.Run(x).GetDataArray();
        ProgramPackagePlanTests.Close(expected, first);
        Assert.Equal(first, second);

        var results = new float[64][];
        Parallel.For(0, results.Length, i => results[i] = program.Run(x).GetDataArray());
        foreach (var r in results) Assert.Equal(first, r);
    }

    [Fact]
    public void Run_TensorCopyIntoAConstant_WritesTheDestination()
    {
        var destination = new Tensor<float>(new[] { 3, 4 });
        Tensor<float> F(Tensor<float> x)
        {
            var e = AiDotNetEngine.Current;
            e.TensorCopy(x, destination);
            return e.TensorMultiplyScalar(destination, 3f);
        }
        var program = TensorProgram.Trace(F, Random(1, 3, 4), new TraceOptions { Verify = false, InferDynamicDimensions = false });
        var node = program.Nodes.Single(n => n.Op.Name == "TensorCopy");
        Assert.Equal("destination", node.Op.Parameters[node.Op.InPlaceParameter].Name);
        var x = Random(2, 3, 4);
        ProgramPackagePlanTests.Close(x.GetDataArray().Select(v => v * 3f).ToArray(), program.Run(x).GetDataArray());
        ProgramPackagePlanTests.Close(x.GetDataArray().Select(v => v * 3f).ToArray(), program.Run(x).GetDataArray());
    }

    [Theory]
    [InlineData("TensorCopy", "destination")]
    [InlineData("TensorAddInPlace", "a")]
    [InlineData("ReLUInPlace", "tensor")]
    [InlineData("MatMulInto", "destination")]
    [InlineData("GroupNormInto", "output")]
    [InlineData("ScatterInPlace", "input")]
    [InlineData("SetRow", "matrix")]
    [InlineData("CopyVectorToTensor", "destination")]
    [InlineData("FFT", null)]
    [InlineData("IFFT", null)]
    [InlineData("SinCos", null)]
    [InlineData("STFT", null)]
    public void Catalog_NamesTheParameterAVoidOpWrites(string op, string? written)
    {
        foreach (var d in EngineOpCatalog.Ops.Where(o => o.Name == op && o.ReturnKind == EngineValueKind.Void))
        {
            if (written is null) Assert.Equal(-1, d.InPlaceParameter);
            else Assert.Equal(written, d.Parameters[d.InPlaceParameter].Name);
        }
        Assert.Contains(EngineOpCatalog.Ops, o => o.Name == op);
    }

    [Fact]
    public void Catalog_EveryInPlaceOpWritesATensorParameter()
    {
        foreach (var d in EngineOpCatalog.Ops.Where(o => o.InPlaceParameter >= 0))
        {
            Assert.Equal(EngineValueKind.Void, d.ReturnKind);
            Assert.Contains(d.Parameters[d.InPlaceParameter].Kind,
                new[] { EngineValueKind.Tensor, EngineValueKind.TensorFixed, EngineValueKind.Vector, EngineValueKind.Matrix });
        }
    }

    [Fact]
    public void TwoTypeParameterOps_Replay_AtAnyElementType()
    {
        var table = Random(3, 10, 4);
        var ids = new Tensor<int>(new[] { 1, 3, 5, 7, 9, 0 }, new[] { 6 });
        Tensor<float> F(Tensor<float> x)
        {
            var e = AiDotNetEngine.Current;
            return e.TensorAdd(e.TensorEmbeddingLookup<float, int>(table, ids), x);
        }
        var op = EngineOpCatalog.Ops.Single(o => o.Name == "TensorEmbeddingLookup");
        Assert.True(op.IsTraceable);

        var program = TensorProgram.Trace(F, Random(4, 6, 4), new TraceOptions { InferDynamicDimensions = false });
        var x = Random(5, 6, 4);
        ProgramPackagePlanTests.Close(F(x).GetDataArray(), program.Run(x).GetDataArray());

        var xd = new Tensor<double>(x.GetDataArray().Select(v => (double)v).ToArray(), new[] { 6, 4 });
        var asDouble = program.Run(AiDotNetEngine.Current, new[] { xd })[0].GetDataArray();
        ProgramPackagePlanTests.Close(F(x).GetDataArray(), asDouble.Select(v => (float)v).ToArray());

        // And through a saved package.
        using var ms = new MemoryStream();
        new ProgramPackage(program).Save(ms);
        var loaded = ProgramPackage.Load(new MemoryStream(ms.ToArray()));
        ProgramPackagePlanTests.Close(F(x).GetDataArray(), loaded.Program.Run(x).GetDataArray());
    }

    [Fact]
    public void EnumArguments_MustBeDefinedMembers()
    {
        Assert.Equal(FusedActivationType.ReLU, EngineArgCoercion.As<FusedActivationType>(new EnumLiteral("ReLU")));
        Assert.Throws<InvalidCastException>(() => EngineArgCoercion.As<FusedActivationType>(new EnumLiteral("999")));
        Assert.Throws<InvalidCastException>(() => EngineArgCoercion.As<FusedActivationType>("999"));
        Assert.Throws<InvalidCastException>(() => EngineArgCoercion.As<FusedActivationType>(999));
        Assert.Throws<InvalidCastException>(() => EngineArgCoercion.As<FusedActivationType>("NoSuchMember"));

        // [Flags] enums accept combinations of defined flags, by name or value, and nothing else.
        Assert.Equal(FileAttributes.ReadOnly | FileAttributes.Hidden, EngineArgCoercion.As<FileAttributes>("ReadOnly, Hidden"));
        Assert.Equal(FileAttributes.ReadOnly | FileAttributes.Hidden, EngineArgCoercion.As<FileAttributes>("3"));
        Assert.Throws<InvalidCastException>(() => EngineArgCoercion.As<FileAttributes>(unchecked((int)0x80000000)));
    }

    [Fact]
    public void Trace_FailsWhenTheFunctionCallsTheEngineFromAnotherThread()
    {
        static Tensor<float> F(Tensor<float> x)
        {
            Tensor<float>? doubled = null;
            var worker = new Thread(() => doubled = AiDotNetEngine.Current.TensorMultiplyScalar(x, 2f));
            worker.Start();
            worker.Join();
            return AiDotNetEngine.Current.TensorAdd(doubled!, x);
        }
        var ex = Assert.Throws<TraceException>(() => TensorProgram.Trace(F, Random(1, 3, 4)));
        Assert.Contains("another thread", ex.Message);

        static Tensor<float> G(Tensor<float> x)
        {
            var parts = new Tensor<float>[4];
            var tasks = Enumerable.Range(0, 4).Select(i => Task.Factory.StartNew(
                () => parts[i] = AiDotNetEngine.Current.TensorMultiplyScalar(x, i + 1f), TaskCreationOptions.LongRunning)).ToArray();
            Task.WaitAll(tasks);
            return parts.Aggregate((a, b) => AiDotNetEngine.Current.TensorAdd(a, b));
        }
        Assert.Throws<TraceException>(() => TensorProgram.Trace(G, Random(1, 3, 4)));
    }

    [Fact]
    public void Trace_AllowsTheEnginesOwnParallelism()
    {
        // A large matmul runs on several threads inside the engine; those calls are nested, not the function's.
        var w = Random(2, 256, 256);
        Tensor<float> F(Tensor<float> x) => AiDotNetEngine.Current.TensorMatMul(x, w);
        var program = TensorProgram.Trace(F, Random(1, 128, 256));
        Assert.Single(program.Nodes);
        var x = Random(3, 128, 256);
        ProgramPackagePlanTests.Close(F(x).GetDataArray(), program.Run(x).GetDataArray(), 1e-3f);
    }
}
