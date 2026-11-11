using System;
using System.IO;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Export;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Export;

[Collection("CompilationGlobalState")]
public class ProgramPackageTests : IDisposable
{
    private readonly IEngine _prior = AiDotNetEngine.Current;
    public ProgramPackageTests() { AiDotNetEngine.Current = new CpuEngine(); }
    public void Dispose() { AiDotNetEngine.Current = _prior; }

    private static Tensor<float> Random(int seed, params int[] shape)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        var d = t.GetDataArray();
        for (int i = 0; i < d.Length; i++) d[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static TensorProgram Mlp(out Tensor<float> w)
    {
        var weight = Random(1, 4, 3);
        w = weight;
        return TensorProgram.Trace(x =>
        {
            var e = AiDotNetEngine.Current;
            return e.Tanh(e.TensorMultiplyScalar(e.TensorMatMul(x, weight), 2f));
        }, Random(2, 5, 4), new TraceOptions { Parameters = new object[] { weight } });
    }

    private static byte[] Bytes(ProgramPackage p)
    {
        using var ms = new MemoryStream();
        p.Save(ms);
        return ms.ToArray();
    }

    [Fact]
    public void SaveLoad_RoundTripsTheProgram_AndIsDeterministic()
    {
        var program = Mlp(out _);
        var bytes = Bytes(new ProgramPackage(program));
        var loaded = ProgramPackage.Load(new MemoryStream(bytes));

        Assert.Equal(program.ToString(), loaded.Program.ToString());
        Assert.Equal(bytes, Bytes(loaded));

        var x = Random(3, 9, 4);
        Assert.Equal(program.Run(x).GetDataArray(), loaded.Program.Run(x).GetDataArray());
        Assert.Single(loaded.Program.GetParameters<float>());
    }

    [Fact]
    public void SaveWithoutParameterValues_RequiresBinding_AndRunsOnTheBoundWeights()
    {
        var program = Mlp(out var w);
        var withValues = Bytes(new ProgramPackage(program));
        using var ms = new MemoryStream();
        new ProgramPackage(program).Save(ms, includeParameterValues: false);
        Assert.True(ms.Length < withValues.Length);

        var loaded = ProgramPackage.Load(new MemoryStream(ms.ToArray())).Program;
        Assert.False(loaded.ParametersBound);
        var x = Random(3, 9, 4);
        Assert.Throws<InvalidOperationException>(() => loaded.Run(x));

        loaded.BindParameters(new[] { w });
        Assert.True(loaded.ParametersBound);
        Assert.Equal(program.Run(x).GetDataArray(), loaded.Run(x).GetDataArray());
        Assert.Throws<ArgumentException>(() => loaded.BindParameters(new[] { Random(1, 2, 2) }));
    }

    [Fact]
    public void Load_RejectsTamperedFiles()
    {
        var bytes = Bytes(new ProgramPackage(Mlp(out _)));
        bytes[bytes.Length / 2] ^= 0x5A;
        Assert.Throws<ProgramFormatException>(() => ProgramPackage.Load(new MemoryStream(bytes)));
        Assert.Throws<ProgramFormatException>(() => ProgramPackage.Load(new MemoryStream(bytes.Take(10).ToArray())));
    }

    [Fact]
    public void Load_RejectsOpsOutsideTheCatalog_EvenWithAValidChecksum()
    {
        var program = Mlp(out _);
        var body = ProgramSerializer.WriteProgram(program);
        var original = System.Text.Encoding.UTF8.GetBytes(program.Nodes[0].Op.Id);
        // Same length, so the rest of the section still parses: only the op id changes.
        var forged = System.Text.Encoding.UTF8.GetBytes(new string('X', original.Length));
        int at = IndexOf(body, original);
        Buffer.BlockCopy(forged, 0, body, at, forged.Length);

        var ex = Assert.Throws<ProgramFormatException>(() => ProgramSerializer.ReadProgram(body, program.Stored));
        Assert.Contains("does not have", ex.Message);
    }

    private static int IndexOf(byte[] haystack, byte[] needle)
    {
        for (int i = 0; i <= haystack.Length - needle.Length; i++)
            if (haystack.AsSpan(i, needle.Length).SequenceEqual(needle)) return i;
        throw new InvalidOperationException("not found");
    }

    [Fact]
    public void GetOrCompile_AddsATargetOnce_AndAReloadedPackageStartsWarm()
    {
        var package = new ProgramPackage(Mlp(out _));
        var engine = AiDotNetEngine.Current;
        var x = Random(4, 5, 4);

        var plan = package.GetOrCompile(engine, new[] { x });
        Assert.NotNull(plan);
        Assert.True(package.IsDirty);
        var target = Assert.Single(package.Targets);
        Assert.Equal(engine.Name, target.Device);
        Assert.Equal(package.Program.Run(x).GetDataArray(), plan!.Execute().GetDataArray());

        var reloaded = ProgramPackage.Load(new MemoryStream(Bytes(package)));
        Assert.False(reloaded.IsDirty);
        var warm = reloaded.GetOrCompile(engine, new[] { x });
        Assert.NotNull(warm);
        Assert.False(reloaded.IsDirty);            // served from the stored target, no recompile
        Assert.Single(reloaded.Targets);
        Assert.Equal(plan.Execute().GetDataArray(), warm!.Execute().GetDataArray());
    }

    [Fact]
    public void GetOrCompile_OnAnotherDevice_KeepsTheForeignTarget_AndAddsItsOwn()
    {
        var package = new ProgramPackage(Mlp(out _));
        var x = Random(5, 5, 4);
        package.GetOrCompile(AiDotNetEngine.Current, new[] { x });

        // Same package, now loaded on "another machine": rewrite the stored target's fingerprint.
        var foreign = package.Targets[0];
        var moved = new CompiledTarget(foreign.Device, "other-machine", foreign.ElementType, foreign.InputShapes.ToArray(),
            foreign.PlanFormatVersion, foreign.CodecVersion, foreign.Payload);
        var onOther = new ProgramPackage(package.Program);
        typeof(ProgramPackage).GetField("_targets", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!
            .SetValue(onOther, new System.Collections.Generic.List<CompiledTarget> { moved });

        Assert.NotNull(onOther.GetOrCompile(AiDotNetEngine.Current, new[] { x }));
        Assert.True(onOther.IsDirty);
        Assert.Equal(2, onOther.Targets.Count);    // self-healed: both machines' plans travel together
    }

    [Fact]
    public void Execute_FallsBackToEagerReplay_ForMultiOutputPrograms()
    {
        var program = TensorProgram.Trace(xs =>
        {
            var e = AiDotNetEngine.Current;
            return new[] { e.TensorNegate(xs[0]), e.TensorAdd(xs[0], xs[1]) };
        }, new[] { Random(6, 3, 2), Random(7, 3, 2) });
        var package = new ProgramPackage(program);
        var a = Random(8, 3, 2);
        var b = Random(9, 3, 2);
        var outs = package.Execute(AiDotNetEngine.Current, new[] { a, b });
        Assert.Equal(2, outs.Length);
        Assert.Empty(package.Targets);
    }
}
