using System;
using System.Collections.Generic;
using Stopwatch = System.Diagnostics.Stopwatch;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Compilation.Serialization;
using AiDotNet.Tensors.Engines.Export;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Export;

/// <summary>
/// Compiled targets of a <see cref="ProgramPackage"/>: they follow their inputs, return owned results, run safely in
/// parallel, run on the program's current weights without storing them, and never serve another program's weights.
/// </summary>
[Collection("CompilationGlobalState")]
public class ProgramPackagePlanTests : IDisposable
{
    private readonly IEngine _prior = AiDotNetEngine.Current;
    private readonly ITestOutputHelper _out;

    public ProgramPackagePlanTests(ITestOutputHelper output)
    {
        _out = output;
        AiDotNetEngine.Current = new CpuEngine();
    }

    public void Dispose() { AiDotNetEngine.Current = _prior; }

    private static IEngine Engine => AiDotNetEngine.Current;

    internal static Tensor<float> Random(int seed, params int[] shape)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        var d = t.GetDataArray();
        for (int i = 0; i < d.Length; i++) d[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    internal static void Close(float[] expected, float[] actual, float tol = 1e-4f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tol * (1 + Math.Abs(expected[i])), $"element {i}: {actual[i]} vs {expected[i]}");
    }

    // Dense(ReLU) -> Dense, either as separate engine ops or as FusedLinear.
    internal static TensorProgram DenseMlp(int seed, out Tensor<float>[] parameters, bool fused = false, int inputs = 4, int hidden = 6, int outputs = 2)
    {
        var w1 = Random(seed, inputs, hidden);
        var b1 = Random(seed + 1, hidden);
        var w2 = Random(seed + 2, hidden, outputs);
        var b2 = Random(seed + 3, outputs);
        parameters = new[] { w1, b1, w2, b2 };
        return TensorProgram.Trace(x =>
        {
            var e = AiDotNetEngine.Current;
            if (fused)
            {
                var h1 = e.FusedLinear(x, w1, b1, FusedActivationType.ReLU);
                return e.FusedLinear(h1, w2, b2, FusedActivationType.None);
            }
            var h = e.ReLU(e.TensorAdd(e.TensorMatMul(x, w1), b1));
            return e.TensorAdd(e.TensorMatMul(h, w2), b2);
        }, Random(100, 3, inputs), new TraceOptions { Parameters = parameters });
    }

    // Conv(3x3) -> ReLU -> Conv(1x1).
    internal static TensorProgram ConvNet(int seed, out Tensor<float>[] parameters)
    {
        var k1 = Random(seed, 4, 2, 3, 3);
        var k2 = Random(seed + 1, 3, 4, 1, 1);
        parameters = new[] { k1, k2 };
        return TensorProgram.Trace(x =>
        {
            var e = AiDotNetEngine.Current;
            return e.Conv2D(e.ReLU(e.Conv2D(x, k1, 1, 1, 1)), k2, 1, 0, 1);
        }, Random(101, 2, 2, 6, 6), new TraceOptions { Parameters = parameters });
    }

    private static byte[] Bytes(ProgramPackage p, bool includeParameterValues = true)
    {
        using var ms = new MemoryStream();
        p.Save(ms, includeParameterValues);
        return ms.ToArray();
    }

    private static ProgramPackage Reload(ProgramPackage p) => ProgramPackage.Load(new MemoryStream(Bytes(p)));

    public static IEnumerable<object[]> Models() => new[]
    {
        new object[] { "mlp", false }, new object[] { "mlp", true },
        new object[] { "fused-mlp", false }, new object[] { "fused-mlp", true },
        new object[] { "conv", false }, new object[] { "conv", true },
    };

    private static (TensorProgram Program, int[] Shape) Model(string name) => name switch
    {
        "mlp" => (DenseMlp(10, out _), new[] { 3, 4 }),
        "fused-mlp" => (DenseMlp(10, out _, fused: true), new[] { 3, 4 }),
        _ => (ConvNet(20, out _), new[] { 2, 2, 6, 6 }),
    };

    [Theory]
    [MemberData(nameof(Models))]
    public void Execute_FollowsEveryInput_OnCompiledAndLoadedTargets(string model, bool loaded)
    {
        var (program, shape) = Model(model);
        var package = new ProgramPackage(program);
        if (loaded)
        {
            package.Precompile(Engine, new[] { new[] { Random(300, shape) } });
            Assert.Single(package.Targets);
            package = Reload(package);
            Assert.Single(package.Targets);
        }

        // Clones share storage copy-on-write: the case that used to pin the plan to its first input.
        var xs = Enumerable.Range(0, 5).Select(i => Random(200 + i, shape)).ToArray();
        var results = xs.Select(x => package.Execute(Engine, new[] { (Tensor<float>)x.Clone() })[0]).ToArray();
        Assert.NotEmpty(package.Targets);
        for (int i = 0; i < xs.Length; i++) Close(program.Run(xs[i]).GetDataArray(), results[i].GetDataArray());
        Assert.NotEqual(results[0].GetDataArray(), results[1].GetDataArray());

        // GetOrCompile's plan follows its inputs too.
        var plan = package.GetOrCompile(Engine, new[] { xs[3] });
        Assert.NotNull(plan);
        Close(program.Run(xs[3]).GetDataArray(), plan!.Execute().GetDataArray());
        plan = package.GetOrCompile(Engine, new[] { (Tensor<float>)xs[4].Clone() });
        Close(program.Run(xs[4]).GetDataArray(), plan!.Execute().GetDataArray());
    }

    [Fact]
    public void Execute_DoesNotOverwriteTheCallersInput()
    {
        var program = DenseMlp(10, out _);
        var package = new ProgramPackage(program);
        var first = Random(1, 3, 4);
        var copy = first.GetDataArray().ToArray();
        package.Execute(Engine, new[] { first });
        package.Execute(Engine, new[] { Random(2, 3, 4) });
        Assert.Equal(copy, first.GetDataArray());
    }

    [Fact]
    public void CompiledModelCache_FollowsClonedInputs()
    {
        var w = Random(5, 4, 3);
        using var cache = new CompiledModelCache<float>();
        var xs = Enumerable.Range(0, 6).Select(i => Random(40 + i, 2, 4)).ToArray();
        foreach (var x in xs)
        {
            var input = (Tensor<float>)x.Clone();
            var plan = cache.GetOrCompileInference(new[] { input }, () => Engine.Tanh(Engine.TensorMatMul(input, w)));
            var expected = Engine.Tanh(Engine.TensorMatMul(x, w)).GetDataArray();
            Close(expected, plan.Execute().GetDataArray());
        }
    }

    [Fact]
    public void CompiledPlan_FollowsItsSlot_AfterTheSlotIsCloned()
    {
        var w = Random(5, 4, 3);
        using var cache = new CompiledModelCache<float>();
        var slot = Random(60, 2, 4);
        var plan = cache.GetOrCompileInference(slot, () => Engine.TensorMatMul(slot, w));
        var peer = (Tensor<float>)slot.Clone();
        var peerData = peer.GetDataArray().ToArray();
        var next = Random(61, 2, 4);
        try
        {
            plan.SetInputs(new[] { next });
            Close(Engine.TensorMatMul(next, w).GetDataArray(), plan.Execute().GetDataArray());
        }
        catch (InvalidOperationException)
        {
            // Refusing is the other correct outcome: the plan cannot be fed without detaching its kernels' array.
        }
        Assert.Equal(peerData, peer.GetDataArray()); // the clone never sees the plan's inputs

        var x = Random(62, 2, 4);
        var replanned = cache.GetOrCompileInference(x, () => Engine.TensorMatMul(x, w));
        Close(Engine.TensorMatMul(x, w).GetDataArray(), replanned.Execute().GetDataArray());
    }

    [Fact]
    public void Execute_ReturnsIndependentResults()
    {
        var program = DenseMlp(10, out _);
        var package = new ProgramPackage(program);
        var a = package.Execute(Engine, new[] { Random(1, 3, 4) })[0];
        var aData = a.GetDataArray().ToArray();
        var b = package.Execute(Engine, new[] { Random(2, 3, 4) })[0];
        Assert.False(ReferenceEquals(a, b));
        Assert.Equal(aData, a.GetDataArray());               // the second call did not overwrite the first result
        a.GetDataArray()[0] = 1234f;                          // and writing a result does not reach the plan
        Close(program.Run(Random(3, 3, 4)).GetDataArray(), package.Execute(Engine, new[] { Random(3, 3, 4) })[0].GetDataArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Execute_IsCorrectUnderParallelCallers(bool loaded)
    {
        var program = DenseMlp(10, out _, inputs: 16, hidden: 32, outputs: 8);
        var package = new ProgramPackage(program);
        if (loaded)
        {
            package.Precompile(Engine, new[] { new[] { Random(1, 5, 16) } });
            package = Reload(package);
        }
        var engine = Engine;
        var xs = Enumerable.Range(0, 200).Select(i => Random(1000 + i, 5, 16)).ToArray();
        var expected = xs.Select(x => program.Run(x).GetDataArray()).ToArray();
        var actual = new float[xs.Length][];
        Parallel.For(0, xs.Length, new ParallelOptions { MaxDegreeOfParallelism = 8 }, i =>
            actual[i] = package.Execute(engine, new[] { xs[i] })[0].GetDataArray());
        for (int i = 0; i < xs.Length; i++) Close(expected[i], actual[i]);
        Assert.Single(package.Targets);
    }

    [Fact]
    public void Targets_DoNotStoreWeights()
    {
        var program = DenseMlp(10, out var parameters, inputs: 64, hidden: 128, outputs: 32);
        long weightBytes = parameters.Sum(p => (long)p.Length * sizeof(float));
        var package = new ProgramPackage(program);
        var shapes = new[] { 1, 2, 3, 4 };
        package.Precompile(Engine, shapes.Select(n => new[] { Random(n, n, 64) }));
        Assert.Equal(shapes.Length, package.Targets.Count);
        foreach (var t in package.Targets)
            Assert.True(t.SizeInBytes < weightBytes / 4, $"a {t.SizeInBytes}-byte target for {weightBytes} bytes of weights");

        var withValues = Bytes(package);
        Assert.True(withValues.Length < weightBytes + weightBytes / 2, $"{withValues.Length} bytes for {weightBytes} bytes of weights");
        var withoutValues = Bytes(package, includeParameterValues: false);
        Assert.True(withoutValues.Length < weightBytes / 2, $"{withoutValues.Length} bytes saved without values");

        // Saved without values, the targets still run once the weights are bound.
        var loaded = ProgramPackage.Load(new MemoryStream(withoutValues));
        loaded.Program.BindParameters(parameters);
        var x = Random(9, 3, 64);
        Close(program.Run(x).GetDataArray(), loaded.Execute(Engine, new[] { x })[0].GetDataArray());
        Assert.False(loaded.IsDirty); // served from the stored target
    }

    [Fact]
    public void ASwappedInTarget_RunsOnTheProgramsOwnWeights()
    {
        var a = new ProgramPackage(DenseMlp(10, out _));
        a.Precompile(Engine, new[] { new[] { Random(1, 3, 4) } });
        var b = new ProgramPackage(DenseMlp(50, out _)); // same architecture, other weights

        // B's program and weights with A's compiled sections.
        var swapped = ProgramPackage.Load(new MemoryStream(WithSections(Bytes(b), CompiledSections(Bytes(a)))));
        Assert.Single(swapped.Targets);
        var x = Random(7, 3, 4);
        var expected = b.Program.Run(x).GetDataArray();
        Close(expected, swapped.Execute(Engine, new[] { x })[0].GetDataArray());
        Assert.False(swapped.IsDirty); // A's target was used, on B's weights
        Assert.NotEqual(a.Program.Run(x).GetDataArray(), expected);

        // A target compiled from a different program is dropped on load.
        var other = new ProgramPackage(DenseMlp(10, out _, hidden: 7));
        var mismatched = ProgramPackage.Load(new MemoryStream(WithSections(Bytes(other), CompiledSections(Bytes(a)))));
        Assert.Empty(mismatched.Targets);
        Close(other.Program.Run(x).GetDataArray(), mismatched.Execute(Engine, new[] { x })[0].GetDataArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Predictions_FollowBindParameters_AndInPlaceWeightUpdates(bool loaded)
    {
        var program = DenseMlp(10, out var parameters);
        var package = new ProgramPackage(program);
        var x = Random(8, 3, 4);
        var before = package.Execute(Engine, new[] { x })[0].GetDataArray();
        if (loaded)
        {
            package = Reload(package);
            package.Program.BindParameters(parameters);
            Close(before, package.Execute(Engine, new[] { x })[0].GetDataArray());
        }

        // An in-place update of a live weight (an optimizer step).
        parameters[0][0, 0] = parameters[0][0, 0] + 3f;
        parameters[2][1, 1] = -2f;
        var updated = package.Execute(Engine, new[] { x })[0].GetDataArray();
        Close(package.Program.Run(x).GetDataArray(), updated);
        Assert.NotEqual(before, updated);

        // New weight tensors.
        var replacement = new[] { Random(70, 4, 6), Random(71, 6), Random(72, 6, 2), Random(73, 2) };
        package.Program.BindParameters(replacement);
        var rebound = package.Execute(Engine, new[] { x })[0].GetDataArray();
        Close(package.Program.Run(x).GetDataArray(), rebound);
        Assert.NotEqual(updated, rebound);

        // A plan handed out before the update is not left serving the old weights.
        var plan = package.GetOrCompile(Engine, new[] { x })!;
        Close(rebound, plan.Execute().GetDataArray());

        // Repeated predictions with unchanged weights do not rebuild.
        int builds = package.BuildCount;
        for (int i = 0; i < 5; i++) package.Execute(Engine, new[] { Random(80 + i, 3, 4) });
        Assert.Equal(builds, package.BuildCount);
    }

    [Fact]
    public void Load_ThrowsOnlyProgramFormatException_OnFuzzedFiles()
    {
        var package = new ProgramPackage(DenseMlp(10, out _));
        package.Precompile(Engine, new[] { new[] { Random(1, 3, 4) } });
        var clean = Bytes(package);
        var shapeOnly = Bytes(package, includeParameterValues: false);
        var rng = new Random(12345);
        int loadedCount = 0;
        for (int iteration = 0; iteration < 3000; iteration++)
        {
            var source = iteration % 2 == 0 ? clean : shapeOnly;
            var body = source.Take(source.Length - 8).ToArray();
            int mutations = 1 + rng.Next(4);
            for (int m = 0; m < mutations; m++)
            {
                int at = rng.Next(body.Length);
                switch (rng.Next(4))
                {
                    case 0: body[at] ^= (byte)(1 << rng.Next(8)); break;
                    case 1: body[at] = (byte)rng.Next(256); break;
                    case 2: body[at] = 0xFF; break;
                    default:
                        // An int-sized field set to an extreme value.
                        int at4 = Math.Min(at, body.Length - 4);
                        var extreme = new[] { int.MaxValue, int.MinValue, -1, 0x7FFF0000, 65536 }[rng.Next(5)];
                        BitConverter.GetBytes(extreme).CopyTo(body, at4);
                        break;
                }
            }
            var file = WithChecksum(body);
            try
            {
                var loaded = ProgramPackage.Load(new MemoryStream(file));
                loadedCount++;
            }
            catch (ProgramFormatException)
            {
            }
        }
        _out.WriteLine($"{loadedCount} of 3000 mutated files still loaded");
    }

    [Fact]
    public void ReadTensors_DoesNotAllocateForShapeOnlyParameters()
    {
        // A 4-GiB parameter declared by shape in a few bytes.
        var stored = new Dictionary<int, object> { [0] = new UnboundTensor(typeof(float), new[] { 1 << 15, 1 << 15 }) };
        var bytes = ProgramSerializer.WriteTensors(stored);
        Assert.True(bytes.Length < 64);
        long before = GC.GetTotalMemory(false);
        var unbound = new HashSet<int>();
        var read = ProgramSerializer.ReadTensors(bytes, unbound, long.MaxValue);
        Assert.True(GC.GetTotalMemory(false) - before < 1 << 20);
        Assert.IsType<UnboundTensor>(read[0]);

        Assert.Throws<ProgramFormatException>(() => ProgramSerializer.ReadTensors(bytes, new HashSet<int>(), maxUnboundElements: 1 << 20));
        var overflow = ProgramSerializer.WriteTensors(new Dictionary<int, object> { [0] = new UnboundTensor(typeof(float), new[] { int.MaxValue, int.MaxValue, int.MaxValue }) });
        Assert.Throws<ProgramFormatException>(() => ProgramSerializer.ReadTensors(overflow, new HashSet<int>()));
    }

    [Fact]
    public void PlanLoader_NeverThrows_OnFuzzedPlans()
    {
        var w = Random(3, 4, 3);
        using var cache = new CompiledModelCache<float>();
        var x = Random(4, 2, 4);
        var plan = cache.GetOrCompileInference(x, () => Engine.Tanh(Engine.TensorMatMul(x, w)));
        using var ms = new MemoryStream();
        plan.SaveAsync(ms).GetAwaiter().GetResult();
        var clean = ms.ToArray();
        var rng = new Random(99);
        for (int iteration = 0; iteration < 2000; iteration++)
        {
            var body = clean.Take(clean.Length - 16).ToArray();
            for (int m = 0, n = 1 + rng.Next(3); m < n; m++)
            {
                int at = rng.Next(body.Length);
                if (rng.Next(2) == 0) body[at] = (byte)rng.Next(256);
                else BitConverter.GetBytes(new[] { int.MaxValue, -1, 1 << 30 }[rng.Next(3)]).CopyTo(body, Math.Min(at, body.Length - 4));
            }
            var file = body.Concat(BitConverter.GetBytes((long)body.Length)).Concat(BitConverter.GetBytes(XXHash64.Compute(body, 0, body.Length))).ToArray();
            CompiledPlanLoader.LoadInferenceAsync<float>(new MemoryStream(file), Engine).GetAwaiter().GetResult()?.Dispose();
        }
    }

    [Fact]
    public void SavedStateEnums_ResolveOnlyThisLibrarysEnums()
    {
        var forged = typeof(System.IO.FileAccess).AssemblyQualifiedName!;
        using var ms = new MemoryStream();
        using (var w = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            w.Write(1);
            w.Write(PlanFormatConstants.TagEnum);
            var name = System.Text.Encoding.UTF8.GetBytes(forged);
            w.Write(name.Length);
            w.Write(name);
            w.Write(1L);
        }
        ms.Position = 0;
        using var reader = new BinaryReader(ms);
        Assert.Throws<InvalidDataException>(() => SavedStateSerializer.Read<float>(reader, Array.Empty<Tensor<float>>()));
    }

    [Fact]
    public void Save_ToPath_ReplacesTheFileAtomically()
    {
        var dir = Path.Combine(Path.GetTempPath(), "aidn-pkg-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var path = Path.Combine(dir, "model.atnp");
            new ProgramPackage(DenseMlp(10, out _)).Save(path);
            new ProgramPackage(DenseMlp(11, out _, hidden: 7)).Save(path); // over an existing file
            Assert.Equal(7, ProgramPackage.Load(path).Program.Values.Max(v => v.Shape.LastOrDefault()));
            Assert.Single(Directory.GetFiles(dir));                        // no temporary file left behind
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void Overhead_OfOwnedResultsAndPooling_IsMeasured()
    {
        foreach (var (inputs, hidden, outputs, batch) in new[] { (4, 6, 2, 3), (256, 256, 64, 32) })
        {
            var program = DenseMlp(10, out _, inputs: inputs, hidden: hidden, outputs: outputs);
            var x = Random(1, batch, inputs);
            // Both the pooled plan and the handed-out one are loaded from the same stored target, so they are the same plan.
            var built = new ProgramPackage(program);
            built.Precompile(Engine, new[] { new[] { x } });
            var package = Reload(built);
            var plan = package.GetOrCompile(Engine, new[] { x })!;
            package.Execute(Engine, new[] { x });
            int n = inputs < 100 ? 20000 : 1000;
            double raw = double.MaxValue, owned = double.MaxValue, eager = double.MaxValue;
            for (int round = 0; round < 7; round++)
            {
                var sw = Stopwatch.StartNew();
                for (int i = 0; i < n; i++) { plan.SetInputs(new[] { x }); plan.Execute(); }
                raw = Math.Min(raw, sw.Elapsed.TotalMilliseconds * 1000 / n);
                sw.Restart();
                for (int i = 0; i < n; i++) package.Execute(Engine, new[] { x });
                owned = Math.Min(owned, sw.Elapsed.TotalMilliseconds * 1000 / n);
                sw.Restart();
                for (int i = 0; i < n / 4; i++) program.Run(x);
                eager = Math.Min(eager, sw.Elapsed.TotalMilliseconds * 1000 / (n / 4));
            }
            _out.WriteLine($"[{batch},{inputs}]x[{inputs},{hidden}]x[{hidden},{outputs}]: plan SetInputs+Execute {raw:F2} us; " +
                           $"package.Execute {owned:F2} us (+{owned - raw:F2}); eager replay {eager:F2} us");
        }
    }

    // ── File surgery helpers ──────────────────────────────────────────────────────────────────────────────────

    private const int HeaderBytes = 4 + 2 + 4;

    private static List<byte[]> CompiledSections(byte[] file)
    {
        var sections = new List<byte[]>();
        int pos = HeaderBytes, end = file.Length - 8;
        while (pos < end)
        {
            byte tag = file[pos];
            int length = BitConverter.ToInt32(file, pos + 1);
            if (tag == ProgramSerializer.SectionCompiled) sections.Add(file.Skip(pos).Take(5 + length).ToArray());
            pos += 5 + length;
        }
        return sections;
    }

    private static byte[] WithSections(byte[] file, IEnumerable<byte[]> sections) =>
        WithChecksum(file.Take(file.Length - 8).Concat(sections.SelectMany(s => s)).ToArray());

    private static byte[] WithChecksum(byte[] body) =>
        body.Concat(BitConverter.GetBytes(XXHash64.Compute(body, 0, body.Length))).ToArray();
}
