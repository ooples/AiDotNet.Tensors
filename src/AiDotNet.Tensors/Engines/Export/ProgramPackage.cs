using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Compilation.Serialization;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Export;

/// <summary>A compiled plan stored in a <see cref="ProgramPackage"/> for one device, element type and input shape set.</summary>
/// <remarks>
/// A target holds no weights: the plan references the program's parameters and constants by value id, and is bound to
/// the program's current tensors whenever it is loaded. It also records a hash of the program it was compiled from, so
/// a target is never run against a different program.
/// </remarks>
public sealed class CompiledTarget
{
    internal CompiledTarget(string device, string hardwareFingerprint, string elementType, int[][] inputShapes, int planFormatVersion,
        int codecVersion, byte[] payload, ulong programHash)
    {
        Device = device;
        HardwareFingerprint = hardwareFingerprint;
        ElementType = elementType;
        InputShapes = inputShapes;
        PlanFormatVersion = planFormatVersion;
        CodecVersion = codecVersion;
        Payload = payload;
        ProgramHash = programHash;
    }

    /// <summary>The engine the plan was compiled for (its <see cref="IEngine.Name"/>).</summary>
    public string Device { get; }

    /// <summary>The hardware fingerprint the plan was compiled and autotuned on.</summary>
    public string HardwareFingerprint { get; }

    /// <summary>The element type key.</summary>
    public string ElementType { get; }

    /// <summary>The concrete input shapes the plan was specialized to.</summary>
    public IReadOnlyList<int[]> InputShapes { get; }

    /// <summary>The compiled-plan format version of the payload.</summary>
    public int PlanFormatVersion { get; }

    /// <summary>The tensor-codec version of the payload.</summary>
    public int CodecVersion { get; }

    internal byte[] Payload { get; }

    /// <summary>Hash of the program section and parameter shapes the plan was compiled from.</summary>
    internal ulong ProgramHash { get; }

    /// <summary>Size of the stored plan in bytes.</summary>
    public int SizeInBytes => Payload.Length;

    internal bool Matches(string device, string fingerprint, string elementType, int[][] shapes, ulong programHash) =>
        Device == device && HardwareFingerprint == fingerprint && ElementType == elementType && ProgramHash == programHash
        && PlanFormatVersion == PlanFormatConstants.CurrentFormatVersion && CodecVersion == PlanFormatConstants.TensorCodecVersion
        && InputShapes.Count == shapes.Length && InputShapes.Zip(shapes, (a, b) => a.SequenceEqual(b)).All(x => x);
}

/// <summary>Options for <see cref="ProgramPackage.Load(Stream, ProgramLoadOptions?)"/>.</summary>
public sealed class ProgramLoadOptions
{
    /// <summary>Maximum file size accepted, in bytes. Default 4 GiB.</summary>
    public long MaxBytes { get; set; } = 4L << 30;

    /// <summary>Whether to keep compiled targets for other devices (so a re-save carries them). Default true.</summary>
    public bool KeepForeignTargets { get; set; } = true;

    /// <summary>
    /// Maximum total elements of the parameters a file may declare by shape only (saved without values). Nothing is
    /// allocated for them on load, but a declared size is still bounded. Default 2^32.
    /// </summary>
    public long MaxUnboundElements { get; set; } = 1L << 32;
}

/// <summary>
/// A <see cref="TensorProgram"/> together with compiled plans for any number of devices. The program is the source of
/// truth and is portable; each compiled target is a cache for one device, element type and shape set. Asking for a
/// plan on a device the package has no (valid) target for compiles it from the program and adds the target, so a
/// package saved after running on new hardware carries that hardware's plan too. Nothing in a package can name code to
/// run: replay binds only to generated IEngine catalog entries.
/// </summary>
/// <remarks>
/// <para>Compiled targets reference the program's parameters instead of copying them, so a plan always runs on the
/// weights the program binds now (<see cref="TensorProgram.BindParameters{T}"/>, or the model's live tensors), and the
/// file holds the weights at most once. A plan is rebuilt when those tensors are replaced or written.</para>
/// <para>Every plan is checked once against eager replay on a second input before it is used; a plan that does not
/// follow its input, or that fails to run, is dropped and its shape is served by eager replay, which is always
/// correct. <see cref="Execute{T}"/> is safe to call from several threads and returns tensors the caller owns.</para>
/// </remarks>
public sealed class ProgramPackage
{
    // Plan instances kept per shape set for concurrent Execute callers (each runs one call at a time).
    private static readonly int MaxPoolSize = Math.Max(1, Math.Min(Environment.ProcessorCount, 8));

    private readonly List<CompiledTarget> _targets = new();
    private readonly ConcurrentDictionary<string, object> _entries = new();
    private readonly object _lock = new();
    private ulong? _programHash;
    private int _builds;

    /// <summary>Wraps a program with no compiled targets.</summary>
    public ProgramPackage(TensorProgram program) => Program = program ?? throw new ArgumentNullException(nameof(program));

    /// <summary>The portable program.</summary>
    public TensorProgram Program { get; }

    /// <summary>The compiled targets this package carries.</summary>
    public IReadOnlyList<CompiledTarget> Targets { get { lock (_lock) return _targets.ToArray(); } }

    /// <summary>True when a target was added since the package was loaded or saved.</summary>
    public bool IsDirty { get; private set; }

    /// <summary>How many plan instances were built (compiled or loaded) so far; for tests.</summary>
    internal int BuildCount => Volatile.Read(ref _builds);

    /// <summary>The hash a target records of the program it was compiled from: the program section plus parameter shapes.</summary>
    internal ulong ProgramHash => _programHash ??= ComputeProgramHash(Program);

    private static ulong ComputeProgramHash(TensorProgram program)
    {
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms, Encoding.UTF8);
        w.Write(ProgramSerializer.WriteProgram(program));
        foreach (var id in program.ParameterIds)
        {
            var shape = program.Values[id].Shape;
            w.Write(id);
            w.Write(shape.Length);
            foreach (var d in shape) w.Write(d);
        }
        w.Flush();
        var bytes = ms.ToArray();
        return XXHash64.Compute(bytes, 0, bytes.Length);
    }

    /// <summary>
    /// Returns a compiled plan for <paramref name="inputs"/> on <paramref name="engine"/>, with its inputs bound. Uses a
    /// stored target when one matches this device, hardware, element type, program and shapes; otherwise compiles from
    /// the program and stores the new target (self-healing). Returns null when the program is served by eager replay
    /// for these shapes (several outputs, an op the plan compiler does not capture, or a plan that failed its check
    /// against eager replay); <see cref="Execute{T}"/> then replays eagerly.
    /// </summary>
    /// <remarks>
    /// The returned plan belongs to this package and is shared by every caller of <see cref="GetOrCompile{T}"/> for
    /// these shapes: run it from one thread at a time, and copy what <see cref="ICompiledPlan{T}.Execute"/> returns
    /// before the next run. It is disposed when the program's parameters are replaced or written (a later call returns
    /// a plan built on the new weights). <see cref="Execute{T}"/> has none of these constraints.
    /// </remarks>
    public ICompiledPlan<T>? GetOrCompile<T>(IEngine engine, Tensor<T>[] inputs)
    {
        var entry = EnsureEntry(engine, inputs);
        if (entry is null) return null;
        lock (_lock)
        {
            if (entry.Eager) return null;
            if (entry.Handout is null || entry.Handout.Generation != entry.Generation)
            {
                entry.Handout?.Plan.Dispose();
                entry.Handout = null;
                var plan = NewInstance(entry, engine, inputs, verifyCompiled: true);
                if (plan is null) return null;
                entry.Handout = new PlanInstance<T>(plan, entry.Generation);
            }
            entry.Handout.Plan.SetInputs(inputs);
            return entry.Handout.Plan;
        }
    }

    /// <summary>
    /// Runs the program: through a compiled plan for this device when one is available or can be built, otherwise by
    /// eager replay. The results are new tensors the caller owns. Safe to call from several threads.
    /// </summary>
    public Tensor<T>[] Execute<T>(IEngine engine, Tensor<T>[] inputs)
    {
        var entry = EnsureEntry(engine, inputs);
        if (entry is null) return Program.Run(engine, inputs).Select(Owned).ToArray();

        var instance = Acquire(entry, engine, inputs);
        if (instance is null) return Program.Run(engine, inputs).Select(Owned).ToArray();
        try
        {
            instance.Plan.SetInputs(inputs);
            var output = instance.Plan.Execute();
            // The plan's output buffer is overwritten by its next run: the caller gets its own copy.
            return new[] { Owned(output) };
        }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException)
        {
            Trace.WriteLine($"[ProgramPackage] The plan for {entry.Device} failed ({ex.Message}); this shape now replays eagerly.");
            lock (_lock) entry.Eager = true;
            return Program.Run(engine, inputs).Select(Owned).ToArray();
        }
        finally
        {
            Release(entry, instance);
        }
    }

    /// <summary>Compiles (or loads) a target for each shape set now, so the saved package starts warm on this device.</summary>
    public void Precompile<T>(IEngine engine, IEnumerable<Tensor<T>[]> inputSets)
    {
        if (inputSets is null) throw new ArgumentNullException(nameof(inputSets));
        foreach (var inputs in inputSets) EnsureEntry(engine, inputs);
    }

    /// <summary>Removes every compiled target (for example before shipping a package that must recompile everywhere).</summary>
    public void ClearTargets()
    {
        lock (_lock)
        {
            _targets.Clear();
            foreach (var entry in _entries.Values) ((IEntry)entry).Retire();
            _entries.Clear();
            IsDirty = true;
        }
    }

    // ── Plan entries ───────────────────────────────────────────────────────────────────────────────────────

    private interface IEntry
    {
        void Retire();
    }

    private sealed class PlanInstance<T>
    {
        internal PlanInstance(ICompiledPlan<T> plan, int generation)
        {
            Plan = plan;
            Generation = generation;
        }

        internal ICompiledPlan<T> Plan { get; }
        internal int Generation { get; }
        internal readonly object Gate = new();
        internal bool Retired;
    }

    /// <summary>What a plan was built against: the program's parameter generation and each external's storage and version.</summary>
    private sealed class WeightStamp<T>
    {
        private readonly int _generation;
        private readonly Tensor<T>[] _tensors;
        private readonly object?[] _backings;
        private readonly object[] _storages;
        private readonly int[] _versions;

        internal WeightStamp(TensorProgram program, Tensor<T>?[] externals)
        {
            _generation = program.ParameterGeneration;
            _tensors = externals.Where(t => t is not null).Select(t => t!).ToArray();
            _backings = _tensors.Select(t => (object?)t.GetLiveBackingArrayIdentityOrNull()).ToArray();
            _storages = _tensors.Select(t => t.StorageIdentity).ToArray();
            _versions = _tensors.Select(t => t.Version).ToArray();
        }

        // Allocation-free: checked on every Execute.
        internal bool IsCurrent(TensorProgram program)
        {
            if (program.ParameterGeneration != _generation) return false;
            for (int i = 0; i < _tensors.Length; i++)
            {
                var t = _tensors[i];
                if (t.Version != _versions[i] || !ReferenceEquals(t.StorageIdentity, _storages[i])
                    || !ReferenceEquals(t.GetLiveBackingArrayIdentityOrNull(), _backings[i])) return false;
            }
            return true;
        }
    }

    private sealed class Entry<T> : IEntry
    {
        internal Entry(string device, string fingerprint, string elementType, int[][] shapes)
        {
            Device = device;
            Fingerprint = fingerprint;
            ElementType = elementType;
            Shapes = shapes;
        }

        internal string Device { get; }
        internal string Fingerprint { get; }
        internal string ElementType { get; }
        internal int[][] Shapes { get; }
        internal CompiledTarget? Target;
        internal volatile bool Eager;
        internal volatile WeightStamp<T>? Stamp;
        internal int Generation;
        internal volatile PlanInstance<T>[] Pool = Array.Empty<PlanInstance<T>>();
        internal PlanInstance<T>? Handout;
        internal Tensor<T>?[]? Externals;

        public void Retire()
        {
            foreach (var instance in Pool) RetireInstance(instance);
            Pool = Array.Empty<PlanInstance<T>>();
            Handout?.Plan.Dispose();
            Handout = null;
        }
    }

    // Disposes an instance now if no caller is running it; otherwise the caller disposes it on release.
    private static void RetireInstance<T>(PlanInstance<T> instance)
    {
        instance.Retired = true;
        if (!Monitor.TryEnter(instance.Gate)) return;
        try { instance.Plan.Dispose(); }
        finally { Monitor.Exit(instance.Gate); }
    }

    /// <summary>
    /// The entry for these inputs, built (or rebuilt, when the program's weights changed since) and verified. Null when
    /// the program has no single output, so it always replays eagerly.
    /// </summary>
    private Entry<T>? EnsureEntry<T>(IEngine engine, Tensor<T>[] inputs)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (inputs is null) throw new ArgumentNullException(nameof(inputs));
        for (int i = 0; i < inputs.Length; i++)
            if (inputs[i] is null) throw new ArgumentNullException(nameof(inputs), $"inputs[{i}] is null.");
        if (Program.Outputs.Count != 1 || inputs.Length != Program.Inputs.Count) return null;

        var shapes = inputs.Select(t => t._shape.ToArray()).ToArray();
        string elementType = ElementTypes.KeyOf(typeof(T)) ?? typeof(T).Name;
        string device = engine.Name;
        string fingerprint = Helpers.Autotune.HardwareFingerprint.Current;
        string key = device + "|" + fingerprint + "|" + elementType + "|" + string.Join(";", shapes.Select(s => string.Join(",", s)));

        var entry = (Entry<T>)_entries.GetOrAdd(key, _ => new Entry<T>(device, fingerprint, elementType, shapes));
        if (entry.Eager) return entry;
        if (entry.Stamp is { } stamp && stamp.IsCurrent(Program)) return entry;

        lock (_lock)
        {
            if (entry.Eager || entry.Stamp is { } current && current.IsCurrent(Program)) return entry;
            Build(entry, engine, inputs);
            return entry;
        }
    }

    /// <summary>
    /// (Re)builds an entry's first plan against the program's current parameters: from the stored target when there is
    /// one, otherwise by compiling the program; then checks it against eager replay. Called under the lock.
    /// </summary>
    private void Build<T>(Entry<T> entry, IEngine engine, Tensor<T>[] inputs)
    {
        entry.Retire();
        entry.Generation++;
        entry.Stamp = null;
        entry.Externals = Program.ExternalTensors<T>();
        var stamp = new WeightStamp<T>(Program, entry.Externals);
        if (entry.Target is null)
            entry.Target = _targets.FirstOrDefault(t => t.Matches(entry.Device, entry.Fingerprint, entry.ElementType, entry.Shapes, ProgramHash));

        // A stored target is tried first; if it cannot be loaded or does not follow its input it is dropped and the
        // program is compiled afresh, and only if that fails too is the shape served eagerly.
        for (int attempt = 0; attempt < 2; attempt++)
        {
            bool fromTarget = entry.Target is not null;
            var plan = fromTarget ? Load(entry, engine) : Compile(entry, engine, inputs);
            if (plan is not null && Verify(plan, engine, inputs))
            {
                entry.Stamp = stamp;
                entry.Pool = new[] { new PlanInstance<T>(plan, entry.Generation) };
                return;
            }
            plan?.Dispose();
            if (entry.Target is { } bad)
            {
                Trace.WriteLine($"[ProgramPackage] Dropping the {entry.Device} target for [{string.Join("; ", entry.Shapes.Select(s => string.Join(",", s)))}]: it does not reproduce the program.");
                _targets.Remove(bad);
                entry.Target = null;
                IsDirty = true;
            }
            if (!fromTarget) break;
        }
        entry.Eager = true;
    }

    /// <summary>A further instance for a built entry: loaded from its target when stored, else compiled (and verified).</summary>
    private ICompiledPlan<T>? NewInstance<T>(Entry<T> entry, IEngine engine, Tensor<T>[] inputs, bool verifyCompiled)
    {
        if (entry.Target is not null) return Load(entry, engine);
        var plan = Compile(entry, engine, inputs, store: false);
        if (plan is not null && verifyCompiled && !Verify(plan, engine, inputs))
        {
            plan.Dispose();
            return null;
        }
        return plan;
    }

    private ICompiledPlan<T>? Load<T>(Entry<T> entry, IEngine engine)
    {
        var target = entry.Target!;
        var registered = RegisteredWeights(entry.Externals!);
        try
        {
            using var ms = new MemoryStream(target.Payload, writable: false);
            var plan = InferencePlanReader.Read<T>(ms, engine, entry.Externals, TableBudget(entry.Shapes));
            Interlocked.Increment(ref _builds);
            return plan;
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            // A plan payload is file data: anything it does wrong only costs this target.
            Trace.WriteLine($"[ProgramPackage] The stored {entry.Device} plan cannot be loaded: {ex.Message}");
            return null;
        }
        finally
        {
            UnregisterNew(entry.Externals!, registered);
        }
    }

    private ICompiledPlan<T>? Compile<T>(Entry<T> entry, IEngine engine, Tensor<T>[] inputs, bool store = true)
    {
        // The plan owns its input slots: private copies, never the caller's tensors (SetInputs writes the slots, and a
        // caller's tensor may share storage copy-on-write or be reused).
        var owned = inputs.Select(Owned).ToArray();
        var registered = RegisteredWeights(entry.Externals!);
        ICompiledPlan<T> compiled;
        try
        {
            using var scope = GraphMode.EnableInference();
            var output = Program.Run(engine, owned)[0];
            compiled = scope.CompileInference<T>(output, owned);
            Interlocked.Increment(ref _builds);
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            Trace.WriteLine($"[ProgramPackage] The program replays eagerly on {entry.Device}: {ex.Message}");
            return null;
        }
        finally
        {
            UnregisterNew(entry.Externals!, registered);
        }

        if (store && compiled is CompiledInferencePlan<T> inference)
        {
            try
            {
                using var ms = new MemoryStream();
                inference.Save(ms, entry.Externals!);
                entry.Target = new CompiledTarget(entry.Device, entry.Fingerprint, entry.ElementType, entry.Shapes,
                    PlanFormatConstants.CurrentFormatVersion, PlanFormatConstants.TensorCodecVersion, ms.ToArray(), ProgramHash);
                _targets.RemoveAll(t => t.Matches(entry.Device, entry.Fingerprint, entry.ElementType, entry.Shapes, ProgramHash));
                _targets.Add(entry.Target);
                IsDirty = true;
            }
            catch (Exception ex) when (ex is NotSupportedException or InvalidOperationException)
            {
                // The plan runs but cannot be serialized (an op without a plan codec): keep it live, not stored.
                Trace.WriteLine($"[ProgramPackage] The plan for {entry.Device} is not stored: {ex.Message}");
            }
        }
        return compiled;
    }

    // Building a plan pre-packs its matmul weights and registers them process-wide (FrozenWeightRegistry), after which
    // eager matmuls on those tensors use the pack too. The program's parameters are live weights that training writes
    // in place, so a registration made here is removed again: the plan keeps its own pack (and is rebuilt when the
    // weights change), while eager replay keeps reading the weights themselves.
    private static bool[] RegisteredWeights<T>(Tensor<T>?[] externals) =>
        externals.Select(t => t is not null && Engines.BlasManaged.FrozenWeightRegistry.TryGetHandle(t) is not null).ToArray();

    private static void UnregisterNew<T>(Tensor<T>?[] externals, bool[] registeredBefore)
    {
        for (int i = 0; i < externals.Length; i++)
            if (externals[i] is { } t && !registeredBefore[i]) Engines.BlasManaged.FrozenWeightRegistry.Unregister(t);
    }

    // Element budget for a stored plan's tensor table: generous for what this program can need at these shapes, but
    // bounded, so a payload cannot make the loader allocate whatever it names.
    private long TableBudget(int[][] shapes)
    {
        long traced = 0, requested = 0;
        for (int i = 0; i < Program.Inputs.Count && i < shapes.Length; i++)
        {
            traced += Product(Program.Values[Program.Inputs[i]].Shape);
            requested += Product(shapes[i]);
        }
        long scale = traced <= 0 ? 1 : Math.Max(1, (requested + traced - 1) / traced);
        long total = 0;
        foreach (var v in Program.Values) total += Product(v.Shape);
        double budget = 16.0 * total * scale + (1 << 20);
        return budget >= PlanFormatConstants.DefaultMaxTableElements ? PlanFormatConstants.DefaultMaxTableElements : (long)budget;
    }

    private static long Product(int[] shape)
    {
        long p = 1;
        foreach (var d in shape) p = Math.Min(p * Math.Max(d, 0), int.MaxValue);
        return p;
    }

    /// <summary>
    /// Checks a plan against eager replay on a second input whose values are the first's permuted: a plan that serves
    /// its trace-time input's result, or reads anything else stale, disagrees there. Also checks the original input.
    /// </summary>
    private bool Verify<T>(ICompiledPlan<T> plan, IEngine engine, Tensor<T>[] inputs)
    {
        try
        {
            foreach (var probe in new[] { Permuted(inputs), inputs })
            {
                Tensor<T> expected;
                using (new NoGradScope<T>())
                    expected = Program.Run(engine, probe)[0];
                plan.SetInputs(probe);
                var actual = plan.Execute();
                if (!Agrees(expected, actual)) return false;
            }
            return true;
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            Trace.WriteLine($"[ProgramPackage] A plan check failed: {ex.Message}");
            return false;
        }
    }

    [Helpers.ScalarKernel("One-time comparison of a freshly built plan with eager replay, with a per-element tolerance and NaN handling.")]
    internal static bool Agrees<T>(Tensor<T> expected, Tensor<T> actual)
    {
        if (!expected._shape.SequenceEqual(actual._shape)) return false;
        var ops = Helpers.MathHelper.GetNumericOperations<T>();
        var e = expected.Contiguous().AsSpan();
        var a = actual.Contiguous().AsSpan();
        double tolerance = typeof(T) == typeof(float) || typeof(T) == typeof(Half) || typeof(T) == typeof(double) ? 1e-3 : 0;
        for (int i = 0; i < e.Length; i++)
        {
            double x = ops.ToDouble(e[i]), y = ops.ToDouble(a[i]);
            if (x == y || (double.IsNaN(x) && double.IsNaN(y))) continue;
            if (double.IsNaN(x) || double.IsNaN(y) || double.IsInfinity(x) || double.IsInfinity(y)
                || !(Math.Abs(x - y) <= tolerance * Math.Max(1, Math.Abs(x)))) return false;
        }
        return true;
    }

    // The inputs with their values reversed (every value stays one the input holds, so ids stay valid ids). A uniform
    // floating-point input, which reversing leaves unchanged, is offset by a small index pattern instead.
    [Helpers.ScalarKernel("One-time construction of the verification input when a plan is built.")]
    private static Tensor<T>[] Permuted<T>(Tensor<T>[] inputs)
    {
        var result = new Tensor<T>[inputs.Length];
        for (int k = 0; k < inputs.Length; k++)
        {
            var data = inputs[k].Contiguous().AsSpan().ToArray();
            var original = (T[])data.Clone();
            Array.Reverse(data);
            if (data.SequenceEqual(original) && (typeof(T) == typeof(float) || typeof(T) == typeof(double) || typeof(T) == typeof(Half)))
            {
                var ops = Helpers.MathHelper.GetNumericOperations<T>();
                for (int i = 0; i < data.Length; i++) data[i] = ops.FromDouble(ops.ToDouble(data[i]) + 0.25 * (i % 7) - 0.75);
            }
            result[k] = new Tensor<T>(data, inputs[k]._shape.ToArray());
        }
        return result;
    }

    private static Tensor<T> Owned<T>(Tensor<T> t) => new(t.Contiguous().AsSpan().ToArray(), t._shape.ToArray());

    /// <summary>
    /// A plan instance for this call, held exclusively until <see cref="Release{T}"/>: an idle pooled one, a new one
    /// loaded from the stored target when all are busy and the pool can grow, else the next one to come free.
    /// </summary>
    private PlanInstance<T>? Acquire<T>(Entry<T> entry, IEngine engine, Tensor<T>[] inputs)
    {
        while (true)
        {
            if (entry.Eager) return null;
            var pool = entry.Pool;
            if (pool.Length == 0) return null;
            foreach (var instance in pool)
            {
                if (!Monitor.TryEnter(instance.Gate)) continue;
                if (!instance.Retired) return instance;
                Monitor.Exit(instance.Gate);
            }

            if (pool.Length < MaxPoolSize && entry.Target is not null)
            {
                lock (_lock)
                {
                    if (ReferenceEquals(entry.Pool, pool) && !entry.Eager)
                    {
                        var plan = NewInstance(entry, engine, inputs, verifyCompiled: false);
                        if (plan is not null)
                        {
                            var added = new PlanInstance<T>(plan, entry.Generation);
                            Monitor.Enter(added.Gate);
                            entry.Pool = pool.Concat(new[] { added }).ToArray();
                            return added;
                        }
                    }
                }
                continue;
            }

            var wait = pool[(Environment.CurrentManagedThreadId & int.MaxValue) % pool.Length];
            Monitor.Enter(wait.Gate);
            if (!wait.Retired) return wait;
            Monitor.Exit(wait.Gate);
        }
    }

    private static void Release<T>(Entry<T> entry, PlanInstance<T> instance)
    {
        try
        {
            if (instance.Retired) instance.Plan.Dispose();
        }
        finally
        {
            Monitor.Exit(instance.Gate);
        }
    }

    // ── Persistence ────────────────────────────────────────────────────────────────────────────────────────

    /// <summary>Writes the package: header, program section, tensor section, one section per compiled target, checksum.</summary>
    public void Save(Stream stream) => Save(stream, includeParameterValues: true);

    /// <summary>
    /// Writes the package. With <paramref name="includeParameterValues"/> false, parameters are written by shape only: a
    /// caller that persists the weights itself (a model file) binds them after loading with
    /// <see cref="TensorProgram.BindParameters{T}"/>, so weights are not stored twice. Compiled targets never store
    /// weights (they reference the program's), so either way the file holds them at most once.
    /// </summary>
    public void Save(Stream stream, bool includeParameterValues)
    {
        if (stream is null) throw new ArgumentNullException(nameof(stream));
        byte[] body;
        using (var ms = new MemoryStream())
        using (var w = new BinaryWriter(ms, Encoding.UTF8))
        {
            w.Write(ProgramSerializer.Magic);
            w.Write(ProgramSerializer.Version);
            w.Write(TensorProgram.FormatVersion);
            WriteSection(w, ProgramSerializer.SectionProgram, ProgramSerializer.WriteProgram(Program));
            WriteSection(w, ProgramSerializer.SectionTensors, ProgramSerializer.WriteTensors(
                includeParameterValues ? Program.StoredForSave() : Program.Stored,
                includeParameterValues ? null : new HashSet<int>(Program.ParameterIds)));
            foreach (var t in Targets) WriteSection(w, ProgramSerializer.SectionCompiled, WriteTarget(t));
            w.Flush();
            body = ms.ToArray();
        }
        ulong hash = XXHash64.Compute(body, 0, body.Length);
        stream.Write(body, 0, body.Length);
        stream.Write(BitConverter.GetBytes(hash), 0, 8);
        IsDirty = false;
    }

    /// <summary>
    /// Writes the package to a file atomically: it is written to a temporary file beside <paramref name="path"/>, which
    /// then replaces it, so a failure leaves the previous file intact (and no temporary file behind).
    /// </summary>
    public void Save(string path)
    {
        if (path is null) throw new ArgumentNullException(nameof(path));
        var full = Path.GetFullPath(path);
        var tmp = full + "." + Guid.NewGuid().ToString("N") + ".tmp";
        try
        {
            using (var fs = new FileStream(tmp, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            {
                Save(fs);
                fs.Flush(flushToDisk: true);
            }
            if (File.Exists(full)) File.Replace(tmp, full, destinationBackupFileName: null);
            else File.Move(tmp, full);
        }
        finally
        {
            try { if (File.Exists(tmp)) File.Delete(tmp); }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }
    }

    /// <summary>Reads a package. Throws <see cref="ProgramFormatException"/> on a malformed, tampered or newer file.</summary>
    public static ProgramPackage Load(Stream stream) => Load(stream, (ProgramLoadOptions?)null);

    /// <summary>Reads a package with options. Throws <see cref="ProgramFormatException"/> on a malformed, tampered or newer file.</summary>
    public static ProgramPackage Load(Stream stream, ProgramLoadOptions? options)
    {
        if (stream is null) throw new ArgumentNullException(nameof(stream));
        options ??= new ProgramLoadOptions();
        byte[] all;
        using (var ms = new MemoryStream())
        {
            var buffer = new byte[81920];
            int n;
            while ((n = stream.Read(buffer, 0, buffer.Length)) > 0)
            {
                if (ms.Length + n > options.MaxBytes) throw new ProgramFormatException($"The program file exceeds {options.MaxBytes} bytes.");
                ms.Write(buffer, 0, n);
            }
            all = ms.ToArray();
        }
        try
        {
            return Parse(all, options);
        }
        catch (ProgramFormatException)
        {
            throw;
        }
        catch (Exception ex) when (ex is not OutOfMemoryException and not ThreadAbortException)
        {
            // Every check the reader makes raises ProgramFormatException; anything else a crafted file provokes is
            // reported the same way, so callers handle one exception for every malformed input.
            throw new ProgramFormatException($"The program file is malformed: {ex.Message}", ex);
        }
    }

    private static ProgramPackage Parse(byte[] all, ProgramLoadOptions options)
    {
        if (all.Length < 18) throw new ProgramFormatException("The file is too short to be a program package.");
        ulong expected = BitConverter.ToUInt64(all, all.Length - 8);
        if (XXHash64.Compute(all, 0, all.Length - 8) != expected) throw new ProgramFormatException("The program file checksum does not match (corrupt or modified).");

        var body = new byte[all.Length - 8];
        Buffer.BlockCopy(all, 0, body, 0, body.Length);
        var r = new ProgramSerializer.Reader(body);
        if ((uint)r.Int() != ProgramSerializer.Magic) throw new ProgramFormatException("Not a program package (bad magic).");
        int containerVersion = (ushort)(r.Byte() | (r.Byte() << 8));
        if (containerVersion == 0 || containerVersion > ProgramSerializer.Version)
            throw new ProgramFormatException($"Program package version {containerVersion} is newer than this runtime supports ({ProgramSerializer.Version}).");
        int programVersion = r.Int();
        if (programVersion <= 0 || programVersion > TensorProgram.FormatVersion)
            throw new ProgramFormatException($"Program format version {programVersion} is newer than this runtime supports ({TensorProgram.FormatVersion}).");

        byte[]? programBytes = null, tensorBytes = null;
        var targets = new List<CompiledTarget>();
        while (r.Remaining > 0)
        {
            byte tag = r.Byte();
            int length = r.Count(1);
            var payload = r.Bytes(length);
            switch (tag)
            {
                case ProgramSerializer.SectionProgram:
                    if (programBytes is not null) throw new ProgramFormatException("The package has two program sections.");
                    programBytes = payload;
                    break;
                case ProgramSerializer.SectionTensors:
                    if (tensorBytes is not null) throw new ProgramFormatException("The package has two tensor sections.");
                    tensorBytes = payload;
                    break;
                case ProgramSerializer.SectionCompiled:
                    // Version 1 targets embedded their own weights and record no program hash: they are dropped and
                    // recompiled, never run against whatever program they now sit beside.
                    if (containerVersion >= 2) targets.Add(ReadTarget(payload));
                    break;
                default: break; // a section a newer writer added that this reader does not need
            }
        }
        if (programBytes is null || tensorBytes is null) throw new ProgramFormatException("The package has no program or tensor section.");

        var unbound = new HashSet<int>();
        var stored = ProgramSerializer.ReadTensors(tensorBytes, unbound, options.MaxUnboundElements);
        var program = ProgramSerializer.ReadProgram(programBytes, stored);
        program.Unbound = unbound;
        var package = new ProgramPackage(program);
        ulong programHash = package.ProgramHash;
        package._targets.AddRange(targets.Where(t => t.ProgramHash == programHash
            && (options.KeepForeignTargets || t.HardwareFingerprint == Helpers.Autotune.HardwareFingerprint.Current)));
        return package;
    }

    /// <summary>Reads a package from a file.</summary>
    public static ProgramPackage Load(string path) => Load(path, (ProgramLoadOptions?)null);

    /// <summary>Reads a package from a file with options.</summary>
    public static ProgramPackage Load(string path, ProgramLoadOptions? options)
    {
        using var fs = File.OpenRead(path);
        return Load(fs, options);
    }

    private static void WriteSection(BinaryWriter w, byte tag, byte[] payload)
    {
        w.Write(tag);
        w.Write(payload.Length);
        w.Write(payload);
    }

    private static byte[] WriteTarget(CompiledTarget t)
    {
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms, Encoding.UTF8);
        w.Write(t.Device);
        w.Write(t.HardwareFingerprint);
        w.Write(t.ElementType);
        w.Write(t.InputShapes.Count);
        foreach (var s in t.InputShapes)
        {
            w.Write(s.Length);
            foreach (var d in s) w.Write(d);
        }
        w.Write(t.PlanFormatVersion);
        w.Write(t.CodecVersion);
        w.Write(t.ProgramHash);
        w.Write(t.Payload.Length);
        w.Write(t.Payload);
        w.Flush();
        return ms.ToArray();
    }

    private static CompiledTarget ReadTarget(byte[] data)
    {
        var r = new ProgramSerializer.Reader(data);
        string device = r.String(), fingerprint = r.String(), elementType = r.String();
        if (ElementTypes.TypeOf(elementType) is null) throw new ProgramFormatException($"A compiled target names unknown element type '{elementType}'.");
        int count = r.Count(4);
        var shapes = new int[count][];
        for (int i = 0; i < count; i++) shapes[i] = r.Ints();
        int planVersion = r.Int(), codec = r.Int();
        ulong programHash = (ulong)r.Long();
        var payload = r.Bytes(r.Count(1));
        return new CompiledTarget(device, fingerprint, elementType, shapes, planVersion, codec, payload, programHash);
    }
}
