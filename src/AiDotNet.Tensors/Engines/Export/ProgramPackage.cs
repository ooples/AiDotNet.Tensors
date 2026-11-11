using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Compilation.Serialization;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Export;

/// <summary>A compiled plan stored in a <see cref="ProgramPackage"/> for one device, element type and input shape set.</summary>
public sealed class CompiledTarget
{
    internal CompiledTarget(string device, string hardwareFingerprint, string elementType, int[][] inputShapes, int planFormatVersion,
        int codecVersion, byte[] payload)
    {
        Device = device;
        HardwareFingerprint = hardwareFingerprint;
        ElementType = elementType;
        InputShapes = inputShapes;
        PlanFormatVersion = planFormatVersion;
        CodecVersion = codecVersion;
        Payload = payload;
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

    /// <summary>Size of the stored plan in bytes.</summary>
    public int SizeInBytes => Payload.Length;

    internal bool Matches(string device, string fingerprint, string elementType, int[][] shapes) =>
        Device == device && HardwareFingerprint == fingerprint && ElementType == elementType
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
}

/// <summary>
/// A <see cref="TensorProgram"/> together with compiled plans for any number of devices. The program is the source of
/// truth and is portable; each compiled target is a cache for one device, element type and shape set. Asking for a
/// plan on a device the package has no (valid) target for compiles it from the program and adds the target, so a
/// package saved after running on new hardware carries that hardware's plan too. Nothing in a package can name code to
/// run: replay binds only to generated IEngine catalog entries.
/// </summary>
public sealed class ProgramPackage
{
    private readonly List<CompiledTarget> _targets = new();
    private readonly Dictionary<string, object> _livePlans = new();
    private readonly object _lock = new();

    /// <summary>Wraps a program with no compiled targets.</summary>
    public ProgramPackage(TensorProgram program) => Program = program ?? throw new ArgumentNullException(nameof(program));

    /// <summary>The portable program.</summary>
    public TensorProgram Program { get; }

    /// <summary>The compiled targets this package carries.</summary>
    public IReadOnlyList<CompiledTarget> Targets { get { lock (_lock) return _targets.ToArray(); } }

    /// <summary>True when a target was added since the package was loaded or saved.</summary>
    public bool IsDirty { get; private set; }

    /// <summary>
    /// Returns a compiled plan for <paramref name="inputs"/> on <paramref name="engine"/>, with its inputs bound. Uses a
    /// stored target when one matches this device, hardware, element type and shapes; otherwise compiles from the
    /// program and stores the new target (self-healing). Returns null when the program cannot be compiled to a single
    /// plan (several outputs, or an op the plan compiler does not capture); <see cref="Execute{T}"/> then replays eagerly.
    /// </summary>
    public ICompiledPlan<T>? GetOrCompile<T>(IEngine engine, Tensor<T>[] inputs)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (inputs is null) throw new ArgumentNullException(nameof(inputs));
        if (Program.Outputs.Count != 1) return null;

        var shapes = inputs.Select(t => t._shape.ToArray()).ToArray();
        string elementType = ElementTypes.KeyOf(typeof(T)) ?? typeof(T).Name;
        string device = engine.Name;
        string fingerprint = Helpers.Autotune.HardwareFingerprint.Current;
        string key = device + "|" + fingerprint + "|" + elementType + "|" + string.Join(";", shapes.Select(s => string.Join(",", s)));

        lock (_lock)
        {
            if (_livePlans.TryGetValue(key, out var live))
            {
                var plan = (ICompiledPlan<T>)live;
                plan.SetInputs(inputs);
                return plan;
            }

            foreach (var target in _targets.Where(t => t.Matches(device, fingerprint, elementType, shapes)).ToList())
            {
                using var ms = new MemoryStream(target.Payload, writable: false);
                var loaded = CompiledPlanLoader.LoadInferenceAsync<T>(ms, engine).GetAwaiter().GetResult();
                if (loaded is null)
                {
                    _targets.Remove(target); // stale for this runtime: drop it and recompile below
                    IsDirty = true;
                    continue;
                }
                loaded.SetInputs(inputs);
                _livePlans[key] = loaded;
                return loaded;
            }

            ICompiledPlan<T> compiled;
            try
            {
                using var scope = GraphMode.EnableInference();
                var output = Program.Run(engine, inputs)[0];
                compiled = scope.CompileInference<T>(output, inputs);
            }
            catch (Exception ex) when (ex is NotSupportedException or InvalidOperationException)
            {
                Trace.WriteLine($"[ProgramPackage] The program replays eagerly on {device}: {ex.Message}");
                return null;
            }

            try
            {
                using var ms = new MemoryStream();
                compiled.SaveAsync(ms).GetAwaiter().GetResult();
                _targets.Add(new CompiledTarget(device, fingerprint, elementType, shapes, PlanFormatConstants.CurrentFormatVersion,
                    PlanFormatConstants.TensorCodecVersion, ms.ToArray()));
                IsDirty = true;
            }
            catch (Exception ex) when (ex is NotSupportedException or InvalidOperationException)
            {
                // The plan runs but cannot be serialized (an op without a plan codec): keep it live, not stored.
                Trace.WriteLine($"[ProgramPackage] The plan for {device} is not stored: {ex.Message}");
            }
            _livePlans[key] = compiled;
            return compiled;
        }
    }

    /// <summary>
    /// Runs the program: through a compiled plan for this device when one is available or can be built, otherwise by
    /// eager replay.
    /// </summary>
    public Tensor<T>[] Execute<T>(IEngine engine, Tensor<T>[] inputs)
    {
        var plan = GetOrCompile(engine, inputs);
        return plan is null ? Program.Run(engine, inputs) : new[] { plan.Execute() };
    }

    /// <summary>Compiles (or loads) a target for each shape set now, so the saved package starts warm on this device.</summary>
    public void Precompile<T>(IEngine engine, IEnumerable<Tensor<T>[]> inputSets)
    {
        foreach (var inputs in inputSets) GetOrCompile(engine, inputs);
    }

    /// <summary>Removes every compiled target (for example before shipping a package that must recompile everywhere).</summary>
    public void ClearTargets()
    {
        lock (_lock)
        {
            _targets.Clear();
            _livePlans.Clear();
            IsDirty = true;
        }
    }

    // ── Persistence ────────────────────────────────────────────────────────────────────────────────────────

    /// <summary>Writes the package: header, program section, tensor section, one section per compiled target, checksum.</summary>
    public void Save(Stream stream) => Save(stream, includeParameterValues: true);

    /// <summary>
    /// Writes the package. With <paramref name="includeParameterValues"/> false, parameters are written by shape only: a
    /// caller that persists the weights itself (a model file) binds them after loading with
    /// <see cref="TensorProgram.BindParameters{T}"/>, so weights are not stored twice.
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
            WriteSection(w, ProgramSerializer.SectionTensors, ProgramSerializer.WriteTensors(Program.Stored, includeParameterValues ? null : new HashSet<int>(Program.ParameterIds)));
            foreach (var t in Targets) WriteSection(w, ProgramSerializer.SectionCompiled, WriteTarget(t));
            w.Flush();
            body = ms.ToArray();
        }
        ulong hash = XXHash64.Compute(body, 0, body.Length);
        stream.Write(body, 0, body.Length);
        stream.Write(BitConverter.GetBytes(hash), 0, 8);
        IsDirty = false;
    }

    /// <summary>Writes the package to a file (atomically: a temporary file is renamed over the target).</summary>
    public void Save(string path)
    {
        var tmp = path + ".tmp";
        using (var fs = File.Create(tmp)) Save(fs);
        if (File.Exists(path)) File.Delete(path);
        File.Move(tmp, path);
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
        while (true)
        {
            byte tag;
            try { tag = r.Byte(); }
            catch (ProgramFormatException) { break; }
            int length = r.Count(1);
            var payload = r.Bytes(length);
            switch (tag)
            {
                case ProgramSerializer.SectionProgram: programBytes = payload; break;
                case ProgramSerializer.SectionTensors: tensorBytes = payload; break;
                case ProgramSerializer.SectionCompiled: targets.Add(ReadTarget(payload)); break;
                default: break; // a section a newer writer added that this reader does not need
            }
        }
        if (programBytes is null || tensorBytes is null) throw new ProgramFormatException("The package has no program or tensor section.");

        var unbound = new HashSet<int>();
        var stored = ProgramSerializer.ReadTensors(tensorBytes, unbound);
        var program = ProgramSerializer.ReadProgram(programBytes, stored);
        program.Unbound = unbound;
        var package = new ProgramPackage(program);
        package._targets.AddRange(options.KeepForeignTargets
            ? targets
            : targets.Where(t => t.HardwareFingerprint == Helpers.Autotune.HardwareFingerprint.Current));
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
        var payload = r.Bytes(r.Count(1));
        return new CompiledTarget(device, fingerprint, elementType, shapes, planVersion, codec, payload);
    }
}
