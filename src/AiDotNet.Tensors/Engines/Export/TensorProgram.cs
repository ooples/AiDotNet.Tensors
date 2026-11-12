using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Export;

/// <summary>Options for <see cref="TensorProgram.Trace{T}(Func{Tensor{T}[], Tensor{T}[]}, Tensor{T}[], TraceOptions?)"/>.</summary>
public sealed class TraceOptions
{
    /// <summary>
    /// Captured tensors that are trainable. They become <see cref="ProgramValueRole.Parameter"/> values: a replay exposes
    /// them through <see cref="TensorProgram.GetParameters{T}"/> as tape leaves. Every other captured tensor is a constant.
    /// </summary>
    public IEnumerable<object>? Parameters { get; set; }

    /// <summary>Names for captured tensors and inputs (by reference), kept in the program for readability and mapping.</summary>
    public IReadOnlyDictionary<object, string>? Names { get; set; }

    /// <summary>
    /// The input dimensions to treat as dynamic, as (input index, axis) pairs. Default: axis 0 of every input of rank
    /// at least 1. Each is probed at two more sizes; integer arguments that follow it become symbolic. A dimension the
    /// program's structure depends on is reported in <see cref="TensorProgram.Diagnostics"/> and kept static.
    /// </summary>
    public IReadOnlyList<(int Input, int Axis)>? DynamicDimensions { get; set; }

    /// <summary>Whether to probe dynamic dimensions at all. Default true.</summary>
    public bool InferDynamicDimensions { get; set; } = true;

    /// <summary>
    /// Whether to replay the program on the example inputs and compare with the traced function's result, which catches
    /// work the function did outside the engine. Default true.
    /// </summary>
    public bool Verify { get; set; } = true;

    /// <summary>
    /// Further inputs to verify the program on (each run through the function and the replay). The example with its
    /// values reversed is always added, which catches values the function reads out of its input on the host.
    /// Each item is a <c>Tensor&lt;T&gt;[]</c> at the traced element type; items of another type are ignored.
    /// </summary>
    public IEnumerable<object>? VerificationInputs { get; set; }

    /// <summary>Relative/absolute tolerance used by <see cref="Verify"/>. Default 1e-5.</summary>
    public double Tolerance { get; set; } = 1e-5;
}

/// <summary>
/// A portable, versioned record of what a function computes, as calls to public <see cref="IEngine"/> ops with named,
/// typed arguments. Unlike a compiled plan it is hardware independent and holds the logical (pre-fusion) graph;
/// it replays eagerly on any engine and element type (recording on the gradient tape, so a replayed program can be
/// trained), compiles to a plan for the current device, and saves with its compiled plans for any number of devices
/// (see <see cref="ProgramPackage"/>).
/// </summary>
public sealed partial class TensorProgram
{
    /// <summary>The current program format version.</summary>
    public const int FormatVersion = 1;

    private readonly ConcurrentDictionary<Type, object[]> _parameterCache = new();
    private readonly ConcurrentDictionary<Type, object> _constantCache = new();

    internal TensorProgram(string elementType, List<ProgramValue> values, List<ProgramNode> nodes, int[] inputs, int[] outputs,
        Dictionary<int, object> stored, SymbolicInt[][] inputDimensions, string[] symbols, List<string> diagnostics)
    {
        TracedElementType = elementType;
        Values = values;
        Nodes = nodes;
        Inputs = inputs;
        Outputs = outputs;
        Stored = stored;
        InputDimensions = inputDimensions;
        Symbols = symbols;
        Diagnostics = diagnostics;
        OpSetHash = EngineOpCatalog.OpSetHash;
    }

    /// <summary>The element type the program was traced at (a key such as <c>float</c>).</summary>
    public string TracedElementType { get; }

    /// <summary>All values, by id.</summary>
    public IReadOnlyList<ProgramValue> Values { get; }

    /// <summary>The recorded engine calls, in execution order.</summary>
    public IReadOnlyList<ProgramNode> Nodes { get; }

    /// <summary>The input value ids, in argument order.</summary>
    public IReadOnlyList<int> Inputs { get; }

    /// <summary>The output value ids, in result order.</summary>
    public IReadOnlyList<int> Outputs { get; }

    /// <summary>Per input, per axis: the dimension as a constant or a symbol.</summary>
    public IReadOnlyList<IReadOnlyList<SymbolicInt>> InputDimensions { get; }

    /// <summary>Symbol names (s0, s1, ...), one per dynamic dimension.</summary>
    public IReadOnlyList<string> Symbols { get; }

    /// <summary>Notes from tracing (dimensions kept static and why).</summary>
    public IReadOnlyList<string> Diagnostics { get; }

    /// <summary>The op-set hash of the engine surface the program was traced against.</summary>
    public string OpSetHash { get; internal set; }

    /// <summary>Stored tensors for constants and parameters, by value id (as traced; converted on replay).</summary>
    internal Dictionary<int, object> Stored { get; }

    /// <summary>Live parameter tensors given at trace time (not persisted), by value id.</summary>
    internal Dictionary<int, object> LiveParameters { get; set; } = new();

    /// <summary>Parameter ids loaded by shape only, which must be bound before a replay.</summary>
    internal HashSet<int> Unbound { get; set; } = new();

    /// <summary>Whether every parameter has a value (false after loading a package saved without parameter values).</summary>
    public bool ParametersBound => Unbound.Count == 0;

    /// <summary>
    /// Makes replays at <typeparamref name="T"/> use <paramref name="parameters"/> (in <see cref="ParameterIds"/> order) as the
    /// parameter tensors: the live weights of a model, so training them updates the program and the program needs no copy.
    /// </summary>
    public void BindParameters<T>(IReadOnlyList<Tensor<T>> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        var ids = ParameterIds.ToArray();
        if (parameters.Count != ids.Length) throw new ArgumentException($"The program has {ids.Length} parameter(s); got {parameters.Count}.", nameof(parameters));
        for (int i = 0; i < ids.Length; i++)
        {
            var expected = Values[ids[i]].Shape;
            if (!parameters[i]._shape.SequenceEqual(expected))
                throw new ArgumentException($"Parameter {i} (%{ids[i]}) must have shape [{string.Join(", ", expected)}]; got [{string.Join(", ", parameters[i]._shape)}].", nameof(parameters));
        }
        // A rebind replaces the live source as well as this dtype's cache. Other replay dtypes must
        // materialize these weights, not the original snapshot (or an unbound placeholder).
        for (int i = 0; i < ids.Length; i++) LiveParameters[ids[i]] = parameters[i];
        _parameterCache.Clear();
        _parameterCache[typeof(T)] = parameters.Cast<object>().ToArray();
        Unbound.Clear();
        Interlocked.Increment(ref _parameterGeneration);
    }

    private int _parameterGeneration;

    /// <summary>Changes whenever the parameter or constant tensors a replay binds are replaced (BindParameters, SetStored).</summary>
    internal int ParameterGeneration => Volatile.Read(ref _parameterGeneration);

    /// <summary>
    /// The tensors a replay at <typeparamref name="T"/> binds for parameters and constants, indexed by value id (null for
    /// every other value, and for a value that is not a <c>Tensor&lt;T&gt;</c>). A compiled plan stores references to these
    /// indices instead of copies, so it runs on whatever the program currently binds.
    /// </summary>
    internal Tensor<T>?[] ExternalTensors<T>()
    {
        var result = new Tensor<T>?[Values.Count];
        var parameters = GetParameterObjects<T>();
        var constants = (object?[])GetConstants<T>();
        int pi = 0;
        foreach (var v in Values)
        {
            var bound = v.Role == ProgramValueRole.Parameter ? parameters[pi++]
                : v.Role == ProgramValueRole.Constant ? constants[v.Id] : null;
            if (bound is not null && !InPlaceTargets.Contains(v.Id)) result[v.Id] = bound as Tensor<T>;
        }
        return result;
    }

    /// <summary>
    /// The current parameter tensors to write when saving with values, converted to their declared element type.
    /// Live updates and rebinding must replace the trace-time snapshot in the saved package.
    /// </summary>
    internal IReadOnlyDictionary<int, object> StoredForSave()
    {
        var traced = ElementTypes.TypeOf(TracedElementType)!;
        var bound = _parameterCache.TryGetValue(traced, out var same) ? same : _parameterCache.Values.FirstOrDefault();
        var ids = ParameterIds.ToArray();
        var result = new Dictionary<int, object>(Stored);
        for (int i = 0; i < ids.Length; i++)
        {
            var value = Values[ids[i]];
            var current = bound is not null ? bound[i]
                : LiveParameters.TryGetValue(ids[i], out var live) ? live : null;
            if (current is not null)
                result[ids[i]] = TensorValues.Materialize(TensorValues.Snapshot(current), value.Container, ElementOf(value, traced));
        }
        return result;
    }

    private HashSet<int>? _inPlaceTargets;

    /// <summary>Parameter and constant value ids some node writes in place: a replay works on a private copy of each.</summary>
    internal HashSet<int> InPlaceTargets => _inPlaceTargets ??= new HashSet<int>(
        Nodes.Where(n => n.InPlaceInput >= 0).Select(n => n.InPlaceInput)
             .Where(id => Values[id].Role is ProgramValueRole.Constant or ProgramValueRole.Parameter));

    /// <summary>The parameter value ids, in definition order.</summary>
    public IEnumerable<int> ParameterIds => Values.Where(v => v.Role == ProgramValueRole.Parameter).Select(v => v.Id);

    // ── Tracing ─────────────────────────────────────────────────────────────────────────────────────────────

    /// <summary>Traces a single-input, single-output function with default options.</summary>
    public static TensorProgram Trace<T>(Func<Tensor<T>, Tensor<T>> function, Tensor<T> example) => Trace(function, example, null);

    /// <summary>Traces a multi-input, multi-output function with default options.</summary>
    public static TensorProgram Trace<T>(Func<Tensor<T>[], Tensor<T>[]> function, Tensor<T>[] examples) => Trace(function, examples, null);

    /// <summary>Traces a single-input, single-output function.</summary>
    public static TensorProgram Trace<T>(Func<Tensor<T>, Tensor<T>> function, Tensor<T> example, TraceOptions? options)
    {
        if (function is null) throw new ArgumentNullException(nameof(function));
        return Trace(xs => new[] { function(xs[0]) }, new[] { example }, options);
    }

    /// <summary>
    /// Runs <paramref name="function"/> on <paramref name="examples"/> with a tracing engine installed for the calling
    /// thread and records every outer IEngine call it makes. Throws <see cref="TraceException"/> when the function uses an
    /// op with no portable form or (with <see cref="TraceOptions.Verify"/>) computes something outside the engine.
    /// </summary>
    public static TensorProgram Trace<T>(Func<Tensor<T>[], Tensor<T>[]> function, Tensor<T>[] examples, TraceOptions? options)
    {
        if (function is null) throw new ArgumentNullException(nameof(function));
        if (examples is null || examples.Length == 0) throw new ArgumentException("At least one example input is required.", nameof(examples));
        options ??= new TraceOptions();
        if (options.Tolerance < 0 || double.IsNaN(options.Tolerance) || double.IsInfinity(options.Tolerance))
            throw new ArgumentOutOfRangeException(nameof(options), "Trace tolerance must be finite and non-negative.");

        var (program, eager) = TraceOnce(function, examples, options);
        var diagnostics = (List<string>)program.Diagnostics;

        if (options.InferDynamicDimensions)
            program = InferSymbols(program, function, examples, options, diagnostics);

        if (options.Verify)
        {
            Tensor<T>[] replayed;
            using (new NoGradScope<T>())
                replayed = program.Run(AiDotNetEngine.Current, examples);
            for (int i = 0; i < eager.Length; i++)
            {
                var why = Compare(eager[i], replayed[i], options.Tolerance);
                if (why is not null)
                    throw new TraceException($"Output {i} of the replayed program differs from the traced function ({why}). " +
                                             "The function computes part of its result outside the engine (element access, " +
                                             "LINQ over tensor data, or an engine held in a field), which a program cannot record.");
            }

            // Replaying on the traced input cannot catch a value the function READ from that input on the host (token
            // ids turned into gather indices, a threshold taken from the data): it became a constant, which is right
            // for that input only. Each further input is run through the function itself and through the program.
            // The default one is the example with its values in reverse order: every value stays one the input holds
            // (so ids stay valid ids) while any value-derived constant changes.
            var extra = new List<Tensor<T>[]>();
            if (options.VerificationInputs is not null) extra.AddRange(options.VerificationInputs.OfType<Tensor<T>[]>());
            var reversed = examples.Select(Reverse).ToArray();
            if (reversed.Where((r, i) => !SameValues(r, examples[i])).Any()) extra.Add(reversed);
            // A dynamic dimension is checked at a size other than the traced one too: a size-dependent value the probes
            // did not expose would replay the traced size's result there.
            if (program.Symbols.Count > 0) extra.Add(AtProbedSize(program, examples));
            foreach (var inputs in extra)
            {
                Tensor<T>[] expected, actual;
                using (new NoGradScope<T>())
                {
                    expected = function(inputs);
                    actual = program.Run(AiDotNetEngine.Current, inputs);
                }
                for (int i = 0; i < expected.Length; i++)
                {
                    var why = Compare(expected[i], actual[i], options.Tolerance);
                    if (why is not null)
                        throw new TraceException($"Output {i} of the replayed program differs from the function on a different input " +
                                                 $"({why}). The function reads values of its input on the host (for example token ids " +
                                                 "used as indices), so the trace captured them as constants and would replay the first " +
                                                 "input's result for every input.");
                }
            }
        }
        return program;
    }

    // The examples with every symbolic axis one larger (rows copied cyclically, so values stay ones the input holds).
    private static Tensor<T>[] AtProbedSize<T>(TensorProgram program, Tensor<T>[] examples) =>
        examples.Select((e, i) =>
        {
            var dims = program.InputDimensions[i];
            for (int axis = 0; axis < dims.Count; axis++)
                if (dims[axis].Symbol >= 0) e = Resize(e, new[] { axis }, e._shape[axis] + 1);
            return e;
        }).ToArray();

    private static Tensor<T> Reverse<T>(Tensor<T> t)
    {
        var data = t.Contiguous().AsSpan().ToArray();
        Array.Reverse(data);
        return new Tensor<T>(data, t._shape.ToArray());
    }

    private static bool SameValues<T>(Tensor<T> a, Tensor<T> b) =>
        a.Contiguous().AsSpan().ToArray().SequenceEqual(b.Contiguous().AsSpan().ToArray(), EqualityComparer<T>.Default);

    private static (TensorProgram Program, Tensor<T>[] Outputs) TraceOnce<T>(Func<Tensor<T>[], Tensor<T>[]> function, Tensor<T>[] examples, TraceOptions options)
    {
        var recorder = new ProgramRecorder(typeof(T), options.Parameters, options.Names);
        var inputs = new int[examples.Length];
        for (int i = 0; i < examples.Length; i++)
        {
            string? name = options.Names is not null && options.Names.TryGetValue(examples[i], out var n) ? n : "x" + i;
            inputs[i] = recorder.DefineInput(examples[i], name);
        }

        Tensor<T>[] result;
        var tracer = new TracingEngine(AiDotNetEngine.Current, recorder);
        using (AiDotNetEngine.PushThreadEngine(tracer))
        using (new NoGradScope<T>())
        {
            result = function(examples) ?? throw new TraceException("The function returned null.");
        }
        recorder.Stop();
        if (recorder.Failure is not null) throw new TraceException(recorder.Failure);

        var outputs = new int[result.Length];
        for (int i = 0; i < result.Length; i++)
        {
            if (result[i] is null) throw new TraceException($"Output {i} is null.");
            outputs[i] = recorder.IdOf(result[i]);
        }
        var dims = examples.Select(e => e._shape.Select(d => new SymbolicInt(-1, 0, d)).ToArray()).ToArray();
        var program = new TensorProgram(ElementTypes.KeyOf(typeof(T)) ?? typeof(T).Name, recorder.Values, recorder.Nodes,
            inputs, outputs, recorder.Snapshots, dims, Array.Empty<string>(), new List<string>())
        {
            LiveParameters = recorder.Live,
        };
        return (program, result);
    }

    private static TensorProgram InferSymbols<T>(TensorProgram program, Func<Tensor<T>[], Tensor<T>[]> function, Tensor<T>[] examples,
        TraceOptions options, List<string> diagnostics)
    {
        var requested = options.DynamicDimensions
            ?? examples.Select((e, i) => (Input: i, Axis: 0)).Where(x => examples[x.Input].Rank >= 1).ToList();
        if (requested.Count == 0) return program;

        // Dimensions that share a size in the example (a batch axis on two inputs) must move together, so each
        // distinct size is one symbol.
        var groups = requested.GroupBy(r => examples[r.Input]._shape[r.Axis]).ToList();
        var symbols = new List<string>();
        var dims = program.InputDimensions.Select(d => d.ToArray()).ToArray();
        var nodes = program.Nodes.ToList();

        foreach (var group in groups)
        {
            int baseSize = group.Key;
            var probes = new List<TensorProgram>();
            bool ok = true;
            foreach (int delta in new[] { 1, 2 })
            {
                var probeInputs = examples.Select((e, i) =>
                {
                    var axes = group.Where(g => g.Input == i).Select(g => g.Axis).ToArray();
                    return axes.Length == 0 ? e : Resize(e, axes, baseSize + delta);
                }).ToArray();
                try
                {
                    probes.Add(TraceOnce(function, probeInputs, options).Program);
                }
                catch (Exception ex)
                {
                    diagnostics.Add($"Size {baseSize} kept static: tracing at {baseSize + delta} failed ({ex.Message}).");
                    ok = false;
                    break;
                }
            }
            if (!ok) continue;

            // The node arguments can be identical while a captured value depends on the size (x * Tensor{1/x.Shape[0]}):
            // the program would then replay the traced size's constant at every size. Such a dimension stays static.
            var capturedWhy = CompareCaptured(program, probes[0]) ?? CompareCaptured(program, probes[1]);
            if (capturedWhy is not null)
            {
                diagnostics.Add($"Size {baseSize} kept static: {capturedWhy}");
                continue;
            }

            int symbol = symbols.Count;
            var updated = Generalize(nodes, probes[0].Nodes, probes[1].Nodes, baseSize, symbol, out var why);
            if (updated is null)
            {
                diagnostics.Add($"Size {baseSize} kept static: {why}");
                continue;
            }
            nodes = updated;
            symbols.Add("s" + symbol);
            foreach (var (input, axis) in group) dims[input][axis] = new SymbolicInt(symbol, 1, 0);
        }

        return new TensorProgram(program.TracedElementType, (List<ProgramValue>)program.Values, nodes, program.Inputs.ToArray(),
            program.Outputs.ToArray(), program.Stored, dims, symbols.ToArray(), diagnostics)
        {
            LiveParameters = program.LiveParameters,
        };
    }

    // Constants and parameters captured by two traces at different sizes must be the same values.
    private static string? CompareCaptured(TensorProgram a, TensorProgram b)
    {
        if (a.Values.Count != b.Values.Count) return "the number of values depends on it.";
        foreach (var v in a.Values)
        {
            if (v.Role is not (ProgramValueRole.Constant or ProgramValueRole.Parameter)) continue;
            var w = b.Values[v.Id];
            if (w.Role != v.Role || w.ElementType != v.ElementType || w.Container != v.Container || !w.Shape.SequenceEqual(v.Shape))
                return $"captured {v.Role.ToString().ToLowerInvariant()} %{v.Id} changes shape with it.";
            if (v.Role == ProgramValueRole.Constant && !TensorValues.SameData(a.Stored[v.Id], b.Stored[w.Id]))
                return $"captured constant %{v.Id} holds values computed from it.";
        }
        return null;
    }

    // The three traces must be the same graph; integer literals that differ must be linear in the size.
    private static List<ProgramNode>? Generalize(List<ProgramNode> a, IReadOnlyList<ProgramNode> b, IReadOnlyList<ProgramNode> c,
        long size, int symbol, out string? why)
    {
        why = null;
        if (a.Count != b.Count || a.Count != c.Count) { why = "the number of engine calls depends on it."; return null; }
        var result = new List<ProgramNode>(a.Count);
        for (int n = 0; n < a.Count; n++)
        {
            var (x, y, z) = (a[n], b[n], c[n]);
            if (x.Op != y.Op || x.Op != z.Op || !x.Outputs.SequenceEqual(y.Outputs) || !x.Outputs.SequenceEqual(z.Outputs))
            { why = $"call {n} ({x.Op.Name}) changes with it."; return null; }
            var args = new ProgramArg[x.Args.Count];
            for (int i = 0; i < args.Length; i++)
            {
                var (p, q, r) = (x.Args[i], y.Args[i], z.Args[i]);
                if (p.Kind != q.Kind || p.Kind != r.Kind || !p.Values.SequenceEqual(q.Values) || !p.Values.SequenceEqual(r.Values))
                { why = $"argument '{p.Name}' of call {n} ({x.Op.Name}) changes with it."; return null; }
                if (p.Kind is not (ProgramArgKind.Literal or ProgramArgKind.Symbolic) || LiteralEquals(p, q) && LiteralEquals(p, r))
                {
                    args[i] = p;
                    continue;
                }
                var fit = Fit(p, q, r, size, symbol);
                if (fit is null) { why = $"argument '{p.Name}' of call {n} ({x.Op.Name}) is not a linear function of it."; return null; }
                args[i] = fit;
            }
            result.Add(new ProgramNode(x.Op, x.TypeArguments.ToArray(), args, x.Outputs.ToArray(), x.InPlaceInput));
        }
        return result;
    }

    private static bool LiteralEquals(ProgramArg p, ProgramArg q)
    {
        if (p.Kind == ProgramArgKind.Symbolic || q.Kind == ProgramArgKind.Symbolic)
            return p.Kind == q.Kind && p.Symbolic!.SequenceEqual(q.Symbolic!);
        return ProgramNode.FormatLiteral(p.Literal) == ProgramNode.FormatLiteral(q.Literal);
    }

    private static long[]? Ints(object? literal) => literal switch
    {
        int i => new long[] { i },
        long l => new[] { l },
        int[] ia => ia.Select(v => (long)v).ToArray(),
        long[] la => la,
        _ => null,
    };

    private static ProgramArg? Fit(ProgramArg p, ProgramArg q, ProgramArg r, long size, int symbol)
    {
        if (p.Kind == ProgramArgKind.Symbolic) return null; // an earlier symbol already claimed it
        var (u, v, w) = (Ints(p.Literal), Ints(q.Literal), Ints(r.Literal));
        if (u is null || v is null || w is null || u.Length != v.Length || u.Length != w.Length) return null;
        var exprs = new SymbolicInt[u.Length];
        for (int k = 0; k < u.Length; k++)
        {
            long scale = v[k] - u[k];
            if (w[k] - v[k] != scale) return null;             // not linear across the three sizes
            exprs[k] = scale == 0 ? new SymbolicInt(-1, 0, u[k]) : new SymbolicInt(symbol, scale, u[k] - scale * size);
        }
        bool isArray = p.Literal is Array;
        return p.WithSymbolic(exprs, isArray);
    }

    private static Tensor<T> Resize<T>(Tensor<T> example, int[] axes, int newSize)
    {
        // Rows are copied cyclically along each axis, so a probe holds only values the example holds (valid indices stay valid).
        var shape = example._shape.ToArray();
        foreach (var a in axes) shape[a] = newSize;
        var src = example.Contiguous();
        var dst = new Tensor<T>(shape);
        var srcData = src.AsSpan();
        var dstData = dst.AsWritableSpan();
        var srcShape = example._shape;
        var idx = new int[shape.Length];
        for (int flat = 0; flat < dstData.Length; flat++)
        {
            int rem = flat, s = 0, stride = 1;
            for (int d = shape.Length - 1; d >= 0; d--) { idx[d] = rem % shape[d]; rem /= shape[d]; }
            for (int d = shape.Length - 1; d >= 0; d--) { s += (idx[d] % srcShape[d]) * stride; stride *= srcShape[d]; }
            dstData[flat] = srcData[s];
        }
        return dst;
    }

    [Helpers.ScalarKernel("Trace-time verification compares two results once, with a per-element tolerance and NaN handling.")]
    private static string? Compare<T>(Tensor<T> expected, Tensor<T> actual, double tol)
    {
        if (!expected._shape.SequenceEqual(actual._shape))
            return $"shape [{string.Join(", ", actual._shape)}] vs [{string.Join(", ", expected._shape)}]";
        var ops = Helpers.MathHelper.GetNumericOperations<T>();
        var e = expected.Contiguous().AsSpan();
        var a = actual.Contiguous().AsSpan();
        for (int i = 0; i < e.Length; i++)
        {
            double x = ops.ToDouble(e[i]), y = ops.ToDouble(a[i]);
            if (x == y || (double.IsNaN(x) && double.IsNaN(y))) continue;
            if (double.IsNaN(x) || double.IsNaN(y) || double.IsInfinity(x) || double.IsInfinity(y)
                || !(Math.Abs(x - y) <= tol + tol * Math.Abs(x))) return $"element {i}: {y} vs {x}";
        }
        return null;
    }

    // ── Replay ──────────────────────────────────────────────────────────────────────────────────────────────

    /// <summary>Replays a single-input, single-output program on the current engine.</summary>
    public Tensor<T> Run<T>(Tensor<T> input) => Run(AiDotNetEngine.Current, new[] { input })[0];

    /// <summary>Replays the program on the current engine.</summary>
    public Tensor<T>[] Run<T>(params Tensor<T>[] inputs) => Run(AiDotNetEngine.Current, inputs);

    /// <summary>
    /// Replays the program eagerly on <paramref name="engine"/> at element type <typeparamref name="T"/> (which need not
    /// be the traced type). Every op goes through the engine, so an active gradient tape records the replay and
    /// <see cref="GetParameters{T}"/> receive gradients.
    /// </summary>
    public Tensor<T>[] Run<T>(IEngine engine, Tensor<T>[] inputs)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (inputs is null) throw new ArgumentNullException(nameof(inputs));
        if (inputs.Length != Inputs.Count) throw new ArgumentException($"The program takes {Inputs.Count} input(s); got {inputs.Length}.", nameof(inputs));

        if (Unbound.Count > 0 && !_parameterCache.ContainsKey(typeof(T)))
            throw new InvalidOperationException("The program was saved without parameter values; call BindParameters before running it.");
        var symbols = BindSymbols(inputs);
        var env = new object?[Values.Count];
        for (int i = 0; i < inputs.Length; i++) env[Inputs[i]] = inputs[i];
        var parameters = GetParameterObjects<T>();
        var constants = GetConstants<T>();
        int pi = 0;
        var inPlace = InPlaceTargets;
        foreach (var v in Values)
        {
            if (v.Role == ProgramValueRole.Parameter) env[v.Id] = parameters[pi++];
            else if (v.Role == ProgramValueRole.Constant) env[v.Id] = ((object?[])constants)[v.Id];
            else continue;
            // A node writes this captured value in place: the shared (cached) tensor must not change, or the next
            // replay, and any replay running concurrently, would read the mutated value. Each replay writes a copy.
            if (inPlace.Count > 0 && inPlace.Contains(v.Id) && env[v.Id] is { } shared)
                env[v.Id] = TensorValues.Copy(shared);
        }

        foreach (var node in Nodes) Execute(engine, node, env, symbols, typeof(T));

        var outputs = new Tensor<T>[Outputs.Count];
        for (int i = 0; i < outputs.Length; i++)
            outputs[i] = env[Outputs[i]] as Tensor<T>
                ?? throw new InvalidOperationException($"Output {i} is not a Tensor<{typeof(T).Name}>.");
        return outputs;
    }

    /// <summary>The program as a function of one input.</summary>
    public Func<Tensor<T>, Tensor<T>> ToFunc<T>() => x => Run(x);

    /// <summary>
    /// The trainable parameters at element type <typeparamref name="T"/>, in <see cref="ParameterIds"/> order. They are
    /// created once per element type from the stored values and reused by every replay, so an optimizer can update them.
    /// </summary>
    public IReadOnlyList<Tensor<T>> GetParameters<T>() => GetParameterObjects<T>().Cast<Tensor<T>>().ToArray();

    private object[] GetParameterObjects<T>() => _parameterCache.GetOrAdd(typeof(T), _ =>
        Values.Where(v => v.Role == ProgramValueRole.Parameter)
              .Select(v => LiveParameters.TryGetValue(v.Id, out var live)
                  ? ElementOf(v, typeof(T)) == TensorValues.Describe(live).Element ? live
                    : TensorValues.Materialize(TensorValues.Snapshot(live), v.Container, ElementOf(v, typeof(T)))
                  : TensorValues.Materialize(Stored[v.Id], v.Container, ElementOf(v, typeof(T))))
              .ToArray());

    private object GetConstants<T>() => _constantCache.GetOrAdd(typeof(T), _ =>
    {
        var arr = new object?[Values.Count];
        foreach (var v in Values)
            if (v.Role == ProgramValueRole.Constant)
                arr[v.Id] = TensorValues.Materialize(Stored[v.Id], v.Container, ElementOf(v, typeof(T)));
        return arr;
    });

    /// <summary>Replaces a stored parameter or constant value (used by loaders and by facades syncing trained weights).</summary>
    internal void SetStored(int id, object tensor)
    {
        Stored[id] = TensorValues.Snapshot(tensor);
        LiveParameters.Remove(id);
        Unbound.Remove(id);
        _parameterCache.Clear();
        _constantCache.Clear();
        Interlocked.Increment(ref _parameterGeneration);
    }

    private static Type ElementOf(ProgramValue v, Type replay) =>
        v.ElementType == "$T" ? replay : ElementTypes.TypeOf(v.ElementType) ?? throw new InvalidOperationException($"Unknown element type {v.ElementType}.");

    private long[] BindSymbols<T>(Tensor<T>[] inputs)
    {
        var symbols = new long[Symbols.Count];
        var bound = new bool[Symbols.Count];
        for (int i = 0; i < inputs.Length; i++)
        {
            var dims = InputDimensions[i];
            var shape = inputs[i]._shape;
            if (shape.Length != dims.Count)
                throw new ArgumentException($"Input {i} must have rank {dims.Count}; got [{string.Join(", ", shape)}].");
            for (int d = 0; d < dims.Count; d++)
            {
                var e = dims[d];
                if (e.Symbol < 0)
                {
                    if (shape[d] != e.Offset)
                        throw new ArgumentException($"Input {i} axis {d} must be {e.Offset} (static in this program); got {shape[d]}.");
                }
                else if (!bound[e.Symbol]) { symbols[e.Symbol] = shape[d]; bound[e.Symbol] = true; }
                else if (symbols[e.Symbol] != shape[d])
                    throw new ArgumentException($"Input {i} axis {d} is {Symbols[e.Symbol]}, already bound to {symbols[e.Symbol]}; got {shape[d]}.");
            }
        }
        return symbols;
    }

    private static readonly ConcurrentDictionary<Type, Func<IEngine, int, object?[], (bool, object?)>> Invokers = new();

    private static Func<IEngine, int, object?[], (bool, object?)> InvokerFor(Type element) => Invokers.GetOrAdd(element, t =>
    {
        var m = typeof(EngineOpInvoker<>).MakeGenericType(t).GetMethod("TryInvoke", BindingFlags.NonPublic | BindingFlags.Static)!;
        return (e, op, a) =>
        {
            var args = new object?[] { e, op, a, null };
            bool ok = (bool)m.Invoke(null, args)!;
            return (ok, args[3]);
        };
    });

    private static readonly ConcurrentDictionary<Type, Func<IEngine, int, Type, object?[], (bool, object?)>> Invokers2 = new();

    private static Func<IEngine, int, Type, object?[], (bool, object?)> InvokerFor2(Type element) => Invokers2.GetOrAdd(element, t =>
    {
        var m = typeof(EngineOpInvoker<>).MakeGenericType(t).GetMethod("TryInvoke2", BindingFlags.NonPublic | BindingFlags.Static)!;
        return (e, op, second, a) =>
        {
            var args = new object?[] { e, op, second, a, null };
            bool ok = (bool)m.Invoke(null, args)!;
            return (ok, args[4]);
        };
    });

    private static void Execute(IEngine engine, ProgramNode node, object?[] env, long[] symbols, Type replay)
    {
        var d = node.Op;
        var args = new object?[node.Args.Count];
        for (int i = 0; i < args.Length; i++)
        {
            var a = node.Args[i];
            args[i] = a.Kind switch
            {
                ProgramArgKind.Value => env[a.Values[0]],
                ProgramArgKind.ValueList => a.Values.Select(v => env[v]).ToArray(),
                ProgramArgKind.Literal => a.Literal,
                ProgramArgKind.Symbolic => a.SymbolicIsArray
                    ? a.Symbolic!.Select(s => checked((int)s.Evaluate(symbols))).ToArray()
                    : checked((int)a.Symbolic![0].Evaluate(symbols)),
                _ => null,
            };
        }

        Type element = node.TypeArguments.Count == 0 ? replay
            : node.TypeArguments[0] == "$T" ? replay
            : ElementTypes.TypeOf(node.TypeArguments[0]) ?? throw new InvalidOperationException($"Unknown element type {node.TypeArguments[0]}.");
        bool ok;
        object? result;
        if (node.TypeArguments.Count == 2)
        {
            // An op with a second type parameter (an index type, as in TensorEmbeddingLookup<TValue, TIndex>): the
            // generated invoker names the call for each allowlisted element type, so no type comes from the file.
            var second = node.TypeArguments[1] == "$T" ? replay
                : ElementTypes.TypeOf(node.TypeArguments[1]) ?? throw new InvalidOperationException($"Unknown element type {node.TypeArguments[1]}.");
            (ok, result) = InvokerFor2(element)(engine, d.Index, second, args);
        }
        else if (element == replay && replay == typeof(float)) ok = EngineOpInvoker<float>.TryInvoke(engine, d.Index, args, out result);
        else if (element == replay && replay == typeof(double)) ok = EngineOpInvoker<double>.TryInvoke(engine, d.Index, args, out result);
        else (ok, result) = InvokerFor(element)(engine, d.Index, args);
        if (!ok) throw new NotSupportedException($"{d.Id} has no typed replay (it takes more than one type parameter).");

        int o = 0;
        switch (d.ReturnKind)
        {
            case EngineValueKind.Void: break;
            case EngineValueKind.TensorArray or EngineValueKind.TensorList:
                foreach (var item in (IEnumerable)result!) env[node.Outputs[o++]] = item;
                break;
            case EngineValueKind.TensorTuple:
                foreach (var item in ProgramRecorder.TupleItems(result!)) env[node.Outputs[o++]] = item;
                break;
            default:
                env[node.Outputs[o++]] = result;
                break;
        }
        for (int i = 0; i < args.Length; i++)
            if (d.Parameters[i].Kind == EngineValueKind.OutTensor) env[node.Outputs[o++]] = args[i];
        if (node.InPlaceInput >= 0) env[node.Outputs[o]] = args[d.InPlaceParameter];
    }

    /// <summary>A readable listing of the program.</summary>
    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.Append("program<").Append(TracedElementType).Append(">(");
        sb.Append(string.Join(", ", Inputs.Select((id, i) =>
            $"%{id}: [{string.Join(", ", InputDimensions[i])}]" + (Values[id].Name is { } n ? " " + n : ""))));
        sb.AppendLine(")");
        foreach (var v in Values.Where(v => v.Role is ProgramValueRole.Constant or ProgramValueRole.Parameter))
            sb.AppendLine($"  %{v.Id} = {v.Role.ToString().ToLowerInvariant()} {v.Container}<{v.ElementType}>[{string.Join(", ", v.Shape)}]" + (v.Name is { } n ? " " + n : ""));
        foreach (var node in Nodes) sb.Append("  ").AppendLine(node.ToString());
        sb.Append("  return ").AppendLine(string.Join(", ", Outputs.Select(o => "%" + o)));
        return sb.ToString();
    }
}
