using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Export;

/// <summary>Builds a program from the outer engine calls a <see cref="TracingEngine"/> reports.</summary>
internal sealed class ProgramRecorder : IEngineCallRecorder
{
    private readonly Type _element;
    private readonly HashSet<object> _parameters;
    private readonly IReadOnlyDictionary<object, string>? _names;
    private readonly Dictionary<object, int> _ids = new(ReferenceEqualityComparer<object>.Instance);
    private int _depth;
    private bool _enabled = true;

    internal readonly List<ProgramValue> Values = new();
    internal readonly List<ProgramNode> Nodes = new();
    /// <summary>For constants and parameters: an immutable snapshot of the captured tensor (by value id).</summary>
    internal readonly Dictionary<int, object> Snapshots = new();
    /// <summary>For parameters: the live tensor the caller passed, so a facade can map parameters back.</summary>
    internal readonly Dictionary<int, object> Live = new();
    internal string? Failure;

    internal ProgramRecorder(Type element, IEnumerable<object>? parameters, IReadOnlyDictionary<object, string>? names)
    {
        _element = element;
        _parameters = new HashSet<object>(parameters ?? Array.Empty<object>(), ReferenceEqualityComparer<object>.Instance);
        _names = names;
    }

    internal int DefineInput(object tensor, string? name)
    {
        var id = Define(tensor, ProgramValueRole.Input);
        Values[id].Name = name;
        return id;
    }

    internal int IdOf(object tensor) => Resolve(tensor);

    internal void Stop() => _enabled = false;

    public bool TryEnter()
    {
        if (!_enabled || _depth > 0) return false;
        _depth++;
        return true;
    }

    public void Exit() => _depth--;

    public void Opaque(int op)
    {
        if (_enabled && _depth == 0)
        {
            var d = EngineOpCatalog.Ops[op];
            Failure ??= $"{d.Name} cannot be traced: {d.UntraceableReason}.";
        }
    }

    public void BeforeInPlace(int op, object?[] args)
    {
        // A constant first seen as the target of an in-place op is snapshotted before the op changes it.
        var d = EngineOpCatalog.Ops[op];
        if (d.InPlaceParameter >= 0 && args[d.InPlaceParameter] is { } target) Resolve(target);
    }

    public void Record(int op, Type[] typeArguments, object?[] args, object? result)
    {
        if (Failure is not null) return;
        var d = EngineOpCatalog.Ops[op];
        try
        {
            var typeArgs = typeArguments.Select(TypeKey).ToArray();
            var pargs = new ProgramArg[d.Parameters.Count];
            for (int i = 0; i < pargs.Length; i++) pargs[i] = Arg(d.Parameters[i], args[i]);

            var outputs = new List<int>();
            switch (d.ReturnKind)
            {
                case EngineValueKind.Void:
                    break;
                case EngineValueKind.Tensor or EngineValueKind.TensorFixed or EngineValueKind.Vector or EngineValueKind.Matrix:
                    if (result is null) throw new TraceException($"{d.Name} returned null.");
                    outputs.Add(Define(result, ProgramValueRole.Intermediate));
                    break;
                case EngineValueKind.TensorArray or EngineValueKind.TensorList:
                    foreach (var item in (IEnumerable)result!) outputs.Add(Define(item!, ProgramValueRole.Intermediate));
                    break;
                case EngineValueKind.TensorTuple:
                    foreach (var item in TupleItems(result!)) outputs.Add(Define(item!, ProgramValueRole.Intermediate));
                    break;
                default:
                    throw new TraceException($"{d.Name} returns {d.ReturnKind}, which a program cannot hold.");
            }
            for (int i = 0; i < pargs.Length; i++)
                if (d.Parameters[i].Kind == EngineValueKind.OutTensor && args[i] is { } outTensor)
                    outputs.Add(Define(outTensor, ProgramValueRole.Intermediate));

            int inPlaceInput = -1;
            if (d.InPlaceParameter >= 0 && args[d.InPlaceParameter] is { } target)
            {
                inPlaceInput = pargs[d.InPlaceParameter].Values[0];
                outputs.Add(Define(target, ProgramValueRole.Intermediate));
            }
            Nodes.Add(new ProgramNode(d, typeArgs, pargs, outputs.ToArray(), inPlaceInput));
        }
        catch (TraceException ex)
        {
            Failure ??= ex.Message;
        }
    }

    internal static IEnumerable<object?> TupleItems(object tuple)
    {
        var type = tuple.GetType();
        for (int i = 1; ; i++)
        {
            var field = type.GetField("Item" + i);
            if (field is null) break;
            yield return field.GetValue(tuple);
        }
    }

    private ProgramArg Arg(EngineOpParameter p, object? value)
    {
        if (p.Kind == EngineValueKind.OutTensor) return new ProgramArg(p.Name, ProgramArgKind.Out);
        if (value is null) return new ProgramArg(p.Name, ProgramArgKind.Null);
        switch (p.Kind)
        {
            case EngineValueKind.Tensor or EngineValueKind.TensorFixed or EngineValueKind.Vector or EngineValueKind.Matrix
                or EngineValueKind.NullableTensor:
                return new ProgramArg(p.Name, ProgramArgKind.Value, new[] { Resolve(value) });
            case EngineValueKind.TensorArray or EngineValueKind.TensorList:
                return new ProgramArg(p.Name, ProgramArgKind.ValueList, ((IEnumerable)value).Cast<object>().Select(Resolve).ToArray());
            case EngineValueKind.Scalar or EngineValueKind.NullableScalar:
                return new ProgramArg(p.Name, ProgramArgKind.Literal, literal: ToDouble(value, p.Name));
            case EngineValueKind.Enum or EngineValueKind.NullableEnum:
                return new ProgramArg(p.Name, ProgramArgKind.Literal, literal: new EnumLiteral(value.ToString()!));
            case EngineValueKind.Primitive or EngineValueKind.NullablePrimitive:
                return new ProgramArg(p.Name, ProgramArgKind.Literal, literal: value);
            case EngineValueKind.PrimitiveArray:
                var arr = (Array)value;
                var copy = arr.GetType().GetElementType()!.IsEnum
                    ? arr.Cast<object>().Select(e => (object)new EnumLiteral(e.ToString()!)).ToArray()
                    : (Array)arr.Clone();
                return new ProgramArg(p.Name, ProgramArgKind.Literal, literal: copy);
            default:
                throw new TraceException($"Parameter '{p.Name}' ({p.TypeKey}) has no portable form.");
        }
    }

    private static double ToDouble(object value, string name) => value switch
    {
        double d => d,
        float f => f,
        int i => i,
        long l => l,
        Half h => (double)h,
        decimal m => (double)m,
        IConvertible c => c.ToDouble(System.Globalization.CultureInfo.InvariantCulture),
        _ => throw new TraceException($"Scalar argument '{name}' of type {value.GetType().Name} has no portable form."),
    };

    private string TypeKey(Type t) => t == _element ? "$T" : ElementTypes.KeyOf(t)
        ?? throw new TraceException($"Element type {t.Name} has no portable form.");

    private int Resolve(object tensor)
    {
        if (_ids.TryGetValue(tensor, out var id)) return id;
        // Not an input and not produced by a recorded op: a tensor the function captured.
        var role = _parameters.Contains(tensor) ? ProgramValueRole.Parameter : ProgramValueRole.Constant;
        id = Define(tensor, role);
        Snapshots[id] = TensorValues.Snapshot(tensor);
        if (role == ProgramValueRole.Parameter) Live[id] = tensor;
        if (_names is not null && _names.TryGetValue(tensor, out var name)) Values[id].Name = name;
        return id;
    }

    private int Define(object tensor, ProgramValueRole role)
    {
        var (container, element, shape) = TensorValues.Describe(tensor);
        var key = element == _element ? "$T" : ElementTypes.KeyOf(element)
            ?? throw new TraceException($"Element type {element.Name} has no portable form.");
        int id = Values.Count;
        Values.Add(new ProgramValue(id, role, key, shape, container));
        _ids[tensor] = id;
        return id;
    }
}

/// <summary>A function could not be traced into a program.</summary>
public sealed class TraceException : Exception
{
    /// <summary>Creates the exception.</summary>
    public TraceException(string message) : base(message) { }
}

/// <summary>The element types a program may name, and their stable keys. Nothing else can be instantiated on load.</summary>
internal static class ElementTypes
{
    private static readonly (string Key, Type Type)[] Known =
    {
        ("float", typeof(float)), ("double", typeof(double)), ("half", typeof(Half)), ("int", typeof(int)),
        ("long", typeof(long)), ("short", typeof(short)), ("byte", typeof(byte)), ("sbyte", typeof(sbyte)),
        ("bool", typeof(bool)), ("uint", typeof(uint)), ("ulong", typeof(ulong)), ("ushort", typeof(ushort)),
        ("decimal", typeof(decimal)),
    };

    internal static string? KeyOf(Type t)
    {
        foreach (var (k, type) in Known) if (type == t) return k;
        return null;
    }

    internal static Type? TypeOf(string key)
    {
        foreach (var (k, type) in Known) if (k == key) return type;
        return null;
    }
}

/// <summary>Type-erased helpers over Tensor/Vector/Matrix of any element type.</summary>
internal static class TensorValues
{
    internal static (string Container, Type Element, int[] Shape) Describe(object value)
    {
        var type = value.GetType();
        while (type is not null && !(type.IsGenericType && IsContainer(type.GetGenericTypeDefinition())))
            type = type.BaseType;
        if (type is null) throw new TraceException($"A value of type {value.GetType().Name} is not a tensor, vector or matrix.");
        var def = type.GetGenericTypeDefinition();
        var element = type.GetGenericArguments()[0];
        var shape = (int[])Generic(nameof(ShapeOf), element).Invoke(null, new[] { value })!;
        return (def == typeof(Tensor<>) ? "Tensor" : def == typeof(Vector<>) ? "Vector" : "Matrix", element, shape);
    }

    private static bool IsContainer(Type def) => def == typeof(Tensor<>) || def == typeof(Vector<>) || def == typeof(Matrix<>);

    /// <summary>A detached copy of a captured value (Tensor, Vector or Matrix), as a Tensor of the same element type.</summary>
    internal static object Snapshot(object value)
    {
        var (_, element, _) = Describe(value);
        return Generic(nameof(SnapshotOf), element).Invoke(null, new[] { value })!;
    }

    /// <summary>Rebuilds a stored tensor as the container the program recorded, converting element type to <typeparamref name="TOut"/> when asked.</summary>
    internal static object Materialize(object stored, string container, Type element)
    {
        var src = stored.GetType().GetGenericArguments()[0];
        return Generic(nameof(MaterializeOf), src, element).Invoke(null, new object[] { stored, container })!;
    }

    private static readonly Dictionary<string, MethodInfo> Cache = new();

    private static MethodInfo Generic(string name, params Type[] args)
    {
        var key = name + "|" + string.Join(",", args.Select(a => a.FullName));
        lock (Cache)
        {
            if (!Cache.TryGetValue(key, out var m))
            {
                m = typeof(TensorValues).GetMethod(name, BindingFlags.NonPublic | BindingFlags.Static)!.MakeGenericMethod(args);
                Cache[key] = m;
            }
            return m;
        }
    }

    private static int[] ShapeOf<T>(object value) => value switch
    {
        Tensor<T> t => t._shape.ToArray(),
        Vector<T> v => new[] { v.Length },
        Matrix<T> m => new[] { m.Rows, m.Columns },
        _ => throw new InvalidOperationException(),
    };

    private static Tensor<T> SnapshotOf<T>(object value) => value switch
    {
        Tensor<T> t => new Tensor<T>(t.Contiguous().AsSpan().ToArray(), t._shape.ToArray()),
        Vector<T> v => new Tensor<T>(v.ToArray(), new[] { v.Length }),
        Matrix<T> m => new Tensor<T>(MatrixData(m), new[] { m.Rows, m.Columns }),
        _ => throw new InvalidOperationException(),
    };

    private static T[] MatrixData<T>(Matrix<T> m) => m.AsSpan().ToArray();

    [Helpers.ScalarKernel("One-time element-type conversion of stored program weights when a program is replayed at a new type; there is no engine cast op.")]
    private static object MaterializeOf<TIn, TOut>(Tensor<TIn> stored, string container)
    {
        var shape = stored._shape.ToArray();
        TOut[] data;
        if (typeof(TIn) == typeof(TOut)) data = (TOut[])(object)stored.AsSpan().ToArray();
        else
        {
            var from = Helpers.MathHelper.GetNumericOperations<TIn>();
            var to = Helpers.MathHelper.GetNumericOperations<TOut>();
            var span = stored.AsSpan();
            data = new TOut[span.Length];
            for (int i = 0; i < data.Length; i++) data[i] = to.FromDouble(from.ToDouble(span[i]));
        }
        return container switch
        {
            "Vector" => new Vector<TOut>(data),
            "Matrix" => MatrixFrom(data, shape[0], shape[1]),
            _ => new Tensor<TOut>(data, shape),
        };
    }

    private static Matrix<T> MatrixFrom<T>(T[] data, int rows, int cols)
    {
        var m = new Matrix<T>(rows, cols);
        data.AsSpan().CopyTo(m.AsWritableSpan());
        return m;
    }
}
