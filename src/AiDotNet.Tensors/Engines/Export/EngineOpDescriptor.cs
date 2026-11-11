using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Export;

/// <summary>How a value crossing the IEngine boundary is represented in a portable program.</summary>
public enum EngineValueKind
{
    /// <summary>No value (a void op).</summary>
    Void,
    /// <summary>A tensor of the op's element type parameter.</summary>
    Tensor,
    /// <summary>A tensor of a fixed element type (e.g. <c>Tensor&lt;int&gt;</c> indices).</summary>
    TensorFixed,
    /// <summary>An array of tensors.</summary>
    TensorArray,
    /// <summary>A list or sequence of tensors.</summary>
    TensorList,
    /// <summary>A value tuple of tensors.</summary>
    TensorTuple,
    /// <summary>A vector of the op's element type.</summary>
    Vector,
    /// <summary>A matrix of the op's element type.</summary>
    Matrix,
    /// <summary>A scalar of the op's element type; stored as a double so it replays at any element type.</summary>
    Scalar,
    /// <summary>bool, int, long, float, double or string.</summary>
    Primitive,
    /// <summary>A one-dimensional array of primitives or enum values.</summary>
    PrimitiveArray,
    /// <summary>An enum value, stored by type name and member name.</summary>
    Enum,
    /// <summary>An array of scalars of the op's element type; stored as doubles.</summary>
    ScalarArray,
    /// <summary>A <see cref="FusedActivationParams"/>, stored as its alpha, beta and theta.</summary>
    ActivationParams,
    /// <summary>An <c>out</c> tensor parameter (an additional output).</summary>
    OutTensor,
    /// <summary>Nullable forms of the above.</summary>
    NullableTensor, NullableScalar, NullablePrimitive, NullableEnum,
    /// <summary>No portable form (delegates, spans, engine-internal objects): an op using it cannot be traced.</summary>
    Unsupported,
}

/// <summary>A named, typed parameter of an engine op.</summary>
public sealed class EngineOpParameter
{
    internal EngineOpParameter(string name, EngineValueKind kind, string typeKey, bool hasDefault)
    {
        Name = name;
        Kind = kind;
        TypeKey = typeKey;
        HasDefault = hasDefault;
    }

    /// <summary>The parameter name as declared on IEngine.</summary>
    public string Name { get; }

    /// <summary>The portable representation.</summary>
    public EngineValueKind Kind { get; }

    /// <summary>The declared type, with method type parameters written positionally (<c>!!0</c>).</summary>
    public string TypeKey { get; }

    /// <summary>Whether IEngine declares a default value.</summary>
    public bool HasDefault { get; }

    /// <inheritdoc/>
    public override string ToString() => $"{TypeKey} {Name}";
}

/// <summary>
/// One public IEngine method, as the portable program format sees it. Generated from IEngine, so the catalog always
/// matches the engine surface; it is also the replay allowlist.
/// </summary>
public sealed class EngineOpDescriptor
{
    internal EngineOpDescriptor(int index, string id, string name, int genericArity, EngineValueKind returnKind,
        int inPlaceParameter, string? untraceableReason, EngineOpParameter[] parameters)
    {
        Index = index;
        Id = id;
        Name = name;
        GenericArity = genericArity;
        ReturnKind = returnKind;
        InPlaceParameter = inPlaceParameter;
        UntraceableReason = untraceableReason;
        Parameters = parameters;
    }

    internal int Index { get; }

    /// <summary>The stable id persisted in programs: name, generic arity and parameter types.</summary>
    public string Id { get; }

    /// <summary>The IEngine method name.</summary>
    public string Name { get; }

    /// <summary>The number of method type parameters.</summary>
    public int GenericArity { get; }

    /// <summary>What the op returns.</summary>
    public EngineValueKind ReturnKind { get; }

    /// <summary>For a void op, the index of the tensor parameter it writes; otherwise -1.</summary>
    public int InPlaceParameter { get; }

    /// <summary>Why the op cannot be traced, or null when it can.</summary>
    public string? UntraceableReason { get; }

    /// <summary>Whether a program may record and replay this op.</summary>
    public bool IsTraceable => UntraceableReason is null;

    /// <summary>The op's parameters, in declaration order.</summary>
    public IReadOnlyList<EngineOpParameter> Parameters { get; }

    /// <inheritdoc/>
    public override string ToString() => Id;
}

internal static partial class EngineOpCatalog
{
    private static Dictionary<string, EngineOpDescriptor>? _byId;

    internal static EngineOpDescriptor? Find(string id)
    {
        var map = _byId;
        if (map is null)
        {
            map = new Dictionary<string, EngineOpDescriptor>(Ops.Length, StringComparer.Ordinal);
            foreach (var op in Ops) map[op.Id] = op;
            _byId = map;
        }
        return map.TryGetValue(id, out var d) ? d : null;
    }
}

/// <summary>Receives the outer IEngine calls a <see cref="TracingEngine"/> sees.</summary>
internal interface IEngineCallRecorder
{
    /// <summary>Enters a recorded call; false when this call is nested inside one already being recorded or recording is off.</summary>
    bool TryEnter();

    /// <summary>Leaves a call entered with <see cref="TryEnter"/>.</summary>
    void Exit();

    /// <summary>Called before a void (in-place) op runs, with its arguments.</summary>
    void BeforeInPlace(int op, object?[] args);

    /// <summary>Records a completed outer call.</summary>
    void Record(int op, Type[] typeArguments, object?[] args, object? result);

    /// <summary>An op with no portable form was called.</summary>
    void Opaque(int op);
}
