using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace AiDotNet.Tensors.Engines.Export;

/// <summary>What a value in a <see cref="TensorProgram"/> is.</summary>
public enum ProgramValueRole
{
    /// <summary>A caller-supplied input.</summary>
    Input,
    /// <summary>A tensor captured by the traced function that is not trained (stored in the program).</summary>
    Constant,
    /// <summary>A captured tensor that is trainable: replay exposes it as a parameter leaf.</summary>
    Parameter,
    /// <summary>Produced by a node.</summary>
    Intermediate,
}

/// <summary>A tensor-valued SSA value of a program.</summary>
public sealed class ProgramValue
{
    internal ProgramValue(int id, ProgramValueRole role, string elementType, int[] shape, string container)
    {
        Id = id;
        Role = role;
        ElementType = elementType;
        Shape = shape;
        Container = container;
    }

    /// <summary>The value id; ids are dense and in definition order.</summary>
    public int Id { get; }

    /// <summary>Input, constant, parameter or intermediate.</summary>
    public ProgramValueRole Role { get; }

    /// <summary>
    /// The element type: <c>$T</c> for the program's element type (so it follows the replay type), or a fixed type key
    /// such as <c>int</c>.
    /// </summary>
    public string ElementType { get; }

    /// <summary>The shape seen when tracing. Symbolic dimensions are recorded on <see cref="TensorProgram.InputDimensions"/>.</summary>
    public int[] Shape { get; }

    /// <summary><c>Tensor</c>, <c>Vector</c> or <c>Matrix</c>.</summary>
    public string Container { get; }

    /// <summary>An optional name (parameters and inputs).</summary>
    public string? Name { get; internal set; }

    /// <inheritdoc/>
    public override string ToString() => $"%{Id}";
}

/// <summary>How a node argument is given.</summary>
public enum ProgramArgKind
{
    /// <summary>One program value.</summary>
    Value,
    /// <summary>An ordered list of program values (a tensor array or list).</summary>
    ValueList,
    /// <summary>A literal (primitive, primitive array, enum member name, or scalar stored as double).</summary>
    Literal,
    /// <summary>A literal int or int array that depends on symbolic input dimensions.</summary>
    Symbolic,
    /// <summary>null.</summary>
    Null,
    /// <summary>An <c>out</c> parameter; its value is one of the node's outputs.</summary>
    Out,
}

/// <summary>A linear function of one symbolic input dimension: <c>Scale * s + Offset</c>, or <c>-1</c> sentinel kept as a literal.</summary>
public readonly struct SymbolicInt : IEquatable<SymbolicInt>
{
    /// <summary>Creates a symbolic int.</summary>
    public SymbolicInt(int symbol, long scale, long offset)
    {
        Symbol = symbol;
        Scale = scale;
        Offset = offset;
    }

    /// <summary>Index into <see cref="TensorProgram.Symbols"/>, or -1 for a constant.</summary>
    public int Symbol { get; }

    /// <summary>Multiplier of the symbol.</summary>
    public long Scale { get; }

    /// <summary>Constant term.</summary>
    public long Offset { get; }

    /// <summary>Evaluates against the bound symbol values.</summary>
    public long Evaluate(IReadOnlyList<long> symbols) => Symbol < 0 ? Offset : checked(Scale * symbols[Symbol] + Offset);

    /// <inheritdoc/>
    public bool Equals(SymbolicInt other) => Symbol == other.Symbol && Scale == other.Scale && Offset == other.Offset;

    /// <inheritdoc/>
    public override bool Equals(object? obj) => obj is SymbolicInt s && Equals(s);

    /// <inheritdoc/>
    public override int GetHashCode() => (Symbol * 397) ^ Scale.GetHashCode() ^ (Offset.GetHashCode() * 31);

    /// <inheritdoc/>
    public override string ToString() =>
        Symbol < 0 ? Offset.ToString(CultureInfo.InvariantCulture)
        : (Scale == 1 ? "" : Scale.ToString(CultureInfo.InvariantCulture) + "*") + "s" + Symbol
          + (Offset == 0 ? "" : (Offset > 0 ? "+" : "") + Offset.ToString(CultureInfo.InvariantCulture));
}

/// <summary>One argument of a node, bound to an IEngine parameter by name.</summary>
public sealed class ProgramArg
{
    internal ProgramArg(string name, ProgramArgKind kind, int[]? values = null, object? literal = null, SymbolicInt[]? symbolic = null, bool symbolicIsArray = false)
    {
        Name = name;
        Kind = kind;
        Values = values ?? Array.Empty<int>();
        Literal = literal;
        Symbolic = symbolic;
        SymbolicIsArray = symbolicIsArray;
    }

    /// <summary>The IEngine parameter name.</summary>
    public string Name { get; }

    /// <summary>How the argument is given.</summary>
    public ProgramArgKind Kind { get; }

    /// <summary>The value ids for <see cref="ProgramArgKind.Value"/> (one) and <see cref="ProgramArgKind.ValueList"/>.</summary>
    public IReadOnlyList<int> Values { get; }

    /// <summary>The literal for <see cref="ProgramArgKind.Literal"/>.</summary>
    public object? Literal { get; }

    /// <summary>The symbolic expression(s) for <see cref="ProgramArgKind.Symbolic"/>.</summary>
    public IReadOnlyList<SymbolicInt>? Symbolic { get; }

    /// <summary>Whether a symbolic argument is an int array (true) or a single int (false).</summary>
    public bool SymbolicIsArray { get; }

    internal ProgramArg WithSymbolic(SymbolicInt[] symbolic, bool isArray) => new(Name, ProgramArgKind.Symbolic, null, null, symbolic, isArray);
}

/// <summary>One recorded call to a public IEngine method.</summary>
public sealed class ProgramNode
{
    internal ProgramNode(EngineOpDescriptor op, string[] typeArguments, ProgramArg[] args, int[] outputs, int inPlaceInput)
    {
        Op = op;
        TypeArguments = typeArguments;
        Args = args;
        Outputs = outputs;
        InPlaceInput = inPlaceInput;
    }

    /// <summary>The engine op.</summary>
    public EngineOpDescriptor Op { get; }

    /// <summary>The method type arguments: <c>$T</c> or a fixed type key.</summary>
    public IReadOnlyList<string> TypeArguments { get; }

    /// <summary>Arguments, one per IEngine parameter, in declaration order.</summary>
    public IReadOnlyList<ProgramArg> Args { get; }

    /// <summary>The values this node defines (its return value(s), then any out parameters, then the in-place result).</summary>
    public IReadOnlyList<int> Outputs { get; }

    /// <summary>For an in-place op, the value it overwrote (the new version is the last output); otherwise -1.</summary>
    public int InPlaceInput { get; }

    /// <inheritdoc/>
    public override string ToString()
    {
        var outs = Outputs.Count == 0 ? "" : string.Join(", ", Outputs.Select(o => "%" + o)) + " = ";
        var targs = TypeArguments.Count == 0 ? "" : "<" + string.Join(", ", TypeArguments) + ">";
        return outs + Op.Name + targs + "(" + string.Join(", ", Args.Select(Format)) + ")";
    }

    private static string Format(ProgramArg a) => a.Name + ": " + a.Kind switch
    {
        ProgramArgKind.Value => "%" + a.Values[0],
        ProgramArgKind.ValueList => "[" + string.Join(", ", a.Values.Select(v => "%" + v)) + "]",
        ProgramArgKind.Null => "null",
        ProgramArgKind.Out => "out",
        ProgramArgKind.Symbolic => a.SymbolicIsArray ? "[" + string.Join(", ", a.Symbolic!) + "]" : a.Symbolic![0].ToString(),
        _ => FormatLiteral(a.Literal),
    };

    internal static string FormatLiteral(object? v) => v switch
    {
        null => "null",
        string s => "\"" + s + "\"",
        bool b => b ? "true" : "false",
        double d => d.ToString("R", CultureInfo.InvariantCulture),
        float f => f.ToString("R", CultureInfo.InvariantCulture) + "f",
        long l => l.ToString(CultureInfo.InvariantCulture) + "L",
        EnumLiteral e => e.ToString(),
        Array arr => "[" + string.Join(", ", arr.Cast<object?>().Select(FormatLiteral)) + "]",
        IFormattable f => f.ToString(null, CultureInfo.InvariantCulture),
        _ => v.ToString() ?? "",
    };
}

/// <summary>An enum argument stored by member name, so a saved program never names a type to load.</summary>
public sealed class EnumLiteral
{
    internal EnumLiteral(string member) => Member = member;

    /// <summary>The enum member name (or its integer value when it has no single name).</summary>
    public string Member { get; }

    /// <inheritdoc/>
    public override string ToString() => "." + Member;
}
