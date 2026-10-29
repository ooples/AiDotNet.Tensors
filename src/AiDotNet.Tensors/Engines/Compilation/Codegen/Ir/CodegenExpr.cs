// Copyright (c) CrowdTrainer LLC. All rights reserved.
// An elementwise expression over a kernel's operands: the general form of a kernel body.
//
// The fixed body -- a product of operands, then one pre-bias, then exp or square -- covers
// every forward operator the catalog declared, and no backward one. LayerNorm's input
// gradient is rstd * (gamma*dy - m1 - xhat*m2) with xhat = (x - mean) * rstd: three terms,
// two of them products, one of them a product with a shifted operand. An expression tree is
// what a gradient is, so the body can now be one.
//
// A CALL names a scalar function of an external math library -- the way generated kernels
// elsewhere call libdevice: the kernel is declared, the scalar sinf or lgammaf it applies is a
// library routine linked in (CudaMathLibrary derives those routines from the hand-written
// kernels' own source, so a declared kernel computes exactly what the kernel it replaces did).
// A SCALAR is a runtime float parameter of the kernel (LeakyRelu's alpha, a clamp's bounds).

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>The node kinds of an elementwise kernel body.</summary>
public enum CodegenExprKind
{
    /// <summary>The value of one input operand at the current index.</summary>
    Operand,

    /// <summary>A literal.</summary>
    Constant,

    /// <summary><c>a + b</c>.</summary>
    Add,

    /// <summary><c>a * b</c>.</summary>
    Multiply,

    /// <summary><c>-a</c>.</summary>
    Negate,

    /// <summary><c>f(a, ...)</c>: a scalar function of an external math library.</summary>
    Call,

    /// <summary>Runtime scalar parameter <see cref="CodegenExpr.Input"/> of the kernel.</summary>
    Scalar,
}

/// <summary>
/// A scalar function of an external math library, called from a kernel body: its link
/// <paramref name="Symbol"/>, the <paramref name="Library"/> that defines it, and how many float
/// arguments it takes. <paramref name="Evaluate"/>, when given, is its fp64 value for the
/// interpreter; without it a body that calls the function cannot be interpreted.
/// </summary>
public sealed record CodegenLibraryFunction(string Symbol, string Library, int Arity, Func<double[], double>? Evaluate = null);

/// <summary>One node of an elementwise body expression.</summary>
public sealed class CodegenExpr
{
    private CodegenExpr(CodegenExprKind kind, int operand, double constant, CodegenExpr? left, CodegenExpr? right,
        CodegenLibraryFunction? function = null, CodegenExpr[]? arguments = null)
    {
        Kind = kind;
        Input = operand;
        Value = constant;
        Left = left;
        Right = right;
        Function = function;
        _arguments = arguments ?? Array.Empty<CodegenExpr>();
    }

    private readonly CodegenExpr[] _arguments;

    /// <summary>The called function, for <see cref="CodegenExprKind.Call"/>.</summary>
    public CodegenLibraryFunction? Function { get; }

    /// <summary>The call's arguments, for <see cref="CodegenExprKind.Call"/>.</summary>
    public IReadOnlyList<CodegenExpr> Arguments => _arguments;

    /// <summary>What this node computes.</summary>
    public CodegenExprKind Kind { get; }

    /// <summary>Input index, for <see cref="CodegenExprKind.Operand"/>.</summary>
    public int Input { get; }

    /// <summary>The literal, for <see cref="CodegenExprKind.Constant"/>.</summary>
    public double Value { get; }

    /// <summary>First (or only) child.</summary>
    public CodegenExpr? Left { get; }

    /// <summary>Second child of a binary node.</summary>
    public CodegenExpr? Right { get; }

    // The child a binary or unary node always has; a missing one is a construction defect, refused loudly.
    private CodegenExpr First => Left ?? throw new InvalidOperationException(Kind + " has no first operand.");
    private CodegenExpr Second => Right ?? throw new InvalidOperationException(Kind + " has no second operand.");

    /// <summary>Input <paramref name="input"/> at the current index.</summary>
    public static CodegenExpr Operand(int input)
    {
        if (input < 0) throw new ArgumentOutOfRangeException(nameof(input));
        return new CodegenExpr(CodegenExprKind.Operand, input, 0, null, null);
    }

    /// <summary>A finite literal.</summary>
    public static CodegenExpr Constant(double value)
    {
        if (double.IsNaN(value) || double.IsInfinity(value))
            throw new ArgumentOutOfRangeException(nameof(value), "A body constant must be finite.");
        return new CodegenExpr(CodegenExprKind.Constant, -1, value, null, null);
    }

    /// <summary><c>a + b</c>.</summary>
    public static CodegenExpr Add(CodegenExpr a, CodegenExpr b) =>
        new(CodegenExprKind.Add, -1, 0, a ?? throw new ArgumentNullException(nameof(a)),
            b ?? throw new ArgumentNullException(nameof(b)));

    /// <summary><c>a * b</c>.</summary>
    public static CodegenExpr Multiply(CodegenExpr a, CodegenExpr b) =>
        new(CodegenExprKind.Multiply, -1, 0, a ?? throw new ArgumentNullException(nameof(a)),
            b ?? throw new ArgumentNullException(nameof(b)));

    /// <summary><c>-a</c>.</summary>
    public static CodegenExpr Negate(CodegenExpr a) =>
        new(CodegenExprKind.Negate, -1, 0, a ?? throw new ArgumentNullException(nameof(a)), null);

    /// <summary><c>f(arguments)</c>: <paramref name="function"/> of the math library.</summary>
    public static CodegenExpr Call(CodegenLibraryFunction function, params CodegenExpr[] arguments)
    {
        if (function is null) throw new ArgumentNullException(nameof(function));
        if (arguments is null) throw new ArgumentNullException(nameof(arguments));
        if (arguments.Length != function.Arity)
            throw new ArgumentException(function.Symbol + " takes " + function.Arity + " arguments; got " + arguments.Length + ".", nameof(arguments));
        foreach (var argument in arguments)
            if (argument is null) throw new ArgumentNullException(nameof(arguments));
        return new CodegenExpr(CodegenExprKind.Call, -1, 0, null, null, function, (CodegenExpr[])arguments.Clone());
    }

    /// <summary>Runtime scalar parameter <paramref name="index"/> of the kernel.</summary>
    public static CodegenExpr Scalar(int index)
    {
        if (index < 0) throw new ArgumentOutOfRangeException(nameof(index));
        return new CodegenExpr(CodegenExprKind.Scalar, index, 0, null, null);
    }

    /// <summary><c>a - b</c>, as <c>a + (-b)</c>.</summary>
    public static CodegenExpr Subtract(CodegenExpr a, CodegenExpr b) => Add(a, Negate(b));

    /// <summary><c>a * b * c ...</c>, left to right.</summary>
    public static CodegenExpr Product(params CodegenExpr[] factors)
    {
        if (factors is null || factors.Length == 0) throw new ArgumentException("A product needs a factor.", nameof(factors));
        var result = factors[0];
        for (int i = 1; i < factors.Length; i++) result = Multiply(result, factors[i]);
        return result;
    }

    /// <summary>Distinct input indices the expression reads, ascending.</summary>
    public int[] Operands()
    {
        var set = new SortedSet<int>();
        Collect(this, set);
        var result = new int[set.Count];
        set.CopyTo(result);
        return result;

        static void Collect(CodegenExpr node, SortedSet<int> into)
        {
            if (node.Kind == CodegenExprKind.Operand) into.Add(node.Input);
            if (node.Left is not null) Collect(node.Left, into);
            if (node.Right is not null) Collect(node.Right, into);
            foreach (var argument in node._arguments) Collect(argument, into);
        }
    }

    /// <summary>How many runtime scalar parameters the expression reads (one past the highest index).</summary>
    public int ScalarCount()
    {
        int count = 0;
        Visit(this);
        return count;

        void Visit(CodegenExpr node)
        {
            if (node.Kind == CodegenExprKind.Scalar) count = Math.Max(count, node.Input + 1);
            if (node.Left is not null) Visit(node.Left);
            if (node.Right is not null) Visit(node.Right);
            foreach (var argument in node._arguments) Visit(argument);
        }
    }

    /// <summary>Every library function the expression calls, each once, in first-call order.</summary>
    public IReadOnlyList<CodegenLibraryFunction> Calls()
    {
        var found = new List<CodegenLibraryFunction>();
        Visit(this);
        return found;

        void Visit(CodegenExpr node)
        {
            if (node.Function is { } function && !found.Exists(f => f.Symbol == function.Symbol)) found.Add(function);
            if (node.Left is not null) Visit(node.Left);
            if (node.Right is not null) Visit(node.Right);
            foreach (var argument in node._arguments) Visit(argument);
        }
    }

    /// <summary>Evaluates in fp64 with <paramref name="operand"/> supplying each input's value.</summary>
    public double Evaluate(Func<int, double> operand) => Evaluate(operand, null);

    /// <summary>
    /// Evaluates in fp64: <paramref name="operand"/> supplies each input's value, <paramref name="scalar"/>
    /// each runtime scalar parameter's.
    /// </summary>
    public double Evaluate(Func<int, double> operand, Func<int, double>? scalar)
    {
        if (operand is null) throw new ArgumentNullException(nameof(operand));
        switch (Kind)
        {
            case CodegenExprKind.Operand: return operand(Input);
            case CodegenExprKind.Constant: return Value;
            case CodegenExprKind.Add: return First.Evaluate(operand, scalar) + Second.Evaluate(operand, scalar);
            case CodegenExprKind.Multiply: return First.Evaluate(operand, scalar) * Second.Evaluate(operand, scalar);
            case CodegenExprKind.Negate: return -First.Evaluate(operand, scalar);
            case CodegenExprKind.Scalar:
                return (scalar ?? throw new InvalidOperationException("The body reads scalar " + Input + " and none was supplied."))(Input);
            case CodegenExprKind.Call:
            {
                var function = Function ?? throw new InvalidOperationException("A call node has no function.");
                var evaluate = function.Evaluate
                    ?? throw new NotSupportedException(function.Symbol + " has no fp64 evaluator; the body cannot be interpreted.");
                var values = new double[_arguments.Length];
                for (int i = 0; i < values.Length; i++) values[i] = _arguments[i].Evaluate(operand, scalar);
                return evaluate(values);
            }
            default: throw new NotSupportedException("Unhandled body node " + Kind + ".");
        }
    }

    /// <summary>The same expression with every input index passed through <paramref name="map"/>.</summary>
    public CodegenExpr Remap(Func<int, int> map)
    {
        if (map is null) throw new ArgumentNullException(nameof(map));
        switch (Kind)
        {
            case CodegenExprKind.Operand: return Operand(map(Input));
            case CodegenExprKind.Constant:
            case CodegenExprKind.Scalar: return this;
            case CodegenExprKind.Negate: return Negate(First.Remap(map));
            case CodegenExprKind.Call:
            {
                var arguments = new CodegenExpr[_arguments.Length];
                for (int i = 0; i < arguments.Length; i++) arguments[i] = _arguments[i].Remap(map);
                return Call(Function ?? throw new InvalidOperationException("A call node has no function."), arguments);
            }
            default: return new CodegenExpr(Kind, -1, 0, First.Remap(map), Second.Remap(map));
        }
    }

    /// <summary>
    /// Emits the expression through <paramref name="emit"/>-style callbacks: <paramref name="operand"/>
    /// returns the register holding an input, <paramref name="constant"/> a literal's text,
    /// and <paramref name="op"/> writes one instruction and returns its destination.
    /// </summary>
    public string Lower(
        Func<int, string> operand, Func<double, string> constant, Func<string, string, string?, string> op) =>
        Lower(operand, constant, op,
            (f, _) => throw new NotSupportedException("This lowering cannot call " + f.Symbol + "."),
            i => throw new NotSupportedException("This lowering has no scalar parameter " + i + "."));

    /// <summary>
    /// <see cref="Lower(Func{int,string},Func{double,string},Func{string,string,string?,string})"/>, with
    /// <paramref name="call"/> emitting a library call (its arguments' registers in) and returning its result,
    /// and <paramref name="scalar"/> the register holding a runtime scalar parameter.
    /// </summary>
    public string Lower(
        Func<int, string> operand, Func<double, string> constant, Func<string, string, string?, string> op,
        Func<CodegenLibraryFunction, string[], string> call, Func<int, string> scalar)
    {
        if (operand is null) throw new ArgumentNullException(nameof(operand));
        if (constant is null) throw new ArgumentNullException(nameof(constant));
        if (op is null) throw new ArgumentNullException(nameof(op));
        if (call is null) throw new ArgumentNullException(nameof(call));
        if (scalar is null) throw new ArgumentNullException(nameof(scalar));
        string Child(CodegenExpr node) => node.Lower(operand, constant, op, call, scalar);
        switch (Kind)
        {
            case CodegenExprKind.Operand: return operand(Input);
            case CodegenExprKind.Constant: return constant(Value);
            case CodegenExprKind.Scalar: return scalar(Input);
            case CodegenExprKind.Add: return op("add.rn.f32", Child(First), Child(Second));
            case CodegenExprKind.Multiply: return op("mul.rn.f32", Child(First), Child(Second));
            case CodegenExprKind.Negate: return op("neg.f32", Child(First), null);
            case CodegenExprKind.Call:
            {
                var arguments = new string[_arguments.Length];
                for (int i = 0; i < arguments.Length; i++) arguments[i] = Child(_arguments[i]);
                return call(Function ?? throw new InvalidOperationException("A call node has no function."), arguments);
            }
            default: throw new NotSupportedException("Unhandled body node " + Kind + ".");
        }
    }

    /// <summary>A canonical text form, for identities and descriptions.</summary>
    public override string ToString()
    {
        var sb = new StringBuilder();
        Write(this, sb);
        return sb.ToString();

        static void Write(CodegenExpr node, StringBuilder sb)
        {
            switch (node.Kind)
            {
                case CodegenExprKind.Operand: sb.Append('$').Append(node.Input.ToString(CultureInfo.InvariantCulture)); break;
                case CodegenExprKind.Constant: sb.Append(node.Value.ToString("R", CultureInfo.InvariantCulture)); break;
                case CodegenExprKind.Negate: sb.Append("-("); Write(node.First, sb); sb.Append(')'); break;
                case CodegenExprKind.Scalar: sb.Append("%s").Append(node.Input.ToString(CultureInfo.InvariantCulture)); break;
                case CodegenExprKind.Call:
                    sb.Append(node.Function?.Symbol ?? "?").Append('(');
                    for (int i = 0; i < node._arguments.Length; i++)
                    {
                        if (i > 0) sb.Append(", ");
                        Write(node._arguments[i], sb);
                    }
                    sb.Append(')');
                    break;
                default:
                    sb.Append('(');
                    Write(node.First, sb);
                    sb.Append(node.Kind == CodegenExprKind.Add ? " + " : " * ");
                    Write(node.Second, sb);
                    sb.Append(')');
                    break;
            }
        }
    }
}
