// Copyright (c) AiDotNet. All rights reserved.
// An elementwise expression over a kernel's operands: the general form of a kernel body.
//
// The fixed body -- a product of operands, then one pre-bias, then exp or square -- covers
// every forward operator the catalog declared, and no backward one. LayerNorm's input
// gradient is rstd * (gamma*dy - m1 - xhat*m2) with xhat = (x - mean) * rstd: three terms,
// two of them products, one of them a product with a shifted operand. An expression tree is
// what a gradient is, so the body can now be one.

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
}

/// <summary>One node of an elementwise body expression.</summary>
public sealed class CodegenExpr
{
    private CodegenExpr(CodegenExprKind kind, int operand, double constant, CodegenExpr? left, CodegenExpr? right)
    {
        Kind = kind;
        Input = operand;
        Value = constant;
        Left = left;
        Right = right;
    }

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
        }
    }

    /// <summary>Evaluates in fp64 with <paramref name="operand"/> supplying each input's value.</summary>
    public double Evaluate(Func<int, double> operand)
    {
        if (operand is null) throw new ArgumentNullException(nameof(operand));
        return Kind switch
        {
            CodegenExprKind.Operand => operand(Input),
            CodegenExprKind.Constant => Value,
            CodegenExprKind.Add => First.Evaluate(operand) + Second.Evaluate(operand),
            CodegenExprKind.Multiply => First.Evaluate(operand) * Second.Evaluate(operand),
            CodegenExprKind.Negate => -First.Evaluate(operand),
            _ => throw new NotSupportedException("Unhandled body node " + Kind + "."),
        };
    }

    /// <summary>The same expression with every input index passed through <paramref name="map"/>.</summary>
    public CodegenExpr Remap(Func<int, int> map)
    {
        if (map is null) throw new ArgumentNullException(nameof(map));
        return Kind switch
        {
            CodegenExprKind.Operand => Operand(map(Input)),
            CodegenExprKind.Constant => this,
            CodegenExprKind.Negate => Negate(First.Remap(map)),
            _ => new CodegenExpr(Kind, -1, 0, First.Remap(map), Second.Remap(map)),
        };
    }

    /// <summary>
    /// Emits the expression through <paramref name="emit"/>-style callbacks: <paramref name="operand"/>
    /// returns the register holding an input, <paramref name="constant"/> a literal's text,
    /// and <paramref name="op"/> writes one instruction and returns its destination.
    /// </summary>
    public string Lower(
        Func<int, string> operand, Func<double, string> constant, Func<string, string, string?, string> op)
    {
        if (operand is null) throw new ArgumentNullException(nameof(operand));
        if (constant is null) throw new ArgumentNullException(nameof(constant));
        if (op is null) throw new ArgumentNullException(nameof(op));
        return Kind switch
        {
            CodegenExprKind.Operand => operand(Input),
            CodegenExprKind.Constant => constant(Value),
            CodegenExprKind.Add => op("add.rn.f32", First.Lower(operand, constant, op), Second.Lower(operand, constant, op)),
            CodegenExprKind.Multiply => op("mul.rn.f32", First.Lower(operand, constant, op), Second.Lower(operand, constant, op)),
            CodegenExprKind.Negate => op("neg.f32", First.Lower(operand, constant, op), null),
            _ => throw new NotSupportedException("Unhandled body node " + Kind + "."),
        };
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
