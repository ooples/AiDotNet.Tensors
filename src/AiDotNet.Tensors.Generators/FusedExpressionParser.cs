// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Generators;

internal enum ExprNodeKind
{
    InputVar,
    ScalarVar,
    Literal,
    Negate,
    Binary,
    Intrinsic
}

internal sealed class ExprNode
{
    public ExprNodeKind Kind { get; }
    public int Index { get; }
    public float Value { get; }
    public char Operator { get; }
    public string IntrinsicName { get; }
    public IReadOnlyList<ExprNode> Children { get; }

    private ExprNode(
        ExprNodeKind kind,
        int index,
        float value,
        char op,
        string intrinsicName,
        IReadOnlyList<ExprNode> children)
    {
        Kind = kind;
        Index = index;
        Value = value;
        Operator = op;
        IntrinsicName = intrinsicName;
        Children = children;
    }

    public static ExprNode Input(int index) => new(ExprNodeKind.InputVar, index, 0f, '\0', string.Empty, Array.Empty<ExprNode>());
    public static ExprNode Scalar(int index) => new(ExprNodeKind.ScalarVar, index, 0f, '\0', string.Empty, Array.Empty<ExprNode>());
    public static ExprNode Literal(float value) => new(ExprNodeKind.Literal, -1, value, '\0', string.Empty, Array.Empty<ExprNode>());
    public static ExprNode Negate(ExprNode operand) => new(ExprNodeKind.Negate, -1, 0f, '-', string.Empty, new[] { operand });
    public static ExprNode Binary(char op, ExprNode left, ExprNode right) => new(ExprNodeKind.Binary, -1, 0f, op, string.Empty, new[] { left, right });
    public static ExprNode Call(string name, IReadOnlyList<ExprNode> args) => new(ExprNodeKind.Intrinsic, -1, 0f, '\0', name, args);

    public bool SupportsPortableVector()
    {
        switch (Kind)
        {
            case ExprNodeKind.InputVar:
            case ExprNodeKind.ScalarVar:
            case ExprNodeKind.Literal:
                return true;
            case ExprNodeKind.Negate:
                return Children[0].SupportsPortableVector();
            case ExprNodeKind.Binary:
                return Children[0].SupportsPortableVector() && Children[1].SupportsPortableVector();
            case ExprNodeKind.Intrinsic:
                if (IntrinsicName is not ("relu" or "abs" or "sqrt" or "rsqrt" or "max" or "min" or "fma"))
                    return false;
                for (int i = 0; i < Children.Count; i++)
                {
                    if (!Children[i].SupportsPortableVector())
                        return false;
                }
                return true;
            default:
                return false;
        }
    }
}

internal static class FusedExpressionParser
{
    public static ExprNode Parse(string expression, int inputCount, int scalarCount)
    {
        if (string.IsNullOrWhiteSpace(expression))
            throw new FormatException("Fused kernel expression cannot be empty.");

        var cursor = new ParserCursor(expression, inputCount, scalarCount);
        var root = cursor.ParseAddSub();
        cursor.SkipWhitespace();
        if (!cursor.IsAtEnd)
            throw new FormatException($"Unexpected trailing characters in fused expression at offset {cursor.Position}: '{expression.Substring(cursor.Position)}'.");
        return root;
    }

    public static string EmitAvx2Expr(ExprNode node, string offsetExpr)
    {
        switch (node.Kind)
        {
            case ExprNodeKind.InputVar:
                return $"global::AiDotNet.Tensors.Engines.Simd.SimdKernels.ReadVector256(x{node.Index}, {offsetExpr})";
            case ExprNodeKind.ScalarVar:
                return $"Vector256.Create(s{node.Index})";
            case ExprNodeKind.Literal:
                return $"Vector256.Create({FormatFloat(node.Value)})";
            case ExprNodeKind.Negate:
                return $"Avx.Subtract(Vector256<float>.Zero, {EmitAvx2Expr(node.Children[0], offsetExpr)})";
            case ExprNodeKind.Binary:
            {
                string lhs = EmitAvx2Expr(node.Children[0], offsetExpr);
                string rhs = EmitAvx2Expr(node.Children[1], offsetExpr);
                return node.Operator switch
                {
                    '+' => $"Avx.Add({lhs}, {rhs})",
                    '-' => $"Avx.Subtract({lhs}, {rhs})",
                    '*' => $"Avx.Multiply({lhs}, {rhs})",
                    '/' => $"Avx.Divide({lhs}, {rhs})",
                    _ => throw new InvalidOperationException($"Unsupported binary operator '{node.Operator}'.")
                };
            }
            case ExprNodeKind.Intrinsic:
            {
                string a0 = EmitAvx2Expr(node.Children[0], offsetExpr);
                switch (node.IntrinsicName)
                {
                    case "relu":
                        return $"Avx.Max(Vector256<float>.Zero, {a0})";
                    case "sigmoid":
                        return $"global::AiDotNet.Tensors.Engines.Simd.SimdKernels.FastSigmoid256({a0})";
                    case "swish":
                    case "silu":
                        return $"GeneratedKernelSimdMath.Swish256({a0})";
                    case "tanh":
                        return $"global::AiDotNet.Tensors.Engines.Simd.SimdKernels.FastTanh256({a0})";
                    case "gelu":
                        return $"GeneratedKernelSimdMath.Gelu256({a0})";
                    case "exp":
                        return $"global::AiDotNet.Tensors.Engines.Simd.SimdKernels.FastExp256({a0})";
                    case "log":
                        return $"global::AiDotNet.Tensors.Engines.Simd.SimdKernels.FastLog256({a0})";
                    case "sqrt":
                        return $"Avx.Sqrt({a0})";
                    case "rsqrt":
                        return $"Avx.Divide(Vector256.Create(1.0f), Avx.Sqrt({a0}))";
                    case "abs":
                        return $"Avx.And({a0}, Vector256.Create(int.MaxValue).AsSingle())";
                    case "max":
                        return $"Avx.Max({a0}, {EmitAvx2Expr(node.Children[1], offsetExpr)})";
                    case "min":
                        return $"Avx.Min({a0}, {EmitAvx2Expr(node.Children[1], offsetExpr)})";
                    case "fma":
                        return $"GeneratedKernelSimdMath.Fma256({a0}, {EmitAvx2Expr(node.Children[1], offsetExpr)}, {EmitAvx2Expr(node.Children[2], offsetExpr)})";
                    default:
                        throw new InvalidOperationException($"Unsupported intrinsic '{node.IntrinsicName}'.");
                }
            }
            default:
                throw new InvalidOperationException($"Unsupported AST kind '{node.Kind}'.");
        }
    }

    public static string EmitPortableVectorExpr(ExprNode node, string vecIndex)
    {
        switch (node.Kind)
        {
            case ExprNodeKind.InputVar:
                return $"vx{node.Index}[{vecIndex}]";
            case ExprNodeKind.ScalarVar:
                return $"vs{node.Index}";
            case ExprNodeKind.Literal:
                return $"new global::System.Numerics.Vector<float>({FormatFloat(node.Value)})";
            case ExprNodeKind.Negate:
                return $"(new global::System.Numerics.Vector<float>(-1.0f) * {EmitPortableVectorExpr(node.Children[0], vecIndex)})";
            case ExprNodeKind.Binary:
                return $"({EmitPortableVectorExpr(node.Children[0], vecIndex)} {node.Operator} {EmitPortableVectorExpr(node.Children[1], vecIndex)})";
            case ExprNodeKind.Intrinsic:
            {
                string a0 = EmitPortableVectorExpr(node.Children[0], vecIndex);
                return node.IntrinsicName switch
                {
                    "relu" => $"global::System.Numerics.Vector.Max(global::System.Numerics.Vector<float>.Zero, {a0})",
                    "abs" => $"global::System.Numerics.Vector.Abs({a0})",
                    "sqrt" => $"global::System.Numerics.Vector.SquareRoot({a0})",
                    "rsqrt" => $"(new global::System.Numerics.Vector<float>(1.0f) / global::System.Numerics.Vector.SquareRoot({a0}))",
                    "max" => $"global::System.Numerics.Vector.Max({a0}, {EmitPortableVectorExpr(node.Children[1], vecIndex)})",
                    "min" => $"global::System.Numerics.Vector.Min({a0}, {EmitPortableVectorExpr(node.Children[1], vecIndex)})",
                    "fma" => $"(({a0} * {EmitPortableVectorExpr(node.Children[1], vecIndex)}) + {EmitPortableVectorExpr(node.Children[2], vecIndex)})",
                    _ => throw new InvalidOperationException($"Intrinsic '{node.IntrinsicName}' is not supported on portable Vector<float>.")
                };
            }
            default:
                throw new InvalidOperationException($"Unsupported AST kind '{node.Kind}'.");
        }
    }

    public static string EmitScalarCsExpr(ExprNode node, string indexExpr)
    {
        switch (node.Kind)
        {
            case ExprNodeKind.InputVar:
                return $"x{node.Index}[{indexExpr}]";
            case ExprNodeKind.ScalarVar:
                return $"s{node.Index}";
            case ExprNodeKind.Literal:
                return FormatFloat(node.Value);
            case ExprNodeKind.Negate:
                return $"(-{EmitScalarCsExpr(node.Children[0], indexExpr)})";
            case ExprNodeKind.Binary:
                return $"({EmitScalarCsExpr(node.Children[0], indexExpr)} {node.Operator} {EmitScalarCsExpr(node.Children[1], indexExpr)})";
            case ExprNodeKind.Intrinsic:
            {
                string a0 = EmitScalarCsExpr(node.Children[0], indexExpr);
                return node.IntrinsicName switch
                {
                    "relu" => $"GeneratedKernelSimdMath.ReluScalar({a0})",
                    "sigmoid" => $"GeneratedKernelSimdMath.SigmoidScalar({a0})",
                    "swish" or "silu" => $"GeneratedKernelSimdMath.SwishScalar({a0})",
                    "tanh" => $"GeneratedKernelSimdMath.TanhScalar({a0})",
                    "gelu" => $"GeneratedKernelSimdMath.GeluScalar({a0})",
                    "exp" => $"GeneratedKernelSimdMath.ExpScalar({a0})",
                    "log" => $"GeneratedKernelSimdMath.LogScalar({a0})",
                    "sqrt" => $"GeneratedKernelSimdMath.SqrtScalar({a0})",
                    "rsqrt" => $"(1.0f / GeneratedKernelSimdMath.SqrtScalar({a0}))",
                    "abs" => $"Math.Abs({a0})",
                    "max" => $"Math.Max({a0}, {EmitScalarCsExpr(node.Children[1], indexExpr)})",
                    "min" => $"Math.Min({a0}, {EmitScalarCsExpr(node.Children[1], indexExpr)})",
                    "fma" => $"(({a0} * {EmitScalarCsExpr(node.Children[1], indexExpr)}) + {EmitScalarCsExpr(node.Children[2], indexExpr)})",
                    _ => throw new InvalidOperationException($"Unsupported intrinsic '{node.IntrinsicName}'.")
                };
            }
            default:
                throw new InvalidOperationException($"Unsupported AST kind '{node.Kind}'.");
        }
    }

    public static string EmitCudaScalarExpr(ExprNode node, string laneSuffix)
    {
        switch (node.Kind)
        {
            case ExprNodeKind.InputVar:
                return string.IsNullOrEmpty(laneSuffix) ? $"x{node.Index}[idx]" : $"v{node.Index}.{laneSuffix}";
            case ExprNodeKind.ScalarVar:
                return $"s{node.Index}";
            case ExprNodeKind.Literal:
                return FormatFloat(node.Value);
            case ExprNodeKind.Negate:
                return $"(-{EmitCudaScalarExpr(node.Children[0], laneSuffix)})";
            case ExprNodeKind.Binary:
                return $"({EmitCudaScalarExpr(node.Children[0], laneSuffix)} {node.Operator} {EmitCudaScalarExpr(node.Children[1], laneSuffix)})";
            case ExprNodeKind.Intrinsic:
            {
                string a0 = EmitCudaScalarExpr(node.Children[0], laneSuffix);
                return node.IntrinsicName switch
                {
                    "relu" => $"fmaxf(0.0f, {a0})",
                    "sigmoid" => $"(1.0f / (1.0f + expf(-({a0}))))",
                    "swish" or "silu" => $"({a0} / (1.0f + expf(-({a0}))))",
                    "tanh" => $"tanhf({a0})",
                    "gelu" => $"aidn_gelu({a0})",
                    "exp" => $"expf({a0})",
                    "log" => $"logf({a0})",
                    "sqrt" => $"sqrtf({a0})",
                    "rsqrt" => $"rsqrtf({a0})",
                    "abs" => $"fabsf({a0})",
                    "max" => $"fmaxf({a0}, {EmitCudaScalarExpr(node.Children[1], laneSuffix)})",
                    "min" => $"fminf({a0}, {EmitCudaScalarExpr(node.Children[1], laneSuffix)})",
                    "fma" => $"fmaf({a0}, {EmitCudaScalarExpr(node.Children[1], laneSuffix)}, {EmitCudaScalarExpr(node.Children[2], laneSuffix)})",
                    _ => throw new InvalidOperationException($"Unsupported CUDA intrinsic '{node.IntrinsicName}'.")
                };
            }
            default:
                throw new InvalidOperationException($"Unsupported AST kind '{node.Kind}'.");
        }
    }

    public static string EmitCudaKernelSource(string kernelName, ExprNode ast, int inputCount, int scalarCount)
    {
        var sb = new StringBuilder(1024);
        sb.AppendLine("__device__ __forceinline__ float aidn_gelu(float x) {");
        sb.AppendLine("    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));");
        sb.AppendLine("}");
        sb.Append("extern \"C\" __global__ void ").Append(kernelName).Append('(');
        for (int i = 0; i < inputCount; i++)
        {
            sb.Append("const float* __restrict__ x").Append(i).Append(", ");
        }
        for (int s = 0; s < scalarCount; s++)
        {
            sb.Append("float s").Append(s).Append(", ");
        }
        sb.AppendLine("float* __restrict__ dst, int n) {");

        // Vectorized float4 fast path when pointers are 16-byte aligned.
        sb.Append("    bool aligned = (((unsigned long long)dst) & 15ULL) == 0ULL");
        for (int i = 0; i < inputCount; i++)
        {
            sb.Append(" && (((unsigned long long)x").Append(i).Append(") & 15ULL) == 0ULL");
        }
        sb.AppendLine(";");
        sb.AppendLine("    int tid = blockIdx.x * blockDim.x + threadIdx.x;");
        sb.AppendLine("    int stride = blockDim.x * gridDim.x;");
        sb.AppendLine("    if (aligned && n >= 4) {");
        sb.AppendLine("        int vecCount = n >> 2;");
        sb.AppendLine("        for (int vecIdx = tid; vecIdx < vecCount; vecIdx += stride) {");
        for (int i = 0; i < inputCount; i++)
        {
            sb.Append("            float4 v").Append(i).Append(" = reinterpret_cast<const float4*>(x").Append(i).AppendLine(")[vecIdx];");
        }
        sb.AppendLine("            float4 out4;");
        sb.Append("            out4.x = ").Append(EmitCudaScalarExpr(ast, "x")).AppendLine(";");
        sb.Append("            out4.y = ").Append(EmitCudaScalarExpr(ast, "y")).AppendLine(";");
        sb.Append("            out4.z = ").Append(EmitCudaScalarExpr(ast, "z")).AppendLine(";");
        sb.Append("            out4.w = ").Append(EmitCudaScalarExpr(ast, "w")).AppendLine(";");
        sb.AppendLine("            reinterpret_cast<float4*>(dst)[vecIdx] = out4;");
        sb.AppendLine("        }");
        sb.AppendLine("        for (int idx = (vecCount << 2) + tid; idx < n; idx += stride) {");
        sb.Append("            dst[idx] = ").Append(EmitCudaScalarExpr(ast, string.Empty)).AppendLine(";");
        sb.AppendLine("        }");
        sb.AppendLine("    } else {");
        sb.AppendLine("        for (int idx = tid; idx < n; idx += stride) {");
        sb.Append("            dst[idx] = ").Append(EmitCudaScalarExpr(ast, string.Empty)).AppendLine(";");
        sb.AppendLine("        }");
        sb.AppendLine("    }");
        sb.AppendLine("}");
        return sb.ToString();
    }

    public static string EmitTileDescriptorInitializer(
        string tileName,
        ExprNode ast,
        int inputCount,
        int scalarCount,
        string cudaConstIdent)
    {
        var instLines = new List<string>(16);

        int EmitInst(string opCodeName, int s0 = -1, int s1 = -1, int s2 = -1, float imm = 0.0f)
        {
            int idx = instLines.Count;
            instLines.Add(
                $"new global::AiDotNet.Tensors.Engines.Compilation.FusionTileIr.FusionTileInstruction(" +
                $"global::AiDotNet.Tensors.Engines.Compilation.FusionTileIr.FusionTileOpCode.{opCodeName}, " +
                $"(short){s0}, (short){s1}, (short){s2}, {FormatFloat(imm)})");
            return idx;
        }

        int LowerNode(ExprNode node)
        {
            switch (node.Kind)
            {
                case ExprNodeKind.InputVar:
                    return EmitInst("LoadInput", node.Index);
                case ExprNodeKind.ScalarVar:
                    return EmitInst("LoadScalar", node.Index);
                case ExprNodeKind.Literal:
                    return EmitInst("Const", -1, -1, -1, node.Value);
                case ExprNodeKind.Negate:
                    return EmitInst("Negate", LowerNode(node.Children[0]));
                case ExprNodeKind.Binary:
                {
                    int l = LowerNode(node.Children[0]);
                    int r = LowerNode(node.Children[1]);
                    string opName = node.Operator switch
                    {
                        '+' => "Add",
                        '-' => "Sub",
                        '*' => "Mul",
                        '/' => "Div",
                        _ => "Add"
                    };
                    return EmitInst(opName, l, r);
                }
                case ExprNodeKind.Intrinsic:
                {
                    int a0 = LowerNode(node.Children[0]);
                    return node.IntrinsicName switch
                    {
                        "relu" => EmitInst("Relu", a0),
                        "sigmoid" => EmitInst("Sigmoid", a0),
                        "swish" or "silu" => EmitInst("Swish", a0),
                        "tanh" => EmitInst("Tanh", a0),
                        "gelu" => EmitInst("Gelu", a0),
                        "exp" => EmitInst("Exp", a0),
                        "log" => EmitInst("Log", a0),
                        "sqrt" => EmitInst("Sqrt", a0),
                        "rsqrt" => EmitInst("Rsqrt", a0),
                        "abs" => EmitInst("Abs", a0),
                        "max" => EmitInst("Max", a0, LowerNode(node.Children[1])),
                        "min" => EmitInst("Min", a0, LowerNode(node.Children[1])),
                        "fma" => EmitInst("Fma", a0, LowerNode(node.Children[1]), LowerNode(node.Children[2])),
                        _ => a0
                    };
                }
                default:
                    return 0;
            }
        }

        int rootIdx = LowerNode(ast);
        EmitInst("StoreOutput", rootIdx, 0);

        var sb = new StringBuilder(512);
        sb.Append("global::AiDotNet.Tensors.Engines.Compilation.FusionTileIr.FusionTileLinker.CreatePrebuiltElementwiseTile(\"")
          .Append(tileName)
          .Append("\", ")
          .Append(inputCount)
          .Append(", ")
          .Append(scalarCount)
          .Append(", new global::AiDotNet.Tensors.Engines.Compilation.FusionTileIr.FusionTileInstruction[] { ")
          .Append(string.Join(", ", instLines))
          .Append(" }, ")
          .Append(cudaConstIdent)
          .Append(')');
        return sb.ToString();
    }

    private static string FormatFloat(float value)
    {
        string text = value.ToString("R", CultureInfo.InvariantCulture);
        if (text.IndexOf('.') < 0 && text.IndexOf('e') < 0 && text.IndexOf('E') < 0)
            text += ".0";
        return text + "f";
    }

    private struct ParserCursor
    {
        private readonly string _text;
        private readonly int _inputCount;
        private readonly int _scalarCount;
        public int Position;

        public ParserCursor(string text, int inputCount, int scalarCount)
        {
            _text = text;
            _inputCount = inputCount;
            _scalarCount = scalarCount;
            Position = 0;
        }

        public bool IsAtEnd => Position >= _text.Length;

        public void SkipWhitespace()
        {
            while (Position < _text.Length && char.IsWhiteSpace(_text[Position]))
                Position++;
        }

        public ExprNode ParseAddSub()
        {
            var left = ParseMulDiv();
            while (true)
            {
                SkipWhitespace();
                if (Position < _text.Length && (_text[Position] == '+' || _text[Position] == '-'))
                {
                    char op = _text[Position++];
                    var right = ParseMulDiv();
                    left = ExprNode.Binary(op, left, right);
                }
                else
                {
                    break;
                }
            }
            return left;
        }

        private ExprNode ParseMulDiv()
        {
            var left = ParseUnary();
            while (true)
            {
                SkipWhitespace();
                if (Position < _text.Length && (_text[Position] == '*' || _text[Position] == '/'))
                {
                    char op = _text[Position++];
                    var right = ParseUnary();
                    left = ExprNode.Binary(op, left, right);
                }
                else
                {
                    break;
                }
            }
            return left;
        }

        private ExprNode ParseUnary()
        {
            SkipWhitespace();
            if (Position < _text.Length && _text[Position] == '-')
            {
                Position++;
                return ExprNode.Negate(ParseUnary());
            }
            if (Position < _text.Length && _text[Position] == '+')
            {
                Position++;
                return ParseUnary();
            }
            return ParsePrimary();
        }

        private ExprNode ParsePrimary()
        {
            SkipWhitespace();
            if (IsAtEnd)
                throw new FormatException("Unexpected end of fused kernel expression.");

            char c = _text[Position];
            if (c == '(')
            {
                Position++;
                var inner = ParseAddSub();
                SkipWhitespace();
                if (IsAtEnd || _text[Position] != ')')
                    throw new FormatException($"Expected ')' at offset {Position}.");
                Position++;
                return inner;
            }

            if (char.IsDigit(c) || c == '.')
            {
                int start = Position;
                while (Position < _text.Length && (char.IsDigit(_text[Position]) || _text[Position] == '.'))
                    Position++;
                if (Position < _text.Length && (_text[Position] == 'e' || _text[Position] == 'E'))
                {
                    Position++;
                    if (Position < _text.Length && (_text[Position] == '+' || _text[Position] == '-'))
                        Position++;
                    while (Position < _text.Length && char.IsDigit(_text[Position]))
                        Position++;
                }
                string numSlice = _text.Substring(start, Position - start);
                if (Position < _text.Length && (_text[Position] == 'f' || _text[Position] == 'F'))
                    Position++;
                if (!float.TryParse(numSlice, NumberStyles.Float, CultureInfo.InvariantCulture, out float parsed))
                    throw new FormatException($"Invalid float literal '{numSlice}' at offset {start}.");
                return ExprNode.Literal(parsed);
            }

            if (char.IsLetter(c) || c == '_')
            {
                int start = Position;
                while (Position < _text.Length && (char.IsLetterOrDigit(_text[Position]) || _text[Position] == '_'))
                    Position++;
                string ident = _text.Substring(start, Position - start);
                SkipWhitespace();

                if (!IsAtEnd && _text[Position] == '(')
                {
                    Position++;
                    var args = new List<ExprNode>(3);
                    SkipWhitespace();
                    if (!IsAtEnd && _text[Position] != ')')
                    {
                        args.Add(ParseAddSub());
                        while (true)
                        {
                            SkipWhitespace();
                            if (!IsAtEnd && _text[Position] == ',')
                            {
                                Position++;
                                args.Add(ParseAddSub());
                            }
                            else
                            {
                                break;
                            }
                        }
                    }
                    SkipWhitespace();
                    if (IsAtEnd || _text[Position] != ')')
                        throw new FormatException($"Expected ')' closing call to '{ident}' at offset {Position}.");
                    Position++;

                    string lower = ident.ToLowerInvariant();
                    int expectedArity = lower switch
                    {
                        "relu" or "sigmoid" or "swish" or "silu" or "tanh" or "gelu" or "exp" or "log" or "sqrt" or "rsqrt" or "abs" => 1,
                        "max" or "min" => 2,
                        "fma" => 3,
                        _ => throw new FormatException($"Unknown fused kernel intrinsic '{ident}'.")
                    };
                    if (args.Count != expectedArity)
                        throw new FormatException($"Intrinsic '{ident}' expects {expectedArity} argument(s), got {args.Count}.");
                    return ExprNode.Call(lower, args);
                }

                if (ident.Length >= 2 && ident[0] == 'x' && int.TryParse(ident.Substring(1), NumberStyles.None, CultureInfo.InvariantCulture, out int xIdx))
                {
                    if (xIdx < 0 || xIdx >= _inputCount)
                        throw new FormatException($"Input variable '{ident}' exceeds declared inputCount {_inputCount}.");
                    return ExprNode.Input(xIdx);
                }

                if (ident.Length >= 2 && ident[0] == 's' && int.TryParse(ident.Substring(1), NumberStyles.None, CultureInfo.InvariantCulture, out int sIdx))
                {
                    if (sIdx < 0 || sIdx >= _scalarCount)
                        throw new FormatException($"Scalar variable '{ident}' exceeds declared scalarCount {_scalarCount}.");
                    return ExprNode.Scalar(sIdx);
                }

                throw new FormatException($"Unknown identifier '{ident}' in fused kernel expression.");
            }

            throw new FormatException($"Unexpected character '{c}' at offset {Position}.");
        }
    }
}
