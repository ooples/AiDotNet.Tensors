using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using Microsoft.CodeAnalysis;

namespace AiDotNet.Tensors.Analyzers;

/// <summary>
/// Generates, from the <c>IEngine</c> interface itself, the three pieces the portable program format is built on:
/// the op catalog (a stable id, named and typed parameters, and an in-place flag per public engine method), the
/// <c>TracingEngine</c> decorator that records every outer call, and the typed replay invoker. Because all three
/// come from the same symbols, an engine method added to <c>IEngine</c> is traceable and replayable with no
/// hand-written registration, and the catalog is the replay allowlist: a saved program can only name these methods.
/// </summary>
[Generator(LanguageNames.CSharp)]
public sealed class EngineSurfaceGenerator : IIncrementalGenerator
{
    private const string EngineMetadataName = "AiDotNet.Tensors.Engines.IEngine";
    private const string TensorMetadataName = "AiDotNet.Tensors.LinearAlgebra.Tensor`1";
    private const string VectorMetadataName = "AiDotNet.Tensors.LinearAlgebra.Vector`1";
    private const string MatrixMetadataName = "AiDotNet.Tensors.LinearAlgebra.Matrix`1";

    private static readonly SymbolDisplayFormat TypeFormat = SymbolDisplayFormat.FullyQualifiedFormat
        .AddMiscellaneousOptions(SymbolDisplayMiscellaneousOptions.IncludeNullableReferenceTypeModifier);

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        // Only the assembly that declares IEngine generates the surface; consumers get it from the package.
        var engine = context.CompilationProvider.Select((c, _) =>
        {
            var symbol = c.GetTypeByMetadataName(EngineMetadataName);
            return symbol is not null && SymbolEqualityComparer.Default.Equals(symbol.ContainingAssembly, c.Assembly)
                ? c
                : null;
        });
        context.RegisterSourceOutput(engine, (spc, compilation) =>
        {
            if (compilation is null) return;
            var model = Build(compilation);
            spc.AddSource("EngineOpCatalog.g.cs", EmitCatalog(model));
            spc.AddSource("TracingEngine.g.cs", EmitTracingEngine(model));
            spc.AddSource("EngineOpInvoker.g.cs", EmitInvoker(model));
        });
    }

    private sealed class Op
    {
        public IMethodSymbol Method = null!;
        public INamedTypeSymbol Declaring = null!;
        public string Id = "";
        public int Index;
        public bool Traceable;
        public string? UntraceableReason;
        public bool InPlace;
        public int InPlaceParam = -1;
        public string ReturnKind = "";
        public string[] ParamKinds = Array.Empty<string>();
    }

    private sealed class Model
    {
        public INamedTypeSymbol Engine = null!;
        public List<Op> Ops = new();
        public List<IPropertySymbol> Properties = new();
        public List<IEventSymbol> Events = new();
        public string OpSetHash = "";
        public INamedTypeSymbol? Tensor, Vector, Matrix;
    }

    private static Model Build(Compilation c)
    {
        var m = new Model
        {
            Engine = c.GetTypeByMetadataName(EngineMetadataName)!,
            Tensor = c.GetTypeByMetadataName(TensorMetadataName),
            Vector = c.GetTypeByMetadataName(VectorMetadataName),
            Matrix = c.GetTypeByMetadataName(MatrixMetadataName),
        };
        var interfaces = new[] { m.Engine }.Concat(m.Engine.AllInterfaces).ToList();
        foreach (var iface in interfaces)
        {
            foreach (var member in iface.GetMembers())
            {
                switch (member)
                {
                    case IMethodSymbol method when method.MethodKind == MethodKind.Ordinary && !method.IsStatic
                                                   && (method.IsAbstract || method.IsVirtual):
                        m.Ops.Add(Describe(m, iface, method));
                        break;
                    case IPropertySymbol p when !p.IsStatic && (p.IsAbstract || p.IsVirtual):
                        m.Properties.Add(p);
                        break;
                    case IEventSymbol e when !e.IsStatic:
                        m.Events.Add(e);
                        break;
                }
            }
        }
        // Stable order (catalog index is not persisted; the id is), deterministic output.
        m.Ops = m.Ops.OrderBy(o => o.Id, StringComparer.Ordinal).ToList();
        for (int i = 0; i < m.Ops.Count; i++) m.Ops[i].Index = i;
        var dupes = m.Ops.GroupBy(o => o.Id).Where(g => g.Count() > 1).Select(g => g.Key).ToList();
        if (dupes.Count > 0) throw new InvalidOperationException("Duplicate engine op ids: " + string.Join(", ", dupes));
        using var sha = SHA256.Create();
        var bytes = sha.ComputeHash(Encoding.UTF8.GetBytes(string.Join("\n", m.Ops.Where(o => o.Traceable).Select(o => o.Id))));
        m.OpSetHash = string.Concat(bytes.Take(8).Select(b => b.ToString("x2")));
        return m;
    }

    private static Op Describe(Model m, INamedTypeSymbol declaring, IMethodSymbol method)
    {
        var op = new Op { Method = method, Declaring = declaring };
        op.Id = method.Name + "`" + method.TypeParameters.Length + "("
                + string.Join(",", method.Parameters.Select(p => RefPrefix(p) + TypeKey(p.Type))) + ")";
        op.ParamKinds = method.Parameters.Select(p => p.RefKind is RefKind.Ref or RefKind.Out
            ? (p.RefKind == RefKind.Out && IsTensorLike(m, p.Type) ? "OutTensor" : "Unsupported")
            : Kind(m, method, p.Type)).ToArray();
        op.ReturnKind = method.ReturnsVoid ? "Void" : Kind(m, method, method.ReturnType);

        var bad = method.Parameters.Select((p, i) => (p, k: op.ParamKinds[i])).FirstOrDefault(x => x.k == "Unsupported");
        if (bad.p is not null) op.UntraceableReason = $"parameter '{bad.p.Name}' of type {bad.p.Type.ToDisplayString()} has no portable form";
        else if (op.ReturnKind == "Unsupported") op.UntraceableReason = $"return type {method.ReturnType.ToDisplayString()} has no portable form";
        else if (method.TypeParameters.Length > 1) op.UntraceableReason = null; // replayed through the reflection binder
        op.Traceable = op.UntraceableReason is null;

        if (method.ReturnsVoid)
        {
            for (int i = 0; i < method.Parameters.Length; i++)
            {
                if (op.ParamKinds[i] is "Tensor" or "TensorFixed" or "Vector" or "Matrix")
                {
                    op.InPlace = true;
                    op.InPlaceParam = i;
                    break;
                }
            }
        }
        return op;
    }

    private static string RefPrefix(IParameterSymbol p) => p.RefKind switch
    {
        RefKind.Ref => "ref ",
        RefKind.Out => "out ",
        RefKind.In => "in ",
        _ => "",
    };

    // Method type parameters are positional (!!0), so renaming T never changes an op id.
    private static string TypeKey(ITypeSymbol t) => t switch
    {
        ITypeParameterSymbol tp when tp.TypeParameterKind == TypeParameterKind.Method => "!!" + tp.Ordinal,
        IArrayTypeSymbol a => TypeKey(a.ElementType) + "[" + new string(',', a.Rank - 1) + "]",
        INamedTypeSymbol n when n.IsGenericType =>
            n.ConstructedFrom.ToDisplayString(SymbolDisplayFormat.CSharpErrorMessageFormat.WithGenericsOptions(SymbolDisplayGenericsOptions.None))
            + "<" + string.Join(",", n.TypeArguments.Select(TypeKey)) + ">",
        _ => t.WithNullableAnnotation(NullableAnnotation.NotAnnotated).ToDisplayString(SymbolDisplayFormat.CSharpErrorMessageFormat),
    };

    private static bool IsTensorLike(Model m, ITypeSymbol t) =>
        t is INamedTypeSymbol n && n.IsGenericType
        && (Eq(n.ConstructedFrom, m.Tensor) || Eq(n.ConstructedFrom, m.Vector) || Eq(n.ConstructedFrom, m.Matrix));

    private static bool Eq(ISymbol? a, ISymbol? b) => SymbolEqualityComparer.Default.Equals(a, b);

    private static string Kind(Model m, IMethodSymbol method, ITypeSymbol t)
    {
        if (t is INamedTypeSymbol nt && nt.OriginalDefinition.SpecialType == SpecialType.System_Nullable_T)
        {
            var inner = Kind(m, method, nt.TypeArguments[0]);
            return inner == "Unsupported" ? inner : "Nullable" + inner;
        }
        if (t is ITypeParameterSymbol tp && tp.TypeParameterKind == TypeParameterKind.Method) return "Scalar";
        if (t.IsRefLikeType) return "Unsupported";
        if (t.TypeKind == TypeKind.Enum) return "Enum";
        switch (t.SpecialType)
        {
            case SpecialType.System_Boolean: case SpecialType.System_Int32: case SpecialType.System_Int64:
            case SpecialType.System_Single: case SpecialType.System_Double: case SpecialType.System_String:
                return "Primitive";
        }
        if (t is IArrayTypeSymbol arr)
        {
            if (arr.Rank != 1) return "Unsupported";
            var ek = Kind(m, method, arr.ElementType);
            return ek switch
            {
                "Primitive" => "PrimitiveArray",
                "Enum" => "PrimitiveArray",
                "Tensor" or "TensorFixed" => "TensorArray",
                _ => "Unsupported",
            };
        }
        if (t is INamedTypeSymbol n && n.IsGenericType)
        {
            if (Eq(n.ConstructedFrom, m.Tensor))
                return n.TypeArguments[0] is ITypeParameterSymbol ? "Tensor" : "TensorFixed";
            if (Eq(n.ConstructedFrom, m.Vector)) return "Vector";
            if (Eq(n.ConstructedFrom, m.Matrix)) return "Matrix";
            if (n.IsTupleType && n.TupleElements.All(e => Kind(m, method, e.Type) is "Tensor" or "TensorFixed"))
                return "TensorTuple";
            var name = n.ConstructedFrom.ToDisplayString();
            if ((name.StartsWith("System.Collections.Generic.IReadOnlyList") || name.StartsWith("System.Collections.Generic.IList")
                 || name.StartsWith("System.Collections.Generic.List") || name.StartsWith("System.Collections.Generic.IEnumerable"))
                && Kind(m, method, n.TypeArguments[0]) is "Tensor" or "TensorFixed")
                return "TensorList";
        }
        return "Unsupported";
    }

    private static string Lit(string s) => "\"" + s.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";

    private static string EmitCatalog(Model m)
    {
        var sb = new StringBuilder();
        sb.AppendLine("// <auto-generated/> by AiDotNet.Tensors.Analyzers.EngineSurfaceGenerator from IEngine.");
        sb.AppendLine("#nullable enable");
        sb.AppendLine("namespace AiDotNet.Tensors.Engines.Export;");
        sb.AppendLine("internal static partial class EngineOpCatalog");
        sb.AppendLine("{");
        sb.AppendLine($"    /// <summary>Hash of every traceable op id. Changes whenever the replayable IEngine surface changes.</summary>");
        sb.AppendLine($"    internal const string OpSetHash = {Lit(m.OpSetHash)};");
        sb.AppendLine("    internal static readonly EngineOpDescriptor[] Ops = new EngineOpDescriptor[]");
        sb.AppendLine("    {");
        foreach (var op in m.Ops)
        {
            var ps = string.Join(", ", op.Method.Parameters.Select((p, i) =>
                $"new EngineOpParameter({Lit(p.Name)}, EngineValueKind.{op.ParamKinds[i]}, {Lit(TypeKey(p.Type))}, {(p.HasExplicitDefaultValue ? "true" : "false")})"));
            sb.AppendLine($"        new EngineOpDescriptor({op.Index}, {Lit(op.Id)}, {Lit(op.Method.Name)}, {op.Method.TypeParameters.Length}, " +
                          $"EngineValueKind.{op.ReturnKind}, {(op.InPlace ? op.InPlaceParam : -1)}, " +
                          $"{(op.UntraceableReason is null ? "null" : Lit(op.UntraceableReason))}, new EngineOpParameter[] {{ {ps} }}),");
        }
        sb.AppendLine("    };");
        sb.AppendLine("}");
        return sb.ToString();
    }

    private static string ParamDecl(IParameterSymbol p) => RefPrefix(p) + p.Type.ToDisplayString(TypeFormat) + " @" + p.Name;

    private static string ArgUse(IParameterSymbol p) => RefPrefix(p) + "@" + p.Name;

    private static string TypeParams(IMethodSymbol method) =>
        method.TypeParameters.Length == 0 ? "" : "<" + string.Join(", ", method.TypeParameters.Select(t => t.Name)) + ">";

    private static string EmitTracingEngine(Model m)
    {
        var sb = new StringBuilder();
        var iface = m.Engine.ToDisplayString(TypeFormat);
        sb.AppendLine("// <auto-generated/> by AiDotNet.Tensors.Analyzers.EngineSurfaceGenerator from IEngine.");
        sb.AppendLine("#nullable enable");
        sb.AppendLine("#pragma warning disable CS0618, CS0612 // forwards obsolete members too");
        sb.AppendLine("namespace AiDotNet.Tensors.Engines.Export;");
        sb.AppendLine($"public sealed partial class TracingEngine : {iface}");
        sb.AppendLine("{");
        foreach (var p in m.Properties)
        {
            var decl = p.ContainingType.ToDisplayString(TypeFormat);
            var type = p.Type.ToDisplayString(TypeFormat);
            if (p.IsIndexer) continue;
            sb.Append($"    {type} {decl}.{p.Name} {{ ");
            if (p.GetMethod is not null) sb.Append($"get => (({decl})_inner).{p.Name}; ");
            if (p.SetMethod is not null) sb.Append($"set => (({decl})_inner).{p.Name} = value; ");
            sb.AppendLine("}");
        }
        foreach (var e in m.Events)
        {
            var decl = e.ContainingType.ToDisplayString(TypeFormat);
            sb.AppendLine($"    event {e.Type.ToDisplayString(TypeFormat)} {decl}.{e.Name} {{ add => (({decl})_inner).{e.Name} += value; remove => (({decl})_inner).{e.Name} -= value; }}");
        }
        foreach (var op in m.Ops)
        {
            var method = op.Method;
            var decl = op.Declaring.ToDisplayString(TypeFormat);
            var ret = method.ReturnsVoid ? "void" : method.ReturnType.ToDisplayString(TypeFormat);
            var tps = TypeParams(method);
            var pars = string.Join(", ", method.Parameters.Select(ParamDecl));
            var args = string.Join(", ", method.Parameters.Select(ArgUse));
            var call = $"(({decl})_inner).{method.Name}{tps}({args})";
            sb.AppendLine($"    {ret} {decl}.{method.Name}{tps}({pars})");
            sb.AppendLine("    {");
            // Out parameters must be assigned on every path, so the pass-through path also calls the inner engine.
            if (!op.Traceable)
            {
                sb.AppendLine($"        _recorder.Opaque({op.Index});");
                sb.AppendLine($"        {(method.ReturnsVoid ? "" : "return ")}{call};");
                sb.AppendLine("    }");
                continue;
            }
            sb.AppendLine($"        if (!_recorder.TryEnter()) {{ {(method.ReturnsVoid ? call + "; return;" : "return " + call + ";")} }}");
            sb.AppendLine("        try");
            sb.AppendLine("        {");
            var typeArgs = method.TypeParameters.Length == 0
                ? "global::System.Type.EmptyTypes"
                : "new global::System.Type[] { " + string.Join(", ", method.TypeParameters.Select(t => $"typeof({t.Name})")) + " }";
            var boxed = method.Parameters.Length == 0
                ? "global::System.Array.Empty<object?>()"
                : "new object?[] { " + string.Join(", ", method.Parameters.Select(p => p.RefKind == RefKind.Out ? "null" : "@" + p.Name)) + " }";
            sb.AppendLine($"            var __args = {boxed};");
            if (method.ReturnsVoid)
            {
                sb.AppendLine($"            _recorder.BeforeInPlace({op.Index}, __args);");
                sb.AppendLine($"            {call};");
                foreach (var (p, i) in method.Parameters.Select((p, i) => (p, i)).Where(x => x.p.RefKind == RefKind.Out))
                    sb.AppendLine($"            __args[{i}] = @{p.Name};");
                sb.AppendLine($"            _recorder.Record({op.Index}, {typeArgs}, __args, null);");
            }
            else
            {
                sb.AppendLine($"            var __result = {call};");
                foreach (var (p, i) in method.Parameters.Select((p, i) => (p, i)).Where(x => x.p.RefKind == RefKind.Out))
                    sb.AppendLine($"            __args[{i}] = @{p.Name};");
                sb.AppendLine($"            _recorder.Record({op.Index}, {typeArgs}, __args, __result);");
                sb.AppendLine("            return __result;");
            }
            sb.AppendLine("        }");
            sb.AppendLine("        finally { _recorder.Exit(); }");
            sb.AppendLine("    }");
        }
        sb.AppendLine("}");
        return sb.ToString();
    }

    // Typed replay: one generic class per element type, so each op is a direct interface call with no reflection.
    // Methods with no type parameter are called the same way from any instantiation; methods with more than one
    // type parameter fall back to the reflection binder, which only ever resolves catalog entries.
    private static string EmitInvoker(Model m)
    {
        var sb = new StringBuilder();
        var iface = m.Engine.ToDisplayString(TypeFormat);
        sb.AppendLine("// <auto-generated/> by AiDotNet.Tensors.Analyzers.EngineSurfaceGenerator from IEngine.");
        sb.AppendLine("#nullable enable");
        sb.AppendLine("#pragma warning disable CS0618, CS0612, CS8600, CS8601, CS8604, CS8605");
        sb.AppendLine("namespace AiDotNet.Tensors.Engines.Export;");
        sb.AppendLine("internal static class EngineOpInvoker<TElem>");
        sb.AppendLine("{");
        sb.AppendLine($"    internal static bool TryInvoke({iface} e, int op, object?[] a, out object? result)");
        sb.AppendLine("    {");
        sb.AppendLine("        result = null;");
        sb.AppendLine("        switch (op)");
        sb.AppendLine("        {");
        foreach (var op in m.Ops)
        {
            var method = op.Method;
            if (!op.Traceable || method.TypeParameters.Length > 1) continue;
            if (method.Parameters.Any(p => p.RefKind is RefKind.Ref or RefKind.In)) continue;
            var decl = op.Declaring.ToDisplayString(TypeFormat);
            string Subst(ITypeSymbol t) => method.TypeParameters.Length == 1
                ? System.Text.RegularExpressions.Regex.Replace(t.ToDisplayString(TypeFormat), @"\b" + method.TypeParameters[0].Name + @"\b", "TElem")
                : t.ToDisplayString(TypeFormat);
            var outDecls = new StringBuilder();
            var args = new List<string>();
            for (int i = 0; i < method.Parameters.Length; i++)
            {
                var p = method.Parameters[i];
                var t = Subst(p.Type);
                if (p.RefKind == RefKind.Out)
                {
                    outDecls.Append($"{t} __o{i}; ");
                    args.Add($"out __o{i}");
                }
                else args.Add($"global::AiDotNet.Tensors.Engines.Export.EngineArgCoercion.As<{t}>(a[{i}])");
            }
            var tps = method.TypeParameters.Length == 1 ? "<TElem>" : "";
            var call = $"(({decl})e).{method.Name}{tps}({string.Join(", ", args)})";
            var outs = string.Concat(method.Parameters.Select((p, i) => p.RefKind == RefKind.Out ? $" a[{i}] = __o{i};" : ""));
            sb.AppendLine($"            case {op.Index}: {{ {outDecls}{(method.ReturnsVoid ? call + ";" : "result = " + call + ";")}{outs} return true; }}");
        }
        sb.AppendLine("            default: return false;");
        sb.AppendLine("        }");
        sb.AppendLine("    }");
        sb.AppendLine("}");
        return sb.ToString();
    }

}
