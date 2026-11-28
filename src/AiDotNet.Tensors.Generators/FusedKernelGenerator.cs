// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace AiDotNet.Tensors.Generators;

/// <summary>
/// Compile-time Roslyn Incremental Source Generator for declarative fused elementwise and row-reduction tensor kernels.
/// Emits zero-overhead vectorized CPU (<c>Vector256&lt;float&gt;</c> / <c>Vector&lt;float&gt;</c>) methods and NVRTC CUDA C kernel
/// source constants at compile time without runtime string-building or JIT expression-tree compilation overhead.
/// </summary>
[Generator(LanguageNames.CSharp)]
public sealed class FusedKernelGenerator : IIncrementalGenerator
{
    private const string ElementwiseAttributeMetadataName =
        "AiDotNet.Tensors.Engines.Compilation.GeneratedKernels.FusedElementwiseKernelAttribute";

    private const string RowSuiteAttributeMetadataName =
        "AiDotNet.Tensors.Engines.Compilation.GeneratedKernels.FusedRowPatternSuiteAttribute";

    private static readonly DiagnosticDescriptor InvalidExpressionDiagnostic = new(
        "TENSGEN001",
        "Invalid fused kernel expression",
        "Fused kernel '{0}' has an invalid expression: {1}",
        "FusedKernelCodegen",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor InvalidSignatureDiagnostic = new(
        "TENSGEN002",
        "Invalid fused kernel method signature",
        "Fused kernel method '{0}' must be a static partial void method in a static partial class taking {1} ReadOnlySpan<float> input(s), {2} float scalar(s), and a trailing Span<float> destination",
        "FusedKernelCodegen",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor OverloadedMethodDiagnostic = new(
        "TENSGEN003",
        "Overloaded fused kernel method",
        "Fused kernel method '{0}' is overloaded in '{1}': each generated kernel needs its own method name, because its companion members (CudaKernelName, CudaSource, TileDescriptor) are named after the method",
        "FusedKernelCodegen",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var elementwiseSpecs = context.SyntaxProvider.ForAttributeWithMetadataName(
            ElementwiseAttributeMetadataName,
            predicate: static (node, _) => node is MethodDeclarationSyntax,
            transform: static (ctx, _) => InspectElementwiseMethod(ctx));

        context.RegisterSourceOutput(elementwiseSpecs.Collect(), static (spc, specs) =>
        {
            if (specs.IsDefaultOrEmpty)
                return;

            var validByOwner = new Dictionary<string, List<ElementwiseMethodSpec>>(StringComparer.Ordinal);
            foreach (var spec in specs)
            {
                if (spec is null)
                    continue;
                if (spec.ErrorMessage is not null)
                {
                    var descriptor = spec.IsSignatureError ? InvalidSignatureDiagnostic : InvalidExpressionDiagnostic;
                    var diag = spec.IsSignatureError
                        ? Diagnostic.Create(descriptor, spec.Location, spec.MethodName, spec.InputCount, spec.ScalarCount)
                        : Diagnostic.Create(descriptor, spec.Location, spec.MethodName, spec.ErrorMessage);
                    spc.ReportDiagnostic(diag);
                    continue;
                }

                string ownerKey = spec.NamespaceName + "|" + spec.ContainingTypeName;
                if (!validByOwner.TryGetValue(ownerKey, out var list))
                {
                    list = new List<ElementwiseMethodSpec>();
                    validByOwner[ownerKey] = list;
                }
                list.Add(spec);
            }

            foreach (var kvp in validByOwner.OrderBy(k => k.Key, StringComparer.Ordinal))
            {
                var methods = kvp.Value.OrderBy(m => m.MethodName, StringComparer.Ordinal).ToList();
                var overloaded = methods.GroupBy(m => m.MethodName, StringComparer.Ordinal).Where(g => g.Count() > 1).ToList();
                foreach (var group in overloaded)
                    foreach (var m in group)
                        spc.ReportDiagnostic(Diagnostic.Create(OverloadedMethodDiagnostic, m.Location, m.MethodName, m.ContainingTypeName));
                if (overloaded.Count > 0)
                    methods = methods.Where(m => overloaded.All(g => g.Key != m.MethodName)).ToList();
                if (methods.Count == 0)
                    continue;
                var first = methods[0];
                string hintName = HintName(first.NamespaceName, first.ContainingTypeName) + ".Elementwise.g.cs";
                spc.AddSource(hintName, EmitElementwiseClass(first.NamespaceName, first.ContainingTypeName, first.TypeAccessibility, methods));
            }
        });

        var rowSuites = context.SyntaxProvider.ForAttributeWithMetadataName(
            RowSuiteAttributeMetadataName,
            predicate: static (node, _) => node is ClassDeclarationSyntax,
            transform: static (ctx, _) => InspectRowSuiteClass(ctx));

        context.RegisterSourceOutput(rowSuites, static (spc, suite) =>
        {
            if (suite is null)
                return;
            string hintName = HintName(suite.NamespaceName, suite.TypeName) + ".RowPatterns.g.cs";
            spc.AddSource(hintName, FusedRowPatternTemplates.EmitRowPatternClass(suite.NamespaceName, suite.TypeName, suite.Accessibility));
        });
    }

    private static RowSuiteSpec? InspectRowSuiteClass(GeneratorAttributeSyntaxContext ctx)
    {
        if (ctx.TargetSymbol is not INamedTypeSymbol typeSymbol)
            return null;
        string ns = typeSymbol.ContainingNamespace.IsGlobalNamespace
            ? string.Empty
            : typeSymbol.ContainingNamespace.ToDisplayString();
        string accessibility = AccessibilityKeyword(typeSymbol.DeclaredAccessibility);
        return new RowSuiteSpec(ns, typeSymbol.Name, accessibility);
    }

    private static ElementwiseMethodSpec? InspectElementwiseMethod(GeneratorAttributeSyntaxContext ctx)
    {
        if (ctx.TargetSymbol is not IMethodSymbol method || ctx.Attributes.Length == 0)
            return null;

        var attr = ctx.Attributes[0];
        if (attr.ConstructorArguments.Length < 3)
            return null;

        string kernelName = attr.ConstructorArguments[0].Value as string ?? method.Name;
        string expression = attr.ConstructorArguments[1].Value as string ?? string.Empty;
        int inputCount = attr.ConstructorArguments[2].Value is int ic ? ic : 0;
        int scalarCount = attr.ConstructorArguments.Length >= 4 && attr.ConstructorArguments[3].Value is int sc ? sc : 0;

        var containingType = method.ContainingType;
        string ns = containingType.ContainingNamespace.IsGlobalNamespace
            ? string.Empty
            : containingType.ContainingNamespace.ToDisplayString();
        string typeAccess = AccessibilityKeyword(containingType.DeclaredAccessibility);
        string methodAccess = AccessibilityKeyword(method.DeclaredAccessibility);
        Location location = ctx.TargetNode.GetLocation();

        // Validate partial static signature and parameter types. A nested owner is refused: the class is emitted at
        // namespace level, so its implementation would land in an unrelated type.
        if (!method.IsStatic || !method.ReturnsVoid || !method.IsPartialDefinition || inputCount < 1 || inputCount > 4 || scalarCount < 0 || scalarCount > 3
            || containingType.ContainingType is not null || containingType.IsGenericType)
        {
            return ElementwiseMethodSpec.SignatureError(ns, containingType.Name, method.Name, inputCount, scalarCount, location);
        }

        int expectedParams = inputCount + scalarCount + 1;
        if (method.Parameters.Length != expectedParams)
        {
            return ElementwiseMethodSpec.SignatureError(ns, containingType.Name, method.Name, inputCount, scalarCount, location);
        }

        var paramNames = new string[expectedParams];
        for (int i = 0; i < expectedParams; i++)
        {
            var p = method.Parameters[i];
            paramNames[i] = p.Name;
            // The emitter writes by-value parameters only: a ref/in/out/params one would not match the declaration.
            if (p.RefKind != RefKind.None || p.IsParams)
                return ElementwiseMethodSpec.SignatureError(ns, containingType.Name, method.Name, inputCount, scalarCount, location);
            string typeDisplay = p.Type.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat);
            if (i < inputCount)
            {
                if (typeDisplay != "ReadOnlySpan<float>" && typeDisplay != "System.ReadOnlySpan<float>")
                    return ElementwiseMethodSpec.SignatureError(ns, containingType.Name, method.Name, inputCount, scalarCount, location);
            }
            else if (i < inputCount + scalarCount)
            {
                if (p.Type.SpecialType != SpecialType.System_Single)
                    return ElementwiseMethodSpec.SignatureError(ns, containingType.Name, method.Name, inputCount, scalarCount, location);
            }
            else
            {
                if (typeDisplay != "Span<float>" && typeDisplay != "System.Span<float>")
                    return ElementwiseMethodSpec.SignatureError(ns, containingType.Name, method.Name, inputCount, scalarCount, location);
            }
        }

        // The kernel name is a CUDA symbol and sits inside C# string literals: an identifier only.
        if (!IsCIdentifier(kernelName))
        {
            return ElementwiseMethodSpec.ExpressionError(ns, containingType.Name, method.Name, inputCount, scalarCount,
                $"kernel name '{kernelName}' is not a C identifier (letters, digits and '_', not starting with a digit)", location);
        }

        try
        {
            var ast = FusedExpressionParser.Parse(expression, inputCount, scalarCount);
            string avx0 = FusedExpressionParser.EmitAvx2Expr(ast, "i");
            string avx8 = FusedExpressionParser.EmitAvx2Expr(ast, "i + 8");
            string avx16 = FusedExpressionParser.EmitAvx2Expr(ast, "i + 16");
            string avx24 = FusedExpressionParser.EmitAvx2Expr(ast, "i + 24");
            string? portableVec = ast.SupportsPortableVector() ? FusedExpressionParser.EmitPortableVectorExpr(ast, "v") : null;
            string scalarExpr = FusedExpressionParser.EmitScalarCsExpr(ast, "i");
            string cudaSource = FusedExpressionParser.EmitCudaKernelSource(kernelName, ast, inputCount, scalarCount);
            string tileInit = FusedExpressionParser.EmitTileDescriptorInitializer(
                kernelName, ast, inputCount, scalarCount, method.Name + "CudaSource");

            return new ElementwiseMethodSpec(
                ns,
                containingType.Name,
                typeAccess,
                methodAccess,
                method.Name,
                kernelName,
                inputCount,
                scalarCount,
                paramNames,
                avx0,
                avx8,
                avx16,
                avx24,
                portableVec,
                scalarExpr,
                cudaSource,
                tileInit,
                null,
                false,
                location);
        }
        catch (Exception ex)
        {
            return ElementwiseMethodSpec.ExpressionError(ns, containingType.Name, method.Name, inputCount, scalarCount, ex.Message, location);
        }
    }

    private static string EmitElementwiseClass(
        string namespaceName,
        string typeName,
        string typeAccessibility,
        IReadOnlyList<ElementwiseMethodSpec> methods)
    {
        var sb = new StringBuilder(8192);
        sb.AppendLine("// <auto-generated />");
        sb.AppendLine("#nullable enable");
        sb.AppendLine("using System;");
        sb.AppendLine("using System.Runtime.CompilerServices;");
        sb.AppendLine("using System.Runtime.InteropServices;");
        sb.AppendLine("#if NET8_0_OR_GREATER");
        sb.AppendLine("using System.Runtime.Intrinsics;");
        sb.AppendLine("using System.Runtime.Intrinsics.X86;");
        sb.AppendLine("#endif");
        sb.AppendLine();
        if (!string.IsNullOrEmpty(namespaceName))
        {
            sb.Append("namespace ").Append(namespaceName).AppendLine(";");
            sb.AppendLine();
        }

        sb.Append(typeAccessibility).Append(" static partial class ").Append(typeName).AppendLine();
        sb.AppendLine("{");

        foreach (var m in methods)
        {
            EmitSingleElementwiseMethod(sb, m);
        }

        sb.AppendLine("}");
        return sb.ToString();
    }

    private static void EmitSingleElementwiseMethod(StringBuilder sb, ElementwiseMethodSpec m)
    {
        string destParam = m.ParamNames[m.InputCount + m.ScalarCount];

        sb.Append("    public const string ").Append(m.MethodName).Append("CudaKernelName = \"").Append(m.KernelName).AppendLine("\";");
        sb.Append("    private static string? _lazy").Append(m.MethodName).AppendLine("CudaSource;");
        sb.Append("    public static string ").Append(m.MethodName).Append("CudaSource => _lazy")
          .Append(m.MethodName).Append("CudaSource ??= global::AiDotNet.Tensors.Engines.Compilation.GeneratedKernels.GeneratedKernelUtf8.Decode(@\"")
          .Append(m.CudaSource.Replace("\"", "\"\"")).AppendLine("\"u8);");
        sb.Append("    internal static readonly global::AiDotNet.Tensors.Engines.Compilation.FusionTileIr.FusionTileDescriptor ")
          .Append(m.MethodName).Append("TileDescriptor = ").Append(m.TileDescriptorInitializer).AppendLine(";");
        sb.AppendLine();

        string kernel = "__" + m.MethodName + "_GeneratedKernel";
        sb.AppendLine("    [MethodImpl(MethodImplOptions.AggressiveInlining)]");
        sb.Append("    ").Append(m.MethodAccessibility).Append(" static partial void ").Append(m.MethodName).Append('(');
        for (int i = 0; i < m.InputCount; i++)
        {
            if (i > 0) sb.Append(", ");
            sb.Append("ReadOnlySpan<float> ").Append(m.ParamNames[i]);
        }
        for (int s = 0; s < m.ScalarCount; s++)
        {
            sb.Append(", float ").Append(m.ParamNames[m.InputCount + s]);
        }
        sb.Append(", Span<float> ").Append(destParam).AppendLine(")");
        sb.Append("        => ").Append(kernel).Append('(').Append(string.Join(", ", m.ParamNames)).AppendLine(");");
        sb.AppendLine();

        // The kernel's parameters are the canonical names x0..x{k-1}, s0..s{m-1}, dst the emitted expressions read.
        sb.AppendLine("    [MethodImpl(MethodImplOptions.AggressiveInlining)]");
        sb.Append("    private static void ").Append(kernel).Append('(');
        for (int i = 0; i < m.InputCount; i++)
        {
            if (i > 0) sb.Append(", ");
            sb.Append("ReadOnlySpan<float> x").Append(i);
        }
        for (int s = 0; s < m.ScalarCount; s++)
        {
            sb.Append(", float s").Append(s);
        }
        sb.AppendLine(", Span<float> dst)");
        sb.AppendLine("    {");
        sb.AppendLine("        int length = dst.Length;");
        for (int i = 0; i < m.InputCount; i++)
        {
            sb.Append("        if (x").Append(i).AppendLine(".Length < length) throw new ArgumentException(\"Input span is shorter than destination span.\");");
        }

        sb.AppendLine("        int i = 0;");
        sb.AppendLine("#if NET8_0_OR_GREATER");
        sb.AppendLine("        if (Avx.IsSupported && Avx2.IsSupported && length >= 32)");
        sb.AppendLine("        {");
        sb.AppendLine("            int simd32 = length & ~31;");
        sb.AppendLine("            for (; i < simd32; i += 32)");
        sb.AppendLine("            {");
        sb.Append("                global::AiDotNet.Tensors.Engines.Simd.SimdKernels.WriteVector256(dst, i, ").Append(m.AvxExpr0).AppendLine(");");
        sb.Append("                global::AiDotNet.Tensors.Engines.Simd.SimdKernels.WriteVector256(dst, i + 8, ").Append(m.AvxExpr8).AppendLine(");");
        sb.Append("                global::AiDotNet.Tensors.Engines.Simd.SimdKernels.WriteVector256(dst, i + 16, ").Append(m.AvxExpr16).AppendLine(");");
        sb.Append("                global::AiDotNet.Tensors.Engines.Simd.SimdKernels.WriteVector256(dst, i + 24, ").Append(m.AvxExpr24).AppendLine(");");
        sb.AppendLine("            }");
        sb.AppendLine("        }");
        sb.AppendLine("        if (Avx.IsSupported && Avx2.IsSupported && length - i >= 8)");
        sb.AppendLine("        {");
        sb.AppendLine("            int simd8 = i + ((length - i) & ~7);");
        sb.AppendLine("            for (; i < simd8; i += 8)");
        sb.AppendLine("            {");
        sb.Append("                global::AiDotNet.Tensors.Engines.Simd.SimdKernels.WriteVector256(dst, i, ").Append(m.AvxExpr0).AppendLine(");");
        sb.AppendLine("            }");
        sb.AppendLine("        }");
        sb.AppendLine("#endif");

        if (m.PortableVecExpr is not null)
        {
            sb.AppendLine("        if (i == 0 && global::System.Numerics.Vector.IsHardwareAccelerated && length >= global::System.Numerics.Vector<float>.Count)");
            sb.AppendLine("        {");
            sb.AppendLine("            int lanes = global::System.Numerics.Vector<float>.Count;");
            sb.AppendLine("            int vecElems = length - (length % lanes);");
            for (int inp = 0; inp < m.InputCount; inp++)
            {
                sb.Append("            var vx").Append(inp).Append(" = MemoryMarshal.Cast<float, global::System.Numerics.Vector<float>>(x").Append(inp).AppendLine(".Slice(0, vecElems));");
            }
            for (int sc = 0; sc < m.ScalarCount; sc++)
            {
                sb.Append("            var vs").Append(sc).Append(" = new global::System.Numerics.Vector<float>(s").Append(sc).AppendLine(");");
            }
            sb.AppendLine("            var vdst = MemoryMarshal.Cast<float, global::System.Numerics.Vector<float>>(dst.Slice(0, vecElems));");
            sb.AppendLine("            for (int v = 0; v < vdst.Length; v++)");
            sb.AppendLine("            {");
            sb.Append("                vdst[v] = ").Append(m.PortableVecExpr).AppendLine(";");
            sb.AppendLine("            }");
            sb.AppendLine("            i = vecElems;");
            sb.AppendLine("        }");
        }

        sb.AppendLine("        for (; i < length; i++)");
        sb.AppendLine("        {");
        sb.Append("            dst[i] = ").Append(m.ScalarExpr).AppendLine(";");
        sb.AppendLine("        }");
        sb.AppendLine("    }");
        sb.AppendLine();
    }

    private static string AccessibilityKeyword(Accessibility accessibility) => accessibility switch
    {
        Accessibility.Public => "public",
        Accessibility.Internal => "internal",
        Accessibility.Private => "private",
        _ => "internal"
    };

    /// <summary>
    /// A hint name unique per owner: the namespace and the type, with every character but an ASCII letter, digit or
    /// '.' written as '_' and its four hex digits. '_' is encoded too, so two distinct identities never meet.
    /// </summary>
    private static string HintName(string namespaceName, string typeName)
    {
        string identity = string.IsNullOrEmpty(namespaceName) ? typeName : namespaceName + "." + typeName;
        var sb = new StringBuilder(identity.Length + 8);
        foreach (char c in identity)
        {
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.')
                sb.Append(c);
            else
                sb.Append('_').Append(((int)c).ToString("X4", System.Globalization.CultureInfo.InvariantCulture));
        }
        return sb.ToString();
    }

    private static bool IsCIdentifier(string name)
    {
        if (string.IsNullOrEmpty(name)) return false;
        for (int i = 0; i < name.Length; i++)
        {
            char c = name[i];
            bool letter = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
            bool digit = c >= '0' && c <= '9';
            if (!letter && !(digit && i > 0)) return false;
        }
        return true;
    }

    private sealed class RowSuiteSpec : IEquatable<RowSuiteSpec>
    {
        public string NamespaceName { get; }
        public string TypeName { get; }
        public string Accessibility { get; }

        public RowSuiteSpec(string namespaceName, string typeName, string accessibility)
        {
            NamespaceName = namespaceName;
            TypeName = typeName;
            Accessibility = accessibility;
        }

        public bool Equals(RowSuiteSpec? other) =>
            other is not null &&
            NamespaceName == other.NamespaceName &&
            TypeName == other.TypeName &&
            Accessibility == other.Accessibility;

        public override bool Equals(object? obj) => obj is RowSuiteSpec other && Equals(other);

        public override int GetHashCode()
        {
            unchecked
            {
                int h = NamespaceName.GetHashCode();
                h = (h * 397) ^ TypeName.GetHashCode();
                h = (h * 397) ^ Accessibility.GetHashCode();
                return h;
            }
        }
    }

    private sealed class ElementwiseMethodSpec : IEquatable<ElementwiseMethodSpec>
    {
        public string NamespaceName { get; }
        public string ContainingTypeName { get; }
        public string TypeAccessibility { get; }
        public string MethodAccessibility { get; }
        public string MethodName { get; }
        public string KernelName { get; }
        public int InputCount { get; }
        public int ScalarCount { get; }
        public string[] ParamNames { get; }
        public string AvxExpr0 { get; }
        public string AvxExpr8 { get; }
        public string AvxExpr16 { get; }
        public string AvxExpr24 { get; }
        public string? PortableVecExpr { get; }
        public string ScalarExpr { get; }
        public string CudaSource { get; }
        public string TileDescriptorInitializer { get; }
        public string? ErrorMessage { get; }
        public bool IsSignatureError { get; }
        public Location Location { get; }

        public ElementwiseMethodSpec(
            string namespaceName,
            string containingTypeName,
            string typeAccessibility,
            string methodAccessibility,
            string methodName,
            string kernelName,
            int inputCount,
            int scalarCount,
            string[] paramNames,
            string avxExpr0,
            string avxExpr8,
            string avxExpr16,
            string avxExpr24,
            string? portableVecExpr,
            string scalarExpr,
            string cudaSource,
            string tileDescriptorInitializer,
            string? errorMessage,
            bool isSignatureError,
            Location location)
        {
            NamespaceName = namespaceName;
            ContainingTypeName = containingTypeName;
            TypeAccessibility = typeAccessibility;
            MethodAccessibility = methodAccessibility;
            MethodName = methodName;
            KernelName = kernelName;
            InputCount = inputCount;
            ScalarCount = scalarCount;
            ParamNames = paramNames;
            AvxExpr0 = avxExpr0;
            AvxExpr8 = avxExpr8;
            AvxExpr16 = avxExpr16;
            AvxExpr24 = avxExpr24;
            PortableVecExpr = portableVecExpr;
            ScalarExpr = scalarExpr;
            CudaSource = cudaSource;
            TileDescriptorInitializer = tileDescriptorInitializer;
            ErrorMessage = errorMessage;
            IsSignatureError = isSignatureError;
            Location = location;
        }

        public static ElementwiseMethodSpec SignatureError(
            string ns, string typeName, string methodName, int inputCount, int scalarCount, Location loc)
            => new(ns, typeName, "internal", "internal", methodName, string.Empty, inputCount, scalarCount,
                Array.Empty<string>(), string.Empty, string.Empty, string.Empty, string.Empty, null, string.Empty,
                string.Empty, string.Empty, "Invalid signature", true, loc);

        public static ElementwiseMethodSpec ExpressionError(
            string ns, string typeName, string methodName, int inputCount, int scalarCount, string error, Location loc)
            => new(ns, typeName, "internal", "internal", methodName, string.Empty, inputCount, scalarCount,
                Array.Empty<string>(), string.Empty, string.Empty, string.Empty, string.Empty, null, string.Empty,
                string.Empty, string.Empty, error, false, loc);

        public bool Equals(ElementwiseMethodSpec? other)
        {
            if (other is null) return false;
            return NamespaceName == other.NamespaceName &&
                   ContainingTypeName == other.ContainingTypeName &&
                   TypeAccessibility == other.TypeAccessibility &&
                   MethodAccessibility == other.MethodAccessibility &&
                   MethodName == other.MethodName &&
                   KernelName == other.KernelName &&
                   InputCount == other.InputCount &&
                   ScalarCount == other.ScalarCount &&
                   ParamNames.SequenceEqual(other.ParamNames) &&
                   CudaSource == other.CudaSource &&
                   ErrorMessage == other.ErrorMessage &&
                   IsSignatureError == other.IsSignatureError;
        }

        public override bool Equals(object? obj) => obj is ElementwiseMethodSpec other && Equals(other);

        public override int GetHashCode()
        {
            unchecked
            {
                int h = NamespaceName.GetHashCode();
                h = (h * 397) ^ ContainingTypeName.GetHashCode();
                h = (h * 397) ^ MethodName.GetHashCode();
                h = (h * 397) ^ KernelName.GetHashCode();
                h = (h * 397) ^ CudaSource.GetHashCode();
                return h;
            }
        }
    }
}
