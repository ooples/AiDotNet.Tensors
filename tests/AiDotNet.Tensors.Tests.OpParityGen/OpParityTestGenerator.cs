// Copyright (c) AiDotNet. All rights reserved.
// Roslyn incremental source generator (#775): emits one discoverable [SkippableFact] per
// tensor-returning IEngine op, each calling the runtime op-parity harness. The generated tests
// auto-sync with IEngine — add an op and it gets a parity test slot for free; ops without a
// registered spec emit a visible NEEDS-SPEC skip so full-surface coverage is auditable in the
// test explorer. This mirrors the ecosystem's source-generator approach; the actual input
// synthesis / tolerances live in the hand-tuned runtime registry (structured op args can't be
// invented from a signature alone).

using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Text;

namespace AiDotNet.Tensors.Tests.OpParityGen
{
    [Generator]
    public sealed class OpParityTestGenerator : IIncrementalGenerator
    {
        private const string EngineMetadataName = "AiDotNet.Tensors.Engines.IEngine";
        private const string CpuMixedResidencyAttributeMetadataName =
            "AiDotNet.Tensors.Engines.CpuMixedResidencyElementwiseAttribute";
        private static readonly DiagnosticDescriptor InvalidMixedResidencySignature = new DiagnosticDescriptor(
            "ADNTP001",
            "Invalid CPU mixed-residency contract",
            "Method '{0}' has CpuMixedResidencyElementwiseAttribute but is not a binary Tensor<T> operation",
            "AiDotNet.Tensors.Tests",
            DiagnosticSeverity.Error,
            isEnabledByDefault: true);
        private static readonly DiagnosticDescriptor DuplicateMixedResidencyContract = new DiagnosticDescriptor(
            "ADNTP002",
            "Duplicate CPU mixed-residency contract",
            "Method name '{0}' has more than one CpuMixedResidencyElementwiseAttribute contract",
            "AiDotNet.Tensors.Tests",
            DiagnosticSeverity.Error,
            isEnabledByDefault: true);

        public void Initialize(IncrementalGeneratorInitializationContext context)
        {
            var opNames = context.CompilationProvider.Select(static (compilation, _) => GetTensorReturningOps(compilation));
            context.RegisterSourceOutput(opNames, static (spc, names) => spc.AddSource("GeneratedOpParityTests.g.cs", Emit(names)));

            var mixedResidencyContracts = context.CompilationProvider.Select(
                static (compilation, _) => GetCpuMixedResidencyContracts(compilation));
            context.RegisterSourceOutput(mixedResidencyContracts, static (spc, result) =>
            {
                foreach (Diagnostic diagnostic in result.Diagnostics)
                    spc.ReportDiagnostic(diagnostic);
                spc.AddSource(
                    "GeneratedCpuMixedResidencyContractTests.g.cs",
                    EmitCpuMixedResidencyContracts(result.MethodNames));
            });
        }

        private static MixedResidencyGenerationResult GetCpuMixedResidencyContracts(Compilation compilation)
        {
            var engine = compilation.GetTypeByMetadataName(EngineMetadataName);
            var marker = compilation.GetTypeByMetadataName(CpuMixedResidencyAttributeMetadataName);
            if (engine is null || marker is null)
                return new MixedResidencyGenerationResult(
                    System.Array.Empty<string>(), System.Array.Empty<Diagnostic>());

            var names = new SortedSet<string>(System.StringComparer.Ordinal);
            var diagnostics = new List<Diagnostic>();
            foreach (IMethodSymbol method in engine.GetMembers().OfType<IMethodSymbol>()
                .Where(method => method.DeclaredAccessibility == Accessibility.Public &&
                                 method.MethodKind == MethodKind.Ordinary &&
                                 method.GetAttributes().Any(attribute =>
                                     SymbolEqualityComparer.Default.Equals(attribute.AttributeClass, marker)))
                .OrderBy(method => method.Name, System.StringComparer.Ordinal))
            {
                bool valid = method.TypeParameters.Length == 1 &&
                    method.Parameters.Length == 2 &&
                    method.Parameters.All(parameter =>
                        parameter.RefKind == RefKind.None &&
                        IsTensorOf(parameter.Type, method.TypeParameters[0])) &&
                    IsTensorOf(method.ReturnType, method.TypeParameters[0]);
                if (!valid)
                {
                    diagnostics.Add(Diagnostic.Create(
                        InvalidMixedResidencySignature,
                        method.Locations.FirstOrDefault() ?? Location.None,
                        method.Name));
                    continue;
                }

                if (!names.Add(method.Name))
                {
                    diagnostics.Add(Diagnostic.Create(
                        DuplicateMixedResidencyContract,
                        method.Locations.FirstOrDefault() ?? Location.None,
                        method.Name));
                }
            }

            return new MixedResidencyGenerationResult(names.ToList(), diagnostics);
        }

        private static bool IsTensorOf(ITypeSymbol type, ITypeParameterSymbol elementType) =>
            type is INamedTypeSymbol { IsGenericType: true, Name: "Tensor" } tensor &&
            tensor.TypeArguments.Length == 1 &&
            SymbolEqualityComparer.Default.Equals(tensor.TypeArguments[0], elementType);

        private static IReadOnlyList<string> GetTensorReturningOps(Compilation compilation)
        {
            var engine = compilation.GetTypeByMetadataName(EngineMetadataName);
            if (engine is null) return System.Array.Empty<string>();

            var names = new SortedSet<string>(System.StringComparer.Ordinal);
            foreach (var member in engine.GetMembers().OfType<IMethodSymbol>())
            {
                if (member.DeclaredAccessibility != Accessibility.Public) continue;
                if (member.MethodKind != MethodKind.Ordinary) continue;
                if (ContainsTensor(member.ReturnType) ||
                    member.Parameters.Any(parameter =>
                        parameter.RefKind == RefKind.Out && ContainsTensor(parameter.Type)))
                    names.Add(member.Name);
            }
            return names.ToList();
        }

        private static bool ContainsTensor(ITypeSymbol type)
        {
            if (type is IArrayTypeSymbol array)
                return ContainsTensor(array.ElementType);
            if (type is INamedTypeSymbol { IsGenericType: true, Name: "Tensor" })
                return true;
            if (type is INamedTypeSymbol { IsTupleType: true } tuple)
                return tuple.TupleElements.Any(element => ContainsTensor(element.Type));
            return false;
        }

        private static SourceText Emit(IReadOnlyList<string> opNames)
        {
            var sb = new StringBuilder();
            sb.AppendLine("// <auto-generated/> CPU-vs-GPU op-parity tests (#775). Do not edit.");
            sb.AppendLine("#if !NETFRAMEWORK");
            sb.AppendLine("using Xunit;");
            sb.AppendLine();
            sb.AppendLine("namespace AiDotNet.Tensors.Tests.Engines.OpParity");
            sb.AppendLine("{");
            sb.AppendLine("    /// <summary>Generated: one discoverable parity fact per tensor-returning IEngine op.</summary>");
            sb.AppendLine("    [Collection(\"OpParity\")]");
            sb.AppendLine("    public sealed partial class GeneratedOpParityTests");
            sb.AppendLine("    {");
            sb.AppendLine("        private readonly OpParityFixture _fx;");
            sb.AppendLine("        public GeneratedOpParityTests(OpParityFixture fx) => _fx = fx;");
            sb.AppendLine();
            foreach (var op in opNames)
            {
                var method = Sanitize(op);
                sb.AppendLine("        [SkippableFact]");
                sb.AppendLine($"        public void Parity_{method}() => GeneratedOpParitySupport.RunForwardByMethod(nameof(global::AiDotNet.Tensors.Engines.IEngine.{op}), _fx);");
            }
            sb.AppendLine("    }");
            sb.AppendLine("}");
            sb.AppendLine("#endif");
            return SourceText.From(sb.ToString(), Encoding.UTF8);
        }

        private static SourceText EmitCpuMixedResidencyContracts(IReadOnlyList<string> methodNames)
        {
            var sb = new StringBuilder();
            sb.AppendLine("// <auto-generated/> CPU mixed-residency elementwise tests. Do not edit.");
            sb.AppendLine("#if !NETFRAMEWORK");
            sb.AppendLine("using Xunit;");
            sb.AppendLine();
            sb.AppendLine("namespace AiDotNet.Tensors.Tests.Engines.OpParity");
            sb.AppendLine("{");
            sb.AppendLine("    /// <summary>Generated from semantic mixed-residency contracts on IEngine.</summary>");
            sb.AppendLine("    [Collection(\"OpParity\")]");
            sb.AppendLine("    public sealed class GeneratedCpuMixedResidencyContractTests");
            sb.AppendLine("    {");
            foreach (var methodName in methodNames)
            {
                string method = Sanitize(methodName);
                sb.AppendLine("        [SkippableFact]");
                sb.AppendLine($"        public void CpuConsumesMixedResidency_{method}() =>");
                sb.AppendLine("            global::AiDotNet.Tensors.Tests.LinearAlgebra.TensorCowInferenceReadPathTests.VerifyCpuConsumesMixedResidencyInputs(");
                sb.AppendLine($"                static (engine, left, right) => engine.{methodName}(left, right));");
            }
            sb.AppendLine("    }");
            sb.AppendLine("}");
            sb.AppendLine("#endif");
            return SourceText.From(sb.ToString(), Encoding.UTF8);
        }

        private static string Sanitize(string name)
        {
            var chars = name.ToCharArray();
            for (int i = 0; i < chars.Length; i++)
                if (!char.IsLetterOrDigit(chars[i]) && chars[i] != '_') chars[i] = '_';
            return new string(chars);
        }

        private sealed class MixedResidencyGenerationResult
        {
            internal MixedResidencyGenerationResult(
                IReadOnlyList<string> methodNames,
                IReadOnlyList<Diagnostic> diagnostics)
            {
                MethodNames = methodNames;
                Diagnostics = diagnostics;
            }

            internal IReadOnlyList<string> MethodNames { get; }
            internal IReadOnlyList<Diagnostic> Diagnostics { get; }
        }
    }
}
