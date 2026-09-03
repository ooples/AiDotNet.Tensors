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

        public void Initialize(IncrementalGeneratorInitializationContext context)
        {
            var opNames = context.CompilationProvider.Select(static (compilation, _) => GetTensorReturningOps(compilation));
            context.RegisterSourceOutput(opNames, static (spc, names) => spc.AddSource("GeneratedOpParityTests.g.cs", Emit(names)));

            var outputContracts = context.CompilationProvider.Select(static (compilation, _) => GetTensorOutputContracts(compilation));
            context.RegisterSourceOutput(outputContracts, static (spc, contracts) =>
                spc.AddSource("GeneratedTensorOutputContractTests.g.cs", EmitOutputContracts(contracts)));

            var captureContracts = context.CompilationProvider.Select(static (compilation, _) => GetGraphCaptureContracts(compilation));
            context.RegisterSourceOutput(captureContracts, static (spc, contracts) =>
                spc.AddSource("GeneratedGraphCaptureSignatureTests.g.cs", EmitGraphCaptureContracts(contracts)));
        }

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

        private static IReadOnlyList<OutputContractSpec> GetTensorOutputContracts(Compilation compilation)
        {
            var engine = compilation.GetTypeByMetadataName(EngineMetadataName);
            if (engine is null) return System.Array.Empty<OutputContractSpec>();

            var contracts = new List<OutputContractSpec>();
            foreach (var method in engine.GetMembers().OfType<IMethodSymbol>())
            {
                if (method.DeclaredAccessibility != Accessibility.Public ||
                    method.MethodKind != MethodKind.Ordinary)
                    continue;

                var tensorOutputs = new List<ITypeSymbol>();
                bool returnedContainer = CollectTensorElementTypes(method.ReturnType, tensorOutputs);
                int returnedTensorCount = tensorOutputs.Count;
                var outTensorElements = method.Parameters
                    .Where(parameter => parameter.RefKind == RefKind.Out)
                    .SelectMany(parameter =>
                    {
                        var elements = new List<ITypeSymbol>();
                        CollectTensorElementTypes(parameter.Type, elements);
                        return elements;
                    })
                    .ToList();
                tensorOutputs.AddRange(outTensorElements);

                bool returnedMultiple = returnedContainer || returnedTensorCount > 1;
                if (tensorOutputs.Count < 2 && !returnedMultiple)
                    continue;

                bool homogeneous = tensorOutputs.All(element => IsMethodElementType(method, element!));

                contracts.Add(new OutputContractSpec(
                    method.Name,
                    homogeneous,
                    BuildOverloadSuffix(method)));
            }

            return contracts
                .GroupBy(contract => contract.Name, System.StringComparer.Ordinal)
                .OrderBy(group => group.Key, System.StringComparer.Ordinal)
                .SelectMany(group =>
                {
                    var overloads = group.OrderBy(contract => contract.OverloadSuffix, System.StringComparer.Ordinal).ToList();
                    if (overloads.Count == 1)
                        return new[] { overloads[0].WithoutOverloadId() };
                    return overloads.Select(contract => contract.WithOverloadId(
                        Sanitize(contract.Name + "_" + contract.OverloadSuffix)));
                })
                .ToList();
        }

        private static IReadOnlyList<GraphCaptureContractSpec> GetGraphCaptureContracts(Compilation compilation)
        {
            var engine = compilation.GetTypeByMetadataName(EngineMetadataName);
            if (engine is null) return System.Array.Empty<GraphCaptureContractSpec>();

            var methods = engine.GetMembers().OfType<IMethodSymbol>()
                .Where(method => method.DeclaredAccessibility == Accessibility.Public &&
                                 method.MethodKind == MethodKind.Ordinary)
                .ToList();
            var overloadCounts = methods
                .GroupBy(method => method.Name, System.StringComparer.Ordinal)
                .ToDictionary(group => group.Key, group => group.Count(), System.StringComparer.Ordinal);

            var contracts = new List<GraphCaptureContractSpec>();
            foreach (var method in methods)
            {
                var inputElements = new List<ITypeSymbol>();
                foreach (var parameter in method.Parameters.Where(parameter => parameter.RefKind != RefKind.Out))
                    CollectTensorElementTypes(parameter.Type, inputElements);

                var outputElements = new List<ITypeSymbol>();
                CollectTensorElementTypes(method.ReturnType, outputElements);
                foreach (var parameter in method.Parameters.Where(parameter => parameter.RefKind == RefKind.Out))
                    CollectTensorElementTypes(parameter.Type, outputElements);

                if (inputElements.Count == 0 && outputElements.Count == 0)
                    continue;

                var allElements = inputElements.Concat(outputElements).ToList();
                var methodTypeParameters = allElements
                    .OfType<ITypeParameterSymbol>()
                    .Where(parameter => method.TypeParameters.Any(candidate =>
                        SymbolEqualityComparer.Default.Equals(candidate, parameter)))
                    .Distinct<ITypeParameterSymbol>(SymbolEqualityComparer.Default)
                    .ToList();

                GraphCaptureConstraintKind? constraint = null;
                if (methodTypeParameters.Count > 1)
                {
                    constraint = GraphCaptureConstraintKind.MixedElementTypes;
                }
                else if (methodTypeParameters.Count == 1)
                {
                    ITypeSymbol primary = methodTypeParameters[0];
                    if (inputElements.Any(element => !SymbolEqualityComparer.Default.Equals(element, primary)))
                        constraint = GraphCaptureConstraintKind.HeterogeneousInput;
                    else if (outputElements.Any(element => !SymbolEqualityComparer.Default.Equals(element, primary)))
                        constraint = GraphCaptureConstraintKind.HeterogeneousOutput;
                }

                if (constraint is null)
                    continue;

                string suffix = BuildGraphCaptureOverloadSuffix(method);
                string? overloadId = overloadCounts[method.Name] > 1
                    ? Sanitize(method.Name + "_" + suffix)
                    : null;
                contracts.Add(new GraphCaptureContractSpec(method.Name, constraint.Value, suffix, overloadId));
            }

            return contracts
                .OrderBy(contract => contract.Name, System.StringComparer.Ordinal)
                .ThenBy(contract => contract.OverloadSuffix, System.StringComparer.Ordinal)
                .ToList();
        }

        private static string BuildOverloadSuffix(IMethodSymbol method)
        {
            if (method.Parameters.Length == 0) return "NoParameters";
            return string.Join("_", method.Parameters.Select(parameter => TypeToken(parameter.Type)));
        }

        private static string BuildGraphCaptureOverloadSuffix(IMethodSymbol method)
        {
            if (method.Parameters.Length == 0) return "NoParameters";
            return string.Join("_", method.Parameters.Select(parameter =>
            {
                string prefix = parameter.RefKind == RefKind.Out ? "Out_" :
                    parameter.RefKind == RefKind.Ref ? "Ref_" :
                    parameter.RefKind == RefKind.In ? "In_" : string.Empty;
                return prefix + TypeToken(parameter.Type);
            }));
        }

        private static string TypeToken(ITypeSymbol type)
        {
            if (type is IArrayTypeSymbol array)
                return TypeToken(array.ElementType) + "Array";
            if (type is ITypeParameterSymbol parameter)
                return parameter.Name;
            if (type is INamedTypeSymbol named)
            {
                if (!named.IsGenericType) return named.Name;
                return named.Name + "_" + string.Join("_", named.TypeArguments.Select(TypeToken));
            }
            return type.Name;
        }

        private static bool ContainsTensor(ITypeSymbol type)
        {
            var elements = new List<ITypeSymbol>();
            CollectTensorElementTypes(type, elements);
            return elements.Count > 0;
        }

        /// <summary>
        /// Adds every tensor element type in a direct tensor, array, or tuple. Returns true when
        /// the declaration is a container that can carry multiple tensors even if it has one
        /// syntactic element type (for example Tensor&lt;T&gt;[]).
        /// </summary>
        private static bool CollectTensorElementTypes(ITypeSymbol type, List<ITypeSymbol> elements)
        {
            if (type is IArrayTypeSymbol array)
            {
                CollectTensorElementTypes(array.ElementType, elements);
                return elements.Count > 0;
            }

            if (type is INamedTypeSymbol { IsGenericType: true, Name: "Tensor" } tensor)
            {
                elements.Add(tensor.TypeArguments[0]);
                return false;
            }

            if (type is INamedTypeSymbol { IsTupleType: true } tuple)
            {
                int before = elements.Count;
                foreach (var element in tuple.TupleElements)
                    CollectTensorElementTypes(element.Type, elements);
                return elements.Count > before;
            }

            return false;
        }

        private static bool IsMethodElementType(IMethodSymbol method, ITypeSymbol element)
            => element is ITypeParameterSymbol parameter &&
               method.TypeParameters.Any(candidate => SymbolEqualityComparer.Default.Equals(candidate, parameter));

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
                sb.AppendLine($"        public void Parity_{method}() => GeneratedOpParitySupport.RunForwardByMethod(\"{op}\", _fx);");
            }
            sb.AppendLine("    }");
            sb.AppendLine("}");
            sb.AppendLine("#endif");
            return SourceText.From(sb.ToString(), Encoding.UTF8);
        }

        private static SourceText EmitOutputContracts(IReadOnlyList<OutputContractSpec> contracts)
        {
            var sb = new StringBuilder();
            sb.AppendLine("// <auto-generated/> Tensor output-contract tests. Do not edit.");
            sb.AppendLine("#if !NETFRAMEWORK");
            sb.AppendLine("using Xunit;");
            sb.AppendLine();
            sb.AppendLine("namespace AiDotNet.Tensors.Tests.Engines.OpParity");
            sb.AppendLine("{");
            sb.AppendLine("    /// <summary>Generated identity for methods whose multi-output overloads need distinct coverage.</summary>");
            sb.AppendLine("    public enum TensorOutputOverload");
            sb.AppendLine("    {");
            sb.AppendLine("        Unspecified,");
            foreach (var contract in contracts.Where(contract => contract.OverloadId is not null))
                sb.AppendLine($"        {contract.OverloadId},");
            sb.AppendLine("    }");
            sb.AppendLine();
            sb.AppendLine("    /// <summary>Generated from every IEngine signature with multiple tensor results.</summary>");
            sb.AppendLine("    public sealed class GeneratedTensorOutputContractTests");
            sb.AppendLine("    {");
            foreach (var contract in contracts)
            {
                string method = contract.OverloadId ?? Sanitize(contract.Name);
                string expectation = contract.IsHomogeneous
                    ? "TensorOutputContract.HomogeneousMultiple"
                    : "TensorOutputContract.HeterogeneousMultiple";
                string overload = contract.OverloadId is null
                    ? "TensorOutputOverload.Unspecified"
                    : $"TensorOutputOverload.{contract.OverloadId}";
                sb.AppendLine("        [Fact]");
                sb.AppendLine($"        public void OutputContract_{method}() => GeneratedOpParitySupport.VerifyTensorOutputContract(\"{contract.Name}\", {expectation}, {overload});");
            }
            sb.AppendLine("    }");
            sb.AppendLine("}");
            sb.AppendLine("#endif");
            return SourceText.From(sb.ToString(), Encoding.UTF8);
        }

        private static SourceText EmitGraphCaptureContracts(IReadOnlyList<GraphCaptureContractSpec> contracts)
        {
            var sb = new StringBuilder();
            sb.AppendLine("// <auto-generated/> Heterogeneous graph-capture signature tests. Do not edit.");
            sb.AppendLine("#if !NETFRAMEWORK");
            sb.AppendLine("using Xunit;");
            sb.AppendLine();
            sb.AppendLine("namespace AiDotNet.Tensors.Tests.Engines.OpParity");
            sb.AppendLine("{");
            sb.AppendLine("    /// <summary>Generated identity for heterogeneous IEngine overloads.</summary>");
            sb.AppendLine("    public enum GraphCaptureSignatureOverload");
            sb.AppendLine("    {");
            sb.AppendLine("        Unspecified,");
            foreach (var contract in contracts.Where(contract => contract.OverloadId is not null))
                sb.AppendLine($"        {contract.OverloadId},");
            sb.AppendLine("    }");
            sb.AppendLine();
            sb.AppendLine("    /// <summary>Generated from tensor element types in every public IEngine signature.</summary>");
            sb.AppendLine("    public sealed class GeneratedGraphCaptureSignatureTests");
            sb.AppendLine("    {");
            foreach (var contract in contracts)
            {
                string method = contract.OverloadId ?? Sanitize(contract.Name);
                string overload = contract.OverloadId is null
                    ? "GraphCaptureSignatureOverload.Unspecified"
                    : $"GraphCaptureSignatureOverload.{contract.OverloadId}";
                sb.AppendLine("        [Fact]");
                sb.AppendLine($"        public void CaptureContract_{method}() => GeneratedOpParitySupport.VerifyGraphCaptureSignature(\"{contract.Name}\", GraphCaptureSignatureConstraint.{contract.Constraint}, {overload});");
            }
            sb.AppendLine("    }");
            sb.AppendLine("}");
            sb.AppendLine("#endif");
            return SourceText.From(sb.ToString(), Encoding.UTF8);
        }

        private sealed class OutputContractSpec
        {
            internal OutputContractSpec(string name, bool isHomogeneous, string overloadSuffix, string? overloadId = null)
            {
                Name = name;
                IsHomogeneous = isHomogeneous;
                OverloadSuffix = overloadSuffix;
                OverloadId = overloadId;
            }

            internal string Name { get; }
            internal bool IsHomogeneous { get; }
            internal string OverloadSuffix { get; }
            internal string? OverloadId { get; }

            internal OutputContractSpec WithoutOverloadId() =>
                new OutputContractSpec(Name, IsHomogeneous, OverloadSuffix);

            internal OutputContractSpec WithOverloadId(string overloadId) =>
                new OutputContractSpec(Name, IsHomogeneous, OverloadSuffix, overloadId);
        }

        private enum GraphCaptureConstraintKind
        {
            HeterogeneousInput,
            HeterogeneousOutput,
            MixedElementTypes,
        }

        private sealed class GraphCaptureContractSpec
        {
            internal GraphCaptureContractSpec(
                string name,
                GraphCaptureConstraintKind constraint,
                string overloadSuffix,
                string? overloadId)
            {
                Name = name;
                Constraint = constraint;
                OverloadSuffix = overloadSuffix;
                OverloadId = overloadId;
            }

            internal string Name { get; }
            internal GraphCaptureConstraintKind Constraint { get; }
            internal string OverloadSuffix { get; }
            internal string? OverloadId { get; }
        }

        private static string Sanitize(string name)
        {
            var chars = name.ToCharArray();
            for (int i = 0; i < chars.Length; i++)
                if (!char.IsLetterOrDigit(chars[i]) && chars[i] != '_') chars[i] = '_';
            return new string(chars);
        }
    }
}
