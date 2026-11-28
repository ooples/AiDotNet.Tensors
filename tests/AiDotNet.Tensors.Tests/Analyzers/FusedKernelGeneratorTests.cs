#if NET10_0_OR_GREATER
using System;
using System.IO;
using System.Linq;
using AiDotNet.Tensors.Generators;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Xunit;

namespace AiDotNet.Tensors.Tests.Analyzers;

/// <summary>
/// The fused kernel generator on declarations it must refuse (with its own diagnostic, not broken output) and on ones
/// it must accept whatever their parameter or owner names.
/// </summary>
public sealed class FusedKernelGeneratorTests
{
    [Fact]
    public void ParameterNamedLikeAGeneratedLocal_StillCompiles()
    {
        var (diagnostics, output) = Run(Owner("Ns", "Kernels",
            """
            [FusedElementwiseKernel("k_add", "x0 + x1", inputCount: 2)]
            public static partial void Add(ReadOnlySpan<float> dst, ReadOnlySpan<float> length, Span<float> i);
            """));

        Assert.Empty(diagnostics);
        Assert.Empty(output.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error));
    }

    [Fact]
    public void SameTypeNameInTwoNamespaces_GetsTwoSources()
    {
        var (diagnostics, output) = Run(
            Owner("A", "Kernels", """
                [FusedElementwiseKernel("k_a", "x0 * 2.0", inputCount: 1)]
                public static partial void Twice(ReadOnlySpan<float> x, Span<float> destination);
                """),
            Owner("B", "Kernels", """
                [FusedElementwiseKernel("k_b", "x0 * 3.0", inputCount: 1)]
                public static partial void Thrice(ReadOnlySpan<float> x, Span<float> destination);
                """));

        Assert.Empty(diagnostics);
        Assert.Empty(output.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error));
    }

    [Theory]
    [InlineData("""[FusedElementwiseKernel("add-relu", "relu(x0 + x1)", inputCount: 2)]""", "TENSGEN001")]
    [InlineData("""[FusedElementwiseKernel("k_big", "x0 * 1e100", inputCount: 2)]""", "TENSGEN001")]
    public void InvalidKernelNameOrLiteral_IsAnExpressionError(string attribute, string id)
    {
        var (diagnostics, _) = Run(Owner("Ns", "Kernels",
            attribute + "\npublic static partial void Add(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> destination);"));

        Assert.Equal(id, Assert.Single(diagnostics).Id);
    }

    [Fact]
    public void RefParameter_IsASignatureError()
    {
        var (diagnostics, _) = Run(Owner("Ns", "Kernels",
            """
            [FusedElementwiseKernel("k_add", "x0 + x1", inputCount: 2)]
            public static partial void Add(ref ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> destination);
            """));

        Assert.Equal("TENSGEN002", Assert.Single(diagnostics).Id);
    }

    [Fact]
    public void NestedOwner_IsASignatureError()
    {
        var (diagnostics, _) = Run(
            """
            namespace Ns
            {
                internal static partial class Outer
                {
                    internal static partial class Kernels
                    {
                        [FusedElementwiseKernel("k_add", "x0 + x1", inputCount: 2)]
                        public static partial void Add(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> destination);
                    }
                }
            }
            """);

        Assert.Equal("TENSGEN002", Assert.Single(diagnostics).Id);
    }

    [Fact]
    public void Overloads_AreRefusedTogether()
    {
        var (diagnostics, _) = Run(Owner("Ns", "Kernels",
            """
            [FusedElementwiseKernel("k_one", "x0 * 2.0", inputCount: 1)]
            public static partial void Run(ReadOnlySpan<float> a, Span<float> destination);
            [FusedElementwiseKernel("k_two", "x0 + x1", inputCount: 2)]
            public static partial void Run(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> destination);
            """));

        Assert.Equal(2, diagnostics.Length);
        Assert.All(diagnostics, d => Assert.Equal("TENSGEN003", d.Id));
    }

    private static string Owner(string ns, string type, string members) => $$"""
        namespace {{ns}}
        {
            internal static partial class {{type}}
            {
        {{members}}
            }
        }
        """;

    private static (Diagnostic[] Diagnostics, Compilation Output) Run(params string[] sources)
    {
        const string Usings = "using System;\nusing AiDotNet.Tensors.Engines.Compilation.GeneratedKernels;\n";
        var references = ((string?)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES") ?? "")
            .Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries)
            .Where(p => Path.GetFileName(p).StartsWith("System.", StringComparison.Ordinal)
                        || Path.GetFileName(p) is "netstandard.dll" or "mscorlib.dll")
            .Select(p => (MetadataReference)MetadataReference.CreateFromFile(p))
            .Append(MetadataReference.CreateFromFile(typeof(AiDotNet.Tensors.LinearAlgebra.Tensor<float>).Assembly.Location));
        // Named like this test assembly, which AiDotNet.Tensors grants its internals (the generated code reads them).
        var compilation = CSharpCompilation.Create("AiDotNet.Tensors.Tests",
            sources.Select(s => CSharpSyntaxTree.ParseText(Usings + s)), references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary, allowUnsafe: true,
                nullableContextOptions: NullableContextOptions.Enable));
        CSharpGeneratorDriver.Create(new FusedKernelGenerator())
            .RunGeneratorsAndUpdateCompilation(compilation, out var output, out var diagnostics);
        return (diagnostics.ToArray(), output);
    }
}
#endif
