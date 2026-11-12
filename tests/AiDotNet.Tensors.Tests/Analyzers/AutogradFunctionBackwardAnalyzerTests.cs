#if NET10_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using AiDotNet.Tensors.Analyzers;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Diagnostics;
using Xunit;

namespace AiDotNet.Tensors.Tests.Analyzers;

/// <summary>
/// TENS0030: a concrete AutogradFunction&lt;T&gt; subclass that overrides neither Backward overload compiled and only
/// threw NotImplementedException when a backward pass reached it. Each case compiles a few classes against a stand-in
/// AutogradFunction&lt;T&gt; with the library's two virtual overloads; one case compiles against the real library.
/// </summary>
public sealed class AutogradFunctionBackwardAnalyzerTests
{
    private const string Rule = "TENS0030";

    private const string PerInput =
        "public override Tensor<float>[] Backward(AutogradContext ctx, Tensor<float> g) => new[] { g };";
    private const string Accumulating =
        "public override void Backward(AutogradContext ctx, Tensor<float> g, IEngine e, Dictionary<Tensor<float>, Tensor<float>> grads) { }";
    private const string ForwardOnly =
        "public override Tensor<float> Forward(AutogradContext ctx, params Tensor<float>[] inputs) => inputs[0];";

    [Fact]
    public void SubclassOverridingNeitherBackward_IsReported()
    {
        var diagnostic = Assert.Single(Analyze($"public sealed class Square : AutogradFunction<float> {{ {ForwardOnly} }}"));
        Assert.Equal(Rule, diagnostic.Id);
        Assert.Equal(DiagnosticSeverity.Warning, diagnostic.Severity);
        Assert.Contains("'Square'", diagnostic.GetMessage(), StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(PerInput)]
    [InlineData(Accumulating)]
    public void SubclassOverridingEitherBackward_IsNotReported(string backward)
    {
        Assert.Empty(Analyze($"public sealed class Square : AutogradFunction<float> {{ {ForwardOnly} {backward} }}"));
    }

    [Fact]
    public void AbstractSubclass_IsNotReported_ButItsConcreteSubclassWithoutBackwardIs()
    {
        var diagnostics = Analyze(
            $"public abstract class Base : AutogradFunction<float> {{ }} " +
            $"public sealed class Leaf : Base {{ {ForwardOnly} }}");
        var diagnostic = Assert.Single(diagnostics);
        Assert.Contains("'Leaf'", diagnostic.GetMessage(), StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(PerInput)]
    [InlineData(Accumulating)]
    public void BackwardInheritedFromAnIntermediateBase_IsNotReported(string backward)
    {
        Assert.Empty(Analyze(
            $"public abstract class Base : AutogradFunction<float> {{ {backward} }} " +
            $"public sealed class Leaf : Base {{ {ForwardOnly} }}"));
        Assert.Empty(Analyze(
            $"public class Mid : AutogradFunction<float> {{ {ForwardOnly} {backward} }} " +
            "public sealed class Leaf : Mid { }"));
    }

    [Fact]
    public void GenericSubclass_IsReported()
    {
        Assert.Single(Analyze(
            "public sealed class Identity<TElement> : AutogradFunction<TElement> " +
            "{ public override Tensor<TElement> Forward(AutogradContext ctx, params Tensor<TElement>[] inputs) => inputs[0]; }"));
    }

    [Fact]
    public void HidingBackwardWithNew_IsStillReported()
    {
        Assert.Single(Analyze(
            $"public sealed class Square : AutogradFunction<float> {{ {ForwardOnly} " +
            "public new Tensor<float>[] Backward(AutogradContext ctx, Tensor<float> g) => new[] { g }; }"));
    }

    [Fact]
    public void UnrelatedClassWithBackward_IsNotReported()
    {
        Assert.Empty(Analyze("public sealed class Other { public Tensor<float>[] Backward(AutogradContext ctx, Tensor<float> g) => null; }"));
    }

    [Fact]
    public void PartialSubclass_IsReportedOnce()
    {
        Assert.Single(Analyze(
            $"public sealed partial class Square : AutogradFunction<float> {{ {ForwardOnly} }} " +
            "public sealed partial class Square { }"));
    }

    // The rule finds the library's real AutogradFunction<T> by its metadata name.
    [Fact]
    public void AgainstTheRealLibrary_SubclassWithoutBackward_IsReported()
    {
        const string source = @"
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
namespace Probe
{
    public sealed class Missing : AutogradFunction<float>
    {
        public override Tensor<float> Forward(AutogradContext ctx, params Tensor<float>[] inputs) => inputs[0];
    }
    public sealed class Present : AutogradFunction<float>
    {
        public override Tensor<float> Forward(AutogradContext ctx, params Tensor<float>[] inputs) => inputs[0];
        public override Tensor<float>?[] Backward(AutogradContext ctx, Tensor<float> gradOutput) => new Tensor<float>?[] { gradOutput };
    }
}";
        var library = MetadataReference.CreateFromFile(typeof(AiDotNet.Tensors.Engines.Autodiff.AutogradFunction<>).Assembly.Location);
        var diagnostic = Assert.Single(Run(source, PlatformReferences().Append(library), NullableContextOptions.Enable));
        Assert.Equal(Rule, diagnostic.Id);
        Assert.Contains("'Missing'", diagnostic.GetMessage(), StringComparison.Ordinal);
    }

    [Fact]
    public void CompilationWithoutAutogradFunction_IsIgnored()
    {
        Assert.Empty(Run("namespace Probe { public sealed class Square { } }", PlatformReferences(), NullableContextOptions.Disable));
    }

    private static ImmutableArray<Diagnostic> Analyze(string declarations)
    {
        string source = @"
namespace AiDotNet.Tensors.Engines { public interface IEngine { } }
namespace AiDotNet.Tensors.LinearAlgebra { public class Tensor<T> { } }
namespace AiDotNet.Tensors.Engines.Autodiff
{
    using System.Collections.Generic;
    using AiDotNet.Tensors.LinearAlgebra;
    public sealed class AutogradContext { }
    public abstract class AutogradFunction<T>
    {
        public abstract Tensor<T> Forward(AutogradContext ctx, params Tensor<T>[] inputs);
        public virtual Tensor<T>[] Backward(AutogradContext ctx, Tensor<T> gradOutput) => throw new System.NotImplementedException();
        public virtual void Backward(AutogradContext ctx, Tensor<T> gradOutput, AiDotNet.Tensors.Engines.IEngine engine,
            Dictionary<Tensor<T>, Tensor<T>> grads) => Backward(ctx, gradOutput);
    }
}
namespace Probe
{
    using System.Collections.Generic;
    using AiDotNet.Tensors.Engines;
    using AiDotNet.Tensors.Engines.Autodiff;
    using AiDotNet.Tensors.LinearAlgebra;
    " + declarations + @"
}
";
        return Run(source, PlatformReferences(), NullableContextOptions.Disable);
    }

    private static ImmutableArray<Diagnostic> Run(string source, IEnumerable<MetadataReference> references, NullableContextOptions nullable)
    {
        var compilation = CSharpCompilation.Create("AutogradFunctionRuleProbe",
            new[] { CSharpSyntaxTree.ParseText(source) }, references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary, nullableContextOptions: nullable));
        Assert.Empty(compilation.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error));
        return compilation
            .WithAnalyzers(ImmutableArray.Create<DiagnosticAnalyzer>(new AutogradFunctionBackwardAnalyzer()))
            .GetAnalyzerDiagnosticsAsync().GetAwaiter().GetResult();
    }

    private static IEnumerable<MetadataReference> PlatformReferences() =>
        ((string?)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES") ?? "")
            .Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries)
            .Where(p => Path.GetFileName(p).StartsWith("System.", StringComparison.Ordinal)
                        || Path.GetFileName(p) is "netstandard.dll" or "mscorlib.dll")
            .Select(p => (MetadataReference)MetadataReference.CreateFromFile(p))
            .ToList();
}
#endif
