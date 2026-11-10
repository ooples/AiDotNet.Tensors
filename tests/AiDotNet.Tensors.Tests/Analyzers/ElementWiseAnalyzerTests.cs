#if NET10_0_OR_GREATER
using System;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using System.Threading;
using AiDotNet.Tensors.Analyzers;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CodeAnalysis.Text;
using Xunit;

namespace AiDotNet.Tensors.Tests.Analyzers;

/// <summary>
/// TENS0010-0012: a tensor indexer or an INumericOperations call inside a loop (or a lambda that runs per element) fails
/// the build unless ElementWiseBaseline.txt counts it, and the baseline may only shrink. Each case compiles one member,
/// Probe.Ops.Body, against stand-in Tensor, INumericOperations and ScalarKernel types.
/// </summary>
public sealed class ElementWiseAnalyzerTests
{
    private const string Indexer = "TENS0010";
    private const string NumericOp = "TENS0011";
    private const string Shrink = "TENS0012";

    [Theory]
    [InlineData("double s = 0; for (int i = 0; i < t.Length; i++) s += t[i]; return s;")]
    [InlineData("double s = 0; int i = 0; while (i < t.Length) { s += t[i]; i++; } return s;")]
    [InlineData("foreach (int i in new[] { 0, 1 }) t[i] = 1; return 0;")]
    [InlineData("System.Threading.Tasks.Parallel.For(0, 4, i => { t[i] = 1; }); return 0;")]
    [InlineData("return System.Linq.Enumerable.Range(0, 4).Sum(i => t[i]);")]
    [InlineData("double s = 0; for (int i = 0; i < 4; i++) { System.Func<double> f = () => t[i]; s += f(); } return s;")]
    public void IndexerInALoop_IsReported(string body)
    {
        var diagnostics = Analyze(body);

        Assert.Contains(diagnostics, d => d.Id == Indexer);
        Assert.All(diagnostics.Where(d => d.Id == Indexer),
            d => Assert.Contains("Probe.Ops.Body(", d.GetMessage(), StringComparison.Ordinal));
    }

    [Fact]
    public void NumericOperationInALoop_IsReported()
    {
        var diagnostics = Analyze("double s = 0; for (int i = 0; i < 4; i++) s = ops.Add(s, 1); return s;");

        Assert.Single(diagnostics, d => d.Id == NumericOp);
        Assert.DoesNotContain(diagnostics, d => d.Id == Indexer);
    }

    [Theory]
    [InlineData("return t[0];")]                                                         // no loop
    [InlineData("System.Func<double> f = () => t[0]; return f();")]                       // a lambda that runs once
    [InlineData("return System.Linq.Enumerable.Range(0, 4).Concat(new[] { 5 }).Count() + t[0];")]
    [InlineData("var a = new double[4]; double s = 0; for (int i = 0; i < 4; i++) s += a[i]; return s;")] // an array
    [InlineData("double s = 0; for (int i = 0; i < 4; i++) s += other[i]; return s;")]   // an indexer of another type
    [InlineData("return ops.Add(1, 2);")]
    public void ScalarAccessOutsideALoop_IsNotReported(string body)
    {
        Assert.Empty(Analyze(body));
    }

    [Fact]
    public void ScalarKernelMember_IsExempt()
    {
        const string body = "double s = 0; for (int i = 0; i < t.Length; i++) s = ops.Add(s, t[i]); return s;";

        Assert.NotEmpty(Analyze(body));
        Assert.Empty(Analyze(body, attribute: "[AiDotNet.Tensors.Helpers.ScalarKernel(\"a reference kernel\")]"));
    }

    // The attribute's constructor rejects a blank reason only under reflection, so the analyzer must: a blank reason,
    // or an attribute of the same name from another namespace, leaves every occurrence reported.
    [Theory]
    [InlineData("[AiDotNet.Tensors.Helpers.ScalarKernel(\"\")]")]
    [InlineData("[AiDotNet.Tensors.Helpers.ScalarKernel(\"   \")]")]
    [InlineData("[Elsewhere.ScalarKernel(\"a reference kernel\")]")]
    public void ScalarKernelWithoutAReasonOrFromElsewhere_IsNotExempt(string attribute)
    {
        const string body = "double s = 0; for (int i = 0; i < t.Length; i++) s = ops.Add(s, t[i]); return s;";

        Assert.Equal(Analyze(body).Length, Analyze(body, attribute: attribute).Length);
    }

    [Fact]
    public void GateRunsOnlyOnTheNewestFramework()
    {
        Assert.Empty(Analyze("double s = 0; for (int i = 0; i < t.Length; i++) s += t[i]; return s;", net10: false));
    }

    [Fact]
    public void Baseline_SuppressesCountedOccurrences_AndReportsAMemberThatGrows()
    {
        const string two = "double s = 0; for (int i = 0; i < t.Length; i++) s += t[i] * t[i]; return s;";
        string key = KeyOf(Analyze(two).First(d => d.Id == Indexer));

        Assert.Empty(Analyze(two, baseline: $"# inventory\n\n2\t{key}\n"));

        // A third occurrence reports every one in the member, so the new one is among them.
        var grown = Analyze("double s = 0; for (int i = 0; i < t.Length; i++) s += t[i] * t[i] + t[0]; return s;",
            baseline: $"2\t{key}\n");
        Assert.Equal(3, grown.Count(d => d.Id == Indexer));
    }

    [Fact]
    public void Baseline_MayOnlyShrink()
    {
        const string one = "double s = 0; for (int i = 0; i < t.Length; i++) s += t[i]; return s;";
        string key = KeyOf(Analyze(one).Single(d => d.Id == Indexer));

        var lower = Assert.Single(Analyze(one, baseline: $"3\t{key}\n"));
        Assert.Equal(Shrink, lower.Id);
        Assert.Contains("lower it to 1", lower.GetMessage(), StringComparison.Ordinal);

        var gone = Assert.Single(Analyze("return t[0];", baseline: $"1\t{key}\n"));
        Assert.Equal(Shrink, gone.Id);
        Assert.Contains("remove its line", gone.GetMessage(), StringComparison.Ordinal);
    }

    [Fact]
    public void Baseline_IgnoresMalformedLines()
    {
        const string one = "double s = 0; for (int i = 0; i < t.Length; i++) s += t[i]; return s;";
        string key = KeyOf(Analyze(one).Single(d => d.Id == Indexer));

        // Neither line parses, so the occurrence stays uncounted and is still reported.
        Assert.Single(Analyze(one, baseline: $"one\t{key}\n\t{key}\n"), d => d.Id == Indexer);
    }

    /// <summary>The baseline key of an occurrence, "rule|member", read back from its message.</summary>
    private static string KeyOf(Diagnostic diagnostic)
    {
        string message = diagnostic.GetMessage();
        const string marker = " inside a loop in ";
        int start = message.IndexOf(marker, StringComparison.Ordinal) + marker.Length;
        int end = message.IndexOf(". Use ", start, StringComparison.Ordinal);
        return diagnostic.Id + "|" + message.Substring(start, end - start);
    }

    private static ImmutableArray<Diagnostic> Analyze(string body, string? baseline = null, string attribute = "",
        bool net10 = true)
    {
        string source = @"
namespace AiDotNet.Tensors.LinearAlgebra
{
    public class Tensor<T>
    {
        public T this[int index] { get { return default(T); } set { } }
        public int Length => 4;
    }
}
namespace AiDotNet.Tensors.Interfaces
{
    public interface INumericOperations<T> { T Add(T a, T b); }
}
namespace AiDotNet.Tensors.Helpers
{
    [System.AttributeUsage(System.AttributeTargets.All)]
    public sealed class ScalarKernelAttribute : System.Attribute { public ScalarKernelAttribute(string reason) { } }
}
namespace Elsewhere
{
    [System.AttributeUsage(System.AttributeTargets.All)]
    public sealed class ScalarKernelAttribute : System.Attribute { public ScalarKernelAttribute(string reason) { } }
}
namespace Probe
{
    using System.Linq;
    using AiDotNet.Tensors.Interfaces;
    using AiDotNet.Tensors.LinearAlgebra;

    public sealed class Other { public double this[int index] => index; }

    public static class Ops
    {
        " + attribute + @"
        public static double Body(Tensor<double> t, INumericOperations<double> ops, Other other) { " + body + @" }
    }
}
";
        var parse = CSharpParseOptions.Default.WithPreprocessorSymbols(net10 ? new[] { "NET10_0_OR_GREATER" } : Array.Empty<string>());
        var references = ((string?)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES") ?? "")
            .Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries)
            .Where(p => Path.GetFileName(p).StartsWith("System.", StringComparison.Ordinal)
                        || Path.GetFileName(p) is "netstandard.dll" or "mscorlib.dll")
            .Select(p => MetadataReference.CreateFromFile(p));
        var compilation = CSharpCompilation.Create("ElementWiseRuleProbe",
            new[] { CSharpSyntaxTree.ParseText(source, parse) }, references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary, nullableContextOptions: NullableContextOptions.Disable));
        Assert.Empty(compilation.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error));

        var additional = baseline is null
            ? ImmutableArray<AdditionalText>.Empty
            : ImmutableArray.Create<AdditionalText>(new InMemoryText("ElementWiseBaseline.txt", baseline));
        return compilation
            .WithAnalyzers(ImmutableArray.Create<DiagnosticAnalyzer>(new ElementWiseAnalyzer()), new AnalyzerOptions(additional))
            .GetAnalyzerDiagnosticsAsync().GetAwaiter().GetResult();
    }

    private sealed class InMemoryText : AdditionalText
    {
        private readonly SourceText _text;

        public InMemoryText(string path, string text)
        {
            Path = path;
            _text = SourceText.From(text);
        }

        public override string Path { get; }

        public override SourceText GetText(CancellationToken cancellationToken = default) => _text;
    }
}
#endif
