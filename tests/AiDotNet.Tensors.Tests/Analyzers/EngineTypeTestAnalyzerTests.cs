#if NET10_0_OR_GREATER
using System;
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
/// TENS0020: a type test of an IEngine-typed expression against anything but IEngine (is, as, casts, patterns in is and
/// switch, GetType comparisons) is reported, because a wrapper engine installed as AiDotNetEngine.Current would take
/// the wrong branch. Each case compiles one member, Probe.Ops.Body, against stand-in engine types.
/// </summary>
public sealed class EngineTypeTestAnalyzerTests
{
    private const string Rule = "TENS0020";

    [Theory]
    [InlineData("return e is CpuEngine;")]
    [InlineData("return e is not CpuEngine;")]
    [InlineData("return e is DirectGpuTensorEngine gpu && gpu.Gpu;")]
    [InlineData("return e is not DirectGpuTensorEngine gpu || gpu.Gpu;")]
    [InlineData("return e is DirectGpuTensorEngine { Gpu: true };")]
    [InlineData("return e is ITensorLevelEngine;")]
    [InlineData("return e is IDisposable;")]
    [InlineData("return (e as CpuEngine) != null;")]
    [InlineData("return ((CpuEngine)e).Cpu;")]
    [InlineData("return ((ITensorLevelEngine)e) != null;")]
    [InlineData("return e.GetType() == typeof(CpuEngine);")]
    [InlineData("return typeof(CpuEngine) != e.GetType();")]
    [InlineData("return typeof(CpuEngine).IsAssignableFrom(e.GetType());")]
    [InlineData("return AiDotNetEngine.Current is DirectGpuTensorEngine;")]
    [InlineData("switch (e) { case DirectGpuTensorEngine g: return g.Gpu; default: return false; }")]
    [InlineData("return e switch { CpuEngine => true, _ => false };")]
    [InlineData("return h is { Engine: CpuEngine };")]
    public void DirectTypeTestOfAnEngine_IsReported(string body)
    {
        var diagnostic = Assert.Single(Analyze(body));
        Assert.Equal(Rule, diagnostic.Id);
        Assert.Equal(DiagnosticSeverity.Warning, diagnostic.Severity);
    }

    [Theory]
    [InlineData("return e.Is<CpuEngine>();")]
    [InlineData("return e.TryGet<DirectGpuTensorEngine>(out var gpu) && gpu.Gpu;")]
    [InlineData("return e.As<CpuEngine>() != null;")]
    [InlineData("return e.Cast<CpuEngine>().Cpu;")]
    [InlineData("return e.Unwrap() is CpuEngine;")]                        // Unwrap() is never a wrapper
    [InlineData("return e.Unwrap().GetType() == typeof(CpuEngine);")]
    [InlineData("return e is IEngine;")]
    [InlineData("return e is IEngineWrapper;")]
    [InlineData("return e is object;")]
    [InlineData("return e is null || e is not null || e is { };")]
    [InlineData("return e is var x && x != null;")]
    [InlineData("return (object)e != null;")]
    [InlineData("return e.GetType().Name.Length > 0;")]                     // GetType without a comparison
    [InlineData("return c is DirectGpuTensorEngine;")]                      // a concrete engine cannot be a wrapper
    [InlineData("return t is DirectGpuTensorEngine;")]                      // nor can another engine interface
    [InlineData("object o = e; return o is CpuEngine;")]                    // not IEngine-typed
    public void WrapperAwareOrUnrelatedTest_IsNotReported(string body)
    {
        Assert.Empty(Analyze(body));
    }

    [Fact]
    public void Message_SuggestsTheHelper()
    {
        Assert.Contains("engine.TryGet<DirectGpuTensorEngine>(out var x)",
            Assert.Single(Analyze("return e is DirectGpuTensorEngine gpu && gpu.Gpu;")).GetMessage(), StringComparison.Ordinal);
        Assert.Contains("engine.As<CpuEngine>()",
            Assert.Single(Analyze("return (e as CpuEngine) != null;")).GetMessage(), StringComparison.Ordinal);
        Assert.Contains("engine.Cast<CpuEngine>()",
            Assert.Single(Analyze("return ((CpuEngine)e).Cpu;")).GetMessage(), StringComparison.Ordinal);
        Assert.Contains("!engine.Is<CpuEngine>()",
            Assert.Single(Analyze("return e is not CpuEngine;")).GetMessage(), StringComparison.Ordinal);
    }

    [Fact]
    public void CompilationWithoutIEngine_IsIgnored()
    {
        Assert.Empty(Analyze("return true;", declareEngine: false));
    }

    private static ImmutableArray<Diagnostic> Analyze(string body, bool declareEngine = true)
    {
        string engines = declareEngine ? @"
namespace AiDotNet.Tensors.Engines
{
    public interface IEngine { bool SupportsGpu { get; } }
    public interface ITensorLevelEngine : IEngine { }
    public interface IEngineWrapper { IEngine Inner { get; } }
    public class CpuEngine : ITensorLevelEngine, System.IDisposable
    {
        public bool SupportsGpu => false;
        public bool Cpu => true;
        public void Dispose() { }
    }
    public class DirectGpuTensorEngine : CpuEngine { public bool Gpu => true; }
    public static class AiDotNetEngine { public static IEngine Current { get; set; } }
    public static class EngineWrapperExtensions
    {
        public static IEngine Unwrap(this IEngine engine) => engine;
        public static bool Is<T>(this IEngine engine) where T : class => engine is T;
        public static T As<T>(this IEngine engine) where T : class => engine as T;
        public static T Cast<T>(this IEngine engine) where T : class => (T)(object)engine;
        public static bool TryGet<T>(this IEngine engine, out T result) where T : class { result = engine as T; return result != null; }
    }
    public sealed class Holder { public IEngine Engine { get; set; } }
}
namespace Probe
{
    using System;
    using AiDotNet.Tensors.Engines;

    public static class Ops
    {
        public static bool Body(IEngine e, CpuEngine c, ITensorLevelEngine t, Holder h) { " + body + @" }
    }
}
" : @"namespace Probe { public static class Ops { public static bool Body() { " + body + @" } } }";
        var references = ((string?)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES") ?? "")
            .Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries)
            .Where(p => Path.GetFileName(p).StartsWith("System.", StringComparison.Ordinal)
                        || Path.GetFileName(p) is "netstandard.dll" or "mscorlib.dll")
            .Select(p => MetadataReference.CreateFromFile(p));
        var compilation = CSharpCompilation.Create("EngineTypeTestRuleProbe",
            new[] { CSharpSyntaxTree.ParseText(engines) }, references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary, nullableContextOptions: NullableContextOptions.Disable));
        Assert.Empty(compilation.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error));

        return compilation
            .WithAnalyzers(ImmutableArray.Create<DiagnosticAnalyzer>(new EngineTypeTestAnalyzer()))
            .GetAnalyzerDiagnosticsAsync().GetAwaiter().GetResult();
    }
}
#endif
