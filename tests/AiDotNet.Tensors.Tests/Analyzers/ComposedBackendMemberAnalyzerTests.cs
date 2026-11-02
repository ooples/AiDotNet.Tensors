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
/// TENS0008: a backend that implements an interface member only by calling two or more of its own other interface members
/// (a fused AddRelu as Add then Relu) fails the build. Each case compiles six stand-in backends that implement AddRelu with
/// one kernel launch, except HipBackend, whose AddRelu body is the case under test.
/// </summary>
public sealed class ComposedBackendMemberAnalyzerTests
{
    private const string RuleId = "TENS0008";
    private const string StaleBaselineId = "TENS0003";

    private const string Fused = "Launch(\"add_relu\");";

    [Theory]
    [InlineData("Add(a, b, c, size); Relu(c, c, size);")]
    [InlineData("if (size <= 0) return; ThrowIfDisposed(); Add(a, b, c, size); Relu(c, c, size);")]
    [InlineData("var scratch = AllocateBuffer(size); Add(a, b, scratch, size); Relu(scratch, c, size);")]
    [InlineData("this.Add(a, b, c, size); this.Relu(c, c, size);")]
    [InlineData("Add(a, b, c, size); Relu(c, c, size); return;")]
    [InlineData("var s = AllocateBuffer(size); Add(a, b, s, size); Relu(s, c, size); s.Dispose();")]           // a released scratch
    [InlineData("var s = AllocateBuffer(size); try { Add(a, b, s, size); Relu(s, c, size); } finally { s.Dispose(); }")]
    [InlineData("var sum = AddNew(a, b, size); Relu(sum, c, size);")]                                         // a launch in a local
    public void Composition_IsReported(string hipBody)
    {
        var diagnostics = Analyze(hipBody, baseline: null);

        var composed = diagnostics.Where(d => d.Id == RuleId).ToList();
        Assert.Single(composed);
        Assert.Contains("HipBackend.AddRelu", composed[0].GetMessage(), StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(Fused)]
    [InlineData("if (TryFused(a, b, c, size)) return; Add(a, b, c, size); Relu(c, c, size);")] // a real kernel first
    [InlineData("Add(a, b, c, size);")]                                                      // one call is a delegation
    [InlineData("Add(a, b, c, size); Launch(\"relu\");")]                                     // a launch of its own
    [InlineData("var s = AllocateBuffer(size); Add(a, b, s, size); s.Dispose();")]            // allocation is not a launch
    [InlineData("var s = AllocateBuffer(size); try { Add(a, b, s, size); Launch(\"relu\"); } finally { s.Dispose(); }")]
    public void NonComposition_IsNotReported(string hipBody)
    {
        var diagnostics = Analyze(hipBody, baseline: null);

        Assert.DoesNotContain(diagnostics, d => d.Id == RuleId);
    }

    [Fact]
    public void BaselinedComposition_IsSuppressed_AndTheEntryGoesStaleOnceFixed()
    {
        const string entry = "composed HipBackend.AddRelu #1";

        var baselined = Analyze("Add(a, b, c, size); Relu(c, c, size);", entry);
        Assert.DoesNotContain(baselined, d => d.Id == RuleId || d.Id == StaleBaselineId);

        var fixedBody = Analyze(Fused, entry);
        Assert.Contains(fixedBody, d => d.Id == StaleBaselineId && d.GetMessage().Contains(entry, StringComparison.Ordinal));
    }

    private static ImmutableArray<Diagnostic> Analyze(string hipBody, string? baseline)
    {
        string[] backends = { "CUDA.CudaBackend", "HIP.HipBackend", "Metal.MetalBackend", "OpenCL.OpenClBackend",
            "Vulkan.VulkanBackend", "WebGpu.WebGpuBackend" };
        var source = new System.Text.StringBuilder(@"
namespace AiDotNet.Tensors.Engines.DirectGpu
{
    public interface IGpuBuffer : System.IDisposable { }
    public interface IDirectGpuBackend
    {
        void Add(IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int size);
        void Relu(IGpuBuffer a, IGpuBuffer b, int size);
        void AddRelu(IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int size);
        IGpuBuffer AllocateBuffer(int size);
        IGpuBuffer AddNew(IGpuBuffer a, IGpuBuffer b, int size);
    }
}
");
        foreach (string qualified in backends)
        {
            string ns = qualified.Substring(0, qualified.IndexOf('.')), name = qualified.Substring(qualified.IndexOf('.') + 1);
            string body = name == "HipBackend" ? hipBody : Fused;
            source.Append($@"
namespace AiDotNet.Tensors.Engines.DirectGpu.{ns}
{{
    public sealed class {name} : IDirectGpuBackend
    {{
        public void Add(IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int size) {{ Launch(""add""); }}
        public void Relu(IGpuBuffer a, IGpuBuffer b, int size) {{ Launch(""relu""); }}
        public void AddRelu(IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int size) {{ {body} }}
        public IGpuBuffer AllocateBuffer(int size) {{ Launch(""alloc""); return null; }}
        public IGpuBuffer AddNew(IGpuBuffer a, IGpuBuffer b, int size) {{ Launch(""add""); return null; }}
        private bool TryFused(IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int size) {{ Launch(""fused""); return true; }}
        private void ThrowIfDisposed() {{ }}
        private void Launch(string kernel) {{ }}
    }}
}}
");
        }

        var references = ((string?)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES") ?? "")
            .Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries)
            .Where(p => Path.GetFileName(p).StartsWith("System.", StringComparison.Ordinal)
                        || Path.GetFileName(p) is "netstandard.dll" or "mscorlib.dll")
            .Select(p => MetadataReference.CreateFromFile(p));
        var compilation = CSharpCompilation.Create("ComposedRuleProbe",
            new[] { CSharpSyntaxTree.ParseText(source.ToString()) }, references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary, nullableContextOptions: NullableContextOptions.Disable));
        Assert.Empty(compilation.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error));

        var additional = baseline is null
            ? ImmutableArray<AdditionalText>.Empty
            : ImmutableArray.Create<AdditionalText>(new InMemoryText("BackendParityBaseline.txt", baseline));
        return compilation
            .WithAnalyzers(ImmutableArray.Create<DiagnosticAnalyzer>(new BackendParityAnalyzer()), new AnalyzerOptions(additional))
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
