// Copyright (c) CrowdTrainer LLC. All rights reserved.
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
/// TENS0040 and TENS0041 against stand-in engine and tensor types. The reported cases are the shapes of the defects the
/// rules were written for: <c>TensorMaxValue</c> returning a host scalar with no trace boundary (frozen into a training
/// plan, a lazy operand read uninitialised), and a convolution reading its streaming weight through
/// <c>AsFloatMemory(kernel.DataUnmarked)</c> (the writable accessor throws for a quantized streaming store).
/// </summary>
public sealed class EngineGraphSafetyAnalyzerTests
{
    private const string Stubs = @"
namespace AiDotNet.Tensors.LinearAlgebra
{
    public abstract class TensorBase<T>
    {
        internal System.Memory<T> DataUnmarked => default;
        internal System.ReadOnlyMemory<T> ReadOnlyDataUnmarked => default;
    }
    public sealed class Tensor<T> : TensorBase<T> { public int Length => 0; }
}
namespace AiDotNet.Tensors.Helpers
{
    [System.AttributeUsage(System.AttributeTargets.Method)]
    internal sealed class HostValueFromMetadataAttribute : System.Attribute { }
}
namespace AiDotNet.Tensors.Engines
{
    using System;
    using AiDotNet.Tensors.Helpers;
    using AiDotNet.Tensors.LinearAlgebra;

    internal static class GraphMode
    {
        internal static void ThrowIfInferenceUnsupported() { }
        internal static void ThrowIfActiveUnsupported() { }
    }

    public struct Cell { public float Value; }

    public interface IEngine
    {
        T TensorMaxValue<T>(Tensor<T> x);
        T TensorSum<T>(Tensor<T> x);
        bool TensorIsSameSize<T>(Tensor<T> a, Tensor<T> b);
        Tensor<T> TensorAbs<T>(Tensor<T> x);
        void Conv<T>(Tensor<T> input, Tensor<T> kernel, Tensor<T> output);
    }

    public class CpuEngine : IEngine
    {
        private protected static void BeginHostScalarRead<T>(Tensor<T> operand) { }
        private static Memory<float> AsFloatMemory<T>(Memory<T> m) => default;
        private static float ReadAll(ReadOnlyMemory<float> m) => 0f;
        private static void Consume(Memory<float> m) { }
        private Tensor<Cell> _cells;
        public virtual T TensorMaxValue<T>(Tensor<T> x) { /*MAX*/ return default; }
        public virtual T TensorSum<T>(Tensor<T> x) { /*SUM*/ return default; }
        public virtual bool TensorIsSameSize<T>(Tensor<T> a, Tensor<T> b) { /*SAME*/ return true; }
        public virtual Tensor<T> TensorAbs<T>(Tensor<T> x) => x;
        public virtual void Conv<T>(Tensor<T> input, Tensor<T> kernel, Tensor<T> output) { /*CONV*/ }
        public T Helper<T>(Tensor<T> x) => default;   // not an IEngine member
    }

    public class DirectGpuTensorEngine : CpuEngine
    {
        public override T TensorMaxValue<T>(Tensor<T> x) { /*GPUMAX*/ return default; }
    }
}
";

    // Fills the stubs' marker comments (an unfilled marker is the rule-satisfying default) and analyzes the result.
    // TensorIsSameSize reads only shapes and is marked [HostValueFromMetadata] unless a case asks otherwise.
    private static ImmutableArray<Diagnostic> Analyze(string? max = null, string? gpuMax = null, string? conv = null,
        bool markSameSize = true)
    {
        string source = Stubs
            .Replace("/*MAX*/", max ?? "BeginHostScalarRead(x);")
            .Replace("/*GPUMAX*/", gpuMax ?? "BeginHostScalarRead(x);")
            .Replace("/*SUM*/", "BeginHostScalarRead(x);")
            .Replace("/*SAME*/", "")
            .Replace("/*CONV*/", conv ?? "");
        if (markSameSize)
            source = source.Replace("public virtual bool TensorIsSameSize", "[HostValueFromMetadata] public virtual bool TensorIsSameSize");
        var references = ((string?)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES") ?? "")
            .Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries)
            .Where(p => Path.GetFileName(p).StartsWith("System.", StringComparison.Ordinal)
                        || Path.GetFileName(p) is "netstandard.dll" or "mscorlib.dll")
            .Select(p => MetadataReference.CreateFromFile(p));
        var compilation = CSharpCompilation.Create("EngineGraphSafetyProbe",
            new[] { CSharpSyntaxTree.ParseText(source) }, references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary, nullableContextOptions: NullableContextOptions.Disable));
        Assert.Empty(compilation.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error));
        return compilation.WithAnalyzers(ImmutableArray.Create<DiagnosticAnalyzer>(new EngineGraphSafetyAnalyzer()))
            .GetAnalyzerDiagnosticsAsync().GetAwaiter().GetResult();
    }

    private static string LineOf(Diagnostic d) =>
        d.Location.SourceTree?.GetText().Lines[d.Location.GetLineSpan().StartLinePosition.Line].ToString() ?? "";

    [Fact]
    public void TheStubsAsWritten_ReportNothing()
    {
        Assert.Empty(Analyze());
    }

    [Fact]
    public void HostValueWithoutBoundary_IsReported()
    {
        var hit = Assert.Single(Analyze(max: "var n = x.Length;"));
        Assert.Equal("TENS0040", hit.Id);
        Assert.Contains("TensorMaxValue", hit.GetMessage(), StringComparison.Ordinal);
        Assert.Equal(DiagnosticSeverity.Error, hit.Severity);
    }

    [Fact]
    public void GpuOverrideWithoutBoundary_IsReported()
    {
        var hit = Assert.Single(Analyze(gpuMax: "var n = x.Length;"));
        Assert.Equal("TENS0040", hit.Id);
        Assert.Contains("override", LineOf(hit), StringComparison.Ordinal);
    }

    [Fact]
    public void ShapeOnlyPredicate_IsReportedUntilMarked()
    {
        var hit = Assert.Single(Analyze(markSameSize: false));
        Assert.Equal("TENS0040", hit.Id);
        Assert.Contains("TensorIsSameSize", hit.GetMessage(), StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("return TensorSum(x);")]                         // delegates to a checked host-value member
    [InlineData("var n = x.Length; BeginHostScalarRead(x); return default;")]
    [InlineData("if (x == null) return default; BeginHostScalarRead(x); return default;")]   // an early return first
    [InlineData("GraphMode.ThrowIfActiveUnsupported(); return default;")]                     // refuses every trace
    public void BoundaryOrDelegation_IsNotReported(string body)
    {
        Assert.Empty(Analyze(max: body));
    }

    [Theory]
    // A boundary that runs on only some calls leaves the others unguarded.
    [InlineData("if (x.Length == 0) BeginHostScalarRead(x); return default;")]
    [InlineData("for (int i = 0; i < x.Length; i++) BeginHostScalarRead(x); return default;")]
    [InlineData("System.Action a = () => BeginHostScalarRead(x); return default;")]
    [InlineData("var ok = x.Length > 0 && TensorSum(x) != null; return default;")]
    [InlineData("return x.Length == 0 ? default : TensorSum(x);")]
    // ThrowIfInferenceUnsupported returns in a training trace, which still freezes the value.
    [InlineData("GraphMode.ThrowIfInferenceUnsupported(); return default;")]
    public void BoundaryOnSomePathsOnly_IsReported(string body)
    {
        var hit = Assert.Single(Analyze(max: body));
        Assert.Equal("TENS0040", hit.Id);
        Assert.Contains("TensorMaxValue", hit.GetMessage(), StringComparison.Ordinal);
    }

    [Theory]
    // The #1750 defect: the streaming kernel reinterpreted and read through the writable accessor.
    [InlineData("var k = AsFloatMemory(kernel.DataUnmarked); ReadAll(k);")]
    [InlineData("ReadAll(AsFloatMemory(kernel.DataUnmarked));")]
    [InlineData("System.ReadOnlySpan<T> s = kernel.DataUnmarked.Span; var v = s[0];")]
    [InlineData("var s = kernel.DataUnmarked.Span; var v = s[0];")]
    [InlineData("kernel.DataUnmarked.Span.CopyTo(output.DataUnmarked.Span);")]   // the source of a copy
    public void WritableAccessorOnlyRead_IsReported(string body)
    {
        var hit = Assert.Single(Analyze(conv: body));
        Assert.Equal("TENS0041", hit.Id);
        Assert.Contains("kernel.DataUnmarked", hit.GetMessage(), StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("output.DataUnmarked.Span[0] = default;")]
    [InlineData("var s = output.DataUnmarked.Span; s[0] = default;")]
    [InlineData("input.ReadOnlyDataUnmarked.Span.CopyTo(output.DataUnmarked.Span);")]   // the destination of a copy
    [InlineData("Consume(AsFloatMemory(output.DataUnmarked));")]                          // a writable argument
    [InlineData("var m = output.DataUnmarked; var h = m.Pin(); h.Dispose();")]           // a pin may write
    [InlineData("_cells.DataUnmarked.Span[0].Value = 1f;")]                               // a struct element's field
    public void WritableAccessorWritten_IsNotReported(string body)
    {
        Assert.Empty(Analyze(conv: body));
    }
}
#endif
