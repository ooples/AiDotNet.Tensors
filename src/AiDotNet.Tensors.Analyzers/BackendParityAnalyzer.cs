using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CodeAnalysis.Text;

namespace AiDotNet.Tensors.Analyzers;

/// <summary>
/// Makes it a build error for a GPU operation to exist on only some of the six DirectGpu backends.
/// </summary>
/// <remarks>
/// <para>
/// The engine reaches optional capabilities through casts such as <c>backend as IGpuBatchExecution</c>.
/// When only one backend implements the interface, every other backend silently takes a slower
/// composed or host route, and nothing fails. This analyzer closes that gap:
/// </para>
/// <list type="bullet">
/// <item><b>TENS0001</b> an interface declared in this assembly is implemented by some backends but not all six.</item>
/// <item><b>TENS0002</b> a backend implements an interface member with a body that only throws
/// <c>NotSupportedException</c> or <c>NotImplementedException</c>, which compiles but is not an implementation.</item>
/// <item><b>TENS0003</b> a baseline entry no longer matches a gap, so the baseline can only shrink.</item>
/// <item><b>TENS0004</b> a baseline line is malformed or does not name a tracking issue.</item>
/// <item><b>TENS0005</b> a <c>[BackendSpecificRoute]</c> exemption has no reason, or names a
/// cross-backend operation that is not present on all six backends.</item>
/// </list>
/// <para>
/// Existing gaps live in <c>BackendParityBaseline.txt</c> (an AdditionalFiles item), one per line:
/// <c>capability &lt;Interface&gt; &lt;Backend&gt; #issue</c> or <c>stub &lt;Backend&gt;.&lt;Member&gt; #issue</c>.
/// </para>
/// </remarks>
[DiagnosticAnalyzer(LanguageNames.CSharp)]
public sealed class BackendParityAnalyzer : DiagnosticAnalyzer
{
    private const string Category = "BackendParity";
    private const string BaselineFileName = "BackendParityBaseline.txt";
    private const string BackendInterfaceName = "AiDotNet.Tensors.Engines.DirectGpu.IDirectGpuBackend";
    private const string RouteAttributeName = "AiDotNet.Tensors.Engines.DirectGpu.BackendSpecificRouteAttribute";

    internal static readonly ImmutableArray<string> BackendTypeNames = ImmutableArray.Create(
        "AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend",
        "AiDotNet.Tensors.Engines.DirectGpu.HIP.HipBackend",
        "AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalBackend",
        "AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend",
        "AiDotNet.Tensors.Engines.DirectGpu.Vulkan.VulkanBackend",
        "AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBackend");

    internal static readonly DiagnosticDescriptor CapabilityNotOnAllBackends = new(
        "TENS0001", "GPU capability is not implemented by every backend",
        "'{0}' is implemented by {1} but not by {2}; implement it natively on every backend or add a tracked baseline entry",
        Category, DiagnosticSeverity.Error, isEnabledByDefault: true);

    internal static readonly DiagnosticDescriptor StubImplementation = new(
        "TENS0002", "Backend member only throws",
        "'{0}.{1}' implements '{2}' with a body that only throws {3}; a stub is not an implementation",
        Category, DiagnosticSeverity.Error, isEnabledByDefault: true);

    internal static readonly DiagnosticDescriptor StaleBaselineEntry = new(
        "TENS0003", "Backend parity baseline entry is stale",
        "Baseline entry '{0}' no longer matches a gap; delete it so the baseline only shrinks",
        Category, DiagnosticSeverity.Error, isEnabledByDefault: true);

    internal static readonly DiagnosticDescriptor MalformedBaselineEntry = new(
        "TENS0004", "Backend parity baseline entry is malformed",
        "Baseline line '{0}' must be 'capability <Interface> <Backend> #issue' or 'stub <Backend>.<Member> #issue'",
        Category, DiagnosticSeverity.Error, isEnabledByDefault: true);

    internal static readonly DiagnosticDescriptor InvalidRouteExemption = new(
        "TENS0005", "Backend-specific route exemption is invalid",
        "'{0}' is marked [BackendSpecificRoute] but {1}",
        Category, DiagnosticSeverity.Error, isEnabledByDefault: true);

    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = ImmutableArray.Create(
        CapabilityNotOnAllBackends, StubImplementation, StaleBaselineEntry, MalformedBaselineEntry, InvalidRouteExemption);

    public override void Initialize(AnalysisContext context)
    {
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.EnableConcurrentExecution();
        context.RegisterCompilationStartAction(Start);
    }

    private static void Start(CompilationStartAnalysisContext context)
    {
        var backendInterface = context.Compilation.GetTypeByMetadataName(BackendInterfaceName);
        if (backendInterface is null) return; // not the AiDotNet.Tensors compilation

        // A backend whose core is compiled out for this target framework (WebGPU needs .NET 7+) leaves
        // only partial-class fragments behind. That type is not a backend here, so it is not compared.
        var backends = new List<INamedTypeSymbol>(BackendTypeNames.Length);
        foreach (string name in BackendTypeNames)
        {
            var type = context.Compilation.GetTypeByMetadataName(name);
            if (type is not null && type.AllInterfaces.Contains(backendInterface, SymbolEqualityComparer.Default))
                backends.Add(type);
        }
        if (backends.Count < 2) return;

        var stubs = new ConcurrentBag<StubCandidate>();
        context.RegisterSymbolAction(c => CollectStub(c, backends, stubs), SymbolKind.Method);
        context.RegisterCompilationEndAction(c => Finish(c, backends, stubs));
    }

    private static void CollectStub(SymbolAnalysisContext context, List<INamedTypeSymbol> backends, ConcurrentBag<StubCandidate> stubs)
    {
        var method = (IMethodSymbol)context.Symbol;
        var owner = method.ContainingType;
        if (owner is null || !backends.Any(b => SymbolEqualityComparer.Default.Equals(b, owner))) return;

        foreach (var reference in method.DeclaringSyntaxReferences)
        {
            if (reference.GetSyntax(context.CancellationToken) is not BaseMethodDeclarationSyntax declaration) continue;
            string? thrown = OnlyThrownType(declaration);
            if (thrown is null) continue;

            var implemented = ImplementedInterfaceMember(owner, method);
            if (implemented is null) continue;

            stubs.Add(new StubCandidate(owner.Name, implemented.Name, implemented.ContainingType.Name + "." + implemented.Name,
                thrown, declaration.GetLocation()));
        }
    }

    private static string? OnlyThrownType(BaseMethodDeclarationSyntax declaration)
    {
        ExpressionSyntax? thrownExpression = null;
        if (declaration.ExpressionBody?.Expression is ThrowExpressionSyntax throwExpression)
            thrownExpression = throwExpression.Expression;
        else if (declaration.Body is { Statements.Count: 1 } body && body.Statements[0] is ThrowStatementSyntax throwStatement)
            thrownExpression = throwStatement.Expression;

        if (thrownExpression is not ObjectCreationExpressionSyntax creation) return null;
        string typeName = creation.Type.ToString();
        int dot = typeName.LastIndexOf('.');
        if (dot >= 0) typeName = typeName.Substring(dot + 1);
        return typeName is "NotSupportedException" or "NotImplementedException" ? typeName : null;
    }

    private static ISymbol? ImplementedInterfaceMember(INamedTypeSymbol owner, IMethodSymbol method)
    {
        foreach (var candidate in method.ExplicitInterfaceImplementations)
            return candidate;
        foreach (var iface in owner.AllInterfaces)
        {
            foreach (var member in iface.GetMembers(method.Name))
            {
                if (SymbolEqualityComparer.Default.Equals(owner.FindImplementationForInterfaceMember(member), method))
                    return member;
            }
        }
        return null;
    }

    private static void Finish(CompilationAnalysisContext context, List<INamedTypeSymbol> backends, ConcurrentBag<StubCandidate> stubs)
    {
        var baseline = Baseline.Load(context);
        var assembly = context.Compilation.Assembly;

        // Every interface any backend implements that this assembly declares.
        var implementers = new Dictionary<INamedTypeSymbol, List<INamedTypeSymbol>>(SymbolEqualityComparer.Default);
        foreach (var backend in backends)
        {
            foreach (var iface in backend.AllInterfaces)
            {
                if (!SymbolEqualityComparer.Default.Equals(iface.ContainingAssembly, assembly)) continue;
                var key = iface.OriginalDefinition;
                if (!implementers.TryGetValue(key, out var list)) implementers[key] = list = new List<INamedTypeSymbol>();
                list.Add(backend);
            }
        }

        foreach (var pair in implementers.OrderBy(p => p.Key.Name, StringComparer.Ordinal))
        {
            var iface = pair.Key;
            var missing = backends.Where(b => !pair.Value.Contains(b, SymbolEqualityComparer.Default)).ToList();
            if (missing.Count == 0) continue;
            if (IsValidRouteExemption(context, iface, backends)) continue;

            var unbaselined = missing.Where(b => !baseline.Consume("capability " + iface.Name + " " + b.Name)).ToList();
            if (unbaselined.Count == 0) continue;
            context.ReportDiagnostic(Diagnostic.Create(CapabilityNotOnAllBackends, iface.Locations.FirstOrDefault(),
                iface.Name, string.Join(", ", pair.Value.Select(b => b.Name)), string.Join(", ", unbaselined.Select(b => b.Name))));
        }

        foreach (var stub in stubs.OrderBy(s => s.Backend, StringComparer.Ordinal).ThenBy(s => s.Member, StringComparer.Ordinal))
        {
            if (baseline.Consume("stub " + stub.Backend + "." + stub.Member)) continue;
            context.ReportDiagnostic(Diagnostic.Create(StubImplementation, stub.Location,
                stub.Backend, stub.Member, stub.Implements, stub.Thrown));
        }

        baseline.ReportProblems(context, new HashSet<string>(backends.Select(b => b.Name), StringComparer.Ordinal));
    }

    private static bool IsValidRouteExemption(CompilationAnalysisContext context, INamedTypeSymbol iface, List<INamedTypeSymbol> backends)
    {
        var attribute = iface.GetAttributes().FirstOrDefault(a => a.AttributeClass?.ToDisplayString() == RouteAttributeName);
        if (attribute is null) return false;

        var location = iface.Locations.FirstOrDefault();
        string reason = attribute.ConstructorArguments.Length > 0 ? attribute.ConstructorArguments[0].Value as string ?? "" : "";
        if (string.IsNullOrWhiteSpace(reason))
        {
            context.ReportDiagnostic(Diagnostic.Create(InvalidRouteExemption, location, iface.Name, "gives no reason"));
            return true;
        }

        var operations = attribute.ConstructorArguments.Length > 1
            ? attribute.ConstructorArguments[1].Values.Select(v => v.Value as string ?? "").ToList()
            : new List<string>();
        if (operations.Count == 0)
        {
            context.ReportDiagnostic(Diagnostic.Create(InvalidRouteExemption, location, iface.Name,
                "names no cross-backend operation that every backend provides natively"));
            return true;
        }

        foreach (string operation in operations)
        {
            var lacking = backends.Where(b => !HasMethod(b, operation)).Select(b => b.Name).ToList();
            if (lacking.Count > 0)
                context.ReportDiagnostic(Diagnostic.Create(InvalidRouteExemption, location, iface.Name,
                    "its operation '" + operation + "' is missing on " + string.Join(", ", lacking)));
        }
        return true;
    }

    private static bool HasMethod(INamedTypeSymbol type, string name)
    {
        for (var current = type; current is not null; current = current.BaseType)
        {
            if (current.GetMembers(name).Any(m => m.Kind == SymbolKind.Method)) return true;
        }
        return false;
    }

    private sealed class StubCandidate
    {
        public StubCandidate(string backend, string member, string implements, string thrown, Location location)
        {
            Backend = backend;
            Member = member;
            Implements = implements;
            Thrown = thrown;
            Location = location;
        }

        public string Backend { get; }
        public string Member { get; }
        public string Implements { get; }
        public string Thrown { get; }
        public Location Location { get; }
    }

    private sealed class Baseline
    {
        private readonly Dictionary<string, Entry> _entries = new(StringComparer.Ordinal);
        private readonly List<Diagnostic> _problems = new();

        public static Baseline Load(CompilationAnalysisContext context)
        {
            var baseline = new Baseline();
            foreach (var file in context.Options.AdditionalFiles)
            {
                if (!string.Equals(Path.GetFileName(file.Path), BaselineFileName, StringComparison.OrdinalIgnoreCase)) continue;
                var text = file.GetText(context.CancellationToken);
                if (text is null) continue;
                foreach (var line in text.Lines)
                    baseline.Parse(file.Path, line);
            }
            return baseline;
        }

        private void Parse(string path, TextLine line)
        {
            string raw = line.ToString();
            string content = raw.Trim();
            if (content.Length == 0 || content.StartsWith("//", StringComparison.Ordinal)) return;

            var location = Location.Create(path, line.Span, line.Text?.Lines.GetLinePositionSpan(line.Span) ?? default);
            var parts = content.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            bool valid = parts.Length >= 3 && parts[parts.Length - 1].Length > 1 && parts[parts.Length - 1][0] == '#'
                && parts[parts.Length - 1].Skip(1).All(char.IsDigit)
                && ((parts[0] == "capability" && parts.Length == 4) || (parts[0] == "stub" && parts.Length == 3 && parts[1].Contains('.')));
            if (!valid)
            {
                _problems.Add(Diagnostic.Create(MalformedBaselineEntry, location, content));
                return;
            }

            string key = string.Join(" ", parts.Take(parts.Length - 1));
            if (_entries.ContainsKey(key))
            {
                _problems.Add(Diagnostic.Create(StaleBaselineEntry, location, content + " (duplicate)"));
                return;
            }
            string backend = parts[0] == "capability" ? parts[2] : parts[1].Substring(0, parts[1].IndexOf('.'));
            _entries[key] = new Entry(content, backend, location);
        }

        public bool Consume(string key)
        {
            if (!_entries.TryGetValue(key, out var entry)) return false;
            entry.Used = true;
            return true;
        }

        /// <summary>Entries for a backend compiled out of this target framework are not stale here.</summary>
        public void ReportProblems(CompilationAnalysisContext context, HashSet<string> presentBackends)
        {
            foreach (var problem in _problems) context.ReportDiagnostic(problem);
            foreach (var entry in _entries.Values.Where(e => !e.Used && presentBackends.Contains(e.Backend)))
                context.ReportDiagnostic(Diagnostic.Create(StaleBaselineEntry, entry.Location, entry.Text));
        }

        private sealed class Entry
        {
            public Entry(string text, string backend, Location location)
            {
                Text = text;
                Backend = backend;
                Location = location;
            }

            public string Text { get; }

            public string Backend { get; }
            public Location Location { get; }
            public bool Used { get; set; }
        }
    }
}
