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
using Microsoft.CodeAnalysis.Operations;
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
/// <item><b>TENS0006</b> a backend implements an interface member by reading device memory back to the
/// host (a direct <c>Download*</c> call), which hides a host round-trip behind a GPU capability.</item>
/// <item><b>TENS0005</b> a <c>[BackendSpecificRoute]</c> exemption has no reason, or names a
/// cross-backend operation that is not present on all six backends.</item>
/// </list>
/// <para>
/// Existing gaps live in <c>BackendParityBaseline.txt</c> (an AdditionalFiles item), one per line:
/// <c>capability &lt;Interface&gt; &lt;Backend&gt; #issue</c>, <c>stub &lt;Backend&gt;.&lt;Member&gt; #issue</c> or
/// <c>host &lt;Backend&gt;.&lt;Member&gt; #issue</c>.
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
        "Baseline line '{0}' must be 'capability <Interface> <Backend> #issue', 'stub <Backend>.<Member> #issue' or 'host <Backend>.<Member> #issue'",
        Category, DiagnosticSeverity.Error, isEnabledByDefault: true);

    internal static readonly DiagnosticDescriptor InvalidRouteExemption = new(
        "TENS0005", "Backend-specific route exemption is invalid",
        "'{0}' is marked [BackendSpecificRoute] but {1}",
        Category, DiagnosticSeverity.Error, isEnabledByDefault: true);

    internal static readonly DiagnosticDescriptor HostRoundTrip = new(
        "TENS0006", "Backend member reads device memory back to the host",
        "'{0}.{1}' implements '{2}' by calling '{3}', a host round-trip; compute it on the device",
        Category, DiagnosticSeverity.Error, isEnabledByDefault: true);

    internal static readonly DiagnosticDescriptor ConcreteBackendDispatch = new(
        "TENS0007", "Code dispatches through a concrete backend class",
        "'{0}.{1}' reaches '{2}' through the concrete class, so no other backend can provide it; put the capability behind an interface every backend implements, or add a tracked baseline entry",
        Category, DiagnosticSeverity.Error, isEnabledByDefault: true);

    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = ImmutableArray.Create(
        CapabilityNotOnAllBackends, StubImplementation, StaleBaselineEntry, MalformedBaselineEntry, InvalidRouteExemption,
        HostRoundTrip, ConcreteBackendDispatch);

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
        var hostReads = new ConcurrentBag<StubCandidate>();
        context.RegisterSymbolAction(c => CollectStub(c, backends, stubs), SymbolKind.Method);
        context.RegisterSyntaxNodeAction(c => CollectHostRead(c, backends, hostReads), SyntaxKind.InvocationExpression);
        var casts = new ConcurrentBag<StubCandidate>();
        context.RegisterOperationAction(c => CollectConcreteCast(c, backends, casts),
            OperationKind.IsType, OperationKind.DeclarationPattern, OperationKind.TypePattern, OperationKind.RecursivePattern,
            OperationKind.Conversion);
        context.RegisterCompilationEndAction(c => Finish(c, backends, stubs, hostReads, casts));
    }

    private static void CollectStub(SymbolAnalysisContext context, List<INamedTypeSymbol> backends, ConcurrentBag<StubCandidate> stubs)
    {
        var method = (IMethodSymbol)context.Symbol;
        var owner = method.ContainingType;
        if (owner is null || !backends.Any(b => SymbolEqualityComparer.Default.Equals(b, owner))) return;

        foreach (var reference in method.DeclaringSyntaxReferences)
        {
            if (reference.GetSyntax(context.CancellationToken) is not BaseMethodDeclarationSyntax declaration) continue;
            string? thrown = OnlyThrownType(declaration, context.Compilation.GetSemanticModel(declaration.SyntaxTree));
            if (thrown is null) continue;

            var implemented = ImplementedInterfaceMember(owner, method);
            if (implemented is null) continue;

            stubs.Add(new StubCandidate(owner.Name, implemented.Name, implemented.ContainingType.Name + "." + implemented.Name,
                thrown, declaration.GetLocation()));
        }
    }

    private static void CollectHostRead(SyntaxNodeAnalysisContext context, List<INamedTypeSymbol> backends, ConcurrentBag<StubCandidate> hostReads)
    {
        var invocation = (InvocationExpressionSyntax)context.Node;
        string calledName = invocation.Expression switch
        {
            MemberAccessExpressionSyntax member => member.Name.Identifier.ValueText,
            IdentifierNameSyntax identifier => identifier.Identifier.ValueText,
            _ => ""
        };
        if (!calledName.StartsWith("Download", StringComparison.Ordinal)) return;

        // A host read inside a lambda or local function still belongs to the member that declares it.
        var containing = context.ContainingSymbol;
        while (containing is IMethodSymbol { MethodKind: MethodKind.AnonymousFunction or MethodKind.LocalFunction } inner)
            containing = inner.ContainingSymbol;
        if (containing is not IMethodSymbol method) return;
        var owner = method.ContainingType;
        if (owner is null || !backends.Any(b => SymbolEqualityComparer.Default.Equals(b, owner))) return;

        var implemented = ImplementedInterfaceMember(owner, method);
        if (implemented is null || IsTransferMember(implemented.Name)) return;
        // A member whose contract returns a host value (a float loss, a norm) must read it back; only
        // results meant to stay resident (void writes into buffers, or a returned buffer) are policed.
        if (implemented is IMethodSymbol contract && !contract.ReturnsVoid
            && !contract.ReturnType.AllInterfaces.Any(i => i.Name == "IGpuBuffer") && contract.ReturnType.Name != "IGpuBuffer")
            return;

        hostReads.Add(new StubCandidate(owner.Name, implemented.Name, implemented.ContainingType.Name + "." + implemented.Name,
            calledName, invocation.GetLocation()));
    }

    /// <summary>
    /// A type test or explicit cast to one backend class outside that backend's own namespace: the engine reaching a
    /// capability that only that backend can then provide. Implicit conversions (passing a backend where its base is
    /// expected) are not dispatch.
    /// </summary>
    private static void CollectConcreteCast(OperationAnalysisContext context, List<INamedTypeSymbol> backends, ConcurrentBag<StubCandidate> casts)
    {
        ITypeSymbol? target = context.Operation switch
        {
            IIsTypeOperation isType => isType.TypeOperand,
            IDeclarationPatternOperation declaration => declaration.MatchedType,
            ITypePatternOperation typePattern => typePattern.MatchedType,
            IRecursivePatternOperation recursive => recursive.MatchedType,
            IConversionOperation conversion when !conversion.IsImplicit && conversion.Operand.Type is not null
                && !SymbolEqualityComparer.Default.Equals(conversion.Operand.Type, conversion.Type) => conversion.Type,
            _ => null
        };
        if (target is not INamedTypeSymbol named) return;
        var backend = backends.FirstOrDefault(b => SymbolEqualityComparer.Default.Equals(b, named.OriginalDefinition));
        if (backend is null) return;

        var containing = context.ContainingSymbol;
        while (containing is IMethodSymbol { MethodKind: MethodKind.AnonymousFunction or MethodKind.LocalFunction } inner)
            containing = inner.ContainingSymbol;
        string backendNamespace = backend.ContainingNamespace.ToDisplayString();
        string callerNamespace = containing.ContainingNamespace?.ToDisplayString() ?? "";
        if (callerNamespace == backendNamespace || callerNamespace.StartsWith(backendNamespace + ".", StringComparison.Ordinal)) return;

        string owner = containing.ContainingType?.Name ?? containing.Name;
        // An explicit interface implementation's name is qualified (IEngine.SoftmaxBackward); keep the member's own name.
        string member = containing is IMethodSymbol { MethodKind: MethodKind.Constructor } ? "ctor"
            : containing.Name.Substring(containing.Name.LastIndexOf('.') + 1);
        casts.Add(new StubCandidate(backend.Name, owner + "." + member, backend.Name, "", context.Operation.Syntax.GetLocation()));
    }

    /// <summary>The transfer API itself may move data to the host; that is its job.</summary>
    private static bool IsTransferMember(string name) =>
        name.StartsWith("Download", StringComparison.Ordinal) || name.StartsWith("Upload", StringComparison.Ordinal)
        || name.StartsWith("Read", StringComparison.Ordinal) || name.StartsWith("Copy", StringComparison.Ordinal)
        || name.StartsWith("Synchronize", StringComparison.Ordinal);

    // The thrown expression is judged by its type, not its spelling, so routing the exception through a helper
    // (throw NoKernel(name)) is still recognised as a stub.
    private static string? OnlyThrownType(BaseMethodDeclarationSyntax declaration, SemanticModel model)
    {
        ExpressionSyntax? thrownExpression = null;
        if (declaration.ExpressionBody?.Expression is ThrowExpressionSyntax throwExpression)
            thrownExpression = throwExpression.Expression;
        else if (declaration.Body is { Statements.Count: 1 } body && body.Statements[0] is ThrowStatementSyntax throwStatement)
            thrownExpression = throwStatement.Expression;

        if (thrownExpression is null) return null;
        for (var type = model.GetTypeInfo(thrownExpression).Type; type is not null; type = type.BaseType)
        {
            if (type.ContainingNamespace?.ToDisplayString() == "System"
                && type.Name is "NotSupportedException" or "NotImplementedException")
                return type.Name;
        }
        return null;
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

    private static void Finish(CompilationAnalysisContext context, List<INamedTypeSymbol> backends, ConcurrentBag<StubCandidate> stubs,
        ConcurrentBag<StubCandidate> hostReads, ConcurrentBag<StubCandidate> casts)
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

        var reportedHost = new HashSet<string>(StringComparer.Ordinal);
        foreach (var read in hostReads.OrderBy(s => s.Backend, StringComparer.Ordinal).ThenBy(s => s.Member, StringComparer.Ordinal))
        {
            string key = "host " + read.Backend + "." + read.Member;
            if (!reportedHost.Add(key + "|" + read.Location.SourceSpan.Start + read.Location.SourceTree?.FilePath)) continue;
            if (baseline.Consume(key)) continue;
            context.ReportDiagnostic(Diagnostic.Create(HostRoundTrip, read.Location,
                read.Backend, read.Member, read.Implements, read.Thrown));
        }

        foreach (var cast in casts.OrderBy(s => s.Member, StringComparer.Ordinal).ThenBy(s => s.Backend, StringComparer.Ordinal))
        {
            if (baseline.Consume("cast " + cast.Member + " " + cast.Backend)) continue;
            int dot = cast.Member.LastIndexOf('.');
            context.ReportDiagnostic(Diagnostic.Create(ConcreteBackendDispatch, cast.Location,
                cast.Member.Substring(0, dot), cast.Member.Substring(dot + 1), cast.Backend));
        }

        // GPU fast paths that cast are partly compiled only for .NET 5+, so only those builds hold every cast site.
        bool allCastSitesCompiled = context.Compilation.SyntaxTrees.FirstOrDefault()?.Options.PreprocessorSymbolNames
            .Contains("NET5_0_OR_GREATER") == true;
        baseline.ReportProblems(context, new HashSet<string>(backends.Select(b => b.Name), StringComparer.Ordinal), allCastSitesCompiled);
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
                && (((parts[0] == "capability" || parts[0] == "cast") && parts.Length == 4)
                    || ((parts[0] == "stub" || parts[0] == "host") && parts.Length == 3 && parts[1].Contains('.')));
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
            string backend = parts[0] == "capability" || parts[0] == "cast" ? parts[2] : parts[1].Substring(0, parts[1].IndexOf('.'));
            _entries[key] = new Entry(content, backend, location);
        }

        public bool Consume(string key)
        {
            if (!_entries.TryGetValue(key, out var entry)) return false;
            entry.Used = true;
            return true;
        }

        /// <summary>Entries for a backend compiled out of this target framework are not stale here.</summary>
        public void ReportProblems(CompilationAnalysisContext context, HashSet<string> presentBackends, bool allCastSitesCompiled)
        {
            foreach (var problem in _problems) context.ReportDiagnostic(problem);
            foreach (var entry in _entries.Values.Where(e => !e.Used && presentBackends.Contains(e.Backend)
                && (allCastSitesCompiled || !e.Text.StartsWith("cast ", StringComparison.Ordinal))))
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
