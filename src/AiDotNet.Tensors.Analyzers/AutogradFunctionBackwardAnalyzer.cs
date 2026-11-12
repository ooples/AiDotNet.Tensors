using System.Collections.Immutable;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Diagnostics;

namespace AiDotNet.Tensors.Analyzers;

/// <summary>
/// TENS0030: flags a concrete subclass of <c>AiDotNet.Tensors.Engines.Autodiff.AutogradFunction&lt;T&gt;</c> that
/// overrides neither <c>Backward</c> overload — not itself, and not through any base class between it and
/// <c>AutogradFunction&lt;T&gt;</c>.
/// </summary>
/// <remarks>
/// <para>
/// <c>AutogradFunction&lt;T&gt;</c> has two virtual <c>Backward</c> overloads, each with a default that defers to the
/// other: the per-input <c>Backward(AutogradContext, Tensor&lt;T&gt;)</c> (torch.autograd.Function.backward) throws
/// <see cref="System.NotImplementedException"/>, and the accumulating
/// <c>Backward(AutogradContext, Tensor&lt;T&gt;, IEngine, Dictionary&lt;...&gt;)</c> calls it. Neither can be
/// abstract without forcing every subclass to override both, so a subclass that overrides neither compiles and only
/// fails when a backward pass first reaches it — possibly deep into training.
/// </para>
/// <para>
/// Not flagged: abstract subclasses (an intermediate base may leave Backward to its own subclasses), and subclasses
/// that override either overload anywhere in their chain of bases.
/// </para>
/// </remarks>
[DiagnosticAnalyzer(LanguageNames.CSharp)]
public sealed class AutogradFunctionBackwardAnalyzer : DiagnosticAnalyzer
{
    internal const string AutogradFunctionMetadataName = "AiDotNet.Tensors.Engines.Autodiff.AutogradFunction`1";

    internal static readonly DiagnosticDescriptor MissingBackward = new(
        "TENS0030", "AutogradFunction subclass does not override Backward",
        "'{0}' derives from AutogradFunction<T> but overrides neither Backward overload, so its backward pass throws "
        + "NotImplementedException. Override Backward(AutogradContext, Tensor<T>) (return one gradient per input) or "
        + "the accumulating Backward(AutogradContext, Tensor<T>, IEngine, Dictionary<Tensor<T>, Tensor<T>>).",
        "Autograd", DiagnosticSeverity.Warning, isEnabledByDefault: true,
        description: "A custom autograd function must define its backward pass. Both Backward overloads are virtual "
                     + "and default to each other, so a subclass that overrides neither compiles but throws at run "
                     + "time when a backward pass reaches it.");

    /// <inheritdoc/>
    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics => ImmutableArray.Create(MissingBackward);

    /// <inheritdoc/>
    public override void Initialize(AnalysisContext context)
    {
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.EnableConcurrentExecution();
        context.RegisterCompilationStartAction(start =>
        {
            var autogradFunction = start.Compilation.GetTypeByMetadataName(AutogradFunctionMetadataName);
            if (autogradFunction is null) return;
            start.RegisterSymbolAction(c => Analyze(c, autogradFunction), SymbolKind.NamedType);
        });
    }

    private static void Analyze(SymbolAnalysisContext context, INamedTypeSymbol autogradFunction)
    {
        var type = (INamedTypeSymbol)context.Symbol;
        if (type.TypeKind != TypeKind.Class || type.IsAbstract) return;

        bool derives = false;
        for (var current = type; current is not null; current = current.BaseType)
        {
            if (SymbolEqualityComparer.Default.Equals(current.OriginalDefinition, autogradFunction))
            {
                derives = true;
                break;
            }
            if (OverridesBackward(current, autogradFunction)) return;
        }
        if (!derives) return;

        foreach (var location in type.Locations)
        {
            if (!location.IsInSource) continue;
            context.ReportDiagnostic(Diagnostic.Create(MissingBackward, location, type.Name));
            return;   // a partial class is reported once
        }
    }

    // True when the type declares an override of a Backward that AutogradFunction<T> introduced.
    private static bool OverridesBackward(INamedTypeSymbol type, INamedTypeSymbol autogradFunction)
    {
        foreach (var member in type.GetMembers("Backward"))
        {
            if (member is not IMethodSymbol { IsOverride: true } method) continue;
            var root = method;
            while (root.OverriddenMethod is { } overridden) root = overridden;
            if (SymbolEqualityComparer.Default.Equals(root.ContainingType.OriginalDefinition, autogradFunction)) return true;
        }
        return false;
    }
}
