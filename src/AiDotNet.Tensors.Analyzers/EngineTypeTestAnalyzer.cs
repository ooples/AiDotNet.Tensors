using System.Collections.Immutable;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CodeAnalysis.Operations;

namespace AiDotNet.Tensors.Analyzers;

/// <summary>
/// TENS0020: flags a direct type test of an <c>IEngine</c>-typed expression — <c>is</c>, <c>as</c>, a cast, a type
/// or declaration pattern (in <c>is</c>, <c>switch</c> or a sub-pattern), or a <c>GetType()</c> comparison — against
/// any type other than <c>IEngine</c> itself.
/// </summary>
/// <remarks>
/// <para>
/// A decorator engine (an <c>AiDotNet.Tensors.Engines.IEngineWrapper</c>, such as the tracing engine that records
/// every call for graph export) is installed as <c>AiDotNetEngine.Current</c> in place of the engine it wraps and
/// implements only <c>IEngine</c>. While it is installed, <c>engine is DirectGpuTensorEngine</c>,
/// <c>engine as ITensorLevelEngine</c> or <c>engine.GetType() == typeof(CpuEngine)</c> silently take the
/// not-that-engine branch. The wrapper-aware helpers in <c>EngineWrapperExtensions</c> (<c>Is</c>, <c>As</c>,
/// <c>TryGet</c>, <c>Cast</c>, <c>Unwrap</c>) look through the wrapper chain instead.
/// </para>
/// <para>
/// Not flagged: tests against <c>IEngine</c>, <c>object</c> or <c>IEngineWrapper</c>; tests of an expression whose
/// static type is a concrete engine class or another interface (it cannot be a wrapper); and tests of
/// <c>engine.Unwrap()</c>, whose result is never a wrapper.
/// </para>
/// </remarks>
[DiagnosticAnalyzer(LanguageNames.CSharp)]
public sealed class EngineTypeTestAnalyzer : DiagnosticAnalyzer
{
    internal const string EngineMetadataName = "AiDotNet.Tensors.Engines.IEngine";
    internal const string WrapperMetadataName = "AiDotNet.Tensors.Engines.IEngineWrapper";
    internal const string ExtensionsMetadataName = "AiDotNet.Tensors.Engines.EngineWrapperExtensions";

    internal static readonly DiagnosticDescriptor DirectEngineTypeTest = new(
        "TENS0020", "Direct type test of an IEngine",
        "'{0}' tests the concrete type of an IEngine, which is wrong while a wrapper engine (such as the tracing "
        + "engine) is installed. Use {1}.",
        "EngineWrapper", DiagnosticSeverity.Warning, isEnabledByDefault: true,
        description: "AiDotNetEngine.Current may be an IEngineWrapper that forwards to the real engine. Type tests "
                     + "of an IEngine must go through EngineWrapperExtensions (Is, As, TryGet, Cast, Unwrap), "
                     + "which look through any chain of wrappers.");

    /// <inheritdoc/>
    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics => ImmutableArray.Create(DirectEngineTypeTest);

    /// <inheritdoc/>
    public override void Initialize(AnalysisContext context)
    {
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.EnableConcurrentExecution();
        context.RegisterCompilationStartAction(start =>
        {
            var engine = start.Compilation.GetTypeByMetadataName(EngineMetadataName);
            if (engine is null) return;
            var wrapper = start.Compilation.GetTypeByMetadataName(WrapperMetadataName);
            var extensions = start.Compilation.GetTypeByMetadataName(ExtensionsMetadataName);
            var rule = new Rule(engine, wrapper, extensions);

            start.RegisterOperationAction(c => rule.AnalyzeIsType(c), OperationKind.IsType);
            start.RegisterOperationAction(c => rule.AnalyzeConversion(c), OperationKind.Conversion);
            start.RegisterOperationAction(c => rule.AnalyzePattern(c),
                OperationKind.DeclarationPattern, OperationKind.TypePattern, OperationKind.RecursivePattern);
            start.RegisterOperationAction(c => rule.AnalyzeGetType(c), OperationKind.Invocation);
        });
    }

    private sealed class Rule
    {
        private readonly INamedTypeSymbol _engine;
        private readonly INamedTypeSymbol? _wrapper;
        private readonly INamedTypeSymbol? _extensions;

        public Rule(INamedTypeSymbol engine, INamedTypeSymbol? wrapper, INamedTypeSymbol? extensions)
        {
            _engine = engine;
            _wrapper = wrapper;
            _extensions = extensions;
        }

        public void AnalyzeIsType(OperationAnalysisContext context)
        {
            var isType = (IIsTypeOperation)context.Operation;
            if (!IsWrappableEngine(isType.ValueOperand) || IsExemptTarget(isType.TypeOperand)) return;
            string name = TypeName(isType.TypeOperand);
            Report(context, isType.IsNegated ? $"!engine.Is<{name}>()" : $"engine.Is<{name}>()");
        }

        public void AnalyzeConversion(OperationAnalysisContext context)
        {
            var conversion = (IConversionOperation)context.Operation;
            if (conversion.IsImplicit || conversion.Conversion.IsImplicit || conversion.Type is null) return;
            if (!IsWrappableEngine(conversion.Operand) || IsExemptTarget(conversion.Type)) return;
            string name = TypeName(conversion.Type);
            Report(context, conversion.IsTryCast ? $"engine.As<{name}>()" : $"engine.Cast<{name}>()");
        }

        public void AnalyzePattern(OperationAnalysisContext context)
        {
            var pattern = (IPatternOperation)context.Operation;
            if (!IsEngine(pattern.InputType) || IsExemptTarget(pattern.NarrowedType)) return;
            if (pattern is IDeclarationPatternOperation { MatchesNull: true }) return; // 'var x'
            if (IsUnwrapCall(GoverningValue(pattern))) return;
            string name = TypeName(pattern.NarrowedType);
            string not = pattern.Parent is INegatedPatternOperation ? "!" : "";
            Report(context, pattern is IDeclarationPatternOperation or IRecursivePatternOperation { DeclaredSymbol: not null }
                ? $"{not}engine.TryGet<{name}>(out var x)"
                : $"{not}engine.Is<{name}>()");
        }

        public void AnalyzeGetType(OperationAnalysisContext context)
        {
            var invocation = (IInvocationOperation)context.Operation;
            if (invocation.TargetMethod is not { Name: "GetType", Parameters.Length: 0, IsStatic: false }
                || invocation.TargetMethod.ContainingType.SpecialType != SpecialType.System_Object
                || invocation.Instance is null
                || !IsWrappableEngine(invocation.Instance))
                return;
            if (!IsTypeComparison(invocation)) return;
            Report(context, "engine.Unwrap().GetType(), or engine.Is<T>()");
        }

        private static void Report(OperationAnalysisContext context, string replacement)
        {
            var syntax = context.Operation.Syntax;
            string text = syntax.ToString().Replace("\r", " ").Replace("\n", " ");
            if (text.Length > 80) text = text.Substring(0, 80) + "...";
            context.ReportDiagnostic(Diagnostic.Create(DirectEngineTypeTest, syntax.GetLocation(), text, replacement));
        }

        private bool IsEngine(ITypeSymbol? type) =>
            type is not null && SymbolEqualityComparer.Default.Equals(type.WithNullableAnnotation(NullableAnnotation.None), _engine);

        // An IEngine-typed value that is not already the result of Unwrap(). Implicit conversions (to object for
        // GetType, nullable wrapping) and parentheses are looked through.
        private bool IsWrappableEngine(IOperation operand)
        {
            var value = StripImplicit(operand);
            return IsEngine(value.Type) && !IsUnwrapCall(value);
        }

        private bool IsExemptTarget(ITypeSymbol? target)
        {
            if (target is null || target.SpecialType == SpecialType.System_Object || target is ITypeParameterSymbol)
                return true;
            var plain = target.WithNullableAnnotation(NullableAnnotation.None);
            return SymbolEqualityComparer.Default.Equals(plain, _engine)
                   || (_wrapper is not null && SymbolEqualityComparer.Default.Equals(plain, _wrapper));
        }

        private bool IsUnwrapCall(IOperation? operation) =>
            operation is not null
            && StripImplicit(operation) is IInvocationOperation { TargetMethod: { Name: "Unwrap" } method }
            && _extensions is not null
            && SymbolEqualityComparer.Default.Equals(method.ContainingType, _extensions);

        private static IOperation StripImplicit(IOperation operation)
        {
            while (true)
            {
                switch (operation)
                {
                    case IConversionOperation { IsImplicit: true } c: operation = c.Operand; continue;
                    case IConversionOperation c when c.Conversion.IsImplicit && c.Conversion.IsReference:
                        operation = c.Operand; continue;
                    case IParenthesizedOperation p: operation = p.Operand; continue;
                    default: return operation;
                }
            }
        }

        // The value a top-level pattern is matched against ('x is P', 'switch (x)', 'x switch { ... }').
        private static IOperation? GoverningValue(IPatternOperation pattern)
        {
            for (IOperation? current = pattern.Parent; current is not null; current = current.Parent)
            {
                switch (current)
                {
                    case IIsPatternOperation isPattern: return isPattern.Value;
                    case ISwitchExpressionOperation switchExpression: return switchExpression.Value;
                    case ISwitchOperation switchStatement: return switchStatement.Value;
                    case IPropertySubpatternOperation: return null; // a sub-pattern matches a member, not the value
                    case IPatternOperation: continue;
                    case ISwitchExpressionArmOperation or IPatternCaseClauseOperation or ISwitchCaseOperation: continue;
                    default: return null;
                }
            }

            return null;
        }

        // engine.GetType() compared with a Type: ==, !=, Equals, IsAssignableFrom, IsSubclassOf, IsAssignableTo,
        // or a pattern ('engine.GetType() is var t && ...' is not, but 'engine.GetType() == typeof(X)' is).
        private static bool IsTypeComparison(IInvocationOperation getType)
        {
            IOperation current = getType;
            while (current.Parent is IConversionOperation { IsImplicit: true } or IParenthesizedOperation)
                current = current.Parent;
            switch (current.Parent)
            {
                case IBinaryOperation { OperatorKind: BinaryOperatorKind.Equals or BinaryOperatorKind.NotEquals }:
                    return true;
                case IInvocationOperation call when IsTypeTestMethod(call.TargetMethod.Name):
                    return true;
                case IArgumentOperation { Parent: IInvocationOperation call } when IsTypeTestMethod(call.TargetMethod.Name):
                    return true;
                case IIsPatternOperation { Pattern: IConstantPatternOperation or INegatedPatternOperation or ITypePatternOperation }:
                    return true;
                default:
                    return false;
            }
        }

        private static bool IsTypeTestMethod(string name) =>
            name is "Equals" or "ReferenceEquals" or "IsAssignableFrom" or "IsAssignableTo" or "IsSubclassOf"
                or "IsInstanceOfType";

        private static string TypeName(ITypeSymbol? type) =>
            type?.WithNullableAnnotation(NullableAnnotation.None).ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat) ?? "?";
    }
}
