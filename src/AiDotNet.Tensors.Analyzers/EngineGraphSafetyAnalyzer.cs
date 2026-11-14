using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CodeAnalysis.Operations;

namespace AiDotNet.Tensors.Analyzers;

/// <summary>
/// Two defects that compiled cleanly, broke at run time, and were found only by a model sweep:
/// <list type="bullet">
/// <item><description>
/// <b>TENS0040</b>: an <c>IEngine</c> operation that returns a host value (a scalar, a bool, a count) computed from a
/// tensor without calling <c>BeginHostScalarRead</c>. A training trace records no node for it, so its trace-time value
/// is frozen into the plan as a constant, and a lazy operand not yet computed is read as whatever its buffer held
/// (<c>TensorDivideScalar(x, TensorMaxValue(TensorAbs(x)))</c> compiled to a NaN loss). An inference trace must record
/// the host escape so compiling refuses the trace.
/// </description></item>
/// <item><description>
/// <b>TENS0041</b>: <c>DataUnmarked</c> (the writable accessor) whose memory is only ever read. The writable accessor
/// makes the tensor writable first, which copies a shared buffer and throws for a bf16/int8/int4 streaming weight, so a
/// convolution over quantized streaming weights failed. <c>ReadOnlyDataUnmarked</c> reads the same memory.
/// </description></item>
/// </list>
/// </summary>
/// <remarks>
/// TENS0040 applies to members of a class implementing <c>IEngine</c> that implement an <c>IEngine</c> member, return a
/// type with no tensor in it, and take a tensor. It is satisfied by a call to <c>BeginHostScalarRead</c>,
/// <c>GraphMode.ThrowIfActiveUnsupported</c> or another such host-value engine member (which is itself checked) that
/// runs on every call (not inside a branch, loop, lambda or short-circuit operand), or by <c>[HostValueFromMetadata]</c>
/// on a member that reads only shapes. TENS0041 follows the accessor's value through <c>AsFloatMemory</c>/<c>AsDoubleMemory</c>, <c>.Span</c>,
/// <c>Slice</c> and locals, and reports only when every use is a read; any use it cannot prove read-only (an element
/// write, a pin, an argument of a writable type) counts as a write.
/// </remarks>
[DiagnosticAnalyzer(LanguageNames.CSharp)]
public sealed class EngineGraphSafetyAnalyzer : DiagnosticAnalyzer
{
    internal const string EngineMetadataName = "AiDotNet.Tensors.Engines.IEngine";
    internal const string TensorNamespace = "AiDotNet.Tensors.LinearAlgebra";
    internal const string BoundaryMethodName = "BeginHostScalarRead";
    internal const string ActiveTraceGuardName = "ThrowIfActiveUnsupported";
    internal const string MetadataOnlyAttributeName = "HostValueFromMetadataAttribute";

    internal static readonly DiagnosticDescriptor HostValueWithoutBoundary = new(
        "TENS0040", "Engine operation returns a host value without a trace boundary",
        "'{0}' returns a host value computed from a tensor but never calls BeginHostScalarRead, so a training trace "
        + "freezes its value into the plan and reads a lazy operand before it is computed. Call "
        + "BeginHostScalarRead(operand) for each tensor it reads, or mark it [HostValueFromMetadata] if it reads only shapes.",
        "GraphSafety", DiagnosticSeverity.Error, isEnabledByDefault: true,
        description: "Graph traces record tensor operations only. An operation whose result leaves the graph as a host "
                     + "value must fail closed in a training trace and record the host escape in an inference trace.");

    internal static readonly DiagnosticDescriptor WritableAccessorForRead = new(
        "TENS0041", "Writable data accessor used only to read",
        "'{0}' is only read; use ReadOnlyDataUnmarked. The writable accessor copies a shared buffer and throws for a "
        + "quantized streaming weight.",
        "GraphSafety", DiagnosticSeverity.Error, isEnabledByDefault: true,
        description: "DataUnmarked makes the tensor's storage writable before returning it. Reading an operand through "
                     + "it breaks read-only storage (bf16/int8/int4 streaming weights) and copies shared buffers.");

    /// <inheritdoc/>
    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } =
        ImmutableArray.Create(HostValueWithoutBoundary, WritableAccessorForRead);

    /// <inheritdoc/>
    public override void Initialize(AnalysisContext context)
    {
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.EnableConcurrentExecution();
        context.RegisterCompilationStartAction(start =>
        {
            var engine = start.Compilation.GetTypeByMetadataName(EngineMetadataName);
            if (engine is not null)
                start.RegisterOperationAction(c => AnalyzeHostValue(c, engine), OperationKind.MethodBody);
            start.RegisterOperationAction(AnalyzeAccessor, OperationKind.PropertyReference);
        });
    }

    // ------------------------------------------------------------------ TENS0040

    private static void AnalyzeHostValue(OperationAnalysisContext context, INamedTypeSymbol engine)
    {
        if (context.ContainingSymbol is not IMethodSymbol method) return;
        if (!IsHostValueEngineMember(method, engine)) return;
        if (method.GetAttributes().Any(a => a.AttributeClass?.Name == MetadataOnlyAttributeName)) return;

        foreach (var invocation in context.Operation.Descendants().OfType<IInvocationOperation>())
        {
            // A boundary in a branch, a loop, a lambda or after a short-circuit protects only some calls.
            if (!RunsOnEveryCall(invocation, context.Operation)) continue;
            var target = invocation.TargetMethod;
            // GraphMode.ThrowIfActiveUnsupported refuses every trace kind, which is stricter than the boundary.
            // ThrowIfInferenceUnsupported and ThrowIfNonInferenceUnsupported each leave one trace kind unguarded.
            if (target.Name is BoundaryMethodName or ActiveTraceGuardName) return;
            if (!SymbolEqualityComparer.Default.Equals(target.OriginalDefinition, method.OriginalDefinition)
                && IsHostValueEngineMember(target, engine))
                return;
        }

        context.ReportDiagnostic(Diagnostic.Create(HostValueWithoutBoundary,
            method.Locations.FirstOrDefault() ?? context.Operation.Syntax.GetLocation(), method.Name));
    }

    // True when the invocation is reached on every call that reaches the end of its top-level statement: no enclosing
    // conditional, loop, switch, try/catch, lambda, local function, null-conditional or short-circuit operand.
    private static bool RunsOnEveryCall(IOperation invocation, IOperation body)
    {
        for (IOperation? child = invocation, parent = invocation.Parent; parent is not null; child = parent, parent = parent.Parent)
        {
            if (ReferenceEquals(parent, body)) return true;
            switch (parent)
            {
                case IConditionalOperation conditional when !ReferenceEquals(conditional.Condition, child):
                case ILoopOperation:
                case ISwitchOperation:
                case ISwitchExpressionOperation:
                case ITryOperation:
                case ICatchClauseOperation:
                case IAnonymousFunctionOperation:
                case ILocalFunctionOperation:
                case IConditionalAccessOperation:
                case ICoalesceOperation coalesce when !ReferenceEquals(coalesce.Value, child):
                case IBinaryOperation { OperatorKind: BinaryOperatorKind.ConditionalAnd or BinaryOperatorKind.ConditionalOr } binary
                    when !ReferenceEquals(binary.LeftOperand, child):
                    return false;
            }
        }
        return false;
    }

    private static bool IsHostValueEngineMember(IMethodSymbol method, INamedTypeSymbol engine)
    {
        method = method.OriginalDefinition;
        if (method.MethodKind is not (MethodKind.Ordinary or MethodKind.ExplicitInterfaceImplementation)) return false;
        if (method.ReturnsVoid || ContainsTensor(method.ReturnType)) return false;
        if (!method.Parameters.Any(p => p.RefKind is RefKind.None or RefKind.In && ContainsTensor(p.Type))) return false;

        if (SymbolEqualityComparer.Default.Equals(method.ContainingType, engine)) return true;
        var type = method.ContainingType;
        if (type is null || !type.AllInterfaces.Contains(engine, SymbolEqualityComparer.Default)) return false;
        foreach (var member in engine.GetMembers(method.Name).OfType<IMethodSymbol>())
        {
            var impl = type.FindImplementationForInterfaceMember(member);
            if (impl is IMethodSymbol m && SymbolEqualityComparer.Default.Equals(m.OriginalDefinition, method)) return true;
            // An override of the implementing virtual (a GPU engine overriding CpuEngine's member).
            for (var o = method.OverriddenMethod; o is not null; o = o.OverriddenMethod)
                if (impl is IMethodSymbol mi && SymbolEqualityComparer.Default.Equals(mi.OriginalDefinition, o.OriginalDefinition))
                    return true;
        }
        return false;
    }

    private static bool ContainsTensor(ITypeSymbol type)
    {
        switch (type)
        {
            case IArrayTypeSymbol array:
                return ContainsTensor(array.ElementType);
            case INamedTypeSymbol named:
                for (ITypeSymbol? t = named; t is not null; t = t.BaseType)
                    if (t.ContainingNamespace?.ToDisplayString() == TensorNamespace
                        && t.Name is "Tensor" or "TensorBase")
                        return true;
                if (named.IsTupleType) return named.TupleElements.Any(e => ContainsTensor(e.Type));
                return named.TypeArguments.Any(ContainsTensor);
            default:
                return false;
        }
    }

    // ------------------------------------------------------------------ TENS0041

    private static void AnalyzeAccessor(OperationAnalysisContext context)
    {
        var reference = (IPropertyReferenceOperation)context.Operation;
        var property = reference.Property;
        if (property.Name != "DataUnmarked"
            || property.ContainingType?.ContainingNamespace?.ToDisplayString() != TensorNamespace)
            return;
        var body = Root(reference);
        if (body is null) return;
        if (Classify(reference, body, depth: 0) == Use.Read)
            context.ReportDiagnostic(Diagnostic.Create(WritableAccessorForRead, reference.Syntax.GetLocation(),
                reference.Syntax.ToString()));
    }

    private enum Use { Read, Write }

    private static IOperation? Root(IOperation operation)
    {
        var current = operation;
        while (current.Parent is not null) current = current.Parent;
        return current;
    }

    // How the value of `value` (a Memory<T>/Span<T>-like expression) is used: Read only when every path is a proven read.
    private static Use Classify(IOperation value, IOperation body, int depth)
    {
        if (depth > 8) return Use.Write;
        var parent = value.Parent;
        while (parent is IParenthesizedOperation) { value = parent; parent = parent.Parent; }
        switch (parent)
        {
            case IConversionOperation conversion:
                if (IsReadOnlyMemoryLike(conversion.Type)) return Use.Read;
                return Classify(conversion, body, depth + 1);

            case IArgumentOperation argument:
                if (argument.Parameter is null) return Use.Write;
                if (IsReadOnlyMemoryLike(argument.Parameter.Type)) return Use.Read;
                if (argument.Parent is IInvocationOperation reinterpret && IsMemoryReinterpret(reinterpret.TargetMethod))
                    return Classify(reinterpret, body, depth + 1);
                return Use.Write;

            case IPropertyReferenceOperation member when ReferenceEquals(member.Instance, value):
                switch (member.Property.Name)
                {
                    case "Length":
                    case "IsEmpty":
                        return Use.Read;
                    case "Span":
                        return Classify(member, body, depth + 1);
                }
                if (member.Property.IsIndexer) return ElementUse(member);
                return Use.Write;

            case IInvocationOperation call when ReferenceEquals(call.Instance, value):
                switch (call.TargetMethod.Name)
                {
                    case "Slice":
                        return Classify(call, body, depth + 1);
                    case "ToArray":
                    case "ToString":
                        return Use.Read;
                    case "CopyTo":
                    case "TryCopyTo":
                        return Use.Read; // the receiver is the source
                }
                return Use.Write;

            case IVariableInitializerOperation initializer when initializer.Parent is IVariableDeclaratorOperation declarator:
                return LocalUses(declarator.Symbol, body, depth);

            case ISimpleAssignmentOperation assignment when ReferenceEquals(assignment.Value, value)
                                                           && assignment.Target is ILocalReferenceOperation local:
                return LocalUses(local.Local, body, depth);

            default:
                return Use.Write;
        }
    }

    private static Use ElementUse(IOperation element)
    {
        var parent = element.Parent;
        switch (parent)
        {
            case ISimpleAssignmentOperation a when ReferenceEquals(a.Target, element): return Use.Write;
            case ICompoundAssignmentOperation c when ReferenceEquals(c.Target, element): return Use.Write;
            case IIncrementOrDecrementOperation: return Use.Write;
            case IArgumentOperation arg when arg.Parameter?.RefKind is RefKind.Ref or RefKind.Out: return Use.Write;
            case IVariableInitializerOperation init
                when init.Parent is IVariableDeclaratorOperation d && d.Symbol.RefKind != RefKind.None:
                return Use.Write;
            // A field or property of a struct element: `span[0].Value = v` writes through the span.
            case IMemberReferenceOperation member when ReferenceEquals(member.Instance, element):
                return MayMutateThrough(element.Type);
            case IInvocationOperation call when ReferenceEquals(call.Instance, element):
                return MayMutateThrough(element.Type);
            default: return Use.Read;
        }
    }

    // A member of a mutable struct element can write through the span; built-in numeric, bool and char elements and
    // readonly structs cannot be written by their members (CompareTo, ToString).
    private static Use MayMutateThrough(ITypeSymbol? type) =>
        type is { IsValueType: true, IsReadOnly: false }
        && type.SpecialType is not (>= SpecialType.System_Boolean and <= SpecialType.System_Double)
            ? Use.Write
            : Use.Read;

    private static Use LocalUses(ILocalSymbol local, IOperation body, int depth)
    {
        if (local.RefKind != RefKind.None) return Use.Write;
        bool any = false;
        foreach (var reference in body.Descendants().OfType<ILocalReferenceOperation>())
        {
            if (!SymbolEqualityComparer.Default.Equals(reference.Local, local)) continue;
            if (reference.Parent is ISimpleAssignmentOperation a && ReferenceEquals(a.Target, reference)) continue;
            any = true;
            if (Classify(reference, body, depth + 1) == Use.Write) return Use.Write;
        }
        return any ? Use.Read : Use.Write;
    }

    private static bool IsReadOnlyMemoryLike(ITypeSymbol? type) =>
        type is INamedTypeSymbol named && named.ContainingNamespace?.ToDisplayString() == "System"
        && named.Name is "ReadOnlyMemory" or "ReadOnlySpan";

    // Helpers that reinterpret a Memory<T> as Memory<float>/Memory<double> without touching it.
    private static bool IsMemoryReinterpret(IMethodSymbol method) =>
        method.Name is "AsFloatMemory" or "AsDoubleMemory";
}
