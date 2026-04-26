// Copyright (c) AiDotNet. All rights reserved.
// CPU reference emitter: lowers a CodegenGraph into a .NET
// Expression tree, compiles it via LambdaExpression.Compile(), and
// wraps the result in a CodegenKernel. The JIT then applies its own
// vectorization + inlining on top of the element-wise code we emit,
// so "AVX-512 CPU" performance is recovered from the baseline JIT
// without us hand-rolling Vector512 IL.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.AvxCs;

/// <summary>
/// Emits a CPU kernel by lowering a <see cref="CodegenGraph"/> to a
/// <see cref="LambdaExpression"/> and compiling it with the
/// standard .NET JIT. Produces real native code (not interpreted
/// tree-walking); the JIT auto-vectorizes the element loop on
/// platforms that support it.
/// </summary>
/// <remarks>
/// <para><b>Why Expression trees and not Roslyn / IL emit:</b></para>
/// <para>
/// Expression trees give us a typed, JIT-backed delegate without
/// pulling in Roslyn as a dependency of the core library. Roslyn
/// adds ~10 MB of assemblies; IL emit is difficult to port across
/// net471/net10.0 and AOT-hostile. Expression trees compose cleanly
/// with the IR's integer-indexed producer model and produce real
/// native code on every target framework the library supports.
/// </para>
/// <para><b>Scope for Phase B:</b></para>
/// <para>
/// This emitter handles pointwise unary chains and binary pointwise
/// ops — the same scope as <see cref="CodegenLowering.LowerUnaryChain{T}"/>
/// / <see cref="CodegenLowering.LowerBinaryPointwise{T}"/>. Reductions
/// and matmuls aren't yet emitted; the emitter declines via
/// <see cref="CodegenEmitResult.Decline"/> so the fusion pass falls
/// back to the pre-existing composed-ops path. Phase C emitters
/// (Triton / HIP / MSL / WGSL / GLSL) build on the same IR so
/// extending this emitter to reductions is a pure additive change.
/// </para>
/// <para><b>Source dump:</b></para>
/// <para>
/// Alongside the compiled delegate the emitter produces a
/// human-readable C# source string — useful for telemetry (Phase G),
/// debug, and as documentation for the other emitters whose target
/// languages we cannot inspect as easily.
/// </para>
/// </remarks>
public sealed class CpuDotNetJitEmitter : IKernelEmitter
{
    /// <inheritdoc/>
    public CodegenTarget Target => CodegenTarget.CpuDotNetJit;

    /// <inheritdoc/>
    public CodegenEmitResult Emit(CodegenGraph graph, CodegenElementType dtype)
    {
        if (graph is null) throw new ArgumentNullException(nameof(graph));

        // Reject empty graphs up front — the shape-uniformity check
        // below indexes Nodes[0] and would otherwise throw an
        // IndexOutOfRangeException, which is a noisier failure mode
        // than the structured Decline path callers already handle.
        if (graph.Count == 0)
            return CodegenEmitResult.Decline("CpuDotNetJitEmitter received an empty graph.");

        // Reject element types we don't implement yet. Keeping this
        // restrictive until Phase D exercises the full CodegenElementType
        // set — silently succeeding on an unsupported dtype would
        // produce wrong results.
        if (dtype != CodegenElementType.Float32 && dtype != CodegenElementType.Float64)
            return CodegenEmitResult.Decline(
                $"CpuDotNetJitEmitter currently only supports Float32/Float64, not {dtype}.");

        // Phase B scope: pointwise-only. Any reduction / matmul /
        // attention / opaque / movement op forces a decline so the
        // fusion pass keeps the existing composed-ops path for those.
        foreach (var node in graph.Nodes)
        {
            var cat = CodegenOpKinds.Categorize(node.Op);
            if (cat != CodegenOpCategory.Pointwise && cat != CodegenOpCategory.Movement)
                return CodegenEmitResult.Decline(
                    $"Phase B CPU emitter does not yet handle {cat} ops (found {node.Op}).");
        }

        // Every output must ultimately be the same shape as every
        // input (pointwise). Shape mismatches indicate broadcast /
        // reshape which isn't pointwise-fusable without a layout
        // rewrite pass.
        var firstShape = graph.Nodes[0].Shape;
        long elementCount = 1;
        for (int i = 0; i < firstShape.Length; i++) elementCount *= firstShape[i];
        foreach (var node in graph.Nodes)
        {
            long nc = 1;
            for (int i = 0; i < node.Shape.Length; i++) nc *= node.Shape[i];
            if (nc != elementCount)
                return CodegenEmitResult.Decline(
                    $"Phase B CPU emitter requires all graph nodes to share element count; "
                  + $"first={elementCount}, mismatch at op {node.Op}={nc}.");
        }

        // The compiled kernel currently passes elementCount as int (the
        // Expression-tree loop counter is int — moving to long would
        // make every array index a long-to-int conversion). Reject
        // graphs that exceed int.MaxValue elements so the (int) cast
        // below at the CompiledCpuKernel ctor cannot silently truncate
        // and produce wrong results.
        if (elementCount > int.MaxValue)
            return CodegenEmitResult.Decline(
                $"CpuDotNetJitEmitter cannot emit kernels for graphs with element count "
              + $"{elementCount} (exceeds int.MaxValue = {int.MaxValue}); the loop counter is int.");

        return dtype switch
        {
            CodegenElementType.Float32 => EmitTyped<float>(graph, dtype),
            CodegenElementType.Float64 => EmitTyped<double>(graph, dtype),
            _ => CodegenEmitResult.Decline($"Unreachable — dtype check above excluded {dtype}."),
        };
    }

    private static CodegenEmitResult EmitTyped<T>(CodegenGraph graph, CodegenElementType dtype)
        where T : unmanaged
    {
        // ─── 1. Build the expression tree ────────────────────────────

        // The emitted delegate signature is:
        //   void Kernel(T[][] inputs, T[][] outputs, int elementCount)
        //
        // Arrays (not Memory<T>/Span<T>) because Span<T> is a ref
        // struct and cannot be stored in Expression-tree locals —
        // the JIT would reject it with "Property cannot have a
        // managed pointer type". Arrays give us array-indexing
        // Expression nodes directly, and the JIT still applies
        // its auto-vectorization pass on the resulting body.

        var inputsParam = Expression.Parameter(typeof(T[][]), "inputs");
        var outputsParam = Expression.Parameter(typeof(T[][]), "outputs");
        var countParam = Expression.Parameter(typeof(int), "count");
        var iVar = Expression.Variable(typeof(int), "i");
        var body = new List<Expression>();

        // Pre-index each input / output buffer once outside the loop.
        var inputArrays = new ParameterExpression[graph.InputNodes.Count];
        var outputArrays = new ParameterExpression[graph.OutputNodes.Count];

        for (int i = 0; i < graph.InputNodes.Count; i++)
        {
            var arr = Expression.Variable(typeof(T[]), $"input_{i}");
            inputArrays[i] = arr;
            body.Add(Expression.Assign(
                arr, Expression.ArrayIndex(inputsParam, Expression.Constant(i))));
        }

        for (int i = 0; i < graph.OutputNodes.Count; i++)
        {
            var arr = Expression.Variable(typeof(T[]), $"output_{i}");
            outputArrays[i] = arr;
            body.Add(Expression.Assign(
                arr, Expression.ArrayIndex(outputsParam, Expression.Constant(i))));
        }

        // Variables holding the per-node intermediate values, one per
        // graph node. LoadInput and StoreOutput have special
        // treatments; interior nodes compute a local T variable.
        var nodeValue = new Expression[graph.Count];

        // ─── 2. Loop body ────────────────────────────────────────────

        var loopBody = new List<Expression>();
        int inputBindingIdx = 0;
        int outputBindingIdx = 0;

        for (int nidx = 0; nidx < graph.Count; nidx++)
        {
            var node = graph[nidx];
            switch (node.Op)
            {
                case CodegenOpKind.LoadInput:
                {
                    // T v = input_k[i];
                    var v = Expression.Variable(typeof(T), $"v{nidx}");
                    loopBody.Add(v);
                    var arrRef = inputArrays[inputBindingIdx++];
                    loopBody.Add(Expression.Assign(v, Expression.ArrayAccess(arrRef, iVar)));
                    nodeValue[nidx] = v;
                    break;
                }

                case CodegenOpKind.StoreOutput:
                {
                    // output_k[i] = v_input;
                    var arrRef = outputArrays[outputBindingIdx++];
                    loopBody.Add(Expression.Assign(
                        Expression.ArrayAccess(arrRef, iVar), nodeValue[node.Inputs[0]]));
                    // Store nodes have no value the graph references later.
                    nodeValue[nidx] = nodeValue[node.Inputs[0]];
                    break;
                }

                default:
                {
                    var v = Expression.Variable(typeof(T), $"v{nidx}");
                    loopBody.Add(v);
                    Expression expr;
                    try
                    {
                        expr = BuildPointwiseOp<T>(node, nodeValue);
                    }
                    catch (NotSupportedException ex)
                    {
                        return CodegenEmitResult.Decline(
                            $"CpuDotNetJitEmitter cannot lower {node.Op}: {ex.Message}");
                    }
                    loopBody.Add(Expression.Assign(v, expr));
                    nodeValue[nidx] = v;
                    break;
                }
            }
        }

        // Wrap the loop: for (int i = 0; i < count; i++) { ... }
        var breakLabel = Expression.Label("break");
        var loop = Expression.Block(
            variables: CollectLocalVars(loopBody),
            expressions: new Expression[]
            {
                Expression.Assign(iVar, Expression.Constant(0)),
                Expression.Loop(
                    Expression.IfThenElse(
                        Expression.LessThan(iVar, countParam),
                        Expression.Block(
                            Expression.Block(loopBody),
                            Expression.PostIncrementAssign(iVar)),
                        Expression.Break(breakLabel)),
                    breakLabel),
            });

        body.Add(loop);

        var outerBlock = Expression.Block(
            variables: CollectOuterVars(inputArrays, outputArrays, iVar),
            expressions: body);

        var lambda = Expression.Lambda<KernelDelegate<T>>(outerBlock, inputsParam, outputsParam, countParam);
        var compiled = lambda.Compile();

        // ─── 3. Produce the source dump ──────────────────────────────

        var source = DumpSource(graph, dtype);

        return CodegenEmitResult.Succeeded(
            new CompiledCpuKernel<T>(graph, dtype, compiled, (int)graph.Nodes[0].ElementCount),
            source);
    }

    // Collects variables declared as the first element of each
    // `Expression.Block(variable, assignment)` triple produced above.
    private static IEnumerable<ParameterExpression> CollectLocalVars(List<Expression> exprs)
    {
        foreach (var e in exprs)
            if (e is ParameterExpression p) yield return p;
    }

    private static IEnumerable<ParameterExpression> CollectOuterVars(
        ParameterExpression[] inputSpans,
        ParameterExpression[] outputSpans,
        ParameterExpression iVar)
    {
        foreach (var s in inputSpans) yield return s;
        foreach (var s in outputSpans) yield return s;
        yield return iVar;
    }

    /// <summary>
    /// Lowers a pointwise op node to the Expression subtree that
    /// computes its value given already-computed producer values.
    /// Unary ops read <c>node.Inputs[0]</c>; binary ops read
    /// <c>node.Inputs[0]</c> and <c>node.Inputs[1]</c>.
    /// </summary>
    private static Expression BuildPointwiseOp<T>(CodegenNode node, Expression[] nodeValue)
    {
        switch (node.Op)
        {
            // Binary arithmetic.
            case CodegenOpKind.Add:
                return Expression.Add(nodeValue[node.Inputs[0]], nodeValue[node.Inputs[1]]);
            case CodegenOpKind.Sub:
                return Expression.Subtract(nodeValue[node.Inputs[0]], nodeValue[node.Inputs[1]]);
            case CodegenOpKind.Mul:
                return Expression.Multiply(nodeValue[node.Inputs[0]], nodeValue[node.Inputs[1]]);
            case CodegenOpKind.Div:
                return Expression.Divide(nodeValue[node.Inputs[0]], nodeValue[node.Inputs[1]]);

            // Unary arithmetic.
            case CodegenOpKind.Negate:
                return Expression.Negate(nodeValue[node.Inputs[0]]);

            // Transcendentals — call the Math / MathF static. Use
            // Math.* for double, MathF.* for float.
            case CodegenOpKind.Exp:
                return CallMath<T>(nameof(Math.Exp), nameof(MathF.Exp), nodeValue[node.Inputs[0]]);
            case CodegenOpKind.Log:
                return CallMath<T>(nameof(Math.Log), nameof(MathF.Log), nodeValue[node.Inputs[0]]);
            case CodegenOpKind.Sqrt:
                return CallMath<T>(nameof(Math.Sqrt), nameof(MathF.Sqrt), nodeValue[node.Inputs[0]]);
            case CodegenOpKind.Sin:
                return CallMath<T>(nameof(Math.Sin), nameof(MathF.Sin), nodeValue[node.Inputs[0]]);
            case CodegenOpKind.Cos:
                return CallMath<T>(nameof(Math.Cos), nameof(MathF.Cos), nodeValue[node.Inputs[0]]);
            case CodegenOpKind.Tanh:
                return CallMath<T>(nameof(Math.Tanh), nameof(MathF.Tanh), nodeValue[node.Inputs[0]]);
            case CodegenOpKind.Abs:
                return CallMath<T>(nameof(Math.Abs), nameof(MathF.Abs), nodeValue[node.Inputs[0]]);

            // Activations — expressed in terms of the primitives above.
            case CodegenOpKind.ReLU:
                // max(x, 0)
                return CallMath<T>(nameof(Math.Max), nameof(MathF.Max),
                    nodeValue[node.Inputs[0]], Expression.Constant(default(T)));
            case CodegenOpKind.Sigmoid:
            {
                // 1 / (1 + exp(-x))
                var x = nodeValue[node.Inputs[0]];
                var negX = Expression.Negate(x);
                var exp = CallMath<T>(nameof(Math.Exp), nameof(MathF.Exp), negX);
                var one = ConstantOne<T>();
                return Expression.Divide(one, Expression.Add(one, exp));
            }
            case CodegenOpKind.Tan:
                return CallMath<T>(nameof(Math.Tan), nameof(MathF.Tan), nodeValue[node.Inputs[0]]);
            case CodegenOpKind.Floor:
                return CallMath<T>(nameof(Math.Floor), nameof(MathF.Floor), nodeValue[node.Inputs[0]]);
            case CodegenOpKind.Ceil:
                return CallMath<T>(nameof(Math.Ceiling), nameof(MathF.Ceiling), nodeValue[node.Inputs[0]]);
            case CodegenOpKind.Round:
                return CallMath<T>(nameof(Math.Round), nameof(MathF.Round), nodeValue[node.Inputs[0]]);
            case CodegenOpKind.Max:
                return CallMath<T>(nameof(Math.Max), nameof(MathF.Max),
                    nodeValue[node.Inputs[0]], nodeValue[node.Inputs[1]]);
            case CodegenOpKind.Min:
                return CallMath<T>(nameof(Math.Min), nameof(MathF.Min),
                    nodeValue[node.Inputs[0]], nodeValue[node.Inputs[1]]);

            default:
                throw new NotSupportedException(
                    $"Pointwise op {node.Op} not yet implemented in CpuDotNetJitEmitter.");
        }
    }

    private static Expression CallMath<T>(string mathDoubleName, string mathfFloatName, params Expression[] args)
    {
        Type host;
        string name;
        Type argType;
        if (typeof(T) == typeof(float)) { host = typeof(MathF); name = mathfFloatName; argType = typeof(float); }
        else if (typeof(T) == typeof(double)) { host = typeof(Math); name = mathDoubleName; argType = typeof(double); }
        else throw new NotSupportedException(
            $"CallMath: unsupported T = {typeof(T).Name}. CpuDotNetJitEmitter specialises only float/double.");

        var parameterTypes = new Type[args.Length];
        for (int i = 0; i < args.Length; i++) parameterTypes[i] = argType;
        var method = host.GetMethod(name, parameterTypes);
        if (method is null)
            throw new NotSupportedException(
                $"{host.Name}.{name}({string.Join(",", parameterTypes.Select(t => t.Name))}) not found — "
              + "runtime may be too old.");
        return Expression.Call(method, args);
    }

    private static Expression ConstantOne<T>()
    {
        if (typeof(T) == typeof(float)) return Expression.Constant(1f);
        if (typeof(T) == typeof(double)) return Expression.Constant(1.0);
        throw new NotSupportedException($"ConstantOne<{typeof(T).Name}> — T must be float or double.");
    }

    // Serializes the graph as human-readable C# — matches what we'd
    // hand-write for the equivalent kernel. Used for Phase G telemetry
    // and for developers to sanity-check the emitted code without
    // stepping through the compiled delegate.
    internal static string DumpSource(CodegenGraph graph, CodegenElementType dtype)
    {
        var sb = new System.Text.StringBuilder();
        string scalar = dtype switch
        {
            CodegenElementType.Float32 => "float",
            CodegenElementType.Float64 => "double",
            _ => dtype.ToString().ToLowerInvariant(),
        };
        sb.AppendLine($"// Auto-generated CPU kernel — {graph.Count} nodes, dtype={dtype}");
        sb.AppendLine($"void Kernel(ReadOnlyMemory<{scalar}>[] inputs, Memory<{scalar}>[] outputs, int count)");
        sb.AppendLine("{");
        for (int i = 0; i < graph.InputNodes.Count; i++)
            sb.AppendLine($"    ReadOnlySpan<{scalar}> input_{i} = inputs[{i}].Span;");
        for (int i = 0; i < graph.OutputNodes.Count; i++)
            sb.AppendLine($"    Span<{scalar}> output_{i} = outputs[{i}].Span;");
        sb.AppendLine("    for (int i = 0; i < count; i++)");
        sb.AppendLine("    {");
        int inputPort = 0, outputPort = 0;
        for (int n = 0; n < graph.Count; n++)
        {
            var node = graph[n];
            switch (node.Op)
            {
                case CodegenOpKind.LoadInput:
                    sb.AppendLine($"        {scalar} v{n} = input_{inputPort++}[i];");
                    break;
                case CodegenOpKind.StoreOutput:
                    sb.AppendLine($"        output_{outputPort++}[i] = v{node.Inputs[0]};");
                    break;
                default:
                    sb.AppendLine($"        {scalar} v{n} = {FormatExpression(node, scalar)};");
                    break;
            }
        }
        sb.AppendLine("    }");
        sb.AppendLine("}");
        return sb.ToString();
    }

    private static string FormatExpression(CodegenNode node, string scalar) => node.Op switch
    {
        CodegenOpKind.Add => $"v{node.Inputs[0]} + v{node.Inputs[1]}",
        CodegenOpKind.Sub => $"v{node.Inputs[0]} - v{node.Inputs[1]}",
        CodegenOpKind.Mul => $"v{node.Inputs[0]} * v{node.Inputs[1]}",
        CodegenOpKind.Div => $"v{node.Inputs[0]} / v{node.Inputs[1]}",
        CodegenOpKind.Negate => $"-v{node.Inputs[0]}",
        CodegenOpKind.Exp => $"{MathHost(scalar)}.Exp(v{node.Inputs[0]})",
        CodegenOpKind.Log => $"{MathHost(scalar)}.Log(v{node.Inputs[0]})",
        CodegenOpKind.Sqrt => $"{MathHost(scalar)}.Sqrt(v{node.Inputs[0]})",
        CodegenOpKind.Sin => $"{MathHost(scalar)}.Sin(v{node.Inputs[0]})",
        CodegenOpKind.Cos => $"{MathHost(scalar)}.Cos(v{node.Inputs[0]})",
        CodegenOpKind.Tan => $"{MathHost(scalar)}.Tan(v{node.Inputs[0]})",
        CodegenOpKind.Tanh => $"{MathHost(scalar)}.Tanh(v{node.Inputs[0]})",
        CodegenOpKind.Abs => $"{MathHost(scalar)}.Abs(v{node.Inputs[0]})",
        CodegenOpKind.Floor => $"{MathHost(scalar)}.Floor(v{node.Inputs[0]})",
        CodegenOpKind.Ceil => $"{MathHost(scalar)}.Ceiling(v{node.Inputs[0]})",
        CodegenOpKind.Round => $"{MathHost(scalar)}.Round(v{node.Inputs[0]})",
        CodegenOpKind.ReLU => $"{MathHost(scalar)}.Max(v{node.Inputs[0]}, 0)",
        CodegenOpKind.Sigmoid => $"1 / (1 + {MathHost(scalar)}.Exp(-v{node.Inputs[0]}))",
        CodegenOpKind.Max => $"{MathHost(scalar)}.Max(v{node.Inputs[0]}, v{node.Inputs[1]})",
        CodegenOpKind.Min => $"{MathHost(scalar)}.Min(v{node.Inputs[0]}, v{node.Inputs[1]})",
        _ => $"/* unsupported {node.Op} */",
    };

    private static string MathHost(string scalar) => scalar == "float" ? "MathF" : "Math";
}

/// <summary>
/// The compiled-delegate signature produced by
/// <see cref="CpuDotNetJitEmitter"/>. Fixed shape so all emitted
/// kernels share the same callable form. Uses jagged
/// <c>T[][]</c> rather than <c>Memory&lt;T&gt;[]</c> because Span-
/// based types are ref structs and cannot be stored in Expression-
/// tree locals.
/// </summary>
public delegate void KernelDelegate<T>(T[][] inputs, T[][] outputs, int count)
    where T : unmanaged;

/// <summary>
/// A kernel backed by a compiled <see cref="KernelDelegate{T}"/>.
/// </summary>
internal sealed class CompiledCpuKernel<TElement> : CodegenKernel
    where TElement : unmanaged
{
    private readonly KernelDelegate<TElement> _delegate;
    private readonly int _elementCount;

    public CompiledCpuKernel(
        CodegenGraph graph,
        CodegenElementType dtype,
        KernelDelegate<TElement> compiled,
        int elementCount)
        : base(dtype, graph, CodegenTarget.CpuDotNetJit)
    {
        _delegate = compiled;
        _elementCount = elementCount;
    }

    /// <inheritdoc/>
    public override void Execute<T>(T[][] inputs, T[][] outputs)
    {
        if (typeof(T) != typeof(TElement))
            throw new ArgumentException(
                $"Kernel specialised for {typeof(TElement).Name} — caller passed {typeof(T).Name}.",
                nameof(inputs));
        if (inputs is null) throw new ArgumentNullException(nameof(inputs));
        if (outputs is null) throw new ArgumentNullException(nameof(outputs));
        if (inputs.Length != InputCount)
            throw new ArgumentException(
                $"Expected {InputCount} input buffers, got {inputs.Length}.", nameof(inputs));
        if (outputs.Length != OutputCount)
            throw new ArgumentException(
                $"Expected {OutputCount} output buffers, got {outputs.Length}.", nameof(outputs));

        // Reinterpret — safe because the type check above guarantees T == TElement.
        var typedInputs = (TElement[][])(object)inputs;
        var typedOutputs = (TElement[][])(object)outputs;
        _delegate(typedInputs, typedOutputs, _elementCount);
    }
}
