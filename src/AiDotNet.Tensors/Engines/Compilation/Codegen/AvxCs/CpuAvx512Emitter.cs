// Copyright (c) AiDotNet. All rights reserved.
// AVX-512 hand-emitted intrinsic kernel emitter — Phase B for issue #225.
//
// Companion to CpuDotNetJitEmitter. Where the JIT emitter relies on
// RyuJIT auto-vectorization (which RyuJIT may or may not perform
// depending on heuristics), this emitter directly emits Vector512<T>
// loads + Avx512F intrinsic calls so vectorisation is a guaranteed
// part of the compiled kernel, not a hope.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
#if NET8_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.AvxCs;

/// <summary>
/// Emits a pointwise kernel that issues real AVX-512F intrinsic calls
/// (16-wide on Float32, 8-wide on Float64) plus a scalar tail for any
/// remainder. Selected over <see cref="CpuDotNetJitEmitter"/> when the
/// running CPU advertises AVX-512F and the graph fits the supported
/// pointwise op subset (Add / Sub / Mul / Negate / Sqrt / ReLU /
/// LoadInput / StoreOutput). Other ops force a clean Decline so the
/// fusion pass falls back to the JIT emitter or the pre-built
/// composed-ops path.
/// </summary>
/// <remarks>
/// <para><b>Why a separate emitter:</b></para>
/// <para>
/// RyuJIT can auto-vectorize the JIT emitter's loop, but the decision
/// is opaque: a small refactor of the emitted IR or a change in JIT
/// heuristics can silently regress the kernel back to scalar.
/// Hand-emitting <see cref="Avx512F"/> intrinsics guarantees the
/// vector ISA is the issued machine code, lets us pick the unroll
/// factor explicitly, and gives the IR a real second emitter to
/// validate the contract Phase C requires (multiple emitters fed by
/// one IR).
/// </para>
/// <para><b>Why no transcendentals (Exp/Tanh/Sigmoid):</b></para>
/// <para>
/// .NET 10 does not yet ship vectorised <c>Avx512F.Exp</c> or
/// equivalents — emitting a ~12-term polynomial approximation is
/// out of scope for issue #225's acceptance criteria. Graphs with
/// Exp/Tanh/Sigmoid Decline cleanly so the JIT emitter (which uses
/// scalar <see cref="Math"/> calls under the loop) handles them
/// without wrong results.
/// </para>
/// </remarks>
public sealed class CpuAvx512Emitter : IKernelEmitter
{
    /// <inheritdoc/>
    public CodegenTarget Target => CodegenTarget.CpuAvx512;

    /// <inheritdoc/>
    public CodegenEmitResult Emit(CodegenGraph graph, CodegenElementType dtype)
    {
        if (graph is null) throw new ArgumentNullException(nameof(graph));

#if !NET8_0_OR_GREATER
        return CodegenEmitResult.Decline(
            "CpuAvx512Emitter requires net8.0+ for System.Runtime.Intrinsics.X86.Avx512F.");
#else
        if (!Avx512F.IsSupported)
            return CodegenEmitResult.Decline("AVX-512F is not available on this CPU.");

        if (graph.Count == 0)
            return CodegenEmitResult.Decline("CpuAvx512Emitter received an empty graph.");

        if (dtype != CodegenElementType.Float32 && dtype != CodegenElementType.Float64)
            return CodegenEmitResult.Decline(
                $"CpuAvx512Emitter currently only supports Float32/Float64, not {dtype}.");

        // Allowed-op gate. Anything outside this set forces a Decline
        // — the JIT emitter or the composed-ops path will handle the
        // op via its scalar Math calls (which we'd need a SIMD math
        // library to vectorise).
        foreach (var node in graph.Nodes)
        {
            if (!IsSupported(node.Op))
                return CodegenEmitResult.Decline(
                    $"CpuAvx512Emitter cannot emit {node.Op} (no direct AVX-512F intrinsic).");
        }

        // Shape uniformity gate (mirrors the JIT emitter's check).
        var firstShape = graph.Nodes[0].Shape;
        long elementCount = 1;
        for (int i = 0; i < firstShape.Length; i++) elementCount *= firstShape[i];
        foreach (var node in graph.Nodes)
        {
            long nc = 1;
            for (int i = 0; i < node.Shape.Length; i++) nc *= node.Shape[i];
            if (nc != elementCount)
                return CodegenEmitResult.Decline(
                    $"CpuAvx512Emitter requires shape uniformity; mismatch at {node.Op}.");
        }
        if (elementCount > int.MaxValue)
            return CodegenEmitResult.Decline(
                $"CpuAvx512Emitter cannot emit graphs with element count {elementCount} > int.MaxValue.");

        return dtype switch
        {
            CodegenElementType.Float32 => EmitFloat32(graph, (int)elementCount),
            CodegenElementType.Float64 => EmitFloat64(graph, (int)elementCount),
            _ => CodegenEmitResult.Decline("Unreachable.")
        };
#endif
    }

    private static bool IsSupported(CodegenOpKind op) => op switch
    {
        CodegenOpKind.LoadInput => true,
        CodegenOpKind.StoreOutput => true,
        CodegenOpKind.Add => true,
        CodegenOpKind.Sub => true,
        CodegenOpKind.Mul => true,
        CodegenOpKind.Negate => true,
        CodegenOpKind.Sqrt => true,
        CodegenOpKind.ReLU => true,
        _ => false,
    };

#if NET8_0_OR_GREATER
    private static CodegenEmitResult EmitFloat32(CodegenGraph graph, int elementCount)
    {
        // Closure captures the graph and produces a delegate that
        // (1) runs the 16-wide AVX-512F loop, (2) handles the scalar
        // tail with Math.Max-style ops. Using a closure rather than
        // Expression trees because Vector512<float> is a struct that
        // Expression supports awkwardly — a hand-coded delegate keeps
        // the SIMD body readable and the scalar tail trivial.

        // Pre-resolve nodes into a flat schedule. nodeKind[i] is the op,
        // nodeIn[i] is the (up to 2) producer indices, nodeBindIdx[i]
        // is the input/output buffer index for LoadInput/StoreOutput.
        int n = graph.Count;
        var nodeKind = new CodegenOpKind[n];
        var nodeIn0 = new int[n];
        var nodeIn1 = new int[n];
        var nodeBindIdx = new int[n];
        int inputCursor = 0;
        int outputCursor = 0;
        for (int i = 0; i < n; i++)
        {
            var node = graph[i];
            nodeKind[i] = node.Op;
            nodeIn0[i] = node.Inputs.Length > 0 ? node.Inputs[0] : -1;
            nodeIn1[i] = node.Inputs.Length > 1 ? node.Inputs[1] : -1;
            nodeBindIdx[i] = node.Op switch
            {
                CodegenOpKind.LoadInput => inputCursor++,
                CodegenOpKind.StoreOutput => outputCursor++,
                _ => -1
            };
        }

        int inputCount = graph.InputNodes.Count;
        int outputCount = graph.OutputNodes.Count;
        int count = elementCount;

        unsafe void Kernel(float[][] inputs, float[][] outputs, int len)
        {
            if (inputs.Length != inputCount || outputs.Length != outputCount)
                throw new ArgumentException("input/output buffer count mismatch.");

            // Bounds-check: the kernel was specialised for 'count'
            // elements; supplied buffers must accommodate it.
            for (int b = 0; b < inputs.Length; b++)
                if (inputs[b].Length < count)
                    throw new ArgumentException($"Input buffer {b} too small for kernel ({inputs[b].Length} < {count}).");
            for (int b = 0; b < outputs.Length; b++)
                if (outputs[b].Length < count)
                    throw new ArgumentException($"Output buffer {b} too small for kernel ({outputs[b].Length} < {count}).");

            // Per-node value slot; v[i] is the current Vector512<float>
            // for node i. SIMD loop fills these top-down each iteration.
            var v = new Vector512<float>[n];

            int simdEnd = count & ~15;
            for (int i = 0; i < simdEnd; i += 16)
            {
                for (int k = 0; k < n; k++)
                {
                    switch (nodeKind[k])
                    {
                        case CodegenOpKind.LoadInput:
                            fixed (float* p = inputs[nodeBindIdx[k]])
                                v[k] = Avx512F.LoadVector512(p + i);
                            break;
                        case CodegenOpKind.StoreOutput:
                            fixed (float* p = outputs[nodeBindIdx[k]])
                                Avx512F.Store(p + i, v[nodeIn0[k]]);
                            break;
                        case CodegenOpKind.Add:
                            v[k] = Avx512F.Add(v[nodeIn0[k]], v[nodeIn1[k]]);
                            break;
                        case CodegenOpKind.Sub:
                            v[k] = Avx512F.Subtract(v[nodeIn0[k]], v[nodeIn1[k]]);
                            break;
                        case CodegenOpKind.Mul:
                            v[k] = Avx512F.Multiply(v[nodeIn0[k]], v[nodeIn1[k]]);
                            break;
                        case CodegenOpKind.Negate:
                            v[k] = Avx512F.Subtract(Vector512<float>.Zero, v[nodeIn0[k]]);
                            break;
                        case CodegenOpKind.Sqrt:
                            v[k] = Avx512F.Sqrt(v[nodeIn0[k]]);
                            break;
                        case CodegenOpKind.ReLU:
                            // NaN-preserving: see CpuFusedOperations.cs comment.
                            // (NOT (v < 0)) AND v keeps NaN bit pattern intact.
                            var lt = Avx512F.CompareLessThan(v[nodeIn0[k]], Vector512<float>.Zero);
                            v[k] = Avx512F.AndNot(lt.AsUInt32(), v[nodeIn0[k]].AsUInt32()).AsSingle();
                            break;
                    }
                }
            }

            // Scalar tail for the trailing < 16 elements.
            var s = new float[n];
            for (int i = simdEnd; i < count; i++)
            {
                for (int k = 0; k < n; k++)
                {
                    switch (nodeKind[k])
                    {
                        case CodegenOpKind.LoadInput: s[k] = inputs[nodeBindIdx[k]][i]; break;
                        case CodegenOpKind.StoreOutput: outputs[nodeBindIdx[k]][i] = s[nodeIn0[k]]; break;
                        case CodegenOpKind.Add: s[k] = s[nodeIn0[k]] + s[nodeIn1[k]]; break;
                        case CodegenOpKind.Sub: s[k] = s[nodeIn0[k]] - s[nodeIn1[k]]; break;
                        case CodegenOpKind.Mul: s[k] = s[nodeIn0[k]] * s[nodeIn1[k]]; break;
                        case CodegenOpKind.Negate: s[k] = -s[nodeIn0[k]]; break;
                        case CodegenOpKind.Sqrt: s[k] = MathF.Sqrt(s[nodeIn0[k]]); break;
                        case CodegenOpKind.ReLU: s[k] = s[nodeIn0[k]] < 0f ? 0f : s[nodeIn0[k]]; break;
                    }
                }
            }
        }

        var source = DumpSource(graph, CodegenElementType.Float32, "float");
        var kernel = new CompiledAvx512Kernel<float>(graph, CodegenElementType.Float32, Kernel, count);
        return CodegenEmitResult.Succeeded(kernel, source);
    }

    private static CodegenEmitResult EmitFloat64(CodegenGraph graph, int elementCount)
    {
        int n = graph.Count;
        var nodeKind = new CodegenOpKind[n];
        var nodeIn0 = new int[n];
        var nodeIn1 = new int[n];
        var nodeBindIdx = new int[n];
        int inputCursor = 0;
        int outputCursor = 0;
        for (int i = 0; i < n; i++)
        {
            var node = graph[i];
            nodeKind[i] = node.Op;
            nodeIn0[i] = node.Inputs.Length > 0 ? node.Inputs[0] : -1;
            nodeIn1[i] = node.Inputs.Length > 1 ? node.Inputs[1] : -1;
            nodeBindIdx[i] = node.Op switch
            {
                CodegenOpKind.LoadInput => inputCursor++,
                CodegenOpKind.StoreOutput => outputCursor++,
                _ => -1
            };
        }

        int inputCount = graph.InputNodes.Count;
        int outputCount = graph.OutputNodes.Count;
        int count = elementCount;

        unsafe void Kernel(double[][] inputs, double[][] outputs, int len)
        {
            if (inputs.Length != inputCount || outputs.Length != outputCount)
                throw new ArgumentException("input/output buffer count mismatch.");
            for (int b = 0; b < inputs.Length; b++)
                if (inputs[b].Length < count)
                    throw new ArgumentException($"Input buffer {b} too small for kernel ({inputs[b].Length} < {count}).");
            for (int b = 0; b < outputs.Length; b++)
                if (outputs[b].Length < count)
                    throw new ArgumentException($"Output buffer {b} too small for kernel ({outputs[b].Length} < {count}).");

            var v = new Vector512<double>[n];
            int simdEnd = count & ~7;
            for (int i = 0; i < simdEnd; i += 8)
            {
                for (int k = 0; k < n; k++)
                {
                    switch (nodeKind[k])
                    {
                        case CodegenOpKind.LoadInput:
                            fixed (double* p = inputs[nodeBindIdx[k]])
                                v[k] = Avx512F.LoadVector512(p + i);
                            break;
                        case CodegenOpKind.StoreOutput:
                            fixed (double* p = outputs[nodeBindIdx[k]])
                                Avx512F.Store(p + i, v[nodeIn0[k]]);
                            break;
                        case CodegenOpKind.Add: v[k] = Avx512F.Add(v[nodeIn0[k]], v[nodeIn1[k]]); break;
                        case CodegenOpKind.Sub: v[k] = Avx512F.Subtract(v[nodeIn0[k]], v[nodeIn1[k]]); break;
                        case CodegenOpKind.Mul: v[k] = Avx512F.Multiply(v[nodeIn0[k]], v[nodeIn1[k]]); break;
                        case CodegenOpKind.Negate: v[k] = Avx512F.Subtract(Vector512<double>.Zero, v[nodeIn0[k]]); break;
                        case CodegenOpKind.Sqrt: v[k] = Avx512F.Sqrt(v[nodeIn0[k]]); break;
                        case CodegenOpKind.ReLU:
                            var lt = Avx512F.CompareLessThan(v[nodeIn0[k]], Vector512<double>.Zero);
                            v[k] = Avx512F.AndNot(lt.AsUInt64(), v[nodeIn0[k]].AsUInt64()).AsDouble();
                            break;
                    }
                }
            }
            var s = new double[n];
            for (int i = simdEnd; i < count; i++)
            {
                for (int k = 0; k < n; k++)
                {
                    switch (nodeKind[k])
                    {
                        case CodegenOpKind.LoadInput: s[k] = inputs[nodeBindIdx[k]][i]; break;
                        case CodegenOpKind.StoreOutput: outputs[nodeBindIdx[k]][i] = s[nodeIn0[k]]; break;
                        case CodegenOpKind.Add: s[k] = s[nodeIn0[k]] + s[nodeIn1[k]]; break;
                        case CodegenOpKind.Sub: s[k] = s[nodeIn0[k]] - s[nodeIn1[k]]; break;
                        case CodegenOpKind.Mul: s[k] = s[nodeIn0[k]] * s[nodeIn1[k]]; break;
                        case CodegenOpKind.Negate: s[k] = -s[nodeIn0[k]]; break;
                        case CodegenOpKind.Sqrt: s[k] = Math.Sqrt(s[nodeIn0[k]]); break;
                        case CodegenOpKind.ReLU: s[k] = s[nodeIn0[k]] < 0.0 ? 0.0 : s[nodeIn0[k]]; break;
                    }
                }
            }
        }

        var source = DumpSource(graph, CodegenElementType.Float64, "double");
        var kernel = new CompiledAvx512Kernel<double>(graph, CodegenElementType.Float64, Kernel, count);
        return CodegenEmitResult.Succeeded(kernel, source);
    }

    private static string DumpSource(CodegenGraph graph, CodegenElementType dtype, string scalar)
    {
        var lanes = scalar == "float" ? 16 : 8;
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"// AVX-512F kernel — emitted by CpuAvx512Emitter ({dtype}).");
        sb.AppendLine($"// {lanes}-wide SIMD body + scalar tail. Pointwise pure.");
        sb.AppendLine($"// Element count: {graph.Nodes[0].Shape.Aggregate(1, (a, b) => a * b)}.");
        sb.AppendLine();
        for (int i = 0; i < graph.Count; i++)
        {
            var n = graph[i];
            sb.Append($"  v{i}: {n.Op}");
            if (n.Inputs.Length > 0) sb.Append(" inputs={" + string.Join(",", n.Inputs.Select(x => $"v{x}")) + "}");
            sb.AppendLine();
        }
        return sb.ToString();
    }
#endif
}

#if NET8_0_OR_GREATER
internal sealed class CompiledAvx512Kernel<TElement> : CodegenKernel
    where TElement : unmanaged
{
    private readonly Action<TElement[][], TElement[][], int> _kernel;
    private readonly int _count;

    public CompiledAvx512Kernel(
        CodegenGraph graph,
        CodegenElementType dtype,
        Action<TElement[][], TElement[][], int> kernel,
        int count)
        : base(dtype, graph, CodegenTarget.CpuAvx512)
    {
        _kernel = kernel;
        _count = count;
    }

    public override void Execute<T>(T[][] inputs, T[][] outputs)
    {
        if (typeof(T) != typeof(TElement))
            throw new ArgumentException(
                $"Kernel specialised for {typeof(TElement).Name} — caller passed {typeof(T).Name}.",
                nameof(inputs));
        if (inputs is null) throw new ArgumentNullException(nameof(inputs));
        if (outputs is null) throw new ArgumentNullException(nameof(outputs));
        if (inputs.Length != InputCount)
            throw new ArgumentException($"Expected {InputCount} input buffers, got {inputs.Length}.");
        if (outputs.Length != OutputCount)
            throw new ArgumentException($"Expected {OutputCount} output buffers, got {outputs.Length}.");

        var typedInputs = (TElement[][])(object)inputs;
        var typedOutputs = (TElement[][])(object)outputs;
        _kernel(typedInputs, typedOutputs, _count);
    }
}
#endif
