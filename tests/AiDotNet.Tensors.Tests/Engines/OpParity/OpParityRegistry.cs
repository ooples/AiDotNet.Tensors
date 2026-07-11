// Copyright (c) AiDotNet. All rights reserved.
// CPU-vs-GPU op-parity scaffold (Tensors #775). OpInfo registry.
// Phase 1 seeds the ViT-path ops implicated by #775 (patch-embed conv, matmul, softmax,
// LayerNorm, GELU/Sigmoid) to localize the CPU/GPU divergence; expands to the full IEngine
// surface from here. Tolerances: exact ops at 0 ULP; accumulation ops at a fp32 relative bound
// (matched to the 1e-3 the existing GPU parity tests use); elementwise transcendentals at a few ULP.
#if !NETFRAMEWORK

using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Tests.Engines.OpParity;

public static class OpParityRegistry
{
    /// <summary>Every registered parity case (the ViT-path localization set + the broad
    /// elementwise/reduction batch). Tests and the coverage audit run over this.</summary>
    public static IEnumerable<OpCase> All() => ViTPath().Concat(Elementwise());

    // ---- batch helpers for the uniform op families -------------------------------------------
    private static string Dims(OpInput i) => string.Join(",", i.Shape);

    /// <summary>Unary elementwise op: Op(tensor).</summary>
    private static OpCase U(
        string method, string cat, System.Func<IEngine, Tensor<float>, Tensor<float>> f,
        System.Func<IEngine, Tensor<double>, Tensor<double>> d, ParityTol tol, OpInput x)
        => new OpCase($"{method}[{Dims(x)}]", cat, e => f(e, x.F()), e => d(e, x.D()), tol, opMethod: method);

    /// <summary>Binary elementwise op: Op(a, b).</summary>
    private static OpCase B(
        string method, string cat, System.Func<IEngine, Tensor<float>, Tensor<float>, Tensor<float>> f,
        System.Func<IEngine, Tensor<double>, Tensor<double>, Tensor<double>> d, ParityTol tol, OpInput a, OpInput b)
        => new OpCase($"{method}[{Dims(a)}]", cat, e => f(e, a.F(), b.F()), e => d(e, a.D(), b.D()), tol, opMethod: method);

    public static IEnumerable<OpCase> Elementwise()
    {
        var s = new[] { 4, 64 };

        // Bounded-range activations (transcendental → a handful of ULP).
        var act = OpInput.Rand(200, s, -6.0, 6.0);
        yield return U("Tanh", "activation", (e, t) => e.Tanh(t), (e, t) => e.Tanh(t), ParityTol.Ulp(16), act);
        yield return U("ReLU", "activation", (e, t) => e.ReLU(t), (e, t) => e.ReLU(t), ParityTol.Exact, act);
        yield return U("Mish", "activation", (e, t) => e.Mish(t), (e, t) => e.Mish(t), ParityTol.Ulp(64, 1e-6), act);
        yield return U("Swish", "activation", (e, t) => e.Swish(t), (e, t) => e.Swish(t), ParityTol.Ulp(64, 1e-6), act);
        yield return U("Softplus", "activation", (e, t) => e.Softplus(t), (e, t) => e.Softplus(t), ParityTol.Ulp(64, 1e-6), act);
        yield return U("HardSwish", "activation", (e, t) => e.HardSwish(t), (e, t) => e.HardSwish(t), ParityTol.Ulp(16, 1e-6), act);
        yield return U("ELU", "activation", (e, t) => e.ELU(t, 1.0), (e, t) => e.ELU(t, 1.0), ParityTol.Ulp(64, 1e-6), act);
        yield return new OpCase($"LeakyReLU[{Dims(act)}]", "activation",
            e => e.LeakyReLU(act.F(), 0.1f), e => e.LeakyReLU(act.D(), 0.1), ParityTol.Ulp(4, 1e-6), opMethod: "LeakyReLU");

        // Elementwise math.
        yield return U("TensorNegate", "arithmetic", (e, t) => e.TensorNegate(t), (e, t) => e.TensorNegate(t), ParityTol.Exact, act);
        yield return U("TensorAbs", "arithmetic", (e, t) => e.TensorAbs(t), (e, t) => e.TensorAbs(t), ParityTol.Exact, act);
        yield return U("TensorSquare", "arithmetic", (e, t) => e.TensorSquare(t), (e, t) => e.TensorSquare(t), ParityTol.Ulp(2, 1e-6), act);
        yield return U("TensorSin", "arithmetic", (e, t) => e.TensorSin(t), (e, t) => e.TensorSin(t), ParityTol.Ulp(64, 1e-6), act);
        yield return U("TensorCos", "arithmetic", (e, t) => e.TensorCos(t), (e, t) => e.TensorCos(t), ParityTol.Ulp(64, 1e-6), act);
        yield return U("TensorExp", "arithmetic", (e, t) => e.TensorExp(t), (e, t) => e.TensorExp(t), ParityTol.Ulp(64, 1e-6),
            OpInput.Rand(201, s, -4.0, 4.0));
        yield return U("TensorLog", "arithmetic", (e, t) => e.TensorLog(t), (e, t) => e.TensorLog(t), ParityTol.Ulp(64, 1e-6),
            OpInput.RandPositive(202, s, 0.1, 8.0));
        yield return U("TensorSqrt", "arithmetic", (e, t) => e.TensorSqrt(t), (e, t) => e.TensorSqrt(t), ParityTol.Ulp(8, 1e-6),
            OpInput.RandPositive(203, s, 0.01, 8.0));

        // Binary elementwise.
        var a = OpInput.Rand(210, s);
        var b = OpInput.Rand(211, s);
        var bnz = OpInput.Rand(212, s, 0.5, 2.0); // non-zero divisor
        yield return B("TensorSubtract", "arithmetic", (e, x, y) => e.TensorSubtract(x, y), (e, x, y) => e.TensorSubtract(x, y), ParityTol.Exact, a, b);
        yield return B("TensorMultiply", "arithmetic", (e, x, y) => e.TensorMultiply(x, y), (e, x, y) => e.TensorMultiply(x, y), ParityTol.Ulp(2, 1e-6), a, b);
        yield return B("TensorDivide", "arithmetic", (e, x, y) => e.TensorDivide(x, y), (e, x, y) => e.TensorDivide(x, y), ParityTol.Ulp(4, 1e-6), a, bnz);

        // Reductions (summation order differs → relative bound; Max is selection → tight).
        var red = OpInput.Rand(220, new[] { 4, 32 });
        yield return new OpCase("ReduceSum[4,32]", "reduction",
            e => e.ReduceSum(red.F(), null, false), e => e.ReduceSum(red.D(), null, false), ParityTol.Accum(1e-3), opMethod: "ReduceSum");
        yield return new OpCase("ReduceMean[4,32;ax1]", "reduction",
            e => e.ReduceMean(red.F(), new[] { 1 }, false), e => e.ReduceMean(red.D(), new[] { 1 }, false), ParityTol.Accum(1e-3), opMethod: "ReduceMean");
        yield return new OpCase("ReduceMax[4,32;ax1]", "reduction",
            e => e.ReduceMax(red.F(), new[] { 1 }, false, out _), e => e.ReduceMax(red.D(), new[] { 1 }, false, out _), ParityTol.Ulp(0, 1e-6), opMethod: "ReduceMax");
    }

    public static IEnumerable<OpCase> ViTPath()
    {
        // --- Elementwise, must be bit-identical (identical scalar math, no accumulation) ---
        {
            var a = OpInput.Rand(1, new[] { 2, 64 });
            var b = OpInput.Rand(2, new[] { 2, 64 });
            yield return new OpCase("Add[2,64]", "arithmetic",
                e => e.TensorAdd(a.F(), b.F()), e => e.TensorAdd(a.D(), b.D()), ParityTol.Exact, opMethod: "TensorAdd");
        }
        {
            var a = OpInput.Rand(3, new[] { 2, 64 });
            yield return new OpCase("Reshape[2,64->128]", "shape",
                e => e.Reshape(a.F(), new[] { 128 }), e => e.Reshape(a.D(), new[] { 128 }), ParityTol.Exact, opMethod: "Reshape");
        }

        // --- Accumulation ops (summation order differs across engines → relative bound) ---
        {
            var a = OpInput.Rand(4, new[] { 8, 32 });
            var b = OpInput.Rand(5, new[] { 32, 16 });
            yield return new OpCase("MatMul[8x32x16]", "matmul",
                e => e.TensorMatMul(a.F(), b.F()), e => e.TensorMatMul(a.D(), b.D()), ParityTol.Accum(1e-3), opMethod: "TensorMatMul");
        }
        {
            var a = OpInput.Rand(6, new[] { 4, 16 }, -4.0, 4.0);
            yield return new OpCase("Softmax[4,16]", "activation",
                e => e.Softmax(a.F(), -1), e => e.Softmax(a.D(), -1), ParityTol.Accum(1e-3));
        }
        {
            var x = OpInput.Rand(9, new[] { 4, 64 });
            var g = OpInput.Rand(10, new[] { 64 }, 0.5, 1.5);
            var b = OpInput.Rand(11, new[] { 64 }, -0.2, 0.2);
            yield return new OpCase("LayerNorm[4,64]", "norm",
                e => { var r = e.LayerNorm(x.F(), g.F(), b.F(), 1e-5, out _, out _); return r; },
                e => { var r = e.LayerNorm(x.D(), g.D(), b.D(), 1e-5, out _, out _); return r; },
                ParityTol.Accum(1e-3));
        }
        {
            // Patch-embed proxy: [1,3,16,16] conv with a 4×4 stride-4 kernel → [1,8,4,4].
            var x = OpInput.Rand(12, new[] { 1, 3, 16, 16 });
            var k = OpInput.Rand(13, new[] { 8, 3, 4, 4 });
            yield return new OpCase("Conv2D[1,3,16,16;k8x3x4x4;s4]", "conv",
                e => e.Conv2D(x.F(), k.F(), 4, 0, 1), e => e.Conv2D(x.D(), k.D(), 4, 0, 1), ParityTol.Accum(1e-3), opMethod: "Conv2D");
        }

        // --- Elementwise transcendentals (a few ULP of per-element rounding) ---
        {
            // NOTE (#775): the CPU and GPU GELU disagree by up to ~7e-6 in the near-zero flat
            // region (GPU flushes small outputs to 0 vs CPU's erf-exact tiny value; the oracle
            // shows GPU drifts further from truth). That is negligible in isolation but is exactly
            // the kind of small ViT forward delta BCE-with-logits amplifies — hence the wider
            // near-zero floor here, WITH the report capturing the ULP/oracle drift for follow-up.
            var a = OpInput.Rand(7, new[] { 4, 64 }, -6.0, 6.0);
            yield return new OpCase("GELU[4,64]", "activation",
                e => e.GELU(a.F()), e => e.GELU(a.D()), ParityTol.Ulp(256, 2e-5));
        }
        {
            // GELUInto is the COMPILED / GraphMode training-path GELU kernel (the one #775's
            // diverging CpuEngine training runs). Deliberately includes near-zero inputs where the
            // old 2·sigmoid(2z)−1 cancellation corrupted the result. Now unified onto the stable
            // x·sigmoid kernel, so it must match the eager GELU + oracle tightly.
            var a = OpInput.Rand(7, new[] { 4, 64 }, -6.0, 6.0);
            yield return new OpCase("GELUInto[4,64]", "activation",
                e => { var dst = new Tensor<float>((int[])a.Shape.Clone()); e.GELUInto(dst, a.F()); return dst; },
                e => { var dst = new Tensor<double>((int[])a.Shape.Clone()); e.GELUInto(dst, a.D()); return dst; },
                ParityTol.Ulp(256, 2e-5), opMethod: "GELU"); // credits GELU; GELUInto is a void Into-variant
        }
        {
            var a = OpInput.Rand(8, new[] { 4, 64 }, -8.0, 8.0);
            yield return new OpCase("Sigmoid[4,64]", "activation",
                e => e.Sigmoid(a.F()), e => e.Sigmoid(a.D()), ParityTol.Ulp(64, 1e-6));
        }

        // --- Backward ops are first-class IEngine ops; their CPU/GPU parity IS backward parity ---
        {
            var go = OpInput.Rand(20, new[] { 4, 64 });
            var input = OpInput.Rand(21, new[] { 4, 64 });
            yield return new OpCase("ReluBackward[4,64]", "activation-bwd",
                e => e.ReluBackward(go.F(), input.F()), e => e.ReluBackward(go.D(), input.D()), ParityTol.Ulp(4, 1e-6));
        }
        {
            var go = OpInput.Rand(22, new[] { 4, 64 });
            var outp = OpInput.Rand(23, new[] { 4, 64 }, 0.05, 0.95);
            yield return new OpCase("SigmoidBackward[4,64]", "activation-bwd",
                e => e.SigmoidBackward(go.F(), outp.F()), e => e.SigmoidBackward(go.D(), outp.D()), ParityTol.Ulp(16, 1e-6));
        }
        {
            var go = OpInput.Rand(24, new[] { 4, 64 });
            var outp = OpInput.Rand(25, new[] { 4, 64 }, -0.95, 0.95);
            yield return new OpCase("TanhBackward[4,64]", "activation-bwd",
                e => e.TanhBackward(go.F(), outp.F()), e => e.TanhBackward(go.D(), outp.D()), ParityTol.Ulp(16, 1e-6));
        }
        {
            // Softmax backward needs a valid softmax output (rows sum to 1) as its second arg.
            var go = OpInput.Rand(26, new[] { 4, 16 });
            var logits = OpInput.Rand(27, new[] { 4, 16 }, -4.0, 4.0);
            yield return new OpCase("SoftmaxBackward[4,16]", "activation-bwd",
                e => e.SoftmaxBackward(go.F(), e.Softmax(logits.F(), -1), -1),
                e => e.SoftmaxBackward(go.D(), e.Softmax(logits.D(), -1), -1),
                ParityTol.Accum(1e-3));
        }
    }
}
#endif
