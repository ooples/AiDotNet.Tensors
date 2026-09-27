// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.Engines.DirectGpu;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Per-op probe: a compiled training plan must recompute every op from the CURRENT parameter values on each Step.
/// Trace with one parameter value, drift the parameter in place, Step at lr = 0, and compare the loss and gradient with a
/// fresh eager tape on the drifted value. The parameter reaches the op through a multiply, so the op's input is an
/// intermediate (as inside a model), not a leaf.
/// </summary>
[Collection("CompilationGlobalState")]
public sealed class CompiledTrainingOpStalenessProbeTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    private readonly ITestOutputHelper _output;

    public CompiledTrainingOpStalenessProbeTests(DirectGpuTensorEngineTestFixture fixture, ITestOutputHelper output)
    {
        _fixture = fixture;
        _output = output;
    }

    private const int B = 2, S = 6, H = 2, M = 4;   // [B, H, S, M] for RoPE-style ops

    private static float[] Rand(int n, int seed, double scale = 1.0, double offset = 0.0)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)((rng.NextDouble() - 0.5) * scale + offset);
        return a;
    }

    private static (Tensor<float> Cos, Tensor<float> Sin) Rope()
    {
        var cos = new Tensor<float>([S, M / 2]);
        var sin = new Tensor<float>([S, M / 2]);
        for (int p = 0; p < S; p++)
            for (int i = 0; i < M / 2; i++)
            {
                double ang = p * Math.Pow(10000.0, -2.0 * i / M);
                cos[p, i] = (float)Math.Cos(ang);
                sin[p, i] = (float)Math.Sin(ang);
            }
        return (cos, sin);
    }

    /// <summary>The op under test, applied to x = w * c (an intermediate). Returns the op output.</summary>
    private static Tensor<float> Apply(string op, IEngine e, Tensor<float> x)
    {
        switch (op)
        {
            case "rope":
            {
                var (cos, sin) = Rope();
                return e.ApplyRoPEInterleaved(x, cos, sin);
            }
            case "clampmin":
                return e.TensorClampMin(x, 0.05f);
            case "bdivide":
            {
                // [B,H,S,M] / [B,H,S,1] (broadcast over the last axis), as a linear-attention normaliser.
                var den = e.TensorAddScalar(e.ReduceSum(e.TensorMultiply(x, x), [3], keepDims: true), 1f);
                return e.TensorDivide(x, den);
            }
            case "mm":        // reshape an intermediate to 2-D, multiply by a constant matrix
                return e.TensorMatMul(e.Reshape(x, [B * S, H * M]), Const([H * M, 4], 21));
            case "mmleft":    // constant on the left
                return e.TensorMatMul(Const([4, B * S], 22), e.Reshape(x, [B * S, H * M]));
            case "v1":        // matmul, then reshape its output
                return e.Reshape(e.TensorMatMul(e.Reshape(x, [B * S, H * M]), Const([H * M, 8], 23)), [B * S * 2, 4]);
            case "v2":        // ... then a second matmul by a constant
                return e.TensorMatMul(e.Reshape(e.TensorMatMul(e.Reshape(x, [B * S, H * M]), Const([H * M, 8], 23)), [B * S * 2, 4]), Const([4, 8], 24));
            case "v3":        // ... then a broadcast add of a [1, d] constant row
                return e.TensorAdd(e.TensorMatMul(e.Reshape(e.TensorMatMul(e.Reshape(x, [B * S, H * M]), Const([H * M, 8], 23)), [B * S * 2, 4]), Const([4, 8], 24)), Const([1, 8], 25));
            case "mm2":       // two chained matmuls, no reshape between
                return e.TensorMatMul(e.TensorMatMul(e.Reshape(x, [B * S, H * M]), Const([H * M, 8], 23)), Const([8, 4], 26));
            case "f1": case "f2": case "f3": case "f4": case "f5": case "f6": case "f5i": case "f5j": case "f5k": case "f5s": case "f5d":
            {
                var (cos, sin) = (new Tensor<float>([S, 1]), new Tensor<float>([S, 1]));
                for (int p = 0; p < S; p++) { cos[p, 0] = (float)Math.Cos(p); sin[p, 0] = (float)Math.Sin(p); }
                // [B*S, 8] @ [8, H*2] -> [B, S, H, 2] -> permute -> [B, H, S, 2]
                var z = e.TensorPermute(e.Reshape(e.TensorMatMul(e.Reshape(x, [B * S, H * M]), Const([H * M, H * 2], 27)), [B, S, H, 2]), [0, 2, 1, 3]);
                if (op == "f1") return z;
                z = e.ApplyRoPEInterleaved(z, cos, sin);
                if (op == "f2") return z;
                var outer = e.TensorMultiply(e.Reshape(z, [B, H, S, 2, 1]), e.Reshape(z, [B, H, S, 1, 2]));
                if (op == "f3") return outer;
                var phi = e.TensorPermute(e.Reshape(outer, [B, H, S, 4]), [0, 2, 1, 3]);
                if (op == "f4") return phi;
                Tensor<float> padM = Const([4, 8], 28);
                if (op == "f5i") { padM = new Tensor<float>([4, 8]); for (int i = 0; i < 4; i++) padM[i, i] = 1f; }
                if (op == "f5j") { var a = new float[32]; for (int i = 0; i < 4; i++) a[i * 8 + i] = 1f; padM = new Tensor<float>([4, 8], new Vector<float>(a)); }
                if (op == "f5s") { var a = Rand(32, 29); var rz = new Random(30); int zeros = 0; while (zeros < 28) { int j = rz.Next(32); if (a[j] != 0f) { a[j] = 0f; zeros++; } } padM = new Tensor<float>([4, 8], new Vector<float>(a)); }
                if (op == "f5d") { var a = Rand(32, 31, scale: 0.02); for (int i = 0; i < 4; i++) a[i * 8 + i] += 1f; padM = new Tensor<float>([4, 8], new Vector<float>(a)); }
                if (op == "f5k") { padM = new Tensor<float>([4, 8]); for (int i = 0; i < 32; i++) padM[i] = 0.1f * (i + 1); }
                var f5 = e.TensorMatMul(e.Reshape(phi, [B * S * H, 4]), padM);
                return op == "f6" ? e.Reshape(f5, [B, S, H * 8]) : f5;
            }
            case "linear_gelu":    // MatMul + bias + GELU: the LinearFusionPattern shape
                return e.GELU(e.TensorAdd(e.TensorMatMul(e.Reshape(x, [B * S, H * M]), Const([H * M, 8], 33, 2.0)), Const([1, 8], 34)));
            case "fusedlinear":   // 1-D bias: matches LinearFusionPattern (MatMul + bias + GELU -> FusedLinear)
                return e.GELU(e.TensorAdd(e.TensorMatMul(e.Reshape(x, [B * S, H * M]), Const([H * M, 8], 33, 2.0)), Const([8], 36)));
            case "linear_sigmoid":
                return e.Sigmoid(e.TensorAdd(e.TensorMatMul(e.Reshape(x, [B * S, H * M]), Const([H * M, 8], 33, 2.0)), Const([1, 8], 34)));
            case "gelu":
                return e.GELU(e.TensorMultiply(x, Const([B, H, S, M], 35, 4.0)));
            case "spectralffn": case "sffn:spec": case "sffn:swap": case "sffn:y":
            case "g:norm": case "g:hidden": case "g:gate": case "g:cre": case "g:bcast": case "g:sign":
                return SpectralFfn(e, x, op);
            // Ops whose graph node used to carry no backward (now a tape-replay backward): the parameter must still
            // receive the eager gradient through them.
            case "layernorm":
            {
                var g = new Tensor<float>([M]); var bta = new Tensor<float>([M]);
                for (int i = 0; i < M; i++) { g[i] = 1f + 0.1f * i; bta[i] = 0.05f * i; }
                return e.TensorLayerNorm(x, g, bta, 1e-5);
            }
            case "reducestd":
                return e.ReduceStd(x, [3], keepDims: true);
            case "lerp":
                return e.TensorLerp(x, e.TensorMultiply(x, x), 0.3f);
            case "addscaled":
                return e.TensorAddScaled(x, e.TensorMultiply(x, x), 0.7f, -1.3f);
            case "softmaxrows":
                return e.TensorSoftmaxRows(e.Reshape(x, [B * H * S, M]));
            case "fft":       // spectral filter: IRFFT(RFFT(x) * c) over the last axis
            {
                var z = e.RFFT(x);   // [B,H,S, 2*(M/2+1)]
                var c = Const(z.Shape.ToArray(), 32);
                return e.IRFFT(e.TensorMultiply(z, c), M);
            }
            case "mm3fan":    // one intermediate consumed by THREE matmuls (the Q/K/V projections of attention)
            {
                var x2 = e.Reshape(x, [B * S, H * M]);
                var q = e.TensorMatMul(x2, Const([H * M, 4], 51));
                var k = e.TensorMatMul(x2, Const([H * M, 4], 52));
                var v = e.TensorMatMul(x2, Const([H * M, 4], 53));
                return e.TensorAdd(e.TensorMultiply(q, k), v);
            }
            case "reshape":
                return e.Reshape(x, [B * S, H * M]);
            case "bornattn":
                return BornAttention(e, x, "full");
            case "born:featq": case "born:vaug": case "born:scan": case "born:num": case "born:den": case "born:scanqk":
                return BornAttention(e, x, op.Substring(5));
            case "clampmax":
                return e.TensorClampMax(x, 0.05f);
            case "divide":
                return e.TensorDivide(x, e.TensorAddScalar(e.TensorMultiply(x, x), 1f));
            case "outer5d":
            {
                var a = e.Reshape(x, [B, H, S, M, 1]);
                var b = e.Reshape(x, [B, H, S, 1, M]);
                return e.TensorMultiply(a, b);
            }
            case "permute":
                return e.TensorPermute(x, [0, 2, 1, 3]);
            case "glascan":
            {
                // [B, S, H*d] with d = M; gate ones [B, S, H].
                var q = e.Reshape(e.TensorPermute(x, [0, 2, 1, 3]), [B, S, H * M]);
                var gate = new Tensor<float>([B, S, H]);
                for (int i = 0; i < gate.Length; i++) gate[i] = 1f;
                return e.GlaScanForward(q, q, q, gate, H);
            }
            case "rmsnorm":
            {
                var g = new Tensor<float>([M]);
                for (int i = 0; i < M; i++) g[i] = 1f + 0.1f * i;
                return e.RMSNorm(x, g, 1e-6, out _);
            }
            default:
                throw new ArgumentException(op);
        }
    }

    private static Tensor<float> Const(int[] shape, int seed, double scale = 0.5)
        => new(shape, new Vector<float>(Rand(shape.Aggregate(1, (p, q) => p * q), seed, scale)));

    /// <summary>A Born-feature linear attention built from generic ops only (x (x) x features, a scan, a clamped
    /// normaliser, a broadcast divide). x = [B,H,S,M] = [2,2,6,4] is read as [b*s, e] = [12, 8].</summary>
    private static Tensor<float> BornAttention(IEngine e, Tensor<float> x, string stage)
    {
        const int b = 2, s = S, em = 8, nh = 2, mb = 2, dh = 4, d = 8;
        var x2 = e.Reshape(x, [b * s, em]);
        var (cos, sin) = (new Tensor<float>([s, mb / 2]), new Tensor<float>([s, mb / 2]));
        for (int p = 0; p < s; p++) { cos[p, 0] = (float)Math.Cos(p); sin[p, 0] = (float)Math.Sin(p); }
        var pad = new Tensor<float>([mb * mb, d]);
        for (int i = 0; i < mb * mb; i++) pad[i, i] = 1f;
        Tensor<float> Features(Tensor<float> w)
        {
            var z = e.TensorPermute(e.Reshape(e.TensorMatMul(x2, w), [b, s, nh, mb]), [0, 2, 1, 3]);
            z = e.ApplyRoPEInterleaved(z, cos, sin);
            var outer = e.TensorMultiply(e.Reshape(z, [b, nh, s, mb, 1]), e.Reshape(z, [b, nh, s, 1, mb]));
            var phi = e.TensorPermute(e.Reshape(outer, [b, nh, s, mb * mb]), [0, 2, 1, 3]);
            return e.Reshape(e.TensorMatMul(e.Reshape(phi, [b * s * nh, mb * mb]), pad), [b, s, nh * d]);
        }
        var phiQ = Features(Const([em, nh * mb], 11));
        if (stage == "featq") return phiQ;
        var phiK = Features(Const([em, nh * mb], 12));
        var v = e.Reshape(e.TensorMatMul(x2, Const([em, nh * dh], 13)), [b * s * nh, dh]);
        var ev = new Tensor<float>([dh, d]);
        for (int i = 0; i < dh; i++) ev[i, i] = 1f;
        var ones = new Tensor<float>([1, d]);
        ones[0, dh] = 1f;
        var vAug = e.Reshape(e.TensorAdd(e.TensorMatMul(v, ev), ones), [b, s, nh * d]);
        if (stage == "vaug") return vAug;
        var gate = new Tensor<float>([b, s, nh]);
        for (int i = 0; i < gate.Length; i++) gate[i] = 1f;
        if (stage == "scanqk")   // scan with constant values: gradient flows only through the Born features
        {
            var cv = Const([b, s, nh * d], 14);
            return e.GlaScanForward(phiQ, phiK, cv, gate, nh);
        }
        var o = e.Reshape(e.GlaScanForward(phiQ, phiK, vAug, gate, nh), [b * s * nh, d]);
        if (stage == "scan") return o;
        var selNum = new Tensor<float>([d, dh]);
        for (int i = 0; i < dh; i++) selNum[i, i] = 1f;
        var selDen = new Tensor<float>([d, 1]);
        selDen[dh, 0] = 1f;
        var num = e.Reshape(e.TensorMatMul(o, selNum), [b, s, nh, dh]);
        if (stage == "num") return num;
        var den = e.TensorClampMin(e.Reshape(e.TensorMatMul(o, selDen), [b, s, nh, 1]), 1e-4f);
        if (stage == "den") return den;
        return e.Reshape(e.TensorDivide(num, den), [b * s, em]);
    }

    /// <summary>A per-token spectral-gate FFN from generic ops: RMSNorm, gate MLP (GELU, sigmoid), RFFT, complex
    /// multiply via a [2,2] swap matmul and a sign vector, IRFFT, residual. x = [2,2,6,4] read as [b,s,e] = [2,6,8].</summary>
    private static Tensor<float> SpectralFfn(IEngine e, Tensor<float> x, string stage)
    {
        if (stage.StartsWith("g:", StringComparison.Ordinal)) return GateStages(e, x, stage);
        const int b = 2, s = S, em = 8, gh = 6, nf = em / 2 + 1;
        var a = e.Reshape(x, [b, s, em]);
        var norm = new Tensor<float>([em]);
        for (int i = 0; i < em; i++) norm[i] = 1f;
        var xn = e.RMSNorm(a, norm, 1e-6, out _);
        var x2 = e.Reshape(xn, [b * s, em]);
        var hidden = e.GELU(e.TensorAdd(e.TensorMatMul(x2, Const([em, gh], 41)), e.Reshape(Const([gh], 42), [1, gh])));
        var gate = e.Reshape(e.Sigmoid(e.TensorAdd(e.TensorMatMul(hidden, Const([gh, nf], 43)), e.Reshape(Const([nf], 44), [1, nf]))), [b, s, nf]);
        var cRe = e.TensorMultiply(gate, e.Reshape(Const([nf], 45), [1, 1, nf]));
        var cIm = e.TensorMultiply(gate, e.Reshape(Const([nf], 46), [1, 1, nf]));
        var z = e.Reshape(e.RFFT(xn), [b, s, nf, 2]);
        if (stage == "sffn:spec") return z;
        var swapM = new Tensor<float>([2, 2]); swapM[0, 1] = 1f; swapM[1, 0] = 1f;
        var sign = new Tensor<float>([2]); sign[0] = -1f; sign[1] = 1f;
        var swapped = e.Reshape(e.TensorMatMul(e.Reshape(z, [b * s * nf, 2]), swapM), [b, s, nf, 2]);
        if (stage == "sffn:swap") return swapped;
        var y = e.TensorAdd(e.TensorMultiply(z, e.Reshape(cRe, [b, s, nf, 1])),
            e.TensorMultiply(swapped, e.TensorMultiply(e.Reshape(cIm, [b, s, nf, 1]), sign)));
        if (stage == "sffn:y") return y;
        return e.TensorAdd(a, e.IRFFT(e.Reshape(y, [b, s, 2 * nf]), em));
    }

    /// <summary>Stages of the spectral FFN's gate path WITHOUT dead branches (each stage feeds the loss).</summary>
    private static Tensor<float> GateStages(IEngine e, Tensor<float> x, string stage)
    {
        const int b = 2, s = S, em = 8, gh = 6, nf = em / 2 + 1;
        var a = e.Reshape(x, [b, s, em]);
        var norm = new Tensor<float>([em]);
        for (int i = 0; i < em; i++) norm[i] = 1f;
        var xn = e.RMSNorm(a, norm, 1e-6, out _);
        if (stage == "g:norm") return xn;
        var hidden = e.GELU(e.TensorAdd(e.TensorMatMul(e.Reshape(xn, [b * s, em]), Const([em, gh], 41)), e.Reshape(Const([gh], 42), [1, gh])));
        if (stage == "g:hidden") return hidden;
        var gate = e.Reshape(e.Sigmoid(e.TensorAdd(e.TensorMatMul(hidden, Const([gh, nf], 43)), e.Reshape(Const([nf], 44), [1, nf]))), [b, s, nf]);
        if (stage == "g:gate") return gate;
        var cRe = e.TensorMultiply(gate, e.Reshape(Const([nf], 45), [1, 1, nf]));
        if (stage == "g:cre") return cRe;
        var bc = e.TensorMultiply(Const([b, s, nf, 2], 47), e.Reshape(cRe, [b, s, nf, 1]));
        if (stage == "g:bcast") return bc;
        var sign = new Tensor<float>([2]); sign[0] = -1f; sign[1] = 1f;
        return e.TensorMultiply(e.Reshape(cRe, [b, s, nf, 1]), sign);   // g:sign
    }

    public static IEnumerable<object[]> Cases()
    {
        foreach (var op in new[] { "rope", "clampmin", "clampmax", "divide", "outer5d", "permute", "glascan", "rmsnorm", "bdivide", "layernorm", "reducestd", "lerp", "addscaled", "softmaxrows", "fft", "spectralffn", "g:norm", "g:hidden", "g:gate", "g:cre", "g:bcast", "g:sign", "linear_gelu", "fusedlinear", "linear_sigmoid", "gelu", "mm", "mm3fan", "mmleft", "reshape", "v1", "v2", "v3", "mm2", "f1", "f2", "f3", "f4", "f5", "f6", "f5i", "f5j", "f5k", "f5s", "f5d", "bornattn", "born:featq", "born:vaug", "born:scanqk", "born:scan", "born:num", "born:den", "sffn:spec", "sffn:swap", "sffn:y" })
        {
            yield return new object[] { op, false };
            yield return new object[] { op, true };
        }
    }

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void CompiledStep_RecomputesOpFromDriftedParameter(string op, bool gpu)
    {
        if (gpu) Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        IEngine engine = gpu ? _fixture.Engine! : new CpuEngine();
        int n = B * H * S * M;
        var cData = Rand(n, 1, offset: 0.2);
        var w0 = Rand(n, 2);
        var w1 = Rand(n, 3, scale: 2.0);
        var rData = Rand(n * M, 4);   // big enough for every op's output

        (float Loss, float[] Grad) Eager(float[] wv)
        {
            var w = new Tensor<float>([B, H, S, M], new Vector<float>((float[])wv.Clone()));
            var c = new Tensor<float>([B, H, S, M], new Vector<float>(cData));
            using var tape = new GradientTape<float>();
            var y = Apply(op, engine, engine.TensorMultiply(w, c));
            var r = new Tensor<float>(y.Shape.ToArray(), new Vector<float>(rData.Take(y.Length).ToArray()));
            var loss = engine.ReduceSum(engine.TensorMultiply(y, r), null);
            var g = tape.ComputeGradients(loss, [w])[w];
            return (loss.ToArray()[0], g.ToArray());
        }

        var wT = new Tensor<float>([B, H, S, M], new Vector<float>((float[])w0.Clone()));
        var cT = new Tensor<float>([B, H, S, M], new Vector<float>(cData));
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                var y = Apply(op, engine, engine.TensorMultiply(wT, cT));
                var r = new Tensor<float>(y.Shape.ToArray(), new Vector<float>(rData.Take(y.Length).ToArray()));
                engine.ReduceSum(engine.TensorMultiply(y, r), null);
                plan = scope.CompileTraining(new[] { wT });
            }

            using (plan)
            {
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
                var l0 = plan.Step().ToArray()[0];
                var g0 = (wT.Grad ?? throw new InvalidOperationException("no grad")).ToArray();
                // Drift the parameter in place (what an optimizer step does), then step again.
                var span = wT.AsWritableSpan();
                for (int i = 0; i < n; i++) span[i] = w1[i];
                wT.IncrementVersion();
                var l1 = plan.Step().ToArray()[0];
                var g1 = (wT.Grad ?? throw new InvalidOperationException("no grad")).ToArray();

                var e0 = Eager(w0);
                var e1 = Eager(w1);
                _output.WriteLine($"{op} gpu={gpu}: dW[0] step0 compiled {g0[0]:G6} eager {e0.Grad[0]:G6} | after drift compiled {g1[0]:G6} eager {e1.Grad[0]:G6}");
                double g0Scale = Math.Max(1e-3, e0.Grad.Max(v => Math.Abs(v)));
                for (int i = 0; i < n; i++)
                    Assert.True(Math.Abs(g0[i] - e0.Grad[i]) <= 1e-3 * g0Scale, $"{op} gpu={gpu}: dW[{i}] compiled {g0[i]} != eager {e0.Grad[i]} on the FIRST step");
                double scale = Math.Max(1.0, Math.Abs(e1.Loss));
                _output.WriteLine($"{op} gpu={gpu}: loss step0 compiled {l0:F5} eager {e0.Loss:F5} | after drift compiled {l1:F5} eager {e1.Loss:F5}");
                Assert.True(Math.Abs(l0 - e0.Loss) <= 1e-3 * scale, $"{op} gpu={gpu}: first compiled loss {l0} != eager {e0.Loss}");
                Assert.True(Math.Abs(l1 - e1.Loss) <= 1e-3 * scale, $"{op} gpu={gpu}: compiled loss after the parameter drifted {l1} != eager {e1.Loss} (stale op)");
                double gScale = Math.Max(1e-3, e1.Grad.Max(v => Math.Abs(v)));
                for (int i = 0; i < n; i++)
                    Assert.True(Math.Abs(g1[i] - e1.Grad[i]) <= 1e-3 * gScale, $"{op} gpu={gpu}: dW[{i}] compiled {g1[i]} != eager {e1.Grad[i]} after drift");
            }
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }
}
