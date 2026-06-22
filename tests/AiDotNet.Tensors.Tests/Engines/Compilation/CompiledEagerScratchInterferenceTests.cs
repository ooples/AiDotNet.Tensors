using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Parity guards for the verify-then-trust gate corruption seen by AiDotNet's diffusion noise predictors
/// (PR #1633): executing a compiled inference plan ONCE must NOT perturb subsequent EAGER forwards of the
/// same op graph (the consumer's gate runs eager then compiled on the verify call, so a rejected shape's
/// eager fallback would otherwise diverge / ping-pong — non-deterministic and wrong on half the calls).
///
/// SCOPE NOTE (investigation result, PR #1633): #632 fixed this for single-input / no-cross-attention
/// graphs. The residual still hits AiDotNet's AudioLDM/AuraFlow Predict_ShouldBeDeterministic with
/// compiled inference enabled on 0.102.2 (exact 3.19-vs-3.55 signature). These isolated patterns —
/// conv→groupnorm→silu→skip-concat→conv, single-input self-attention, and multi-input CROSS-attention —
/// are all PROVEN CLEAN here on the current source, which narrows the trigger to the full DiT/U-Net block
/// stack (AdaLN + attention + residual + secondary-consumer reuse across many ops, 3 input leaves), not
/// any short chain. Reliable repro for the residual: build AiDotNet's pr1633 consumer against this source
/// (-p:UseLocalTensors=true) and run AudioLDMModelTests.Predict_ShouldBeDeterministic with
/// AIDOTNET_ENABLE_AUTO_COMPILE=1; bisect the MemoryPlanning aliasing in the attention compiled path from
/// there. Until fixed, the consumer keeps compiled inference opt-in (default eager).
/// </summary>
public class CompiledEagerScratchInterferenceTests
{
    private readonly ITestOutputHelper _output;
    public CompiledEagerScratchInterferenceTests(ITestOutputHelper output) { _output = output; }

    private static long Checksum(Tensor<double> t)
    {
        long h = unchecked((long)0xcbf29ce484222325L);
        var s = t.AsSpan();
        for (int i = 0; i < s.Length; i++)
        {
            h ^= System.BitConverter.DoubleToInt64Bits(s[i]);
            h *= unchecked((long)0x100000001b3L);
        }
        return h;
    }

    private static Tensor<double> Forward(CpuEngine engine, Tensor<double> x, Tensor<double> k1, Tensor<double> k2,
        Tensor<double> gamma, Tensor<double> beta)
    {
        var enc = engine.Conv2D(x, k1, stride: 1, padding: 1);   // [1,8,8,8] — "encoder" output (skip)
        var h = engine.GroupNorm(enc, 4, gamma, beta, 1e-5, out _, out _);
        h = engine.Swish(h);                                      // SiLU
        // U-Net signature: concat the skip connection back along channels in the "decoder".
        h = engine.Concat(new[] { h, enc }, axis: 1);             // [1,16,8,8]
        h = engine.Conv2D(h, k2, stride: 1, padding: 1);
        return h;
    }

    [Fact]
    public void EagerForward_AfterCompiledExecute_StaysDeterministic()
    {
        var engine = new CpuEngine();

        var x = new Tensor<double>(new[] { 1, 4, 8, 8 });
        var k1 = new Tensor<double>(new[] { 8, 4, 3, 3 });
        var k2 = new Tensor<double>(new[] { 4, 16, 3, 3 });
        var gamma = new Tensor<double>(new[] { 8 });
        var beta = new Tensor<double>(new[] { 8 });
        for (int i = 0; i < x.Length; i++) x[i] = (i % 7) * 0.1 - 0.3;
        for (int i = 0; i < k1.Length; i++) k1[i] = (i % 5) * 0.05 - 0.1;
        for (int i = 0; i < k2.Length; i++) k2[i] = (i % 3) * 0.07 - 0.05;
        for (int i = 0; i < gamma.Length; i++) { gamma[i] = 1.0 + i * 0.01; beta[i] = i * 0.02 - 0.05; }

        // Pure-eager baseline: three forwards must be bit-identical.
        long e0 = Checksum(Forward(engine, x, k1, k2, gamma, beta));
        long e1 = Checksum(Forward(engine, x, k1, k2, gamma, beta));
        Assert.Equal(e0, e1);

        // Run the compiled plan ONCE (the gate's verify "candidate").
        using var cache = new CompiledModelCache<double>();
        var plan = cache.GetOrCompileInference(x._shape, () => Forward(engine, x, k1, k2, gamma, beta));
        _ = plan.Execute();

        // Every eager forward AFTER the compiled execute must STILL equal the eager baseline.
        long a1 = Checksum(Forward(engine, x, k1, k2, gamma, beta));
        long a2 = Checksum(Forward(engine, x, k1, k2, gamma, beta));
        long a3 = Checksum(Forward(engine, x, k1, k2, gamma, beta));
        _output.WriteLine($"eager baseline={e0:X16}  post-compiled: {a1:X16} {a2:X16} {a3:X16}");

        Assert.True(a1 == e0 && a2 == e0 && a3 == e0,
            $"eager forward must stay bit-identical after a compiled execute (baseline={e0:X16}, got {a1:X16},{a2:X16},{a3:X16})");
    }

    // The residual that #632 did NOT fix: attention inference. A DiT/cross-attention block routes
    // ScaledDotProductAttention through the multi-input compiled path; the consumer's cross-attention
    // diffusion models (AudioLDM/AuraFlow, contextDim>0) still replay non-deterministically with compiled
    // inference on 0.102.2. This drives ScaledDotProductAttention through a compiled plan and asserts the
    // subsequent eager forward stays bit-identical.
    // MULTI-INPUT CROSS-ATTENTION — matches the consumer's failing diffusion path: the sample provides
    // Q, the conditioning (a distinct per-call input leaf) provides K/V. This is the shape the gate's
    // PredictCompiledMulti compiles ([noisySample, conditioning]); single-input attention did NOT
    // reproduce, and the existing MultiInput aliasing tests use matmul+relu (no attention).
    private static Tensor<double> CrossAttnForward(CpuEngine engine, Tensor<double> x, Tensor<double> cond,
        Tensor<double> wk, Tensor<double> wv)
    {
        var q = x;                              // [B,H,Sq,D]
        var k = engine.TensorMatMul(cond, wk);  // [B,H,Sk,D] @ [D,D]
        var v = engine.TensorMatMul(cond, wv);
        var attn = engine.ScaledDotProductAttention(q, k, v, null, null, out _);  // [B,H,Sq,D]
        return attn.Contiguous();
    }

    [Fact]
    public void EagerCrossAttention_AfterCompiledExecute_StaysDeterministic()
    {
        var engine = new CpuEngine();
        int B = 1, H = 4, Sq = 16, Sk = 12, D = 8;
        var x = new Tensor<double>(new[] { B, H, Sq, D });
        var cond = new Tensor<double>(new[] { B, H, Sk, D });
        var wk = new Tensor<double>(new[] { D, D });
        var wv = new Tensor<double>(new[] { D, D });
        for (int i = 0; i < x.Length; i++) x[i] = (i % 11) * 0.03 - 0.15;
        for (int i = 0; i < cond.Length; i++) cond[i] = (i % 9) * 0.04 - 0.12;
        for (int i = 0; i < wk.Length; i++) { wk[i] = (i % 7) * 0.05 - 0.1; wv[i] = (i % 5) * 0.04 - 0.08; }

        long e0 = Checksum(CrossAttnForward(engine, x, cond, wk, wv));
        Assert.Equal(e0, Checksum(CrossAttnForward(engine, x, cond, wk, wv)));

        using var cache = new CompiledModelCache<double>();
        var plan = cache.GetOrCompileInference(new[] { x, cond }, () => CrossAttnForward(engine, x, cond, wk, wv));

        // (a) compiled REPLAY determinism (the denoising loop replays the trusted plan).
        plan.SetInputs(new[] { x, cond });
        long r1 = Checksum(plan.Execute());
        plan.SetInputs(new[] { x, cond });
        long r2 = Checksum(plan.Execute());
        _output.WriteLine($"cross-attn eager={e0:X16}  compiled replay: {r1:X16} {r2:X16}");

        // (b) eager-after-compiled determinism (the gate runs eager then compiled on verify; a rejected
        //     shape's eager fallback must stay bit-identical).
        long a1 = Checksum(CrossAttnForward(engine, x, cond, wk, wv));
        long a2 = Checksum(CrossAttnForward(engine, x, cond, wk, wv));
        long a3 = Checksum(CrossAttnForward(engine, x, cond, wk, wv));
        _output.WriteLine($"  eager-after-compiled: {a1:X16} {a2:X16} {a3:X16}");

        Assert.True(r1 == r2, $"compiled replay must be deterministic (got {r1:X16},{r2:X16})");
        Assert.True(a1 == e0 && a2 == e0 && a3 == e0,
            $"eager cross-attention must stay bit-identical after a compiled execute (baseline={e0:X16}, got {a1:X16},{a2:X16},{a3:X16})");
    }
}
