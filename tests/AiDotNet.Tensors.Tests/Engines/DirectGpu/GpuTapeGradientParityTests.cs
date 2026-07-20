using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Gradient parity, CPU vs GPU, for every op whose forward stopped bailing on an active tape.
///
/// WHY THE SOURCE AUDIT IS NOT ENOUGH: TapeBailAuditTests proves a recording EXISTS. It cannot prove the
/// recording is CORRECT — wrong saved state, the wrong backward, or the wrong overload all leave the source
/// looking right while gradients come out silently wrong. TensorMax/TensorMin are the standing example: the
/// GPU implements (tensor, scalar) while CpuEngine records only (tensor, tensor), so reusing that backward
/// would attribute gradient to an operand that does not exist. Only running backprop catches that class.
///
/// GUARDS AGAINST A VACUOUS PASS (each has already produced a false green in this codebase):
///   1. Calls go through IEngine. DirectGpuTensorEngine uses EXPLICIT interface implementations, so
///      gpu.Op(x) resolves to the inherited CpuEngine method and compares CPU against itself.
///   2. IsGpuAvailable is asserted — constructing the engine succeeds with no backend, and the ops fall
///      back to base silently by design.
///   3. maxAbs > 0.0 is asserted — two independent float32 implementations never agree to the last bit, so
///      an exact zero delta IS the fallback signature. A first run passed 6/6 entirely on CPU and the
///      exact 0.000E+000 was the only tell.
/// </summary>
public class GpuTapeGradientParityTests : IDisposable
{
    private readonly ITestOutputHelper _out;
    private readonly IEngine _prior = AiDotNetEngine.Current;

    public GpuTapeGradientParityTests(ITestOutputHelper output) => _out = output;
    public void Dispose() => AiDotNetEngine.Current = _prior;

    private static bool TryGpu(out DirectGpuTensorEngine? engine)
    {
        try
        {
            var candidate = new DirectGpuTensorEngine();
            if (!candidate.IsGpuAvailable) { candidate.Dispose(); engine = null; return false; }
            engine = candidate;
            return true;
        }
        catch (Exception) { engine = null; return false; }
    }

    private static Tensor<float> Rand(int[] shape, int seed, double lo = -1.0, double hi = 1.0)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(lo + rng.NextDouble() * (hi - lo));
        return t;
    }

    /// <summary>
    /// Runs forward+backward on one engine and returns the flattened input gradient.
    /// A non-uniform output weighting keeps the upstream gradient from being constant, which is what makes
    /// a wrong backward observable rather than accidentally matching.
    /// </summary>
    private static float[] GradientOf(IEngine engine, Tensor<float> x, Func<IEngine, Tensor<float>, Tensor<float>> op)
    {
        AiDotNetEngine.Current = engine;

        var input = new Tensor<float>(x.Shape.ToArray());
        for (int i = 0; i < x.Length; i++) input[i] = x[i];

        using var tape = new GradientTape<float>();
        var y = op(engine, input);

        var weight = new Tensor<float>(y.Shape.ToArray());
        for (int i = 0; i < weight.Length; i++) weight[i] = 0.13f + 0.017f * (i % 11);
        var loss = engine.ReduceSum(engine.TensorMultiply(y, weight), null);

        var grads = tape.ComputeGradients(loss, new[] { input });
        Assert.True(grads.TryGetValue(input, out var g) && g is not null,
            "no gradient reached the input — the forward recorded no tape node");

        var flat = new float[g.Length];
        for (int i = 0; i < g.Length; i++) flat[i] = g[i];
        return flat;
    }

    /// <summary>
    /// Whether bit-identical CPU/GPU results are expected for this op, so the divergence guard must not
    /// be used as the engagement probe.
    /// </summary>
    /// <remarks>
    /// The maxAbs &gt; 0 guard assumes two independent float32 implementations cannot agree to the last bit.
    /// That holds for anything doing arithmetic, but NOT for ops that only move data: Upsample3D replicates
    /// elements in the forward and its backward sums gradOutput over each block — exact float adds of the
    /// same values on both engines, so bit-identity is the CORRECT answer and the guard false-positives.
    /// Those ops use the deferred-materialisation counter instead, which observes GPU residency directly.
    /// </remarks>
    private enum Engagement { ExpectDivergence, UseResidencyCounter }

    private void AssertGradientParity(string opName, Tensor<float> x,
        Func<IEngine, Tensor<float>, Tensor<float>> op, double tol = 1e-4,
        Engagement probe = Engagement.ExpectDivergence)
    {
        Assert.True(TryGpu(out var gpu) && gpu is not null,
            $"GPU backend did not resolve, so {opName} would have been compared against itself. Copy the "
            + "CUDA natives into the test output directory AFTER building.");

        using (gpu)
        {
            var cpuGrad = GradientOf(new CpuEngine(), x, op);

            // Count deferred GPU->host materialisations across the GPU run. A non-zero delta means results
            // were GPU-resident and had to be downloaded — direct evidence the device path ran, independent
            // of whether the numbers happen to match the CPU exactly.
            AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.ResetMaterializeCount();
            var gpuGrad = GradientOf(gpu, x, op);
            long materialisations = AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.MaterializeCount;

            Assert.Equal(cpuGrad.Length, gpuGrad.Length);

            double maxAbs = 0, maxRel = 0;
            for (int i = 0; i < cpuGrad.Length; i++)
            {
                double d = Math.Abs(cpuGrad[i] - gpuGrad[i]);
                maxAbs = Math.Max(maxAbs, d);
                maxRel = Math.Max(maxRel, d / Math.Max(1.0, Math.Abs(cpuGrad[i])));
            }

            _out.WriteLine($"{opName,-14} maxAbs={maxAbs:E3}  maxRel={maxRel:E3}  materialisations={materialisations}");

            if (probe == Engagement.ExpectDivergence)
            {
                Assert.True(maxAbs > 0.0,
                    $"{opName}: CPU and GPU gradients were BIT-IDENTICAL, so the GPU path did not run and "
                    + "this compared the CPU implementation with itself.");
            }
            else
            {
                Assert.True(materialisations > 0,
                    $"{opName}: no deferred GPU->host materialisation occurred, so nothing was ever "
                    + "GPU-resident and the device path did not run. (This op is exact on both engines, so "
                    + "bit-identical output cannot be used as the engagement signal.)");
            }
            Assert.True(maxRel <= tol,
                $"{opName}: GPU gradient diverged from CPU (maxRel={maxRel:E3}). A gross mismatch means the "
                + "recorded backward, its saved state, or the overload is wrong — not float rounding.");
        }
    }

    [Fact]
    public void TensorCosh_gradients_match_cpu() =>
        AssertGradientParity("TensorCosh", Rand([4, 16], seed: 21, lo: -2.0, hi: 2.0),
            static (e, t) => e.TensorCosh(t));

    [Fact]
    public void TensorSinh_gradients_match_cpu() =>
        AssertGradientParity("TensorSinh", Rand([4, 16], seed: 22, lo: -2.0, hi: 2.0),
            static (e, t) => e.TensorSinh(t));

    /// <summary>Rank-5 [N,C,D,H,W] is the only shape the GPU Upsample3D kernel accepts.</summary>
    [Fact]
    public void Upsample3D_gradients_match_cpu() =>
        AssertGradientParity("Upsample3D", Rand([2, 2, 3, 4, 4], seed: 23),
            static (e, t) => e.Upsample3D(t, 2, 2, 2),
            probe: Engagement.UseResidencyCounter);

    /// <summary>
    /// AffineGrid: theta is [batch, 2, 3], grid is [batch, H, W, 2].
    /// </summary>
    /// <remarks>
    /// Uses the RESIDENCY probe, not divergence. d(grid)/d(theta) is the normalized coordinate grid, fixed
    /// entirely by the output geometry (H, W) — it does NOT depend on any value the forward produced. So
    /// forward precision never reaches the gradient and CPU/GPU agree to the last bit, exactly as for
    /// Upsample3D. Measured: maxAbs 0.000E+000 with 4 materialisations, i.e. the device path ran and the
    /// exact agreement is correct.
    ///
    /// The general rule this establishes: the divergence probe is only valid when the BACKWARD consumes
    /// values the FORWARD computed. "Computes rather than selects" is not sufficient — that was the
    /// distinction I first reasoned from, and it predicted a non-zero delta here, wrongly.
    /// </remarks>
    [Fact]
    public void AffineGrid_gradients_match_cpu() =>
        AssertGradientParity("AffineGrid", Rand([2, 2, 3], seed: 41),
            static (e, t) => e.AffineGrid(t, 5, 7),
            probe: Engagement.UseResidencyCounter);

    /// <summary>
    /// RBFKernel records THREE differentiable inputs (input, centers, epsilons), so all three are checked.
    /// Scatter showed one operand can be exactly right while another is corrupt.
    /// </summary>
    /// <remarks>
    /// EPSILONS ARE DELIBERATELY UNIFORM. CpuEngine saves only epsilons[0] as backward state, so
    /// RBFKernelBackward treats the width as shared across all centers — with varying epsilons the CPU
    /// reference is itself wrong for centers 1..n, and agreeing with a wrong reference would prove nothing.
    /// Uniform epsilons keep the reference correct so this test measures the GPU, not the shared defect.
    /// </remarks>
    [Fact]
    public void RBFKernel_gradients_match_cpu_for_all_three_operands()
    {
        Assert.True(TryGpu(out var gpu) && gpu is not null,
            "GPU backend did not resolve — RBFKernel would have been compared against itself.");

        using (gpu)
        {
            const int batch = 4, features = 3, centers = 5;
            var inputSeed = Rand([batch, features], seed: 51);
            var centerSeed = Rand([centers, features], seed: 52);

            (float[] gi, float[] gc, float[] ge, long mat) Run(IEngine engine)
            {
                AiDotNetEngine.Current = engine;
                var x = new Tensor<float>([batch, features]);
                for (int i = 0; i < x.Length; i++) x[i] = inputSeed[i];
                var c = new Tensor<float>([centers, features]);
                for (int i = 0; i < c.Length; i++) c[i] = centerSeed[i];
                var eps = new Tensor<float>([centers]);
                for (int i = 0; i < centers; i++) eps[i] = 0.75f;   // uniform — see remarks

                AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.ResetMaterializeCount();
                using var tape = new GradientTape<float>();
                var y = engine.RBFKernel(x, c, eps);

                var w = new Tensor<float>(y.Shape.ToArray());
                for (int i = 0; i < w.Length; i++) w[i] = 0.17f + 0.011f * (i % 7);
                var loss = engine.ReduceSum(engine.TensorMultiply(y, w), null);

                var grads = tape.ComputeGradients(loss, new[] { x, c, eps });
                Assert.True(grads.TryGetValue(x, out var g1) && g1 is not null, "no gradient to input");
                Assert.True(grads.TryGetValue(c, out var g2) && g2 is not null, "no gradient to centers");
                Assert.True(grads.TryGetValue(eps, out var g3) && g3 is not null, "no gradient to epsilons");

                float[] Flat(Tensor<float> t) { var a = new float[t.Length]; for (int i = 0; i < a.Length; i++) a[i] = t[i]; return a; }
                return (Flat(g1), Flat(g2), Flat(g3),
                        AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.MaterializeCount);
            }

            var (cI, cC, cE, _) = Run(new CpuEngine());
            var (gI, gC, gE, mat) = Run(gpu);

            double dI = 0, dC = 0, dE = 0;
            for (int i = 0; i < cI.Length; i++) dI = Math.Max(dI, Math.Abs(cI[i] - gI[i]));
            for (int i = 0; i < cC.Length; i++) dC = Math.Max(dC, Math.Abs(cC[i] - gC[i]));
            for (int i = 0; i < cE.Length; i++) dE = Math.Max(dE, Math.Abs(cE[i] - gE[i]));

            _out.WriteLine($"RBFKernel      d(input)={dI:E3}  d(centers)={dC:E3}  d(epsilons)={dE:E3}  materialisations={mat}");

            Assert.True(mat > 0, "RBFKernel: nothing was GPU-resident, so the device path did not run.");
            Assert.True(dI <= 1e-4, $"RBFKernel input gradient diverged: {dI:E3}");
            Assert.True(dC <= 1e-4, $"RBFKernel centers gradient diverged: {dC:E3}");
            Assert.True(dE <= 1e-4, $"RBFKernel epsilons gradient diverged: {dE:E3}");
        }
    }

    // Scatter has NO test here: its bail was RESTORED because this very test caught a real defect —
    // recording CpuEngine's ScatterBackward on the GPU result gave d(values) exactly right but d(input)
    // wrong by 3.14e-01. Scatter overwrites input at the scattered positions, so d/d(input) must be zero
    // there, and the reused recording did not reproduce that mask. Restore this test together with a
    // correct GPU-side backward. Keeping it while the op bails would only assert CPU against CPU.

    // Sparsemax has NO test here on purpose: its bail was restored because the GPU path throws
    // InvalidOperationException("CUDA kernel not found: where_select"). Add the test back together with
    // the where_select kernel.
}
