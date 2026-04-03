using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

public class BackwardBottleneckProfile
{
    private readonly ITestOutputHelper _output;
    private readonly IEngine _engine = AiDotNetEngine.Current;

    public BackwardBottleneckProfile(ITestOutputHelper output) => _output = output;

    [Fact(Skip = "Profiling benchmark - run manually")]
    public void Profile_FullStep_Microbreakdown()
    {
        var input = Tensor<float>.CreateRandom([32, 128]);
        var w1 = Tensor<float>.CreateRandom([128, 64]);
        var w2 = Tensor<float>.CreateRandom([64, 32]);
        var w3 = Tensor<float>.CreateRandom([32, 10]);

        int warmup = 30;
        int iterations = 300;

        // Warmup
        for (int i = 0; i < warmup; i++)
        {
            using var tape = new GradientTape<float>();
            var h1 = _engine.ReLU(_engine.TensorMatMul(input, w1));
            var h2 = _engine.ReLU(_engine.TensorMatMul(h1, w2));
            var o = _engine.TensorMatMul(h2, w3);
            var l = _engine.ReduceSum(o, null);
            tape.ComputeGradients(l, new[] { w1, w2, w3 });
        }

        // Measure each piece individually in a training step
        double totalTapeCreateUs = 0, totalFwd1Us = 0, totalRelu1Us = 0;
        double totalFwd2Us = 0, totalRelu2Us = 0, totalFwd3Us = 0;
        double totalReduceUs = 0, totalBackwardUs = 0, totalTapeDisposeUs = 0;

        var sw = new Stopwatch();

        for (int iter = 0; iter < iterations; iter++)
        {
            sw.Restart(); var tape = new GradientTape<float>(); sw.Stop();
            totalTapeCreateUs += sw.Elapsed.TotalMilliseconds * 1000;

            sw.Restart(); var mm1 = _engine.TensorMatMul(input, w1); sw.Stop();
            totalFwd1Us += sw.Elapsed.TotalMilliseconds * 1000;

            sw.Restart(); var h1 = _engine.ReLU(mm1); sw.Stop();
            totalRelu1Us += sw.Elapsed.TotalMilliseconds * 1000;

            sw.Restart(); var mm2 = _engine.TensorMatMul(h1, w2); sw.Stop();
            totalFwd2Us += sw.Elapsed.TotalMilliseconds * 1000;

            sw.Restart(); var h2 = _engine.ReLU(mm2); sw.Stop();
            totalRelu2Us += sw.Elapsed.TotalMilliseconds * 1000;

            sw.Restart(); var o = _engine.TensorMatMul(h2, w3); sw.Stop();
            totalFwd3Us += sw.Elapsed.TotalMilliseconds * 1000;

            sw.Restart(); var l = _engine.ReduceSum(o, null); sw.Stop();
            totalReduceUs += sw.Elapsed.TotalMilliseconds * 1000;

            sw.Restart(); var grads = tape.ComputeGradients(l, new[] { w1, w2, w3 }); sw.Stop();
            totalBackwardUs += sw.Elapsed.TotalMilliseconds * 1000;

            sw.Restart(); tape.Dispose(); sw.Stop();
            totalTapeDisposeUs += sw.Elapsed.TotalMilliseconds * 1000;
        }

        double total = totalTapeCreateUs + totalFwd1Us + totalRelu1Us + totalFwd2Us +
            totalRelu2Us + totalFwd3Us + totalReduceUs + totalBackwardUs + totalTapeDisposeUs;

        _output.WriteLine($"=== Micro-breakdown per step ({iterations} iters) ===");
        _output.WriteLine($"Tape create:    {totalTapeCreateUs / iterations:F1}us");
        _output.WriteLine($"MatMul1 32x128@128x64: {totalFwd1Us / iterations:F1}us");
        _output.WriteLine($"ReLU1 32x64:    {totalRelu1Us / iterations:F1}us");
        _output.WriteLine($"MatMul2 32x64@64x32:   {totalFwd2Us / iterations:F1}us");
        _output.WriteLine($"ReLU2 32x32:    {totalRelu2Us / iterations:F1}us");
        _output.WriteLine($"MatMul3 32x32@32x10:   {totalFwd3Us / iterations:F1}us");
        _output.WriteLine($"ReduceSum:      {totalReduceUs / iterations:F1}us");
        _output.WriteLine($"ComputeGradients: {totalBackwardUs / iterations:F1}us");
        _output.WriteLine($"Tape dispose:   {totalTapeDisposeUs / iterations:F1}us");
        _output.WriteLine($"---");
        _output.WriteLine($"Total:          {total / iterations:F1}us ({1_000_000 / (total / iterations):F0} steps/sec)");
        _output.WriteLine($"Forward total:  {(totalFwd1Us + totalRelu1Us + totalFwd2Us + totalRelu2Us + totalFwd3Us + totalReduceUs) / iterations:F1}us");
        _output.WriteLine($"Backward:       {totalBackwardUs / iterations:F1}us");
        _output.WriteLine($"Overhead:       {(totalTapeCreateUs + totalTapeDisposeUs) / iterations:F1}us");
    }

    [Fact(Skip = "Profiling benchmark - run manually")]
    public void Profile_BackwardMatMul_Components()
    {
        // The backward for matmul C = A @ B is:
        //   dA = dC @ B^T
        //   dB = A^T @ dC
        // Each involves a transpose + matmul. Let's measure each piece.

        var A = Tensor<float>.CreateRandom([32, 128]);
        var B = Tensor<float>.CreateRandom([128, 64]);
        var dC = Tensor<float>.CreateRandom([32, 64]);

        int warmup = 50;
        int iterations = 500;

        // Warmup
        for (int i = 0; i < warmup; i++)
        {
            var bt = _engine.TensorTranspose(B);
            _engine.TensorMatMul(dC, bt);
        }

        // Measure transpose of B (128x64 -> 64x128)
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            _engine.TensorTranspose(B);
        sw.Stop();
        double transposeBUs = sw.Elapsed.TotalMilliseconds * 1000 / iterations;

        // Measure matmul dC @ B^T (32x64 @ 64x128 = 32x128)
        var BT = _engine.TensorTranspose(B);
        sw.Restart();
        for (int i = 0; i < iterations; i++)
            _engine.TensorMatMul(dC, BT);
        sw.Stop();
        double matmulGradAUs = sw.Elapsed.TotalMilliseconds * 1000 / iterations;

        // Is B^T contiguous?
        _output.WriteLine($"B^T IsContiguous: {BT.IsContiguous}");

        // Measure transpose of A (32x128 -> 128x32)
        sw.Restart();
        for (int i = 0; i < iterations; i++)
            _engine.TensorTranspose(A);
        sw.Stop();
        double transposeAUs = sw.Elapsed.TotalMilliseconds * 1000 / iterations;

        // Measure matmul A^T @ dC (128x32 @ 32x64 = 128x64)
        var AT = _engine.TensorTranspose(A);
        _output.WriteLine($"A^T IsContiguous: {AT.IsContiguous}");

        sw.Restart();
        for (int i = 0; i < iterations; i++)
            _engine.TensorMatMul(AT, dC);
        sw.Stop();
        double matmulGradBUs = sw.Elapsed.TotalMilliseconds * 1000 / iterations;

        // Measure matmul with pre-contiguous transpose
        var BT_contig = BT.IsContiguous ? BT : BT.Contiguous();
        var AT_contig = AT.IsContiguous ? AT : AT.Contiguous();

        sw.Restart();
        for (int i = 0; i < iterations; i++)
            _engine.TensorMatMul(dC, BT_contig);
        sw.Stop();
        double matmulGradA_contigUs = sw.Elapsed.TotalMilliseconds * 1000 / iterations;

        sw.Restart();
        for (int i = 0; i < iterations; i++)
            _engine.TensorMatMul(AT_contig, dC);
        sw.Stop();
        double matmulGradB_contigUs = sw.Elapsed.TotalMilliseconds * 1000 / iterations;

        // Measure forward matmul for comparison
        sw.Restart();
        for (int i = 0; i < iterations; i++)
            _engine.TensorMatMul(A, B);
        sw.Stop();
        double forwardUs = sw.Elapsed.TotalMilliseconds * 1000 / iterations;

        double backwardTotal = transposeBUs + matmulGradAUs + transposeAUs + matmulGradBUs;
        double backwardOptimal = matmulGradA_contigUs + matmulGradB_contigUs;

        _output.WriteLine($"");
        _output.WriteLine($"=== MatMul Backward Breakdown [32x128 @ 128x64] ===");
        _output.WriteLine($"Forward:              {forwardUs:F1}us");
        _output.WriteLine($"");
        _output.WriteLine($"Backward components:");
        _output.WriteLine($"  Transpose B:        {transposeBUs:F1}us");
        _output.WriteLine($"  MatMul dC@B^T:      {matmulGradAUs:F1}us (B^T contiguous={BT.IsContiguous})");
        _output.WriteLine($"  Transpose A:        {transposeAUs:F1}us");
        _output.WriteLine($"  MatMul A^T@dC:      {matmulGradBUs:F1}us (A^T contiguous={AT.IsContiguous})");
        _output.WriteLine($"  Total backward:     {backwardTotal:F1}us");
        _output.WriteLine($"");
        _output.WriteLine($"With pre-contiguous transposes:");
        _output.WriteLine($"  MatMul dC@B^T:      {matmulGradA_contigUs:F1}us");
        _output.WriteLine($"  MatMul A^T@dC:      {matmulGradB_contigUs:F1}us");
        _output.WriteLine($"  Optimal backward:   {backwardOptimal:F1}us (no transpose cost)");
        _output.WriteLine($"");
        _output.WriteLine($"Backward/Forward ratio: {backwardTotal / forwardUs:F1}x");
        _output.WriteLine($"Transpose overhead:   {(transposeBUs + transposeAUs) / backwardTotal * 100:F0}% of backward");
    }
}
