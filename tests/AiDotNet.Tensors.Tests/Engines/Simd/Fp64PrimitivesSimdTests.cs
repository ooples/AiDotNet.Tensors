using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// Stage 4 (#415) parity tests for FP64 SIMD primitives ported from
/// scalar paths: SoftmaxBackward, LayerNormBackward, GELUBackward.
/// Each test compares the new SIMD output to a scalar reference computed
/// inline; ulp ≤ 4 (absolute tolerance 1e-11 — these are FMA chains, not
/// transcendental sums, so should be very close to bit-identical).
/// </summary>
public class Fp64PrimitivesSimdTests
{
    [Theory]
    [InlineData(1, 16)]
    [InlineData(1, 17)]   // odd axis exercises scalar tail
    [InlineData(2, 64)]
    [InlineData(4, 128)]
    [InlineData(8, 256)]
    [InlineData(1, 1024)]
    public void SoftmaxBackward_Double_Matches_Scalar_Reference(int outer, int axis)
    {
        var engine = new CpuEngine();
        var rng = new System.Random(42);

        var gradOut = new Tensor<double>(new[] { outer, axis });
        var output = new Tensor<double>(new[] { outer, axis });
        for (int i = 0; i < gradOut.Length; i++)
            gradOut[i] = rng.NextDouble() - 0.5;
        // Generate a valid softmax output (positive, rows sum to 1).
        for (int b = 0; b < outer; b++)
        {
            double sum = 0;
            for (int j = 0; j < axis; j++)
            {
                output[b * axis + j] = Math.Exp(rng.NextDouble());
                sum += output[b * axis + j];
            }
            for (int j = 0; j < axis; j++)
                output[b * axis + j] /= sum;
        }

        var actual = engine.SoftmaxBackward(gradOut, output, axis: -1);

        // Scalar reference.
        var expected = new double[gradOut.Length];
        for (int b = 0; b < outer; b++)
        {
            double dot = 0;
            for (int j = 0; j < axis; j++) dot += gradOut[b * axis + j] * output[b * axis + j];
            for (int j = 0; j < axis; j++)
                expected[b * axis + j] = output[b * axis + j] * (gradOut[b * axis + j] - dot);
        }

        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-11,
                $"[{i}] expected={expected[i]:F14} actual={actual[i]:F14}");
    }

    [Theory]
    [InlineData(1, 16)]
    [InlineData(1, 17)]
    [InlineData(2, 64)]
    [InlineData(4, 128)]
    [InlineData(8, 768)]    // ViT-Base hidden dim
    [InlineData(2, 4096)]   // VGG FC dim
    public void LayerNormBackward_Double_Matches_Scalar_Reference(int batchSize, int featureSize)
    {
        var engine = new CpuEngine();
        var rng = new System.Random(7);

        var input = new Tensor<double>(new[] { batchSize, featureSize });
        var gamma = new Tensor<double>(new[] { featureSize });
        var beta = new Tensor<double>(new[] { featureSize });
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() - 0.5;
        for (int i = 0; i < featureSize; i++)
        {
            gamma[i] = 0.5 + rng.NextDouble();
            beta[i] = rng.NextDouble() - 0.5;
        }
        double epsilon = 1e-5;

        // Forward to get mean/variance.
        var y = engine.LayerNorm(input, gamma, beta, epsilon, out var mean, out var variance);

        // gradOutput sampled normally.
        var gradOut = new Tensor<double>(new[] { batchSize, featureSize });
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = rng.NextDouble() - 0.5;

        var actualGI = engine.LayerNormBackward(gradOut, input, gamma, mean, variance, epsilon,
            out var actualGG, out var actualGB);

        // Scalar reference inline (re-implements pre-Stage-4 path).
        var dInput = new double[input.Length];
        var dGG = new double[featureSize];
        var dGB = new double[featureSize];
        for (int b = 0; b < batchSize; b++)
        {
            int off = b * featureSize;
            double invStd = 1.0 / Math.Sqrt(variance[b] + epsilon);
            double m = mean[b];
            for (int f = 0; f < featureSize; f++)
            {
                double go = gradOut[off + f];
                double normalized = (input[off + f] - m) * invStd;
                dGG[f] += go * normalized;
                dGB[f] += go;
            }
        }
        for (int b = 0; b < batchSize; b++)
        {
            int off = b * featureSize;
            double invStd = 1.0 / Math.Sqrt(variance[b] + epsilon);
            double m = mean[b];
            double sumGrad = 0, sumGradX = 0;
            for (int f = 0; f < featureSize; f++)
            {
                double scaledGrad = gamma[f] * gradOut[off + f];
                sumGrad += scaledGrad;
                sumGradX += scaledGrad * (input[off + f] - m);
            }
            double scale = invStd / featureSize;
            for (int f = 0; f < featureSize; f++)
            {
                double normalized = (input[off + f] - m) * invStd;
                double gradNorm = gamma[f] * gradOut[off + f];
                double term1 = featureSize * gradNorm;
                double term3 = normalized * invStd * sumGradX;
                dInput[off + f] = scale * (term1 - sumGrad - term3);
            }
        }

        // GradInput parity (loose tolerance because FMA reordering changes
        // last-ulp summation).
        for (int i = 0; i < dInput.Length; i++)
            Assert.True(Math.Abs(dInput[i] - actualGI[i]) < 1e-10,
                $"GI[{i}] expected={dInput[i]:F14} actual={actualGI[i]:F14}");
        for (int f = 0; f < featureSize; f++)
        {
            Assert.True(Math.Abs(dGG[f] - actualGG[f]) < 1e-10,
                $"GG[{f}] expected={dGG[f]:F14} actual={actualGG[f]:F14}");
            Assert.True(Math.Abs(dGB[f] - actualGB[f]) < 1e-10,
                $"GB[{f}] expected={dGB[f]:F14} actual={actualGB[f]:F14}");
        }
    }

    [Theory]
    [InlineData(4)]
    [InlineData(7)]      // scalar tail
    [InlineData(16)]
    [InlineData(128)]
    [InlineData(2048)]
    public void GeluBackward_Double_Matches_Scalar_Reference(int length)
    {
        var engine = new CpuEngine();
        var rng = new System.Random(13);

        var input = new Tensor<double>(new[] { length });
        var gradOut = new Tensor<double>(new[] { length });
        for (int i = 0; i < length; i++)
        {
            input[i] = (rng.NextDouble() - 0.5) * 4.0;  // exercise full range incl. negative
            gradOut[i] = rng.NextDouble() - 0.5;
        }

        var actual = engine.GeluBackward(gradOut, input);

        // Scalar reference using Math.Tanh.
        const double sqrtTwoPi = 0.7978845608028654;
        const double coeff = 0.044715;
        for (int i = 0; i < length; i++)
        {
            double x = input[i];
            double tanhArg = sqrtTwoPi * (x + coeff * x * x * x);
            double tanhVal = Math.Tanh(tanhArg);
            double sechSq = 1.0 - tanhVal * tanhVal;
            double derivative = 0.5 * (1.0 + tanhVal) + 0.5 * x * sechSq * sqrtTwoPi * (1.0 + 3.0 * coeff * x * x);
            double expected = gradOut[i] * derivative;
            // FastExp polynomial approximation: ulp tolerance ~1e-6 relative.
            double tol = Math.Max(1e-7, Math.Abs(expected) * 1e-6);
            Assert.True(Math.Abs(expected - actual[i]) < tol,
                $"[{i}] x={x:F4} expected={expected:F10} actual={actual[i]:F10} tol={tol:E2}");
        }
    }
}
