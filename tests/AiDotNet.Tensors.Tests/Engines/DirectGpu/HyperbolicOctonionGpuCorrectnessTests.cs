// Copyright (c) CrowdTrainer LLC. All rights reserved.
// Tests verifying GPU hyperbolic geometry and octonion algebra kernels
// produce identical results to the CPU reference implementation.

using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// GPU correctness tests for hyperbolic geometry (Poincaré ball model) and
/// octonion algebra operations. Each test runs the operation on CPU, then on GPU,
/// and verifies element-wise equality within floating-point tolerance.
/// </summary>
/// <remarks>
/// The tests run on every backend that derives from this base; each derived class supplies its backend, or the reason
/// it cannot run. HIP joined after its octonion backward kernels turned out never to have been compiled.
/// </remarks>
public abstract class HyperbolicOctonionGpuCorrectnessTestsBase : IDisposable
{
    protected readonly IDirectGpuBackend? _gpu;
    private readonly string? _skipReason;
    private protected const float Tolerance = 1e-4f;
    private protected const float GradTolerance = 1e-3f; // Gradients accumulate error

    protected HyperbolicOctonionGpuCorrectnessTestsBase(IDirectGpuBackend? gpu, string? skipReason)
    {
        _gpu = gpu;
        _skipReason = gpu is null && skipReason is null ? "GPU backend not available" : skipReason;
    }

    public virtual void Dispose()
    {
    }

    private protected void SkipIfNoGpu() => Skip.If(_skipReason is not null, _skipReason);

    // The backend a test runs on; SkipIfNoGpu has already skipped when there is none.
    private protected IDirectGpuBackend Gpu => _gpu ?? throw new InvalidOperationException("No GPU backend; call SkipIfNoGpu first.");

    private static float[] RandomFloats(int count, int seed, float scale = 1f)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        return Enumerable.Range(0, count).Select(_ => (float)(rng.NextDouble() * 2 - 1) * scale).ToArray();
    }

    /// <summary>
    /// Generates random points inside the Poincaré ball (||x|| &lt; 1/sqrt(c)).
    /// </summary>
    private static float[] RandomPoincarePoints(int batchSize, int dim, float curvature, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        float maxRadius = 0.9f / MathF.Sqrt(curvature); // Stay well inside the ball
        var data = new float[batchSize * dim];
        for (int b = 0; b < batchSize; b++)
        {
            float norm = 0;
            for (int d = 0; d < dim; d++)
            {
                data[b * dim + d] = (float)(rng.NextDouble() * 2 - 1);
                norm += data[b * dim + d] * data[b * dim + d];
            }
            norm = MathF.Sqrt(norm);
            float targetRadius = (float)(rng.NextDouble() * maxRadius);
            float scale = (norm > 1e-8f) ? targetRadius / norm : 0f;
            for (int d = 0; d < dim; d++)
                data[b * dim + d] *= scale;
        }
        return data;
    }

    private static float[] CpuMobiusAdd(float[] x, float[] y, int batchSize, int dim, float c)
    {
        var result = new float[batchSize * dim];
        for (int b = 0; b < batchSize; b++)
        {
            int off = b * dim;
            float xSq = 0, ySq = 0, xy = 0;
            for (int d = 0; d < dim; d++)
            {
                xSq += x[off + d] * x[off + d];
                ySq += y[off + d] * y[off + d];
                xy += x[off + d] * y[off + d];
            }
            float denom = 1f + 2f * c * xy + c * c * xSq * ySq;
            if (MathF.Abs(denom) < 1e-10f) denom = 1e-10f;
            float numX = 1f + 2f * c * xy + c * ySq;
            float numY = 1f - c * xSq;
            for (int d = 0; d < dim; d++)
                result[off + d] = (numX * x[off + d] + numY * y[off + d]) / denom;
        }
        return result;
    }

    private static float[] CpuPoincareProject(float[] input, int batchSize, int dim, float curvature, float epsilon)
    {
        var result = new float[batchSize * dim];
        float maxNorm = 1f / MathF.Sqrt(curvature) - epsilon;
        for (int b = 0; b < batchSize; b++)
        {
            int off = b * dim;
            float norm = 0;
            for (int d = 0; d < dim; d++) norm += input[off + d] * input[off + d];
            norm = MathF.Sqrt(norm);
            float scale = norm > maxNorm ? maxNorm / norm : 1f;
            for (int d = 0; d < dim; d++) result[off + d] = input[off + d] * scale;
        }
        return result;
    }

    private static float[] CpuPoincareExpMap(float[] bp, float[] tv, int batchSize, int dim, float c)
    {
        var result = new float[batchSize * dim];
        float sqrtC = MathF.Sqrt(c);
        for (int b = 0; b < batchSize; b++)
        {
            int off = b * dim;
            float bpSq = 0, tvNorm = 0;
            for (int d = 0; d < dim; d++)
            {
                bpSq += bp[off + d] * bp[off + d];
                tvNorm += tv[off + d] * tv[off + d];
            }
            tvNorm = MathF.Sqrt(tvNorm);
            float lambda = 2f / MathF.Max(1f - c * bpSq, 1e-10f);
            if (tvNorm < 1e-10f)
            {
                Array.Copy(bp, off, result, off, dim);
                continue;
            }
            float t = MathF.Tanh(sqrtC * lambda * tvNorm / 2f) / (sqrtC * tvNorm);
            float sSq = 0, bs = 0;
            for (int d = 0; d < dim; d++)
            {
                float sv = t * tv[off + d];
                sSq += sv * sv;
                bs += bp[off + d] * sv;
            }
            float denom = 1f + 2f * c * bs + c * c * bpSq * sSq;
            if (MathF.Abs(denom) < 1e-10f) denom = 1e-10f;
            float numX = 1f + 2f * c * bs + c * sSq;
            float numY = 1f - c * bpSq;
            for (int d = 0; d < dim; d++)
                result[off + d] = (numX * bp[off + d] + numY * t * tv[off + d]) / denom;
        }
        return result;
    }

    private static float[] CpuPoincareDistance(float[] x, float[] y, int batchSize, int dim, float c)
    {
        var result = new float[batchSize];
        for (int b = 0; b < batchSize; b++)
        {
            int off = b * dim;
            float diffSq = 0, xSq = 0, ySq = 0;
            for (int d = 0; d < dim; d++)
            {
                float diff = x[off + d] - y[off + d];
                diffSq += diff * diff;
                xSq += x[off + d] * x[off + d];
                ySq += y[off + d] * y[off + d];
            }
            float denomX = MathF.Max(MathF.Abs(1f - c * xSq), 1e-10f);
            float denomY = MathF.Max(MathF.Abs(1f - c * ySq), 1e-10f);
            float arg = MathF.Max(1f + 2f * c * diffSq / (denomX * denomY), 1f);
            result[b] = MathF.Log(arg + MathF.Sqrt(MathF.Max(arg * arg - 1f, 0f))) / MathF.Sqrt(c);
        }
        return result;
    }

    private static float[] CpuOctonionMultiply(float[] a, float[] b, int count)
    {
        var result = new float[count * 8];
        for (int n = 0; n < count; n++)
        {
            int o = n * 8;
            float a0=a[o],a1=a[o+1],a2=a[o+2],a3=a[o+3],a4=a[o+4],a5=a[o+5],a6=a[o+6],a7=a[o+7];
            float b0=b[o],b1=b[o+1],b2=b[o+2],b3=b[o+3],b4=b[o+4],b5=b[o+5],b6=b[o+6],b7=b[o+7];
            result[o+0]=a0*b0-a1*b1-a2*b2-a3*b3-a4*b4-a5*b5-a6*b6-a7*b7;
            result[o+1]=a0*b1+a1*b0+a2*b3-a3*b2+a4*b5-a5*b4-a6*b7+a7*b6;
            result[o+2]=a0*b2-a1*b3+a2*b0+a3*b1+a4*b6+a5*b7-a6*b4-a7*b5;
            result[o+3]=a0*b3+a1*b2-a2*b1+a3*b0+a4*b7-a5*b6+a6*b5-a7*b4;
            result[o+4]=a0*b4-a1*b5-a2*b6-a3*b7+a4*b0+a5*b1+a6*b2+a7*b3;
            result[o+5]=a0*b5+a1*b4-a2*b7+a3*b6-a4*b1+a5*b0-a6*b3+a7*b2;
            result[o+6]=a0*b6+a1*b7+a2*b4-a3*b5-a4*b2+a5*b3+a6*b0-a7*b1;
            result[o+7]=a0*b7-a1*b6+a2*b5+a3*b4-a4*b3-a5*b2+a6*b1+a7*b0;
        }
        return result;
    }

    private static float[] CpuOctonionLinearForward(float[] input, float[] weights, float[] biases,
        int batchSize, int inputFeatures, int outputFeatures)
    {
        var output = new float[batchSize * outputFeatures * 8];
        for (int b = 0; b < batchSize; b++)
        {
            for (int o = 0; o < outputFeatures; o++)
            {
                var accum = new float[8];
                for (int i = 0; i < inputFeatures; i++)
                {
                    int wi = (o * inputFeatures + i) * 8;
                    int ii = (b * inputFeatures + i) * 8;
                    // weight * input (non-commutative order)
                    float w0=weights[wi],w1=weights[wi+1],w2=weights[wi+2],w3=weights[wi+3],
                          w4=weights[wi+4],w5=weights[wi+5],w6=weights[wi+6],w7=weights[wi+7];
                    float a0=input[ii],a1=input[ii+1],a2=input[ii+2],a3=input[ii+3],
                          a4=input[ii+4],a5=input[ii+5],a6=input[ii+6],a7=input[ii+7];
                    accum[0]+=w0*a0-w1*a1-w2*a2-w3*a3-w4*a4-w5*a5-w6*a6-w7*a7;
                    accum[1]+=w0*a1+w1*a0+w2*a3-w3*a2+w4*a5-w5*a4-w6*a7+w7*a6;
                    accum[2]+=w0*a2-w1*a3+w2*a0+w3*a1+w4*a6+w5*a7-w6*a4-w7*a5;
                    accum[3]+=w0*a3+w1*a2-w2*a1+w3*a0+w4*a7-w5*a6+w6*a5-w7*a4;
                    accum[4]+=w0*a4-w1*a5-w2*a6-w3*a7+w4*a0+w5*a1+w6*a2+w7*a3;
                    accum[5]+=w0*a5+w1*a4-w2*a7+w3*a6-w4*a1+w5*a0-w6*a3+w7*a2;
                    accum[6]+=w0*a6+w1*a7+w2*a4-w3*a5-w4*a2+w5*a3+w6*a0-w7*a1;
                    accum[7]+=w0*a7-w1*a6+w2*a5+w3*a4-w4*a3-w5*a2+w6*a1+w7*a0;
                }
                int oo = (b * outputFeatures + o) * 8;
                for (int c = 0; c < 8; c++)
                    output[oo + c] = accum[c] + biases[o * 8 + c];
            }
        }
        return output;
    }

    private static float[] CpuOctonionLinearBackwardBiases(float[] gradOutput, int batchSize, int outputFeatures)
    {
        var gb = new float[outputFeatures * 8];
        for (int b = 0; b < batchSize; b++)
            for (int i = 0; i < outputFeatures * 8; i++)
                gb[i] += gradOutput[b * outputFeatures * 8 + i];
        return gb;
    }

    private static void AssertArraysClose(float[] expected, float[] actual, float tolerance, string context)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            if (float.IsNaN(expected[i]) && float.IsNaN(actual[i])) continue;
            Assert.True(MathF.Abs(expected[i] - actual[i]) < tolerance,
                $"{context} mismatch at [{i}]: CPU={expected[i]:G8}, GPU={actual[i]:G8}, diff={MathF.Abs(expected[i] - actual[i]):G6}");
        }
    }

    // =====================================================================
    // Poincaré Ball Model Tests
    // =====================================================================

    [SkippableTheory]
    [InlineData(1, 8, 1.0f)]
    [InlineData(16, 32, 1.0f)]
    [InlineData(64, 128, 1.0f)]
    [InlineData(16, 32, 0.5f)]
    [InlineData(16, 32, 2.0f)]
    public void PoincareProject_GpuMatchesCpu(int batchSize, int dim, float curvature)
    {
        SkipIfNoGpu();
        // Points outside the ball to test clamping
        var input = RandomFloats(batchSize * dim, 42, scale: 2f);
        var cpuResult = CpuPoincareProject(input, batchSize, dim, curvature, 1e-5f);

        using var gpuIn = Gpu.AllocateBuffer(input);
        using var gpuOut = Gpu.AllocateBuffer(batchSize * dim);
        Gpu.PoincareProject(gpuIn, gpuOut, batchSize, dim, curvature);
        var gpuResult = Gpu.DownloadBuffer(gpuOut);

        AssertArraysClose(cpuResult, gpuResult, Tolerance, "PoincareProject");
    }

    [SkippableTheory]
    [InlineData(1, 8, 1.0f)]
    [InlineData(16, 32, 1.0f)]
    [InlineData(64, 128, 1.0f)]
    [InlineData(16, 32, 0.5f)]
    public void MobiusAdd_GpuMatchesCpu(int batchSize, int dim, float curvature)
    {
        SkipIfNoGpu();
        var x = RandomPoincarePoints(batchSize, dim, curvature, 42);
        var y = RandomPoincarePoints(batchSize, dim, curvature, 99);
        var cpuResult = CpuMobiusAdd(x, y, batchSize, dim, curvature);

        using var gpuX = Gpu.AllocateBuffer(x);
        using var gpuY = Gpu.AllocateBuffer(y);
        using var gpuOut = Gpu.AllocateBuffer(batchSize * dim);
        Gpu.MobiusAdd(gpuX, gpuY, gpuOut, batchSize, dim, curvature);
        var gpuResult = Gpu.DownloadBuffer(gpuOut);

        AssertArraysClose(cpuResult, gpuResult, Tolerance, "MobiusAdd");
    }

    [SkippableTheory]
    [InlineData(1, 8, 1.0f)]
    [InlineData(16, 32, 1.0f)]
    [InlineData(32, 64, 1.0f)]
    [InlineData(16, 32, 0.5f)]
    public void PoincareExpMap_GpuMatchesCpu(int batchSize, int dim, float curvature)
    {
        SkipIfNoGpu();
        var basePoints = RandomPoincarePoints(batchSize, dim, curvature, 42);
        var tangentVecs = RandomFloats(batchSize * dim, 99, scale: 0.5f); // Keep tangent small
        var cpuResult = CpuPoincareExpMap(basePoints, tangentVecs, batchSize, dim, curvature);

        using var gpuBp = Gpu.AllocateBuffer(basePoints);
        using var gpuTv = Gpu.AllocateBuffer(tangentVecs);
        using var gpuOut = Gpu.AllocateBuffer(batchSize * dim);
        Gpu.PoincareExpMap(gpuBp, gpuTv, gpuOut, batchSize, dim, curvature);
        var gpuResult = Gpu.DownloadBuffer(gpuOut);

        AssertArraysClose(cpuResult, gpuResult, Tolerance, "PoincareExpMap");
    }

    [SkippableTheory]
    [InlineData(1, 8, 1.0f)]
    [InlineData(16, 32, 1.0f)]
    [InlineData(32, 64, 1.0f)]
    public void PoincareDistance_GpuMatchesCpu(int batchSize, int dim, float curvature)
    {
        SkipIfNoGpu();
        var x = RandomPoincarePoints(batchSize, dim, curvature, 42);
        var y = RandomPoincarePoints(batchSize, dim, curvature, 99);
        var cpuResult = CpuPoincareDistance(x, y, batchSize, dim, curvature);

        using var gpuX = Gpu.AllocateBuffer(x);
        using var gpuY = Gpu.AllocateBuffer(y);
        using var gpuOut = Gpu.AllocateBuffer(batchSize);
        Gpu.PoincareDistance(gpuX, gpuY, gpuOut, batchSize, dim, curvature);
        var gpuResult = Gpu.DownloadBuffer(gpuOut);

        AssertArraysClose(cpuResult, gpuResult, Tolerance, "PoincareDistance");
    }

    [SkippableFact]
    public void PoincareDistance_IsNonNegative()
    {
        SkipIfNoGpu();
        var x = RandomPoincarePoints(32, 16, 1f, 42);
        var y = RandomPoincarePoints(32, 16, 1f, 99);

        using var gpuX = Gpu.AllocateBuffer(x);
        using var gpuY = Gpu.AllocateBuffer(y);
        using var gpuOut = Gpu.AllocateBuffer(32);
        Gpu.PoincareDistance(gpuX, gpuY, gpuOut, 32, 16, 1f);
        var result = Gpu.DownloadBuffer(gpuOut);

        foreach (var d in result)
            Assert.True(d >= 0, $"Distance must be non-negative, got {d}");
    }

    [SkippableFact]
    public void PoincareDistance_SamePointIsZero()
    {
        SkipIfNoGpu();
        var x = RandomPoincarePoints(8, 16, 1f, 42);

        using var gpuX = Gpu.AllocateBuffer(x);
        using var gpuOut = Gpu.AllocateBuffer(8);
        Gpu.PoincareDistance(gpuX, gpuX, gpuOut, 8, 16, 1f);
        var result = Gpu.DownloadBuffer(gpuOut);

        foreach (var d in result)
            Assert.True(d < 1e-5f, $"Distance to self must be ~0, got {d}");
    }

    // =====================================================================
    // Octonion Algebra Tests
    // =====================================================================

    // A base point near the boundary and a long tangent saturate tanh, so the Mobius sum's float numerator and
    // denominator can cancel; the result must still be finite and strictly inside the ball (the CPU engine projects it
    // back to (1 - 1e-5) / sqrt(c), and so do the kernels).
    [SkippableTheory]
    [InlineData(1.0f)]
    [InlineData(0.5f)]
    public void PoincareExpMap_NearTheBoundary_StaysInsideTheBall(float curvature)
    {
        SkipIfNoGpu();
        const int batch = 32, dim = 16;
        float radius = 1f / MathF.Sqrt(curvature);
        var bp = new float[batch * dim];
        var tv = new float[batch * dim];
        var rng = RandomHelper.CreateSeededRandom(1260);
        for (int b = 0; b < batch; b++)
        {
            // x at 0.999 of the radius along a random direction; v long, along +x for even rows and -x for odd rows.
            double norm = 0;
            for (int d = 0; d < dim; d++) { bp[b * dim + d] = (float)(rng.NextDouble() * 2 - 1); norm += bp[b * dim + d] * bp[b * dim + d]; }
            float s = (float)(0.999 * radius / Math.Sqrt(norm));
            float sign = b % 2 == 0 ? 1f : -1f;
            for (int d = 0; d < dim; d++) { bp[b * dim + d] *= s; tv[b * dim + d] = sign * 50f * bp[b * dim + d]; }
        }
        using var gBp = Gpu.AllocateBuffer(bp);
        using var gTv = Gpu.AllocateBuffer(tv);
        using var gOut = Gpu.AllocateBuffer(batch * dim);
        Gpu.PoincareExpMap(gBp, gTv, gOut, batch, dim, curvature);
        var result = Gpu.DownloadBuffer(gOut);
        for (int b = 0; b < batch; b++)
        {
            double norm = 0;
            for (int d = 0; d < dim; d++)
            {
                Assert.False(float.IsNaN(result[b * dim + d]) || float.IsInfinity(result[b * dim + d]), $"row {b} is not finite");
                norm += (double)result[b * dim + d] * result[b * dim + d];
            }
            Assert.True(Math.Sqrt(norm) < radius, $"row {b}: norm {Math.Sqrt(norm)} is not inside the ball of radius {radius}");
        }
    }
    [SkippableTheory]
    [InlineData(1)]
    [InlineData(32)]
    [InlineData(256)]
    [InlineData(1024)]
    public void OctonionMultiply_GpuMatchesCpu(int count)
    {
        SkipIfNoGpu();
        var a = RandomFloats(count * 8, 42);
        var b = RandomFloats(count * 8, 99);
        var cpuResult = CpuOctonionMultiply(a, b, count);

        using var gpuA = Gpu.AllocateBuffer(a);
        using var gpuB = Gpu.AllocateBuffer(b);
        using var gpuOut = Gpu.AllocateBuffer(count * 8);
        Gpu.OctonionMultiply(gpuA, gpuB, gpuOut, count);
        var gpuResult = Gpu.DownloadBuffer(gpuOut);

        AssertArraysClose(cpuResult, gpuResult, Tolerance, "OctonionMultiply");
    }

    [SkippableFact]
    public void OctonionMultiply_IsNonCommutative()
    {
        SkipIfNoGpu();
        var a = RandomFloats(8, 42);
        var b = RandomFloats(8, 99);

        using var gpuA = Gpu.AllocateBuffer(a);
        using var gpuB = Gpu.AllocateBuffer(b);
        using var gpuAB = Gpu.AllocateBuffer(8);
        using var gpuBA = Gpu.AllocateBuffer(8);
        Gpu.OctonionMultiply(gpuA, gpuB, gpuAB, 1);
        Gpu.OctonionMultiply(gpuB, gpuA, gpuBA, 1);
        var ab = Gpu.DownloadBuffer(gpuAB);
        var ba = Gpu.DownloadBuffer(gpuBA);

        bool anyDifferent = false;
        for (int i = 0; i < 8; i++)
            if (MathF.Abs(ab[i] - ba[i]) > 1e-6f) anyDifferent = true;
        Assert.True(anyDifferent, "Octonion multiplication should be non-commutative");
    }

    [SkippableTheory]
    [InlineData(1, 4, 2)]
    [InlineData(4, 8, 4)]
    [InlineData(8, 16, 8)]
    [InlineData(16, 32, 16)]
    public void OctonionLinearForward_GpuMatchesCpu(int batchSize, int inputFeatures, int outputFeatures)
    {
        SkipIfNoGpu();
        var input = RandomFloats(batchSize * inputFeatures * 8, 42, scale: 0.5f);
        var weights = RandomFloats(outputFeatures * inputFeatures * 8, 99, scale: 0.1f);
        var biases = RandomFloats(outputFeatures * 8, 77, scale: 0.1f);
        var cpuResult = CpuOctonionLinearForward(input, weights, biases, batchSize, inputFeatures, outputFeatures);

        using var gpuIn = Gpu.AllocateBuffer(input);
        using var gpuW = Gpu.AllocateBuffer(weights);
        using var gpuB = Gpu.AllocateBuffer(biases);
        using var gpuOut = Gpu.AllocateBuffer(batchSize * outputFeatures * 8);
        Gpu.OctonionLinearForward(gpuIn, gpuW, gpuB, gpuOut, batchSize, inputFeatures, outputFeatures);
        var gpuResult = Gpu.DownloadBuffer(gpuOut);

        AssertArraysClose(cpuResult, gpuResult, Tolerance, "OctonionLinearForward");
    }

    [SkippableTheory]
    [InlineData(2, 4, 2)]
    [InlineData(4, 8, 4)]
    public void OctonionLinearBackwardBiases_GpuMatchesCpu(int batchSize, int inputFeatures, int outputFeatures)
    {
        SkipIfNoGpu();
        var gradOutput = RandomFloats(batchSize * outputFeatures * 8, 42);
        var cpuResult = CpuOctonionLinearBackwardBiases(gradOutput, batchSize, outputFeatures);

        using var gpuGo = Gpu.AllocateBuffer(gradOutput);
        using var gpuGb = Gpu.AllocateBuffer(outputFeatures * 8);
        Gpu.OctonionLinearBackwardBiases(gpuGo, gpuGb, batchSize, outputFeatures);
        var gpuResult = Gpu.DownloadBuffer(gpuGb);

        AssertArraysClose(cpuResult, gpuResult, Tolerance, "OctonionLinearBackwardBiases");
    }

    [SkippableTheory]
    [InlineData(2, 4, 2)]
    [InlineData(4, 8, 4)]
    public void OctonionLinearBackwardInput_GpuMatchesCpu(int batchSize, int inputFeatures, int outputFeatures)
    {
        SkipIfNoGpu();
        var gradOutput = RandomFloats(batchSize * outputFeatures * 8, 42, scale: 0.5f);
        var weights = RandomFloats(outputFeatures * inputFeatures * 8, 99, scale: 0.1f);

        // CPU reference: full Jacobian backward
        var cpuGradInput = new float[batchSize * inputFeatures * 8];
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < inputFeatures; i++)
            {
                int giOff = (b * inputFeatures + i) * 8;
                var ga = new float[8];
                for (int o = 0; o < outputFeatures; o++)
                {
                    int goOff = (b * outputFeatures + o) * 8;
                    int wOff = (o * inputFeatures + i) * 8;
                    float w0=weights[wOff],w1=weights[wOff+1],w2=weights[wOff+2],w3=weights[wOff+3],
                          w4=weights[wOff+4],w5=weights[wOff+5],w6=weights[wOff+6],w7=weights[wOff+7];
                    float g0=gradOutput[goOff],g1=gradOutput[goOff+1],g2=gradOutput[goOff+2],g3=gradOutput[goOff+3],
                          g4=gradOutput[goOff+4],g5=gradOutput[goOff+5],g6=gradOutput[goOff+6],g7=gradOutput[goOff+7];
                    // Column c of the Jacobian of r = w * a with respect to a: the coefficient of
                    // a_c in each output component r_r, summed against g_r. This is the mirror of the
                    // weight-gradient table in OctonionLinearBackwardWeights_GpuMatchesCpu, and the two
                    // had been transposed into each other.
                    ga[0]+=g0*w0+g1*w1+g2*w2+g3*w3+g4*w4+g5*w5+g6*w6+g7*w7;
                    ga[1]+=g0*(-w1)+g1*w0+g2*w3+g3*(-w2)+g4*w5+g5*(-w4)+g6*(-w7)+g7*w6;
                    ga[2]+=g0*(-w2)+g1*(-w3)+g2*w0+g3*w1+g4*w6+g5*w7+g6*(-w4)+g7*(-w5);
                    ga[3]+=g0*(-w3)+g1*w2+g2*(-w1)+g3*w0+g4*w7+g5*(-w6)+g6*w5+g7*(-w4);
                    ga[4]+=g0*(-w4)+g1*(-w5)+g2*(-w6)+g3*(-w7)+g4*w0+g5*w1+g6*w2+g7*w3;
                    ga[5]+=g0*(-w5)+g1*w4+g2*(-w7)+g3*w6+g4*(-w1)+g5*w0+g6*(-w3)+g7*w2;
                    ga[6]+=g0*(-w6)+g1*w7+g2*w4+g3*(-w5)+g4*(-w2)+g5*w3+g6*w0+g7*(-w1);
                    ga[7]+=g0*(-w7)+g1*(-w6)+g2*w5+g3*w4+g4*(-w3)+g5*(-w2)+g6*w1+g7*w0;
                }
                for (int c = 0; c < 8; c++) cpuGradInput[giOff + c] = ga[c];
            }
        }

        var input = RandomFloats(batchSize * inputFeatures * 8, 55, scale: 0.5f);
        using var gpuGo = Gpu.AllocateBuffer(gradOutput);
        using var gpuIn = Gpu.AllocateBuffer(input);
        using var gpuW = Gpu.AllocateBuffer(weights);
        using var gpuGi = Gpu.AllocateBuffer(batchSize * inputFeatures * 8);
        Gpu.OctonionLinearBackwardInput(gpuGo, gpuIn, gpuW, gpuGi, batchSize, inputFeatures, outputFeatures);
        var gpuGradInput = Gpu.DownloadBuffer(gpuGi);

        AssertArraysClose(cpuGradInput, gpuGradInput, GradTolerance, "OctonionLinearBackwardInput");
    }

    [SkippableTheory]
    [InlineData(2, 4, 2)]
    [InlineData(4, 8, 4)]
    public void OctonionLinearBackwardWeights_GpuMatchesCpu(int batchSize, int inputFeatures, int outputFeatures)
    {
        SkipIfNoGpu();
        var gradOutput = RandomFloats(batchSize * outputFeatures * 8, 42, scale: 0.5f);
        var input = RandomFloats(batchSize * inputFeatures * 8, 99, scale: 0.5f);

        // CPU reference: full Jacobian backward for weights
        var cpuGradWeights = new float[outputFeatures * inputFeatures * 8];
        for (int o = 0; o < outputFeatures; o++)
        {
            for (int i = 0; i < inputFeatures; i++)
            {
                int gwOff = (o * inputFeatures + i) * 8;
                var gw = new float[8];
                for (int b = 0; b < batchSize; b++)
                {
                    int inOff = (b * inputFeatures + i) * 8;
                    int goOff = (b * outputFeatures + o) * 8;
                    float a0=input[inOff],a1=input[inOff+1],a2=input[inOff+2],a3=input[inOff+3],
                          a4=input[inOff+4],a5=input[inOff+5],a6=input[inOff+6],a7=input[inOff+7];
                    float g0=gradOutput[goOff],g1=gradOutput[goOff+1],g2=gradOutput[goOff+2],g3=gradOutput[goOff+3],
                          g4=gradOutput[goOff+4],g5=gradOutput[goOff+5],g6=gradOutput[goOff+6],g7=gradOutput[goOff+7];
                    // Column c of the Jacobian of r = w * a with respect to w: the coefficient of
                    // w_c in each output component r_r, summed against g_r. The arbiter for these
                    // rows is OctonionLinearForward_NumericalGradientCheck below, which differentiates
                    // CpuOctonionLinearForward numerically and never consults this table.
                    gw[0]+=g0*a0+g1*a1+g2*a2+g3*a3+g4*a4+g5*a5+g6*a6+g7*a7;
                    gw[1]+=g0*(-a1)+g1*a0+g2*(-a3)+g3*a2+g4*(-a5)+g5*a4+g6*a7+g7*(-a6);
                    gw[2]+=g0*(-a2)+g1*a3+g2*a0+g3*(-a1)+g4*(-a6)+g5*(-a7)+g6*a4+g7*a5;
                    gw[3]+=g0*(-a3)+g1*(-a2)+g2*a1+g3*a0+g4*(-a7)+g5*a6+g6*(-a5)+g7*a4;
                    gw[4]+=g0*(-a4)+g1*a5+g2*a6+g3*a7+g4*a0+g5*(-a1)+g6*(-a2)+g7*(-a3);
                    gw[5]+=g0*(-a5)+g1*(-a4)+g2*a7+g3*(-a6)+g4*a1+g5*a0+g6*a3+g7*(-a2);
                    gw[6]+=g0*(-a6)+g1*(-a7)+g2*(-a4)+g3*a5+g4*a2+g5*(-a3)+g6*a0+g7*a1;
                    gw[7]+=g0*(-a7)+g1*a6+g2*(-a5)+g3*(-a4)+g4*a3+g5*a2+g6*(-a1)+g7*a0;
                }
                for (int c = 0; c < 8; c++) cpuGradWeights[gwOff + c] = gw[c];
            }
        }

        using var gpuGo = Gpu.AllocateBuffer(gradOutput);
        using var gpuIn = Gpu.AllocateBuffer(input);
        using var gpuGw = Gpu.AllocateBuffer(outputFeatures * inputFeatures * 8);
        Gpu.OctonionLinearBackwardWeights(gpuGo, gpuIn, gpuGw, batchSize, inputFeatures, outputFeatures);
        var gpuGradWeights = Gpu.DownloadBuffer(gpuGw);

        AssertArraysClose(cpuGradWeights, gpuGradWeights, GradTolerance, "OctonionLinearBackwardWeights");
    }

    // =====================================================================
    // Numerical Gradient Check (finite differences)
    // =====================================================================

    [SkippableFact]
    public void OctonionLinearForward_NumericalGradientCheck()
    {
        SkipIfNoGpu();
        const int B = 2, I = 2, O = 2;
        const float eps = 1e-3f;
        // Bracketed by measurement, not guessed. With the kernels correct the worst element of this
        // check lands at 2.4E-06 (weights) / 3.5E-06 (input); with the backward Jacobian tables
        // transposed it lands at 1.2E-01 / 4.6E-02. 1E-03 sits ~300x above the float32 finite-
        // difference floor and ~50x below the defect. The previous 0.05f was inside the input
        // defect's own magnitude, so that arm could not fail against the bug it guards.
        const float gradTol = 1e-3f;
        var input = RandomFloats(B * I * 8, 42, scale: 0.3f);
        var weights = RandomFloats(O * I * 8, 99, scale: 0.1f);
        var biases = new float[O * 8];

        // Compute forward and analytic gradients via GPU backward
        var output = CpuOctonionLinearForward(input, weights, biases, B, I, O);
        var gradOutput = (float[])output.Clone(); // dLoss/dOutput = output (for L = ||out||^2 / 2)

        // GPU backward for weight gradients
        using var gpuGo = Gpu.AllocateBuffer(gradOutput);
        using var gpuIn = Gpu.AllocateBuffer(input);
        using var gpuGw = Gpu.AllocateBuffer(O * I * 8);
        Gpu.OctonionLinearBackwardWeights(gpuGo, gpuIn, gpuGw, B, I, O);
        var analyticGrad = Gpu.DownloadBuffer(gpuGw);

        // Compare against numerical gradient via finite differences
        float worstDiff = 0f;
        int worstIdx = -1;
        float worstNumerical = 0f;
        for (int wIdx = 0; wIdx < weights.Length; wIdx++)
        {
            var wPlus = (float[])weights.Clone();
            var wMinus = (float[])weights.Clone();
            wPlus[wIdx] += eps;
            wMinus[wIdx] -= eps;
            var outPlus = CpuOctonionLinearForward(input, wPlus, biases, B, I, O);
            var outMinus = CpuOctonionLinearForward(input, wMinus, biases, B, I, O);
            float lossPlus = 0, lossMinus = 0;
            for (int j = 0; j < outPlus.Length; j++)
            {
                lossPlus += outPlus[j] * outPlus[j] / 2;
                lossMinus += outMinus[j] * outMinus[j] / 2;
            }
            float numericalGrad = (lossPlus - lossMinus) / (2 * eps);

            Assert.True(!float.IsNaN(numericalGrad) && !float.IsInfinity(numericalGrad),
                $"Numerical gradient at weight[{wIdx}] is {numericalGrad}");

            float diff = MathF.Abs(numericalGrad - analyticGrad[wIdx]);
            if (diff > worstDiff)
            {
                worstDiff = diff;
                worstIdx = wIdx;
                worstNumerical = numericalGrad;
            }
        }

        // Asserted on the worst element rather than element-by-element, because the first element is
        // gw0 — the one row the two octonion backward tables share — so an abort there reported a
        // clean 2E-07 while later rows were out by 1E-01.
        Assert.True(worstDiff < gradTol,
            $"Gradient mismatch at weight[{worstIdx}]: numerical={worstNumerical:G6}, " +
            $"analytic(GPU)={(worstIdx >= 0 ? analyticGrad[worstIdx] : 0f):G6}, diff={worstDiff:G6}");
    }

    /// <summary>
    /// The same independent arbiter for the INPUT gradient. Without it the input kernel and the
    /// CPU table in <see cref="OctonionLinearBackwardInput_GpuMatchesCpu"/> can agree with each other
    /// and still both be wrong — which is exactly how the two backward tables came to be transposed
    /// into each other. Finite differences consult only the forward pass.
    /// </summary>
    [SkippableFact]
    public void OctonionLinearBackwardInput_NumericalGradientCheck()
    {
        SkipIfNoGpu();
        const int B = 2, I = 2, O = 2;
        const float eps = 1e-3f;
        // Bracketed by measurement, not guessed. With the kernels correct the worst element of this
        // check lands at 2.4E-06 (weights) / 3.5E-06 (input); with the backward Jacobian tables
        // transposed it lands at 1.2E-01 / 4.6E-02. 1E-03 sits ~300x above the float32 finite-
        // difference floor and ~50x below the defect. The previous 0.05f was inside the input
        // defect's own magnitude, so that arm could not fail against the bug it guards.
        const float gradTol = 1e-3f;
        var input = RandomFloats(B * I * 8, 42, scale: 0.3f);
        var weights = RandomFloats(O * I * 8, 99, scale: 0.1f);
        var biases = new float[O * 8];

        var output = CpuOctonionLinearForward(input, weights, biases, B, I, O);
        var gradOutput = (float[])output.Clone(); // dLoss/dOutput = output (for L = ||out||^2 / 2)

        using var gpuGo = Gpu.AllocateBuffer(gradOutput);
        using var gpuIn = Gpu.AllocateBuffer(input);
        using var gpuW = Gpu.AllocateBuffer(weights);
        using var gpuGi = Gpu.AllocateBuffer(B * I * 8);
        Gpu.OctonionLinearBackwardInput(gpuGo, gpuIn, gpuW, gpuGi, B, I, O);
        var analyticGrad = Gpu.DownloadBuffer(gpuGi);

        float worstDiff = 0f;
        int worstIdx = -1;
        float worstNumerical = 0f;
        for (int xIdx = 0; xIdx < input.Length; xIdx++)
        {
            var xPlus = (float[])input.Clone();
            var xMinus = (float[])input.Clone();
            xPlus[xIdx] += eps;
            xMinus[xIdx] -= eps;
            var outPlus = CpuOctonionLinearForward(xPlus, weights, biases, B, I, O);
            var outMinus = CpuOctonionLinearForward(xMinus, weights, biases, B, I, O);
            float lossPlus = 0, lossMinus = 0;
            for (int j = 0; j < outPlus.Length; j++)
            {
                lossPlus += outPlus[j] * outPlus[j] / 2;
                lossMinus += outMinus[j] * outMinus[j] / 2;
            }
            float numericalGrad = (lossPlus - lossMinus) / (2 * eps);

            Assert.True(!float.IsNaN(numericalGrad) && !float.IsInfinity(numericalGrad),
                $"Numerical gradient at input[{xIdx}] is {numericalGrad}");

            float diff = MathF.Abs(numericalGrad - analyticGrad[xIdx]);
            if (diff > worstDiff)
            {
                worstDiff = diff;
                worstIdx = xIdx;
                worstNumerical = numericalGrad;
            }
        }

        // Asserted on the worst element rather than element-by-element so the failure message
        // reports the largest deviation instead of whichever index happens to come first.
        Assert.True(worstDiff < gradTol,
            $"Gradient mismatch at input[{worstIdx}]: numerical={worstNumerical:G6}, " +
            $"analytic(GPU)={(worstIdx >= 0 ? analyticGrad[worstIdx] : 0f):G6}, diff={worstDiff:G6}");
    }
}

[Collection("VulkanGlobalState")]
public sealed class HyperbolicOctonionGpuCorrectnessTests : HyperbolicOctonionGpuCorrectnessTestsBase
{
    public HyperbolicOctonionGpuCorrectnessTests() : base(CreateVulkan(out string? reason), reason) { }

    // VulkanBackend is a singleton, so it is not disposed here.
    private static VulkanBackend? CreateVulkan(out string? skipReason)
    {
        skipReason = null;
        try
        {
            var vulkan = VulkanBackend.Instance;
            if (!vulkan.Initialize()) { skipReason = "Vulkan GPU backend not available"; return null; }
            // Octonion + hyperbolic kernels are emitted as GLSL and compiled at runtime. When libshaderc is missing the
            // pipelines silently no-op (GlslQuadOp returns early), the output buffer stays at its zero-allocation fill,
            // and the CPU-GPU comparison reads "GPU=0" vs a non-zero CPU result. That's an environment gap, not a
            // kernel bug, so skip instead of failing on every dev box that hasn't installed shaderc.
            if (!vulkan.IsGlslCompilerAvailable)
                skipReason = "GLSL runtime compiler (libshaderc) unavailable — install to run octonion/hyperbolic GPU tests.";
            return vulkan;
        }
        catch (Exception ex) when (ex is DllNotFoundException or PlatformNotSupportedException or TypeInitializationException
            or InvalidOperationException)
        {
            skipReason = "Vulkan GPU backend not available";
            return null;
        }
    }
}

[Collection("DirectGpuSerial")]
public sealed class HipHyperbolicOctonionGpuCorrectnessTests : HyperbolicOctonionGpuCorrectnessTestsBase,
    IClassFixture<HipTensorEngineTestFixture>
{
    // The fixture owns the backend: constructing a HipBackend compiles every module, too slow to repeat per test.
    public HipHyperbolicOctonionGpuCorrectnessTests(HipTensorEngineTestFixture fixture)
        : base(fixture.IsAvailable ? fixture.Engine?.GetBackend() : null, fixture.IsAvailable ? null : "HIP GPU backend not available") { }
}

/// <summary>One CUDA backend per test class, or none (and a reason) when there is no NVIDIA driver and GPU.</summary>
public sealed class CudaHyperbolicOctonionFixture : IDisposable
{
    public CudaBackend? Backend { get; }

    public CudaHyperbolicOctonionFixture()
    {
        try
        {
            if (!DirectPtxRuntime.IsAvailable) return;
            var backend = new CudaBackend();
            if (backend.IsAvailable) Backend = backend;
            else backend.Dispose();
        }
        catch (Exception ex) when (ex is DllNotFoundException or PlatformNotSupportedException or TypeInitializationException
            or InvalidOperationException)
        {
            Backend = null;
        }
    }

    public void Dispose() => Backend?.Dispose();
}

[Collection("DirectGpuSerial")]
public sealed class CudaHyperbolicOctonionGpuCorrectnessTests : HyperbolicOctonionGpuCorrectnessTestsBase,
    IClassFixture<CudaHyperbolicOctonionFixture>
{
    public CudaHyperbolicOctonionGpuCorrectnessTests(CudaHyperbolicOctonionFixture fixture)
        : base(fixture.Backend, fixture.Backend is null ? "Requires an NVIDIA CUDA driver and GPU." : null) { }
}
#if NET6_0_OR_GREATER
// OpenClBackendTestFixture exists only on .NET 6+.
[Collection("OpenClBackend")]
public sealed class OpenClHyperbolicOctonionGpuCorrectnessTests : HyperbolicOctonionGpuCorrectnessTestsBase,
    IClassFixture<OpenClBackendTestFixture>
{
    public OpenClHyperbolicOctonionGpuCorrectnessTests(OpenClBackendTestFixture fixture)
        : base(fixture.IsAvailable ? fixture.Backend : null, fixture.IsAvailable ? null : "OpenCL GPU backend not available") { }
}
#endif
