using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Integration tests verifying GPU fused kernels produce correct results
/// by comparing against CPU reference implementations.
/// Tests are skipped when no GPU backend is available.
/// </summary>
public class GpuFusedKernelCorrectnessTests
{
    private readonly CpuEngine _cpu = new();
    private readonly DirectGpuTensorEngine? _gpu;
    private const float Tolerance = 1e-4f;

    public GpuFusedKernelCorrectnessTests()
    {
        try
        {
            _gpu = new DirectGpuTensorEngine();
            if (!_gpu.IsGpuAvailable)
                _gpu = null;
        }
        catch
        {
            _gpu = null;
        }
    }

    private void SkipIfNoGpu()
    {
        Skip.If(_gpu is null, "No GPU backend available");
    }

    private static Tensor<float> RandomTensor(int[] shape, int seed = 42)
    {
        var rng = new Random(seed);
        int total = 1;
        foreach (var d in shape) total *= d;
        var data = new float[total];
        for (int i = 0; i < total; i++)
            data[i] = (float)(rng.NextDouble() * 2 - 1);
        return new Tensor<float>(data, shape);
    }

    private static void AssertClose(float expected, float actual, float tolerance = Tolerance)
    {
        Assert.True(Math.Abs(expected - actual) < tolerance,
            $"Expected {expected}, got {actual}, diff {Math.Abs(expected - actual)}");
    }

    private static void AssertTensorsClose(Tensor<float> expected, Tensor<float> actual, float tolerance = Tolerance)
    {
        Assert.Equal(expected.Shape, actual.Shape);
        var expData = expected.GetDataArray();
        var actData = actual.GetDataArray();
        for (int i = 0; i < expData.Length; i++)
        {
            Assert.True(Math.Abs(expData[i] - actData[i]) < tolerance,
                $"Element [{i}]: expected {expData[i]}, got {actData[i]}, diff {Math.Abs(expData[i] - actData[i])}");
        }
    }

    #region Element-wise ops

    [Fact]
    public void TensorAdd_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var a = RandomTensor(new[] { 64, 128 }, 1);
        var b = RandomTensor(new[] { 64, 128 }, 2);
        var cpuResult = _cpu.TensorAdd(a, b);
        var gpuResult = _gpu!.TensorAdd(a, b);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [Fact]
    public void TensorMultiply_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var a = RandomTensor(new[] { 64, 128 }, 3);
        var b = RandomTensor(new[] { 64, 128 }, 4);
        var cpuResult = _cpu.TensorMultiply(a, b);
        var gpuResult = _gpu!.TensorMultiply(a, b);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [Fact]
    public void TensorSubtract_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var a = RandomTensor(new[] { 64, 128 }, 5);
        var b = RandomTensor(new[] { 64, 128 }, 6);
        var cpuResult = _cpu.TensorSubtract(a, b);
        var gpuResult = _gpu!.TensorSubtract(a, b);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [Fact]
    public void TensorDivide_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var a = RandomTensor(new[] { 64, 128 }, 7);
        // Avoid division by zero
        var rng = new Random(8);
        var bData = new float[64 * 128];
        for (int i = 0; i < bData.Length; i++)
            bData[i] = (float)(rng.NextDouble() * 1.8 + 0.1); // [0.1, 1.9]
        var b = new Tensor<float>(bData, new[] { 64, 128 });
        var cpuResult = _cpu.TensorDivide(a, b);
        var gpuResult = _gpu!.TensorDivide(a, b);
        AssertTensorsClose(cpuResult, gpuResult, 1e-3f);
    }

    #endregion

    #region Reductions

    [Fact]
    public void TensorSum_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 1024 }, 10);
        var cpuResult = _cpu.TensorSum(input);
        var gpuResult = _gpu!.TensorSum(input);
        AssertClose(cpuResult, gpuResult, 1e-2f); // Reductions have higher tolerance due to order of operations
    }

    [Fact]
    public void TensorMean_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 1024 }, 11);
        var cpuResult = _cpu.TensorMean(input);
        var gpuResult = _gpu!.TensorMean(input);
        AssertClose(cpuResult, gpuResult, 1e-3f);
    }

    #endregion

    #region Activations

    [Fact]
    public void TensorExp_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 256 }, 20);
        var cpuResult = _cpu.TensorExp(input);
        var gpuResult = _gpu!.TensorExp(input);
        AssertTensorsClose(cpuResult, gpuResult, 1e-3f);
    }

    [Fact]
    public void TensorLog_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        // Positive values only
        var rng = new Random(21);
        var data = new float[256];
        for (int i = 0; i < data.Length; i++)
            data[i] = (float)(rng.NextDouble() * 9.9 + 0.1); // [0.1, 10]
        var input = new Tensor<float>(data, new[] { 256 });
        var cpuResult = _cpu.TensorLog(input);
        var gpuResult = _gpu!.TensorLog(input);
        AssertTensorsClose(cpuResult, gpuResult, 1e-3f);
    }

    [Fact]
    public void TensorSigmoid_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 512 }, 22);
        var cpuResult = _cpu.TensorSigmoid(input);
        var gpuResult = _gpu!.TensorSigmoid(input);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [Fact]
    public void TensorTanh_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 512 }, 23);
        var cpuResult = _cpu.TensorTanh(input);
        var gpuResult = _gpu!.TensorTanh(input);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [Fact]
    public void TensorReLU_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 512 }, 24);
        var cpuResult = _cpu.TensorReLU(input);
        var gpuResult = _gpu!.TensorReLU(input);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [Fact]
    public void TensorGELU_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 512 }, 25);
        var cpuResult = _cpu.TensorGELU(input);
        var gpuResult = _gpu!.TensorGELU(input);
        AssertTensorsClose(cpuResult, gpuResult, 1e-3f);
    }

    #endregion

    #region Fused kernels

    [Fact]
    public void TensorClip_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 256 }, 30);
        var cpuResult = _cpu.TensorClip(input, -0.5f, 0.5f);
        var gpuResult = _gpu!.TensorClip(input, -0.5f, 0.5f);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [Fact]
    public void TensorPow_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        // Positive values for pow
        var rng = new Random(31);
        var data = new float[256];
        for (int i = 0; i < data.Length; i++)
            data[i] = (float)(rng.NextDouble() * 4 + 0.1);
        var input = new Tensor<float>(data, new[] { 256 });
        var cpuResult = _cpu.TensorPow(input, 2.5f);
        var gpuResult = _gpu!.TensorPow(input, 2.5f);
        AssertTensorsClose(cpuResult, gpuResult, 1e-2f);
    }

    [Fact]
    public void TensorFrac_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = new Tensor<float>(new float[] { 1.5f, -2.3f, 3.7f, 0.1f, -0.9f, 4.0f }, new[] { 6 });
        var cpuResult = _cpu.TensorFrac(input);
        var gpuResult = _gpu!.TensorFrac(input);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [Fact]
    public void TensorEye_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var cpuResult = _cpu.TensorEye<float>(4);
        var gpuResult = _gpu!.TensorEye<float>(4);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [Fact]
    public void TensorEquals_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var a = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 4 });
        var b = new Tensor<float>(new float[] { 1, 5, 3, 6 }, new[] { 4 });
        var cpuResult = _cpu.TensorEquals(a, b);
        var gpuResult = _gpu!.TensorEquals(a, b);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [Fact]
    public void TensorOuter_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var a = new Tensor<float>(new float[] { 1, 2, 3 }, new[] { 3 });
        var b = new Tensor<float>(new float[] { 4, 5 }, new[] { 2 });
        var cpuResult = _cpu.TensorOuter(a, b);
        var gpuResult = _gpu!.TensorOuter(a, b);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [Fact]
    public void DotProduct_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var a = new Vector<float>(new float[] { 1, 2, 3, 4 });
        var b = new Vector<float>(new float[] { 5, 6, 7, 8 });
        var cpuResult = _cpu.DotProduct(a, b);
        var gpuResult = _gpu!.DotProduct(a, b);
        AssertClose(cpuResult, gpuResult);
    }

    [Fact]
    public void GLU_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        // GLU splits last dim in half: [batch, 2*dim] -> [batch, dim]
        var input = RandomTensor(new[] { 4, 16 }, 40);
        var cpuResult = _cpu.GLU(input, -1);
        var gpuResult = _gpu!.GLU(input, -1);
        AssertTensorsClose(cpuResult, gpuResult, 1e-3f);
    }

    [Fact]
    public void GeGLU_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 4, 16 }, 41);
        var cpuResult = _cpu.GeGLU(input, -1);
        var gpuResult = _gpu!.GeGLU(input, -1);
        AssertTensorsClose(cpuResult, gpuResult, 1e-3f);
    }

    #endregion

    #region Matrix ops

    [Fact]
    public void MatrixMultiply_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var a = new Matrix<float>(4, 8);
        var b = new Matrix<float>(8, 6);
        var rng = new Random(50);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 8; j++)
                a[i, j] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 6; j++)
                b[i, j] = (float)(rng.NextDouble() * 2 - 1);
        var cpuResult = _cpu.MatrixMultiply(a, b);
        var gpuResult = _gpu!.MatrixMultiply(a, b);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 6; j++)
                AssertClose(cpuResult[i, j], gpuResult[i, j], 1e-3f);
    }

    [Fact]
    public void MatrixAdd_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var a = new Matrix<float>(4, 8);
        var b = new Matrix<float>(4, 8);
        var rng = new Random(51);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 8; j++)
            {
                a[i, j] = (float)(rng.NextDouble() * 2 - 1);
                b[i, j] = (float)(rng.NextDouble() * 2 - 1);
            }
        var cpuResult = _cpu.MatrixAdd(a, b);
        var gpuResult = _gpu!.MatrixAdd(a, b);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 8; j++)
                AssertClose(cpuResult[i, j], gpuResult[i, j]);
    }

    #endregion

    #region Softmax

    [Fact]
    public void Softmax_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 4, 32 }, 60);
        var cpuResult = _cpu.Softmax(input, -1);
        var gpuResult = _gpu!.Softmax(input, -1);
        AssertTensorsClose(cpuResult, gpuResult, 1e-3f);
    }

    #endregion

    #region Normalization

    [Fact]
    public void BatchNorm_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 2, 4, 8, 8 }, 70);
        var gamma = new Tensor<float>(new float[] { 1, 1, 1, 1 }, new[] { 4 });
        var beta = new Tensor<float>(new float[] { 0, 0, 0, 0 }, new[] { 4 });
        var cpuResult = _cpu.BatchNorm(input, gamma, beta, 1e-5, out _, out _);
        var gpuResult = _gpu!.BatchNorm(input, gamma, beta, 1e-5, out _, out _);
        AssertTensorsClose(cpuResult, gpuResult, 1e-3f);
    }

    #endregion

    #region Convolution

    [Fact]
    public void Conv2D_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 1, 3, 8, 8 }, 80);
        var kernel = RandomTensor(new[] { 16, 3, 3, 3 }, 81);
        var cpuResult = _cpu.Conv2D(input, kernel, 1, 1, 1);
        var gpuResult = _gpu!.Conv2D(input, kernel, 1, 1, 1);
        AssertTensorsClose(cpuResult, gpuResult, 1e-2f);
    }

    #endregion

    #region Large tensor stress test

    [Fact]
    public void TensorAdd_LargeTensor_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var a = RandomTensor(new[] { 1, 64, 32, 32 }, 90);
        var b = RandomTensor(new[] { 1, 64, 32, 32 }, 91);
        var cpuResult = _cpu.TensorAdd(a, b);
        var gpuResult = _gpu!.TensorAdd(a, b);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    #endregion

    #region Newly wired ops

    [Fact]
    public void TensorAddScalar_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 256 }, 100);
        var cpuResult = _cpu.TensorAddScalar(input, 3.14f);
        var gpuResult = _gpu!.TensorAddScalar(input, 3.14f);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [Fact]
    public void TensorBroadcastMultiply_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var a = RandomTensor(new[] { 4, 8 }, 101);
        var b = RandomTensor(new[] { 1, 8 }, 102);
        var cpuResult = _cpu.TensorBroadcastMultiply(a, b);
        var gpuResult = _gpu!.TensorBroadcastMultiply(a, b);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [Fact]
    public void TensorSiLU_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 256 }, 103);
        var cpuResult = _cpu.TensorSiLU(input);
        var gpuResult = _gpu!.TensorSiLU(input);
        AssertTensorsClose(cpuResult, gpuResult, 1e-3f);
    }

    [Fact]
    public void TensorMish_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 256 }, 104);
        var cpuResult = _cpu.TensorMish(input);
        var gpuResult = _gpu!.TensorMish(input);
        AssertTensorsClose(cpuResult, gpuResult, 1e-3f);
    }

    [Fact]
    public void TensorDiag_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var diag = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 4 });
        var cpuResult = _cpu.TensorDiag(diag);
        var gpuResult = _gpu!.TensorDiag(diag);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [Fact]
    public void TensorLinspace_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var cpuResult = _cpu.TensorLinspace(0f, 10f, 100);
        var gpuResult = _gpu!.TensorLinspace(0f, 10f, 100);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [Fact]
    public void ReduceSum_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 4, 64 }, 106);
        var cpuResult = _cpu.ReduceSum(input, new[] { 1 }, false);
        var gpuResult = _gpu!.ReduceSum(input, new[] { 1 }, false);
        AssertTensorsClose(cpuResult, gpuResult, 1e-2f);
    }

    [Fact]
    public void ReduceMean_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 4, 64 }, 107);
        var cpuResult = _cpu.ReduceMean(input, new[] { 1 }, false);
        var gpuResult = _gpu!.ReduceMean(input, new[] { 1 }, false);
        AssertTensorsClose(cpuResult, gpuResult, 1e-3f);
    }

    [Fact]
    public void Pad_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 1, 3, 8, 8 }, 108);
        var cpuResult = _cpu.Pad(input, 1, 1, 1, 1, 0f);
        var gpuResult = _gpu!.Pad(input, 1, 1, 1, 1, 0f);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [Fact]
    public void TensorSumOfSquares_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 256 }, 109);
        var cpuResult = _cpu.TensorSumOfSquares(input);
        var gpuResult = _gpu!.TensorSumOfSquares(input);
        AssertClose(cpuResult, gpuResult, 1e-1f);
    }

    [Fact]
    public void TensorTriangularMask_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var cpuResult = _cpu.TensorTriangularMask<float>(8, true, 0);
        var gpuResult = _gpu!.TensorTriangularMask<float>(8, true, 0);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    #endregion

    #region Softmax and advanced ops

    [Fact]
    public void TensorSoftmax_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 4, 32 }, 110);
        var cpuResult = _cpu.TensorSoftmax(input, -1);
        var gpuResult = _gpu!.TensorSoftmax(input, -1);
        AssertTensorsClose(cpuResult, gpuResult, 1e-3f);
    }

    [Fact]
    public void TensorLogSoftmax_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 4, 32 }, 111);
        var cpuResult = _cpu.TensorLogSoftmax(input, -1);
        var gpuResult = _gpu!.TensorLogSoftmax(input, -1);
        AssertTensorsClose(cpuResult, gpuResult, 1e-3f);
    }

    [Fact]
    public void TensorBroadcastAdd_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var a = RandomTensor(new[] { 8, 16 }, 112);
        var b = RandomTensor(new[] { 1, 16 }, 113);
        var cpuResult = _cpu.TensorBroadcastAdd(a, b);
        var gpuResult = _gpu!.TensorBroadcastAdd(a, b);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [Fact]
    public void TensorCumSum_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 4, 8 }, 114);
        var cpuResult = _cpu.TensorCumSum(input, 1);
        var gpuResult = _gpu!.TensorCumSum(input, 1);
        AssertTensorsClose(cpuResult, gpuResult, 1e-3f);
    }

    [Fact]
    public void TensorRandomUniform_HasCorrectShape()
    {
        SkipIfNoGpu();
        var result = _gpu!.TensorRandomUniform<float>(new[] { 4, 8 });
        Assert.Equal(new[] { 4, 8 }, result.Shape);
        Assert.Equal(32, result.Length);
        // Values should be in [0, 1)
        var data = result.GetDataArray();
        foreach (var v in data)
        {
            Assert.True(v >= 0f && v < 1f, $"Value {v} out of range [0, 1)");
        }
    }

    [Fact]
    public void ReduceSum_SingleAxis_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 4, 8, 16 }, 116);
        var cpuResult = _cpu.ReduceSum(input, new[] { 2 }, false);
        var gpuResult = _gpu!.ReduceSum(input, new[] { 2 }, false);
        AssertTensorsClose(cpuResult, gpuResult, 1e-2f);
    }

    [Fact]
    public void Upsample_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 1, 3, 4, 4 }, 117);
        var cpuResult = _cpu.Upsample(input, 2, 2);
        var gpuResult = _gpu!.Upsample(input, 2, 2);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [Fact]
    public void PixelShuffle_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 1, 12, 4, 4 }, 118); // 12 = 3 * 2^2
        var cpuResult = _cpu.PixelShuffle(input, 2);
        var gpuResult = _gpu!.PixelShuffle(input, 2);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [Fact]
    public void TensorBatchMatMul_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var a = RandomTensor(new[] { 2, 4, 8 }, 119);
        var b = RandomTensor(new[] { 2, 8, 6 }, 120);
        var cpuResult = _cpu.TensorBatchMatMul(a, b);
        var gpuResult = _gpu!.TensorBatchMatMul(a, b);
        AssertTensorsClose(cpuResult, gpuResult, 1e-2f);
    }

    #endregion
}

/// <summary>
/// Exception to skip tests when GPU is not available.
// SkipException removed — using Skip.If from Xunit.SkippableFact instead
