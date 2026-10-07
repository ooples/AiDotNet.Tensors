using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Integration tests verifying GPU fused kernels produce correct results
/// by comparing against CPU reference implementations.
/// Tests are skipped when no GPU backend is available.
/// </summary>
[Collection("DirectGpuSerial")]
public class GpuFusedKernelCorrectnessTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly CpuEngine _cpu = new();
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    private const float Tolerance = 1e-4f;
    // Typed as IEngine on purpose: most GPU overrides are explicit IEngine implementations, so a call on the
    // concrete DirectGpuTensorEngine binds to the inherited CpuEngine method and the test compares the CPU with itself.
    private IEngine Gpu => _fixture.Engine ?? throw new InvalidOperationException(
        "Direct GPU engine was not initialized.", _fixture.InitializationException);

    public GpuFusedKernelCorrectnessTests(DirectGpuTensorEngineTestFixture fixture)
    {
        _fixture = fixture;
    }

    private void SkipIfNoGpu()
    {
        Skip.If(!_fixture.IsAvailable, "No GPU backend available");
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

    [SkippableFact]
    public void TensorAdd_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var a = RandomTensor(new[] { 64, 128 }, 1);
        var b = RandomTensor(new[] { 64, 128 }, 2);
        var cpuResult = _cpu.TensorAdd(a, b);
        var gpuResult = Gpu.TensorAdd(a, b);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [SkippableFact]
    public void TensorMultiply_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var a = RandomTensor(new[] { 64, 128 }, 3);
        var b = RandomTensor(new[] { 64, 128 }, 4);
        var cpuResult = _cpu.TensorMultiply(a, b);
        var gpuResult = Gpu.TensorMultiply(a, b);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [SkippableFact]
    public void TensorSubtract_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var a = RandomTensor(new[] { 64, 128 }, 5);
        var b = RandomTensor(new[] { 64, 128 }, 6);
        var cpuResult = _cpu.TensorSubtract(a, b);
        var gpuResult = Gpu.TensorSubtract(a, b);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [SkippableFact]
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
        var gpuResult = Gpu.TensorDivide(a, b);
        AssertTensorsClose(cpuResult, gpuResult, 1e-3f);
    }

    #endregion

    #region Reductions

    [SkippableFact]
    public void TensorSum_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 1024 }, 10);
        var cpuResult = _cpu.TensorSum(input);
        var gpuResult = Gpu.TensorSum(input);
        AssertClose(cpuResult, gpuResult, 1e-2f); // Reductions have higher tolerance due to order of operations
    }

    [SkippableFact]
    public void TensorMean_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 1024 }, 11);
        var cpuResult = _cpu.TensorMean(input);
        var gpuResult = Gpu.TensorMean(input);
        AssertClose(cpuResult, gpuResult, 1e-3f);
    }

    #endregion

    #region Activations

    [SkippableFact]
    public void TensorExp_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 256 }, 20);
        var cpuResult = _cpu.TensorExp(input);
        var gpuResult = Gpu.TensorExp(input);
        AssertTensorsClose(cpuResult, gpuResult, 1e-3f);
    }

    [SkippableFact]
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
        var gpuResult = Gpu.TensorLog(input);
        AssertTensorsClose(cpuResult, gpuResult, 1e-3f);
    }

    [SkippableFact]
    public void TensorSigmoid_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 512 }, 22);
        var cpuResult = _cpu.TensorSigmoid(input);
        var gpuResult = Gpu.TensorSigmoid(input);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [SkippableFact]
    public void TensorTanh_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 512 }, 23);
        var cpuResult = _cpu.TensorTanh(input);
        var gpuResult = Gpu.TensorTanh(input);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [SkippableFact]
    public void TensorReLU_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 512 }, 24);
        var cpuResult = _cpu.TensorReLU(input);
        var gpuResult = Gpu.TensorReLU(input);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [SkippableFact]
    public void TensorGELU_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 512 }, 25);
        var cpuResult = _cpu.TensorGELU(input);
        var gpuResult = Gpu.TensorGELU(input);
        AssertTensorsClose(cpuResult, gpuResult, 1e-3f);
    }

    #endregion

    #region Fused kernels

    [SkippableFact]
    public void TensorClip_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 256 }, 30);
        var cpuResult = _cpu.TensorClip(input, -0.5f, 0.5f);
        var gpuResult = Gpu.TensorClip(input, -0.5f, 0.5f);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [SkippableFact]
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
        var gpuResult = Gpu.TensorPow(input, 2.5f);
        AssertTensorsClose(cpuResult, gpuResult, 1e-2f);
    }

    [SkippableFact]
    public void TensorFrac_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = new Tensor<float>(new float[] { 1.5f, -2.3f, 3.7f, 0.1f, -0.9f, 4.0f }, new[] { 6 });
        var cpuResult = _cpu.TensorFrac(input);
        var gpuResult = Gpu.TensorFrac(input);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [SkippableFact]
    public void TensorEye_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var cpuResult = _cpu.TensorEye<float>(4);
        var gpuResult = Gpu.TensorEye<float>(4);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [SkippableFact]
    public void TensorEquals_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var a = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 4 });
        var b = new Tensor<float>(new float[] { 1, 5, 3, 6 }, new[] { 4 });
        var cpuResult = _cpu.TensorEquals(a, b);
        var gpuResult = Gpu.TensorEquals(a, b);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [SkippableFact]
    public void TensorOuter_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var a = new Tensor<float>(new float[] { 1, 2, 3 }, new[] { 3 });
        var b = new Tensor<float>(new float[] { 4, 5 }, new[] { 2 });
        var cpuResult = _cpu.TensorOuter(a, b);
        var gpuResult = Gpu.TensorOuter(a, b);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [SkippableFact]
    public void DotProduct_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var a = new Vector<float>(new float[] { 1, 2, 3, 4 });
        var b = new Vector<float>(new float[] { 5, 6, 7, 8 });
        var cpuResult = _cpu.DotProduct(a, b);
        var gpuResult = Gpu.DotProduct(a, b);
        AssertClose(cpuResult, gpuResult);
    }

    [SkippableFact]
    public void GLU_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        // GLU splits last dim in half: [batch, 2*dim] -> [batch, dim]
        var input = RandomTensor(new[] { 4, 16 }, 40);
        var cpuResult = _cpu.GLU(input, -1);
        var gpuResult = Gpu.GLU(input, -1);
        AssertTensorsClose(cpuResult, gpuResult, 1e-3f);
    }

    [SkippableFact]
    public void GeGLU_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 4, 16 }, 41);
        var cpuResult = _cpu.GeGLU(input, -1);
        var gpuResult = Gpu.GeGLU(input, -1);
        AssertTensorsClose(cpuResult, gpuResult, 1e-3f);
    }

    #endregion

    #region Matrix ops

    [SkippableFact]
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
        var gpuResult = Gpu.MatrixMultiply(a, b);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 6; j++)
                AssertClose(cpuResult[i, j], gpuResult[i, j], 1e-3f);
    }

    [SkippableFact]
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
        var gpuResult = Gpu.MatrixAdd(a, b);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 8; j++)
                AssertClose(cpuResult[i, j], gpuResult[i, j]);
    }

    #endregion

    #region Softmax

    [SkippableFact]
    public void Softmax_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 4, 32 }, 60);
        var cpuResult = _cpu.Softmax(input, -1);
        var gpuResult = Gpu.Softmax(input, -1);
        AssertTensorsClose(cpuResult, gpuResult, 1e-3f);
    }

    #endregion

    #region Normalization

    [SkippableFact]
    public void BatchNorm_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 2, 4, 8, 8 }, 70);
        var gamma = new Tensor<float>(new float[] { 1, 1, 1, 1 }, new[] { 4 });
        var beta = new Tensor<float>(new float[] { 0, 0, 0, 0 }, new[] { 4 });
        var cpuResult = _cpu.BatchNorm(input, gamma, beta, 1e-5, out _, out _);
        var gpuResult = Gpu.BatchNorm(input, gamma, beta, 1e-5, out _, out _);
        AssertTensorsClose(cpuResult, gpuResult, 1e-3f);
    }

    #endregion

    #region Convolution

    [SkippableFact]
    public void Conv2D_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 1, 3, 8, 8 }, 80);
        var kernel = RandomTensor(new[] { 16, 3, 3, 3 }, 81);
        var cpuResult = _cpu.Conv2D(input, kernel, 1, 1, 1);
        var gpuResult = Gpu.Conv2D(input, kernel, 1, 1, 1);
        AssertTensorsClose(cpuResult, gpuResult, 1e-2f);
    }

    #endregion

    #region Large tensor stress test

    [SkippableFact]
    public void TensorAdd_LargeTensor_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var a = RandomTensor(new[] { 1, 64, 32, 32 }, 90);
        var b = RandomTensor(new[] { 1, 64, 32, 32 }, 91);
        var cpuResult = _cpu.TensorAdd(a, b);
        var gpuResult = Gpu.TensorAdd(a, b);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    #endregion

    #region Newly wired ops

    [SkippableFact]
    public void TensorAddScalar_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 256 }, 100);
        var cpuResult = _cpu.TensorAddScalar(input, 3.14f);
        var gpuResult = Gpu.TensorAddScalar(input, 3.14f);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [SkippableFact]
    public void TensorMultiply_BroadcastShapes_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var a = RandomTensor(new[] { 4, 8 }, 101);
        var b = RandomTensor(new[] { 1, 8 }, 102);
        var cpuResult = _cpu.TensorMultiply(a, b);
        var gpuResult = Gpu.TensorMultiply(a, b);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [SkippableFact]
    public void TensorSiLU_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 256 }, 103);
        var cpuResult = _cpu.TensorSiLU(input);
        var gpuResult = Gpu.TensorSiLU(input);
        AssertTensorsClose(cpuResult, gpuResult, 1e-3f);
    }

    [SkippableFact]
    public void TensorMish_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 256 }, 104);
        var cpuResult = _cpu.TensorMish(input);
        var gpuResult = Gpu.TensorMish(input);
        AssertTensorsClose(cpuResult, gpuResult, 1e-3f);
    }

    [SkippableFact]
    public void TensorDiag_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var diag = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 4 });
        var cpuResult = _cpu.TensorDiag(diag);
        var gpuResult = Gpu.TensorDiag(diag);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [SkippableFact]
    public void TensorLinspace_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var cpuResult = _cpu.TensorLinspace(0f, 10f, 100);
        var gpuResult = Gpu.TensorLinspace(0f, 10f, 100);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [SkippableFact]
    public void ReduceSum_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 4, 64 }, 106);
        var cpuResult = _cpu.ReduceSum(input, new[] { 1 }, false);
        var gpuResult = Gpu.ReduceSum(input, new[] { 1 }, false);
        AssertTensorsClose(cpuResult, gpuResult, 1e-2f);
    }

    [SkippableFact]
    public void ReduceMean_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 4, 64 }, 107);
        var cpuResult = _cpu.ReduceMean(input, new[] { 1 }, false);
        var gpuResult = Gpu.ReduceMean(input, new[] { 1 }, false);
        AssertTensorsClose(cpuResult, gpuResult, 1e-3f);
    }

    [SkippableFact]
    public void Pad_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 1, 3, 8, 8 }, 108);
        var cpuResult = _cpu.Pad(input, 1, 1, 1, 1, 0f);
        var gpuResult = Gpu.Pad(input, 1, 1, 1, 1, 0f);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [SkippableFact]
    public void TensorSumOfSquares_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 256 }, 109);
        var cpuResult = _cpu.TensorSumOfSquares(input);
        var gpuResult = Gpu.TensorSumOfSquares(input);
        AssertClose(cpuResult, gpuResult, 1e-1f);
    }

    /// <summary>
    /// The GPU path went through IGpuBatchExecution, which only Vulkan implements, so on CUDA/OpenCL/HIP the value
    /// above was computed by the CPU base after downloading the whole tensor. A device-resident input must be
    /// reduced on the device with only the scalar read back.
    /// </summary>
    [SkippableFact]
    public void TensorSumOfSquares_OnResidentInput_ReadsBackOnlyTheScalar()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 4096 }, 110);
        var expected = _cpu.TensorSumOfSquares(input);
        var resident = Gpu.TensorMultiplyScalar(input, 1f);   // a deferred GPU result: lives on the device
        bool savedCapture = GpuLaunchProbe.CaptureReadbackSites;
        try
        {
            GpuLaunchProbe.CaptureReadbackSites = true;
            GpuLaunchProbe.Reset();
            var got = Gpu.TensorSumOfSquares(resident);
            Assert.True(GpuLaunchProbe.ReadbackBytes <= 16,
                $"sum of squares read back {GpuLaunchProbe.ReadbackBytes} bytes: " + string.Join("; ", GpuLaunchProbe.ReadbackSites));
            Assert.Empty(GpuLaunchProbe.Fallbacks);
            AssertClose(expected, got, 1e-1f);
        }
        finally
        {
            GpuLaunchProbe.CaptureReadbackSites = savedCapture;
        }
    }

    [SkippableFact]
    public void TensorTriangularMask_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var cpuResult = _cpu.TensorTriangularMask<float>(8, true, 0);
        var gpuResult = Gpu.TensorTriangularMask<float>(8, true, 0);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    #endregion

    #region Softmax and advanced ops

    [SkippableFact]
    public void TensorSoftmax_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 4, 32 }, 110);
        var cpuResult = _cpu.TensorSoftmax(input, -1);
        var gpuResult = Gpu.TensorSoftmax(input, -1);
        AssertTensorsClose(cpuResult, gpuResult, 1e-3f);
    }

    [SkippableFact]
    public void TensorLogSoftmax_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 4, 32 }, 111);
        var cpuResult = _cpu.TensorLogSoftmax(input, -1);
        var gpuResult = Gpu.TensorLogSoftmax(input, -1);
        AssertTensorsClose(cpuResult, gpuResult, 1e-3f);
    }

    /// <summary>
    /// LayerNorm's gamma/beta gradients reduce every row of a column: the CUDA kernel splits the rows over 32 lanes per
    /// column tile. Covers a tall narrow shape (many rows, one column tile), a width that is not a multiple of 32, and a
    /// row count below the lane count.
    /// </summary>
    [SkippableTheory]
    [InlineData(2048, 64)]
    [InlineData(37, 100)]
    [InlineData(5, 33)]
    public void LayerNormBackward_ParamGradients_GpuMatchesCpu(int rows, int features)
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { rows, features }, 300 + features);
        var gamma = RandomTensor(new[] { features }, 301);
        var beta = RandomTensor(new[] { features }, 302);
        var gradOut = RandomTensor(new[] { rows, features }, 303);
        _cpu.LayerNorm(input, gamma, beta, 1e-5, out var cpuMean, out var cpuVar);
        var cpuGradIn = _cpu.LayerNormBackward(gradOut, input, gamma, cpuMean, cpuVar, 1e-5, out var cpuGG, out var cpuGB);
        Gpu.LayerNorm(input, gamma, beta, 1e-5, out var gpuMean, out var gpuVar);
        var gpuGradIn = Gpu.LayerNormBackward(gradOut, input, gamma, gpuMean, gpuVar, 1e-5, out var gpuGG, out var gpuGB);
        // Sums over up to 2048 rows: float accumulation order differs between the CPU loop and the lane split.
        float tol = 1e-4f * rows;
        AssertTensorsClose(cpuGG, gpuGG, tol);
        AssertTensorsClose(cpuGB, gpuGB, tol);
        AssertTensorsClose(cpuGradIn, gpuGradIn, 1e-3f);
    }

    /// <summary>
    /// CUDA permutes are reduced to a canonical form (size-1 axes dropped, adjacent axes merged) and then run as a
    /// copy, a transpose, a batched transpose or the by-value-geometry kernel (float4 when the innermost axis is kept).
    /// Each shape below lands on a different one of those; all must equal the CPU permute exactly.
    /// </summary>
    [SkippableTheory]
    [InlineData(new[] { 4, 32, 4, 16 }, new[] { 0, 2, 1, 3 })]   // attention head split: innermost kept, float4
    [InlineData(new[] { 4, 4, 32, 16 }, new[] { 0, 2, 1, 3 })]   // head merge
    [InlineData(new[] { 3, 5, 7, 6 }, new[] { 0, 2, 1, 3 })]     // innermost kept, not a multiple of 4
    [InlineData(new[] { 2, 3, 4, 5 }, new[] { 3, 1, 0, 2 })]     // innermost moved
    [InlineData(new[] { 2, 3, 4, 5 }, new[] { 0, 1, 3, 2 })]     // merges to a batched transpose
    [InlineData(new[] { 2, 1, 6, 4 }, new[] { 0, 2, 1, 3 })]     // a size-1 axis: a plain copy
    [InlineData(new[] { 5, 1, 3 }, new[] { 2, 1, 0 })]           // a size-1 axis: a 2-D transpose
    [InlineData(new[] { 2, 3, 2, 3, 2 }, new[] { 4, 2, 0, 3, 1 })]
    public void Permute_CanonicalForms_GpuMatchesCpu(int[] shape, int[] permutation)
    {
        SkipIfNoGpu();
        var input = RandomTensor(shape, 700 + shape.Length);
        var expected = _cpu.TensorPermute(input, permutation).Contiguous().GetDataArray();
        // The backend kernel itself: the engine's TensorPermute is a strided view outside a resident training step.
        Skip.IfNot(_fixture.Engine is DirectGpuTensorEngine, "Fixture engine is not a DirectGpuTensorEngine");
        var backend = ((DirectGpuTensorEngine)Gpu).GetBackend();
        using var source = backend.AllocateBuffer(input.GetDataArray());
        using var destination = backend.AllocateBuffer(input.Length);
        backend.Permute(source, destination, shape, permutation);
        var actual = backend.DownloadBuffer(destination);
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(expected[i] == actual[i], $"Element [{i}]: expected {expected[i]}, got {actual[i]}");
    }

    /// <summary>
    /// LayerNorm rows up to 1K wide run one warp per row on CUDA, forward and backward; wider rows keep a block per row.
    /// Rows narrower than a warp, not a multiple of it, the 1024 boundary and the first wider row must all match the CPU:
    /// output, saved mean and variance, and the input gradient.
    /// </summary>
    [SkippableTheory]
    [InlineData(2048, 64)]
    [InlineData(9, 7)]
    [InlineData(13, 1024)]
    [InlineData(3, 1025)]
    public void LayerNorm_ForwardAndBackward_RowShapes_GpuMatchesCpu(int rows, int features)
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { rows, features }, 600 + features);
        var gamma = RandomTensor(new[] { features }, 601);
        var beta = RandomTensor(new[] { features }, 602);
        var gradOut = RandomTensor(new[] { rows, features }, 603);
        var cpuOut = _cpu.LayerNorm(input, gamma, beta, 1e-5, out var cpuMean, out var cpuVar);
        var gpuOut = Gpu.LayerNorm(input, gamma, beta, 1e-5, out var gpuMean, out var gpuVar);
        AssertTensorsClose(cpuOut, gpuOut, 1e-4f);
        AssertTensorsClose(cpuMean, gpuMean, 1e-5f);
        AssertTensorsClose(cpuVar, gpuVar, 1e-4f);
        var cpuGradIn = _cpu.LayerNormBackward(gradOut, input, gamma, cpuMean, cpuVar, 1e-5, out _, out _);
        var gpuGradIn = Gpu.LayerNormBackward(gradOut, input, gamma, gpuMean, gpuVar, 1e-5, out _, out _);
        AssertTensorsClose(cpuGradIn, gpuGradIn, 1e-3f);
    }

    /// <summary>
    /// The deterministic CUDA max-pool backward visits only the windows that contain each input cell. Overlapping
    /// windows (kernel 3, stride 2) route several output gradients to one input cell, and a plain 2x2/2 pool and a
    /// stride-1 pool cover the non-overlapping and densest cases; each must match the CPU scatter exactly.
    /// </summary>
    [SkippableTheory]
    [InlineData(28, 2, 2, true)]
    [InlineData(13, 3, 2, true)]
    [InlineData(9, 3, 1, true)]
    [InlineData(13, 3, 2, false)]
    public void MaxPool2DBackward_WindowGeometries_GpuMatchesCpu(int size, int pool, int stride, bool deterministic)
    {
        SkipIfNoGpu();
        bool prior = AiDotNetEngine.DeterministicMode;
        AiDotNetEngine.SetDeterministicMode(deterministic);
        try
        {
            AssertMaxPoolBackwardMatches(size, pool, stride);
        }
        finally
        {
            AiDotNetEngine.SetDeterministicMode(prior);
        }
    }

    private void AssertMaxPoolBackwardMatches(int size, int pool, int stride)
    {
        var input = RandomTensor(new[] { 2, 3, size, size }, 400 + size);
        var cpuOut = _cpu.MaxPool2DWithIndices(input, new[] { pool, pool }, new[] { stride, stride }, out var cpuIdx);
        var gradOut = RandomTensor(cpuOut.Shape.ToArray(), 401);
        var cpuGrad = _cpu.MaxPool2DBackward(gradOut, cpuIdx, input.Shape.ToArray(), new[] { pool, pool }, new[] { stride, stride });
        Gpu.MaxPool2DWithIndices(input, new[] { pool, pool }, new[] { stride, stride }, out var gpuIdx);
        var gpuGrad = Gpu.MaxPool2DBackward(gradOut, gpuIdx, input.Shape.ToArray(), new[] { pool, pool }, new[] { stride, stride });
        AssertTensorsClose(cpuGrad, gpuGrad, 1e-5f);
    }

    /// <summary>
    /// Softmax rows up to 1K wide run one warp per row on CUDA (wider rows a block per row), and the softmax backward
    /// always runs one warp per row. Short rows, rows not a multiple of 32, a row count that leaves the last block
    /// partly empty, the 1024 boundary and a vocabulary-sized row must all match the CPU, forward and backward.
    /// </summary>
    [SkippableTheory]
    [InlineData(8192, 32)]
    [InlineData(9, 7)]
    [InlineData(13, 1024)]
    [InlineData(3, 1025)]
    [InlineData(2, 50257)]
    public void Softmax_ForwardAndBackward_RowShapes_GpuMatchesCpu(int rows, int features)
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { rows, features }, 500 + features);
        var gradOut = RandomTensor(new[] { rows, features }, 501);
        var cpuOut = _cpu.Softmax(input, -1);
        var gpuOut = Gpu.Softmax(input, -1);
        AssertTensorsClose(cpuOut, gpuOut, 1e-5f);
        var cpuGrad = _cpu.SoftmaxBackward(gradOut, cpuOut, -1);
        var gpuGrad = Gpu.SoftmaxBackward(gradOut, cpuOut, -1);
        AssertTensorsClose(cpuGrad, gpuGrad, 1e-4f);
    }

    /// <summary>
    /// The CUDA log-softmax kernel reduces each row with one warp: rows shorter than a warp, not a multiple of it,
    /// far longer than it (an LM vocabulary), and a row count that leaves the last block partly empty must all match
    /// the CPU, including a row whose entries span ~100 in log space (where log(softmax) underflows).
    /// </summary>
    [SkippableTheory]
    [InlineData(1, 1)]
    [InlineData(3, 7)]
    [InlineData(1001, 10)]
    [InlineData(9, 33)]
    [InlineData(5, 50257)]
    public void TensorLogSoftmax_RowShapes_GpuMatchesCpu(int rows, int features)
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { rows, features }, 1000 + features);
        var span = input.AsWritableSpan();
        for (int i = 0; i < features; i++) span[i] *= 60f;   // row 0 spans ~120: exp underflows for most of it
        var cpuResult = _cpu.TensorLogSoftmax(input, -1);
        var gpuResult = Gpu.TensorLogSoftmax(input, -1);
        AssertTensorsClose(cpuResult, gpuResult, 1e-3f);
    }

    [SkippableFact]
    public void TensorAdd_BroadcastShapes_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var a = RandomTensor(new[] { 8, 16 }, 112);
        var b = RandomTensor(new[] { 1, 16 }, 113);
        var cpuResult = _cpu.TensorAdd(a, b);
        var gpuResult = Gpu.TensorAdd(a, b);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [SkippableFact]
    public void TensorCumSum_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 4, 8 }, 114);
        var cpuResult = _cpu.TensorCumSum(input, 1);
        var gpuResult = Gpu.TensorCumSum(input, 1);
        AssertTensorsClose(cpuResult, gpuResult, 1e-3f);
    }

    [SkippableFact]
    public void TensorRandomUniform_HasCorrectShape()
    {
        SkipIfNoGpu();
        var result = Gpu.TensorRandomUniform<float>(new[] { 4, 8 });
        Assert.Equal(new[] { 4, 8 }, result.Shape.ToArray());
        Assert.Equal(32, result.Length);
        // Values should be in [0, 1)
        var data = result.GetDataArray();
        foreach (var v in data)
        {
            Assert.True(v >= 0f && v < 1f, $"Value {v} out of range [0, 1)");
        }
    }

    [SkippableFact]
    public void ReduceSum_SingleAxis_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 4, 8, 16 }, 116);
        var cpuResult = _cpu.ReduceSum(input, new[] { 2 }, false);
        var gpuResult = Gpu.ReduceSum(input, new[] { 2 }, false);
        AssertTensorsClose(cpuResult, gpuResult, 1e-2f);
    }

    [SkippableFact]
    public void Upsample_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 1, 3, 4, 4 }, 117);
        var cpuResult = _cpu.Upsample(input, 2, 2);
        var gpuResult = Gpu.Upsample(input, 2, 2);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [SkippableFact]
    public void PixelShuffle_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var input = RandomTensor(new[] { 1, 12, 4, 4 }, 118); // 12 = 3 * 2^2
        var cpuResult = _cpu.PixelShuffle(input, 2);
        var gpuResult = Gpu.PixelShuffle(input, 2);
        AssertTensorsClose(cpuResult, gpuResult);
    }

    [SkippableFact]
    public void TensorBatchMatMul_GpuMatchesCpu()
    {
        SkipIfNoGpu();
        var a = RandomTensor(new[] { 2, 4, 8 }, 119);
        var b = RandomTensor(new[] { 2, 8, 6 }, 120);
        var cpuResult = _cpu.TensorBatchMatMul(a, b);
        var gpuResult = Gpu.TensorBatchMatMul(a, b);
        AssertTensorsClose(cpuResult, gpuResult, 1e-2f);
    }

    #endregion
}

/// <summary>
/// Exception to skip tests when GPU is not available.
// SkipException removed — using Skip.If from Xunit.SkippableFact instead
