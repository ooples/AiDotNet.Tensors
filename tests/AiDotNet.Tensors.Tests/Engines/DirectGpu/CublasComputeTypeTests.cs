// Copyright (c) CrowdTrainer LLC. All rights reserved.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// <see cref="CublasComputeType"/> carries cublas_api.h's values (Float32Fast16F and Float32FastBF16 were swapped), and
/// matmul autocast really multiplies in the precision it names: BF16 (8-bit significand) rounds the same product
/// several times coarser than FP16 (11-bit), which a swap of the two compute types would invert.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class CublasComputeTypeTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public CublasComputeTypeTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    [Fact]
    public void ValuesMatchCublasApiHeader()
    {
        // cublasComputeType_t: CUBLAS_COMPUTE_32F = 68, _32F_PEDANTIC = 69, _32F_FAST_16F = 74, _32F_FAST_16BF = 75,
        // _32F_FAST_TF32 = 77.
        Assert.Equal(68, (int)CublasComputeType.Float32);
        Assert.Equal(69, (int)CublasComputeType.Float32Pedantic);
        Assert.Equal(74, (int)CublasComputeType.Float32Fast16F);
        Assert.Equal(75, (int)CublasComputeType.Float32FastBF16);
        Assert.Equal(77, (int)CublasComputeType.Float32FastTf32);
        Assert.Equal(CuBlasNative.CUBLAS_COMPUTE_32F_FAST_16F, (int)CublasComputeType.Float32Fast16F);
        Assert.Equal(CuBlasNative.CUBLAS_COMPUTE_32F_FAST_16BF, (int)CublasComputeType.Float32FastBF16);
    }

    private static float[] Data(int n, double a, double b)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float)Math.Sin(a * i + b);
        return v;
    }

    [SkippableTheory]
    // Shapes large enough for tensor cores: the FAST_16F/_16BF compute types permit the reduced precision rather than
    // require it, and cuBLAS keeps FP32 kernels where it prefers them (64 x 256 x 64 came back FP32-exact for both, and
    // 512 x 1024 x 512 for BF16 on GA102). These shapes take tensor cores for both on GA102.
    [InlineData(672, 256, 12288)]   // a slice of the language-model head's forward
    [InlineData(2048, 256, 2048)]
    public void Bf16Autocast_RoundsCoarserThanFp16(int m, int k, int n)
    {
        Skip.If(!_fixture.IsAvailable, "No GPU device.");
        var engine = _fixture.Engine!;
        Skip.If(engine.GetBackend() is not AiDotNet.Tensors.Engines.DirectGpu.IGemmAutocastBackend,
            "The backend has no matmul autocast.");
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var a = new Tensor<float>(new[] { m, k }, new Vector<float>(Data(m * k, 0.731, 0.2))).Gpu();
            var b = new Tensor<float>(new[] { k, n }, new Vector<float>(Data(k * n, 0.293, 0.5))).Gpu();
            var exact = engine.TensorMatMul(a, b).ToArray();
            double Error(PrecisionMode precision)
            {
                float[] mixed;
                using (engine.EnterMatMulAutocast(precision)) mixed = engine.TensorMatMul(a, b).ToArray();
                double sum = 0;
                for (int i = 0; i < exact.Length; i++) sum += Math.Abs(exact[i] - mixed[i]);
                return sum / exact.Length;
            }
            double fp16 = Error(PrecisionMode.Float16), bf16 = Error(PrecisionMode.BFloat16);
            Assert.True(fp16 > 0, "FP16 autocast was bit-identical to FP32");
            // Unit roundoff 2^-8 against 2^-11: the mean error ratio is ~8; require at least 3.
            Assert.True(bf16 > 3 * fp16, $"BF16 mean error {bf16:E2} is not coarser than FP16's {fp16:E2}");
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }
}
