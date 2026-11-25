// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
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
        foreach (var mode in new[] { MatMulAutocastMode.CastOperands, MatMulAutocastMode.FastComputeType })
            Bf16RoundsCoarserThanFp16(m, k, n, mode);
    }

    private void Bf16RoundsCoarserThanFp16(int m, int k, int n, MatMulAutocastMode mode)
    {
        Skip.If(!_fixture.IsAvailable, "No GPU device.");
        var engine = TestRequire.NotNull(_fixture.Engine, "the fixture's engine");
        Skip.If(engine.GetBackend() is not AiDotNet.Tensors.Engines.DirectGpu.IGemmAutocastBackend,
            "The backend has no matmul autocast.");
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var hostA = Data(m * k, 0.731, 0.2);
            var hostB = Data(k * n, 0.293, 0.5);
            var a = new Tensor<float>(new[] { m, k }, new Vector<float>(hostA)).Gpu();
            var b = new Tensor<float>(new[] { k, n }, new Vector<float>(hostB)).Gpu();
            // The reference is the host product in double: a device FP32 GEMM may itself run TF32 on Ampere.
            var exact = new double[m * n];
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double acc = 0;
                    for (int t = 0; t < k; t++) acc += (double)hostA[i * k + t] * hostB[t * n + j];
                    exact[i * n + j] = acc;
                }
            double Error(PrecisionMode precision)
            {
                float[] mixed;
                using (engine.EnterMatMulAutocast(precision, mode)) mixed = engine.TensorMatMul(a, b).ToArray();
                double sum = 0;
                for (int i = 0; i < exact.Length; i++) sum += Math.Abs(exact[i] - mixed[i]);
                return sum / exact.Length;
            }
            double fp16 = Error(PrecisionMode.Float16), bf16 = Error(PrecisionMode.BFloat16);
            Assert.True(fp16 > 0, $"{mode}: FP16 autocast was bit-identical to FP32");
            // Unit roundoff 2^-8 against 2^-11: the mean error ratio is ~8; require at least 3.
            Assert.True(bf16 > 3 * fp16, $"{mode}: BF16 mean error {bf16:E2} is not coarser than FP16's {fp16:E2}");
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    private static int Bits(float f) => BitConverter.ToInt32(BitConverter.GetBytes(f), 0);

    // f rounded to FP16 (round to nearest even; |f| within FP16's range), as __float2half_rn does.
    private static float RoundToHalf(float f)
    {
        if (f == 0) return f;
        int exponent = ((Bits(f) >> 23) & 0xFF) - 127;
        double quantum = Math.Pow(2, Math.Max(exponent, -14) - 10);   // below 2^-14, FP16's subnormal step 2^-24
        return (float)(Math.Round(f / quantum, MidpointRounding.ToEven) * quantum);
    }

    // f rounded to BF16 (round to nearest even): the top 16 bits of the float.
    private static float RoundToBFloat16(float f)
    {
        int bits = Bits(f);
        bits += 0x7FFF + ((bits >> 16) & 1);
        return BitConverter.ToSingle(BitConverter.GetBytes(bits & unchecked((int)0xFFFF0000)), 0);
    }

    // The default autocast (CastOperands) is PyTorch's: every product is that of the operands rounded to 16 bits, on a
    // small GEMM as on a large one. The fast compute type only permits 16-bit math, and cuBLAS kept FP32 kernels for
    // small shapes (a 16-32-8 MLP's GEMMs came back FP32-exact on GA102), so a model's precision depended on its batch.
    [SkippableTheory]
    [InlineData(16, 16, 32, 1)]
    [InlineData(32, 32, 8, 1)]
    [InlineData(64, 256, 64, 1)]
    [InlineData(672, 256, 1024, 1)]
    [InlineData(16, 32, 32, 3)]     // a strided batch
    public void CastOperands_RoundsTheOperandsTo16Bits_AtEveryShape(int m, int k, int n, int batch)
    {
        Skip.If(!_fixture.IsAvailable, "No GPU device.");
        var engine = TestRequire.NotNull(_fixture.Engine, "the fixture's engine");
        Skip.If(engine.GetBackend() is not CudaBackend, "Matmul autocast is CUDA's.");
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var hostA = Data(batch * m * k, 0.731, 0.2);
            var hostB = Data(batch * k * n, 0.293, 0.5);
            int[] aShape = batch == 1 ? new[] { m, k } : new[] { batch, m, k };
            int[] bShape = batch == 1 ? new[] { k, n } : new[] { batch, k, n };
            var a = new Tensor<float>(aShape, new Vector<float>(hostA)).Gpu();
            var b = new Tensor<float>(bShape, new Vector<float>(hostB)).Gpu();
            foreach (var precision in new[] { PrecisionMode.Float16, PrecisionMode.BFloat16 })
            {
                Func<float, float> round = precision == PrecisionMode.Float16 ? RoundToHalf : RoundToBFloat16;
                float[] mixed;
                var cuda = (CudaBackend)TestRequire.NotNull(engine.GetBackend(), "the engine's backend");
                long declines = cuda.AutocastCastDeclinesForTest;
                using (engine.EnterMatMulAutocast(precision)) mixed = engine.TensorMatMul(a, b).ToArray();
                Assert.Equal(declines, cuda.AutocastCastDeclinesForTest);   // cast, not the fast fallback
                double roundedErr = 0, exactErr = 0;
                for (int z = 0; z < batch; z++)
                    for (int i = 0; i < m; i++)
                        for (int j = 0; j < n; j++)
                        {
                            double rounded = 0, exact = 0, magnitude = 0;
                            for (int t = 0; t < k; t++)
                            {
                                float x = hostA[(z * m + i) * k + t], y = hostB[(z * k + t) * n + j];
                                double product = (double)round(x) * round(y);
                                rounded += product;
                                magnitude += Math.Abs(product);
                                exact += (double)x * y;
                            }
                            double got = mixed[(z * m + i) * n + j];
                            // FP32 accumulation of the exact 16-bit products: within k float roundings of the magnitude.
                            double bound = k * Math.Pow(2, -23) * magnitude + 1e-30;
                            Assert.True(Math.Abs(got - rounded) <= bound,
                                $"{precision} [{z},{i},{j}]: {got:R} is {Math.Abs(got - rounded):E2} from the rounded-operand product {rounded:R} (bound {bound:E2})");
                            roundedErr += Math.Abs(got - rounded);
                            exactErr += Math.Abs(got - exact);
                        }
                // The operands' rounding shows: the result is far closer to the rounded-operand product than to the
                // exact one (an FP32 or TF32 GEMM is the other way round).
                Assert.True(exactErr > 10 * roundedErr,
                    $"{precision} {m}x{k}x{n}: mean distance to the exact product {exactErr / mixed.Length:E2} is not well above the distance to the rounded-operand product {roundedErr / mixed.Length:E2}");
            }
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    // The fast compute type keeps its behaviour behind the option: FP32 operands, no cast, the permissive compute type.
    [SkippableFact]
    public void FastComputeType_keeps_FP32_operands_and_the_permissive_compute_type()
    {
        Skip.If(!_fixture.IsAvailable, "No GPU device.");
        Skip.IfNot(TestRequire.NotNull(_fixture.Engine, "the fixture's engine").GetBackend() is CudaBackend, "CUDA only.");
        var cuda = (CudaBackend)TestRequire.NotNull(TestRequire.NotNull(_fixture.Engine, "the fixture's engine").GetBackend(), "the engine's backend");
        using (TestRequire.NotNull(_fixture.Engine, "the fixture's engine").EnterMatMulAutocast(PrecisionMode.Float16, MatMulAutocastMode.FastComputeType))
        {
            Assert.Equal((int)CublasComputeType.Float32Fast16F, cuda.GemmAutocastComputeTypeForTest);
            Assert.Equal(0, cuda.GemmAutocastCastTypeForTest);
        }
        using (TestRequire.NotNull(_fixture.Engine, "the fixture's engine").EnterMatMulAutocast(PrecisionMode.BFloat16))
        {
            Assert.Equal((int)CublasComputeType.Float32FastBF16, cuda.GemmAutocastComputeTypeForTest);
            Assert.Equal(CuBlasNative.CUDA_R_16BF, cuda.GemmAutocastCastTypeForTest);
        }
        Assert.Equal(0, cuda.GemmAutocastCastTypeForTest);
    }

    // Scopes that close out of order (two threads' scopes overlapping) leave the newest open one in force and FP32 once
    // all have closed: restoring each scope's predecessor left BF16 on for good after A(FP16) B(BF16) closed A then B.
    [SkippableFact]
    public void Autocast_scopes_closing_out_of_order_return_to_fp32()
    {
        Skip.If(!_fixture.IsAvailable, "No GPU device.");
        Skip.IfNot(TestRequire.NotNull(_fixture.Engine, "the fixture's engine").GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend, "CUDA only.");
        var cuda = (AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend)TestRequire.NotNull(TestRequire.NotNull(_fixture.Engine, "the fixture's engine").GetBackend(), "the engine's backend");
        var autocast = (AiDotNet.Tensors.Engines.DirectGpu.IGemmAutocastBackend)cuda;
        Assert.Equal(0, cuda.GemmAutocastComputeTypeForTest);
        var a = autocast.EnterGemmAutocast(PrecisionMode.Float16);
        var b = autocast.EnterGemmAutocast(PrecisionMode.BFloat16);
        try
        {
            Assert.Equal((int)CublasComputeType.Float32FastBF16, cuda.GemmAutocastComputeTypeForTest);
            a.Dispose();
            Assert.Equal((int)CublasComputeType.Float32FastBF16, cuda.GemmAutocastComputeTypeForTest);
            b.Dispose();
            Assert.Equal(0, cuda.GemmAutocastComputeTypeForTest);
            b.Dispose();   // twice is a no-op
            Assert.Equal(0, cuda.GemmAutocastComputeTypeForTest);
        }
        finally
        {
            // A failed assertion must not leave the shared fixture's backend in autocast.
            a.Dispose();
            b.Dispose();
        }
    }

    // Every autocast route (cuBLASLt, the classic GemmEx, the fixed-order split-K) computes
    // D = alpha * op(A) op(B) + beta * D within the reduced-precision rounding bound, and the split-K is bit-reproducible.
    [SkippableTheory]
    [InlineData(0, 0, true, false)]
    [InlineData(1, 0, true, false)]
    [InlineData(2, 8, true, false)]
    [InlineData(2, 16, false, true)]
    [InlineData(2, 8, false, false)]
    public void AutocastRoutes_ComputeTheGemm(int route, int splits, bool transA, bool transB)
    {
        Skip.If(!_fixture.IsAvailable, "No GPU device.");
        if (TestRequire.NotNull(_fixture.Engine, "the fixture's engine").GetBackend() is not CudaBackend backend) { Skip.If(true, "Not the CUDA backend."); return; }
        // .NET Framework has no resolver for cuBLASLt, and a CUDA install may lack it: that route is then not offered.
        Skip.If(route == 0 && !CuBlasLtMatmul.IsAvailable, "cuBLASLt is not loadable here.");
        const int m = 96, n = 80, k = 4096;
        const float alpha = 0.5f, beta = 2f;
        int lda = transA ? k : m, ldb = transB ? n : k, ldd = m;
        var ha = Data(lda * (transA ? m : k), 0.731, 0.2);
        var hb = Data(ldb * (transB ? k : n), 0.293, 0.5);
        var hd = Data(ldd * n, 0.111, 0.9);
        float A(int i, int kk) => transA ? ha[i * lda + kk] : ha[kk * lda + i];   // column-major op(A)[i, kk]
        float B(int kk, int j) => transB ? hb[kk * ldb + j] : hb[j * ldb + kk];
        using var da = backend.AllocateBuffer(ha);
        using var db = backend.AllocateBuffer(hb);
        foreach (var precision in new[] { PrecisionMode.Float16, PrecisionMode.BFloat16 })
        {
            double u = precision == PrecisionMode.Float16 ? Math.Pow(2, -11) : Math.Pow(2, -8);
            double factor = 2 * u + u * u + 2.0 * k * Math.Pow(2, -24);
            float[]? first = null;
            for (int run = 0; run < 2; run++)
            {
                using var dd = backend.AllocateBuffer(hd);
                Assert.True(backend.RunAutocastGemmRouteForTest(route, splits, transA, transB, m, n, k, da, lda, db, ldb,
                    dd, ldd, alpha, beta, precision), $"route {route} declined");
                var got = backend.DownloadBuffer(dd);
                for (int j = 0; j < n; j++)
                    for (int i = 0; i < m; i++)
                    {
                        double exact = 0, abs = 0;
                        for (int kk = 0; kk < k; kk++) { double t = (double)A(i, kk) * B(kk, j); exact += t; abs += Math.Abs(t); }
                        double want = alpha * exact + beta * hd[j * ldd + i];
                        double bound = Math.Abs(alpha) * factor * abs + 4 * Math.Pow(2, -24) * Math.Abs(want) + 1e-6;
                        Assert.True(Math.Abs(got[j * ldd + i] - want) <= bound,
                            $"{precision} route {route}: D[{i},{j}] = {got[j * ldd + i]}, want {want} (bound {bound:E2})");
                    }
                if (first is null) first = got;
                else if (route == 2) Assert.Equal(first, got);   // the fixed-order reduction is bit-reproducible
            }
        }
    }

    // The cast-operands routes (the classic GemmEx and the fixed-order split-K on 16-bit inputs) compute
    // D = alpha * op(rA) op(rB) + beta * D, rA and rB the operands rounded to 16 bits, for transposed operands and leading
    // dimensions past the rows (the backward GEMMs' layouts); the split-K stays bit-reproducible.
    [SkippableTheory]
    [InlineData(1, 0, false, false, 0)]
    [InlineData(1, 0, true, false, 3)]
    [InlineData(1, 0, false, true, 5)]
    [InlineData(2, 8, true, false, 0)]
    [InlineData(2, 16, false, true, 2)]
    public void CastOperandRoutes_ComputeTheRoundedProduct(int route, int splits, bool transA, bool transB, int pad)
    {
        Skip.If(!_fixture.IsAvailable, "No GPU device.");
        if (TestRequire.NotNull(_fixture.Engine, "the fixture's engine").GetBackend() is not CudaBackend backend) { Skip.If(true, "Not the CUDA backend."); return; }
        const int m = 64, n = 48, k = 2048;
        const float alpha = 0.5f, beta = 2f;
        int lda = (transA ? k : m) + pad, ldb = (transB ? n : k) + pad, ldd = m;
        var ha = Data(lda * (transA ? m : k), 0.731, 0.2);
        var hb = Data(ldb * (transB ? k : n), 0.293, 0.5);
        var hd = Data(ldd * n, 0.111, 0.9);
        using var da = backend.AllocateBuffer(ha);
        using var db = backend.AllocateBuffer(hb);
        foreach (var precision in new[] { PrecisionMode.Float16, PrecisionMode.BFloat16 })
        {
            if (precision == PrecisionMode.BFloat16 && backend.ComputeCapabilityMajorForTest < 8) continue;   // no BF16 GEMM
            Func<float, float> round = precision == PrecisionMode.Float16 ? RoundToHalf : RoundToBFloat16;
            float A(int i, int kk) => round(transA ? ha[i * lda + kk] : ha[kk * lda + i]);
            float B(int kk, int j) => round(transB ? hb[kk * ldb + j] : hb[j * ldb + kk]);
            float[]? first = null;
            for (int run = 0; run < 2; run++)
            {
                using var dd = backend.AllocateBuffer(hd);
                Assert.True(backend.RunAutocastGemmRouteForTest(route, splits, transA, transB, m, n, k, da, lda, db, ldb,
                    dd, ldd, alpha, beta, precision, castOperands: true), $"cast route {route} declined");
                var got = backend.DownloadBuffer(dd);
                for (int j = 0; j < n; j++)
                    for (int i = 0; i < m; i++)
                    {
                        double exact = 0, abs = 0;
                        for (int kk = 0; kk < k; kk++) { double t = (double)A(i, kk) * B(kk, j); exact += t; abs += Math.Abs(t); }
                        double want = alpha * exact + beta * hd[j * ldd + i];
                        // FP32 accumulation of exact 16-bit products, then the alpha/beta epilogue.
                        double bound = Math.Abs(alpha) * k * Math.Pow(2, -23) * abs + 4 * Math.Pow(2, -24) * Math.Abs(want) + 1e-7;
                        Assert.True(Math.Abs(got[j * ldd + i] - want) <= bound,
                            $"{precision} cast route {route}: D[{i},{j}] = {got[j * ldd + i]}, want {want} (bound {bound:E2})");
                    }
                if (first is null) first = got;
                else if (route == 2) Assert.Equal(first, got);
            }
        }
    }
}
