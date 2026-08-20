using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.NumericOperations;
using AiDotNet.Tensors.Tests.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

public sealed class GpuPrecisionExecutionTests
{
    [Fact]
    public void SpeedFirst_DoubleMatMulExecutesThroughFp32AndReturnsDouble()
    {
        using var fixture = new Fixture();
        var a = new Tensor<double>(new[] { 1d, 2d, 3d, 4d }, new[] { 2, 2 });
        var b = new Tensor<double>(new[] { 5d, 6d, 7d, 8d }, new[] { 2, 2 });

        var result = fixture.Engine.TensorMatMul(a, b);

        Assert.Equal(new[] { 19d, 22d, 43d, 50d }, result.GetDataArray());
        Assert.Equal(1, fixture.Backend.Fp32GemmCalls);
        Assert.Equal(0, fixture.Backend.Fp16GemmCalls);
        Assert.Equal(typeof(double), GpuPrecisionDiagnostics.LastPlan!.PublicType);
        Assert.Equal(GpuScalarType.Float32, GpuPrecisionDiagnostics.LastPlan.MultiplyType);
    }

    [Fact]
    public void SpeedFirst_IntegerMatMulConvertsBackToTheDeclaredType()
    {
        using var fixture = new Fixture();
        var a = new Tensor<int>(new[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        var b = new Tensor<int>(new[] { 5, 6, 7, 8 }, new[] { 2, 2 });

        var result = fixture.Engine.TensorMatMul(a, b);

        Assert.Equal(new[] { 19, 22, 43, 50 }, result.GetDataArray());
        Assert.Equal(1, fixture.Backend.Fp32GemmCalls);
        Assert.Equal(typeof(int), GpuPrecisionDiagnostics.LastPlan!.PublicType);
    }

    [Fact]
    public void PreserveInputType_DoubleMatMulUsesExactCpuRoute()
    {
        using var fixture = new Fixture();
        var a = new Tensor<double>(new[] { 0.1d, 0.2d, 0.3d, 0.4d }, new[] { 2, 2 });
        var b = new Tensor<double>(new[] { 0.5d, 0.6d, 0.7d, 0.8d }, new[] { 2, 2 });
        using var policy = new GpuExecutionPolicyScope(GpuExecutionPolicy.Preserve);

        var result = fixture.Engine.TensorMatMul(a, b);

        Assert.Equal(0, fixture.Backend.Fp32GemmCalls);
        Assert.Equal(0, fixture.Backend.Fp16GemmCalls);
        Assert.Equal(GpuExecutionRoute.Cpu, GpuPrecisionDiagnostics.LastPlan!.Route);
        Assert.Equal(0.19d, result.GetDataArray()[0], 14);
    }

    [Fact]
    public void Fp16Autocast_MatMulUsesTypedHalfInputsAndFp32Accumulation()
    {
        using var fixture = new Fixture();
        var a = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 });
        var b = new Tensor<float>(new[] { 5f, 6f, 7f, 8f }, new[] { 2, 2 });
        using var autocast = new AutocastScope(PrecisionMode.Float16);

        var result = fixture.Engine.TensorMatMul(a, b);

        Assert.Equal(new[] { 19f, 22f, 43f, 50f }, result.GetDataArray());
        Assert.Equal(0, fixture.Backend.Fp32GemmCalls);
        Assert.Equal(1, fixture.Backend.Fp16GemmCalls);
        Assert.True(fixture.Backend.Fp16Conversions >= 2);
        Assert.Equal(GpuScalarType.Float16, GpuPrecisionDiagnostics.LastPlan!.InputStorage);
        Assert.Equal(GpuScalarType.Float32, GpuPrecisionDiagnostics.LastPlan.AccumulatorType);
        Assert.Equal(GpuScalarType.Float32, GpuPrecisionDiagnostics.LastPlan.OutputStorage);
    }

    [Fact]
    public void Fp16Autocast_DoubleReluUsesTypedKernelAndReturnsDouble()
    {
        using var fixture = new Fixture();
        var input = new Tensor<double>(new[] { -2d, -0d, 1.5d, 9d }, new[] { 4 });
        using var autocast = new AutocastScope(PrecisionMode.Float16);

        var result = fixture.Engine.TensorReLU(input);

        Assert.Equal(new[] { 0d, 0d, 1.5d, 9d }, result.GetDataArray());
        Assert.Equal(1, fixture.Backend.Fp16ReluCalls);
        Assert.Equal(0, fixture.Backend.Fp32ReluCalls);
        Assert.Equal(GpuScalarType.Float16, GpuPrecisionDiagnostics.LastPlan!.InputStorage);
        Assert.Equal(GpuScalarType.Float16, GpuPrecisionDiagnostics.LastPlan.OutputStorage);
    }

    [Fact]
    public void PreserveInputType_HalfReluUsesExactTypedGpuRoute()
    {
        using var fixture = new Fixture();
        var input = new Tensor<Half>(new Half[] { (Half)(-2), (Half)1.5 }, new[] { 2 });
        using var policy = new GpuExecutionPolicyScope(GpuExecutionPolicy.Preserve);

        var result = fixture.Engine.TensorReLU(input);

        Assert.Equal(new Half[] { (Half)0, (Half)1.5 }, result.GetDataArray());
        Assert.Equal(1, fixture.Backend.Fp16ReluCalls);
        Assert.Equal(0, fixture.Backend.Fp32ReluCalls);
        Assert.Equal(GpuExecutionRoute.Gpu, GpuPrecisionDiagnostics.LastPlan!.Route);
        Assert.Equal(GpuScalarType.Float16, GpuPrecisionDiagnostics.LastPlan.ComputeFormat);
    }

    [Fact]
    public void Fp16Autocast_AddUsesTypedKernelInsteadOfFp32Delegate()
    {
        using var fixture = new Fixture();
        var left = new Tensor<float>(new[] { 1f, 2f, 3f }, new[] { 3 });
        var right = new Tensor<float>(new[] { 4f, 5f, 6f }, new[] { 3 });
        using var autocast = new AutocastScope(PrecisionMode.Float16);

        var result = fixture.Engine.TensorAdd(left, right);

        Assert.Equal(new[] { 5f, 7f, 9f }, result.GetDataArray());
        Assert.Equal(1, fixture.Backend.Fp16AddCalls);
        Assert.Equal(0, fixture.Backend.Fp32AddCalls);
    }

    [Fact]
    public void Rank4MatMulCollapsesEveryLeadingDimensionIntoTheBatchCount()
    {
        using var fixture = new Fixture();
        var leftValues = Enumerable.Range(1, 24).Select(value => (double)value).ToArray();
        var rightValues = new double[24];
        for (int batch = 0; batch < 6; batch++)
        {
            rightValues[batch * 4] = 1;
            rightValues[batch * 4 + 3] = 1;
        }
        var left = new Tensor<double>(leftValues, new[] { 2, 3, 2, 2 });
        var right = new Tensor<double>(rightValues, new[] { 2, 3, 2, 2 });

        var result = fixture.Engine.TensorMatMul(left, right);

        Assert.Equal(leftValues, result.GetDataArray());
        Assert.Equal(1, fixture.Backend.BatchedGemmCalls);
        Assert.Equal(6, fixture.Backend.LastBatchCount);
        Assert.Equal(new[] { 2, 3, 2, 2 }, result.Shape._dims);
    }

    [Fact]
    public void SpeedFirst_LongAndDecimalMatMulReturnTheDeclaredTypes()
    {
        using var fixture = new Fixture();
        var longLeft = new Tensor<long>(new long[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        var longIdentity = new Tensor<long>(new long[] { 1, 0, 0, 1 }, new[] { 2, 2 });
        Assert.Equal(longLeft.GetDataArray(), fixture.Engine.TensorMatMul(longLeft, longIdentity).GetDataArray());

        var decimalLeft = new Tensor<decimal>(new decimal[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        var decimalIdentity = new Tensor<decimal>(new decimal[] { 1, 0, 0, 1 }, new[] { 2, 2 });
        Assert.Equal(decimalLeft.GetDataArray(), fixture.Engine.TensorMatMul(decimalLeft, decimalIdentity).GetDataArray());
    }

    [Fact]
    public void SpeedFirst_AllReducedPublicTypesExecuteAndConvertBack()
    {
        using var fixture = new Fixture();

        AssertReducedRoundTrip(fixture.Engine, new Half[] { (Half)1, (Half)2, (Half)3, (Half)4 });
        AssertReducedRoundTrip(fixture.Engine, new[]
        {
            BFloat16.FromFloat(1), BFloat16.FromFloat(2),
            BFloat16.FromFloat(3), BFloat16.FromFloat(4),
        });
        AssertReducedRoundTrip(fixture.Engine, new[]
        {
            Float8E4M3.FromFloat(1), Float8E4M3.FromFloat(2),
            Float8E4M3.FromFloat(3), Float8E4M3.FromFloat(4),
        });
        AssertReducedRoundTrip(fixture.Engine, new[]
        {
            Float8E5M2.FromFloat(1), Float8E5M2.FromFloat(2),
            Float8E5M2.FromFloat(3), Float8E5M2.FromFloat(4),
        });
    }

    private static void AssertReducedRoundTrip<T>(DirectGpuTensorEngine engine, T[] values)
    {
        var left = new Tensor<T>(values, new[] { 2, 2 });
        var operations = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var identityValues = new[]
        {
            operations.One, operations.Zero,
            operations.Zero, operations.One,
        };
        var identity = new Tensor<T>(identityValues, new[] { 2, 2 });

        var result = engine.TensorMatMul(left, identity).GetDataArray();

        Assert.Equal(values, result);
        Assert.Equal(typeof(T), GpuPrecisionDiagnostics.LastPlan!.PublicType);
        Assert.Equal(GpuExecutionRoute.Gpu, GpuPrecisionDiagnostics.LastPlan.Route);
    }

    private sealed class Fixture : IDisposable
    {
        private readonly DirectGpuEngine _direct;

        internal Fixture()
        {
            Backend = new ExecutablePrecisionBackend(
                MockDirectGpuBackend.Create(new MockBackendState()));
            _direct = new DirectGpuEngine(Backend);
            Engine = new DirectGpuTensorEngine(_direct);
            GpuPrecisionDiagnostics.Clear();
        }

        internal ExecutablePrecisionBackend Backend { get; }
        internal DirectGpuTensorEngine Engine { get; }

        public void Dispose()
        {
            Engine.Dispose();
            _direct.Dispose();
            GpuPrecisionDiagnostics.Clear();
        }
    }

    private sealed class ExecutablePrecisionBackend : DelegatingGpuBackend,
        IGpuPrecisionBackend,
        IGpuHalfPrecisionBackend,
        IGpuFp16ElementwiseBackend
    {
        internal ExecutablePrecisionBackend(IDirectGpuBackend inner) : base(inner) { }

        internal int Fp32GemmCalls { get; private set; }
        internal int Fp16GemmCalls { get; private set; }
        internal int Fp16Conversions { get; private set; }
        internal int Fp16ReluCalls { get; private set; }
        internal int Fp32ReluCalls { get; private set; }
        internal int Fp16AddCalls { get; private set; }
        internal int Fp32AddCalls { get; private set; }
        internal int BatchedGemmCalls { get; private set; }
        internal int LastBatchCount { get; private set; }

        public IReadOnlyList<GpuPrecisionCapability> GetPrecisionCapabilities(
            GpuPrecisionOperation operation)
        {
            var fp32 = new GpuPrecisionCapability(
                GpuScalarType.Float32, GpuScalarType.Float32, GpuScalarType.Float32,
                GpuScalarType.Float32, GpuPrecisionImplementation.Emulated, false);
            if (operation is not (GpuPrecisionOperation.MatMul
                or GpuPrecisionOperation.Add
                or GpuPrecisionOperation.Relu
                or GpuPrecisionOperation.Gelu))
                return new[] { fp32 };

            var outputStorage = operation == GpuPrecisionOperation.MatMul
                ? GpuScalarType.Float32
                : GpuScalarType.Float16;
            return new[]
            {
                fp32,
                new GpuPrecisionCapability(
                    GpuScalarType.Float16, GpuScalarType.Float16, GpuScalarType.Float32,
                    outputStorage, GpuPrecisionImplementation.Emulated, false),
            };
        }

        public override IGpuBuffer AllocateBuffer(float[] data)
            => new MockGpuBuffer((float[])data.Clone());

        public override IGpuBuffer AllocateBuffer(int size)
            => new MockGpuBuffer(new float[size]);

        public override float[] DownloadBuffer(IGpuBuffer buffer)
            => (float[])Buffer(buffer).Data.Clone();

        public override void DownloadBuffer(IGpuBuffer buffer, float[] destination)
            => Array.Copy(Buffer(buffer).Data, destination, destination.Length);

        public override void ConvertToFp16(IGpuBuffer input, IGpuBuffer output, int size)
        {
            Fp16Conversions++;
            var source = Buffer(input).Data;
            var destination = Buffer(output).Data;
            for (int i = 0; i < size; i++)
                destination[i] = (float)(Half)source[i];
        }

        public override void ConvertToFp32(IGpuBuffer input, IGpuBuffer output, int size)
            => Array.Copy(Buffer(input).Data, Buffer(output).Data, size);

        public override void Gemm(
            IGpuBuffer a,
            IGpuBuffer b,
            IGpuBuffer c,
            int m,
            int n,
            int k,
            float alpha = 1f,
            float beta = 0f)
        {
            Fp32GemmCalls++;
            GemmCore(a, b, c, m, n, k, alpha, beta);
        }

        public override void BatchedGemm(
            IGpuBuffer a,
            IGpuBuffer b,
            IGpuBuffer c,
            int m,
            int n,
            int k,
            int batchCount,
            float alpha = 1f,
            float beta = 0f)
        {
            BatchedGemmCalls++;
            LastBatchCount = batchCount;
            int aStride = m * k;
            int bStride = k * n;
            int cStride = m * n;
            var left = Buffer(a).Data;
            var right = Buffer(b).Data;
            var output = Buffer(c).Data;
            for (int batch = 0; batch < batchCount; batch++)
            {
                for (int row = 0; row < m; row++)
                {
                    for (int column = 0; column < n; column++)
                    {
                        float sum = 0;
                        for (int inner = 0; inner < k; inner++)
                        {
                            sum += left[batch * aStride + row * k + inner]
                                * right[batch * bStride + inner * n + column];
                        }
                        int index = batch * cStride + row * n + column;
                        output[index] = alpha * sum + beta * output[index];
                    }
                }
            }
        }

        public override void Relu(IGpuBuffer input, IGpuBuffer output, int size)
        {
            Fp32ReluCalls++;
            var source = Buffer(input).Data;
            var destination = Buffer(output).Data;
            for (int i = 0; i < size; i++)
                destination[i] = MathF.Max(source[i], 0f);
        }

        public override void Add(IGpuBuffer left, IGpuBuffer right, IGpuBuffer output, int size)
        {
            Fp32AddCalls++;
            var a = Buffer(left).Data;
            var b = Buffer(right).Data;
            var destination = Buffer(output).Data;
            for (int i = 0; i < size; i++)
                destination[i] = a[i] + b[i];
        }

        public bool SupportsHgemm => true;
        public bool SupportsFp16FusedBackward => false;
        public bool SupportsFp16NativeOps => true;
        public bool Fp16Im2colAvailable => false;

        public void Hgemm(IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int m, int n, int k)
            => GemmCore(a, b, c, m, n, k, 1f, 0f);

        public void GemmFp16In32fOut(IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int m, int n, int k)
        {
            Fp16GemmCalls++;
            GemmCore(a, b, c, m, n, k, 1f, 0f);
        }

        public void MatMulBackwardFp16Fused(
            IGpuBuffer gradC,
            IGpuBuffer a,
            IGpuBuffer b,
            IGpuBuffer gradA,
            IGpuBuffer gradB,
            int m,
            int n,
            int k,
            bool gradOutHalf)
            => throw new NotSupportedException();

        public void Fp16Gelu(IGpuBuffer input, IGpuBuffer output, int size)
        {
            var source = Buffer(input).Data;
            var destination = Buffer(output).Data;
            for (int i = 0; i < size; i++)
            {
                float x = source[i];
                destination[i] = (float)(Half)(0.5f * x *
                    (1f + MathF.Tanh(0.7978845608f * (x + 0.044715f * x * x * x))));
            }
        }

        public void Fp16Relu(IGpuBuffer input, IGpuBuffer output, int size)
        {
            Fp16ReluCalls++;
            var source = Buffer(input).Data;
            var destination = Buffer(output).Data;
            for (int i = 0; i < size; i++)
                destination[i] = (float)(Half)MathF.Max(source[i], 0f);
        }

        public void Fp16Add(IGpuBuffer left, IGpuBuffer right, IGpuBuffer output, int size)
        {
            Fp16AddCalls++;
            var a = Buffer(left).Data;
            var b = Buffer(right).Data;
            var destination = Buffer(output).Data;
            for (int i = 0; i < size; i++)
                destination[i] = (float)(Half)(a[i] + b[i]);
        }

        public void Fp16Softmax(IGpuBuffer input, IGpuBuffer output, int rows, int cols)
            => throw new NotSupportedException();

        public void Fp16LayerNorm(
            IGpuBuffer input,
            IGpuBuffer gamma,
            IGpuBuffer beta,
            IGpuBuffer output,
            IGpuBuffer mean,
            IGpuBuffer variance,
            int rows,
            int cols,
            float epsilon)
            => throw new NotSupportedException();

        public void Im2colKNFp16(
            IGpuBuffer input,
            IGpuBuffer output,
            int batch,
            int channels,
            int height,
            int width,
            int kernelH,
            int kernelW,
            int strideH,
            int strideW,
            int padH,
            int padW,
            int dilationH,
            int dilationW)
            => throw new NotSupportedException();

        private static MockGpuBuffer Buffer(IGpuBuffer buffer) => Assert.IsType<MockGpuBuffer>(buffer);

        private static void GemmCore(
            IGpuBuffer a,
            IGpuBuffer b,
            IGpuBuffer c,
            int m,
            int n,
            int k,
            float alpha,
            float beta)
        {
            var left = Buffer(a).Data;
            var right = Buffer(b).Data;
            var output = Buffer(c).Data;
            for (int row = 0; row < m; row++)
            {
                for (int column = 0; column < n; column++)
                {
                    float sum = 0;
                    for (int inner = 0; inner < k; inner++)
                        sum += left[row * k + inner] * right[inner * n + column];
                    output[row * n + column] = alpha * sum + beta * output[row * n + column];
                }
            }
        }
    }
}
