// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

#if NET7_0_OR_GREATER
using AiDotNet.Tensors.Engines.DirectGpu.WebGpu;
#endif

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class CompressedMomentGpuOptimizerTests
{
    public enum BackendKind
    {
        Cuda,
        Hip,
        OpenCl,
        Metal,
        Vulkan,
#if NET7_0_OR_GREATER
        WebGpu,
#endif
    }

    public static TheoryData<BackendKind> Backends
    {
        get
        {
            var data = new TheoryData<BackendKind>
            {
                BackendKind.Cuda,
                BackendKind.Hip,
                BackendKind.OpenCl,
                BackendKind.Metal,
                BackendKind.Vulkan,
            };
#if NET7_0_OR_GREATER
            data.Add(BackendKind.WebGpu);
#endif
            return data;
        }
    }

    public static TheoryData<BackendKind, OptimizerType> DenseOptimizerBackends
    {
        get
        {
            var data = new TheoryData<BackendKind, OptimizerType>();
            foreach (BackendKind backend in Backends)
            {
                data.Add(backend, OptimizerType.SGD);
                data.Add(backend, OptimizerType.SGDMomentum);
                data.Add(backend, OptimizerType.Adam);
                data.Add(backend, OptimizerType.AdamW);
                data.Add(backend, OptimizerType.AMSGrad);
                data.Add(backend, OptimizerType.Nadam);
                data.Add(backend, OptimizerType.RMSprop);
                data.Add(backend, OptimizerType.Adagrad);
                data.Add(backend, OptimizerType.Lion);
                data.Add(backend, OptimizerType.AdaMax);
            }
            return data;
        }
    }

#if NET7_0_OR_GREATER
    [Fact]
    public void WebGpuOptimizerUniforms_PackStepAsFloatForBiasCorrection()
    {
        var uniforms = WebGpuBackend.MakeOptimizerUniforms(
            size: 17,
            lr: 0.01f,
            beta1: 0.9f,
            beta2: 0.999f,
            epsilon: 1e-8f,
            weightDecay: 0.0125f,
            t: 12);

        Assert.Equal(8, uniforms.Length);
        Assert.Equal(17, BitConverter.SingleToInt32Bits(uniforms[0]));
        Assert.Equal(12f, uniforms[6]);
    }
#endif

    private static bool RequireGpu =>
        string.Equals(Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS"), "1", StringComparison.Ordinal);

    [SkippableTheory]
    [MemberData(nameof(Backends))]
    public void ByteBufferUploadDownload_RoundTripsRawBytes(BackendKind kind)
    {
        var acquired = TryCreate(kind);
        using var scope = acquired;
        var backend = RequireReady(kind, acquired);
        byte[] expected =
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            15, 16, 17, 127, 128, 129, 254, 255, 42
        ];

        using var buffer = backend.AllocateByteBuffer(expected.Length);
        backend.UploadByteBuffer(buffer, expected);

        Assert.Equal(expected, backend.DownloadByteBuffer(buffer, expected.Length));
    }

    [SkippableTheory]
    [MemberData(nameof(Backends))]
    public void Fp32AdamAndAdamW_MatchCpuReference(BackendKind kind)
    {
        var acquired = TryCreate(kind);
        using var scope = acquired;
        var backend = RequireReady(kind, acquired);

        const int length = 41;
        const float lr = 0.01f;
        const float beta1 = 0.9f;
        const float beta2 = 0.997f;
        const float eps = 1e-7f;
        const float weightDecay = 0.0125f;
        var initial = Vector(length, 0.25f, -0.0075f);
        var grad = Gradient(length);

        using var adamParam = backend.AllocateBuffer((float[])initial.Clone());
        using var adamGrad = backend.AllocateBuffer(grad);
        using var adamM = backend.AllocateBuffer(new float[length]);
        using var adamV = backend.AllocateBuffer(new float[length]);
        for (int step = 1; step <= 3; step++)
            backend.AdamUpdate(adamParam, adamGrad, adamM, adamV, lr, beta1, beta2, eps, weightDecay, step, length);
        AssertClose(SimulateFp32Adam(initial, grad, 3, lr, beta1, beta2, eps, weightDecay, adamW: false),
            backend.DownloadBuffer(adamParam), length, 2e-5f, $"{kind} fp32 Adam");

        using var adamWParam = backend.AllocateBuffer((float[])initial.Clone());
        using var adamWGrad = backend.AllocateBuffer(grad);
        using var adamWM = backend.AllocateBuffer(new float[length]);
        using var adamWV = backend.AllocateBuffer(new float[length]);
        for (int step = 1; step <= 3; step++)
            backend.AdamWUpdate(adamWParam, adamWGrad, adamWM, adamWV, lr, beta1, beta2, eps, weightDecay, step, length);
        AssertClose(SimulateFp32Adam(initial, grad, 3, lr, beta1, beta2, eps, weightDecay, adamW: true),
            backend.DownloadBuffer(adamWParam), length, 2e-5f, $"{kind} fp32 AdamW");
    }

    [SkippableTheory]
    [MemberData(nameof(DenseOptimizerBackends))]
    public void DensePlanSupportedOptimizer_MatchesCpuReference(BackendKind kind, OptimizerType optimizer)
    {
        var acquired = TryCreate(kind);
        using var scope = acquired;
        var backend = RequireReady(kind, acquired);

        const int length = 47;
        const int steps = 4;
        const float lr = 0.01f;
        const float beta1 = 0.87f;
        const float beta2 = 0.993f;
        const float eps = 1e-6f;
        const float weightDecay = 0.0175f;
        var initial = Vector(length, 0.45f, -0.0085f);
        var grad = Gradient(length);

        using var param = backend.AllocateBuffer((float[])initial.Clone());
        using var gradBuffer = backend.AllocateBuffer(grad);
        using var state1 = backend.AllocateBuffer(new float[length]);
        using var state2 = backend.AllocateBuffer(new float[length]);
        using var state3 = backend.AllocateBuffer(new float[length]);

        for (int step = 1; step <= steps; step++)
        {
            switch (optimizer)
            {
                case OptimizerType.SGD:
                    backend.SgdUpdate(param, gradBuffer, lr, weightDecay, length);
                    break;
                case OptimizerType.SGDMomentum:
                    backend.SgdMomentumUpdate(param, gradBuffer, state1, lr, beta1, weightDecay, length);
                    break;
                case OptimizerType.Adam:
                    backend.AdamUpdate(param, gradBuffer, state1, state2, lr, beta1, beta2, eps, weightDecay, step, length);
                    break;
                case OptimizerType.AdamW:
                    backend.AdamWUpdate(param, gradBuffer, state1, state2, lr, beta1, beta2, eps, weightDecay, step, length);
                    break;
                case OptimizerType.AMSGrad:
                    backend.AmsgradUpdate(param, gradBuffer, state1, state2, state3, lr, beta1, beta2, eps, weightDecay, step, length);
                    break;
                case OptimizerType.Nadam:
                    backend.NadamUpdate(param, gradBuffer, state1, state2, lr, beta1, beta2, eps, weightDecay, step, length);
                    break;
                case OptimizerType.RMSprop:
                    backend.RmspropUpdate(param, gradBuffer, state1, lr, beta2, eps, weightDecay, length);
                    break;
                case OptimizerType.Adagrad:
                    backend.AdagradUpdate(param, gradBuffer, state1, lr, eps, weightDecay, length);
                    break;
                case OptimizerType.Lion:
                    backend.LionUpdate(param, gradBuffer, state1, lr, beta1, beta2, weightDecay, length);
                    break;
                case OptimizerType.AdaMax:
                    backend.AdamaxUpdate(param, gradBuffer, state1, state2, lr, beta1, beta2, eps, weightDecay, step, length);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(optimizer), optimizer, null);
            }
        }

        AssertClose(
            SimulateDenseOptimizer(optimizer, initial, grad, steps, lr, beta1, beta2, eps, weightDecay),
            backend.DownloadBuffer(param),
            length,
            3e-4f,
            $"{kind} dense {optimizer}");
    }

    [SkippableTheory]
    [MemberData(nameof(Backends))]
    public void Bf16AdamAndAdamW_MatchCpuReference(BackendKind kind)
    {
        var acquired = TryCreate(kind);
        using var scope = acquired;
        var backend = RequireReady(kind, acquired);
        var compressed = Assert.IsAssignableFrom<ICompressedMomentGpuOptimizerBackend>(backend);

        const int length = 43;
        const float lr = 0.01f;
        const float beta1 = 0.88f;
        const float beta2 = 0.996f;
        const float eps = 1e-7f;
        const float weightDecay = 0.0075f;
        var initial = Vector(length, -0.33f, 0.011f);
        var grad = Gradient(length);

        using var adamParam = backend.AllocateBuffer((float[])initial.Clone());
        using var adamGrad = backend.AllocateBuffer(grad);
        using var adamM = backend.AllocateByteBuffer(length * sizeof(ushort));
        using var adamV = backend.AllocateByteBuffer(length * sizeof(ushort));
        for (int step = 1; step <= 3; step++)
            compressed.AdamUpdateBf16(adamParam, adamGrad, adamM, adamV, lr, beta1, beta2, eps, weightDecay, step, length);
        AssertClose(SimulateBf16Adam(initial, grad, 3, lr, beta1, beta2, eps, weightDecay, adamW: false),
            backend.DownloadBuffer(adamParam), length, 1e-3f, $"{kind} bf16 Adam");

        using var adamWParam = backend.AllocateBuffer((float[])initial.Clone());
        using var adamWGrad = backend.AllocateBuffer(grad);
        using var adamWM = backend.AllocateByteBuffer(length * sizeof(ushort));
        using var adamWV = backend.AllocateByteBuffer(length * sizeof(ushort));
        for (int step = 1; step <= 3; step++)
            compressed.AdamWUpdateBf16(adamWParam, adamWGrad, adamWM, adamWV, lr, beta1, beta2, eps, weightDecay, step, length);
        AssertClose(SimulateBf16Adam(initial, grad, 3, lr, beta1, beta2, eps, weightDecay, adamW: true),
            backend.DownloadBuffer(adamWParam), length, 1e-3f, $"{kind} bf16 AdamW");
    }

    [SkippableTheory]
    [MemberData(nameof(Backends))]
    public void Int8Adam_MatchesCpuReference(BackendKind kind)
    {
        var acquired = TryCreate(kind);
        using var scope = acquired;
        var backend = RequireReady(kind, acquired);
        var compressed = Assert.IsAssignableFrom<ICompressedMomentGpuOptimizerBackend>(backend);

        const int length = 67;
        const int blockSize = 16;
        const float lr = 0.00625f;
        const float beta1 = 0.91f;
        const float beta2 = 0.995f;
        const float eps = 1e-7f;
        int numBlocks = (length + blockSize - 1) / blockSize;
        var initial = Vector(length, 0.5f, -0.006f);
        var grad = Gradient(length);

        using var param = backend.AllocateBuffer((float[])initial.Clone());
        using var gradBuffer = backend.AllocateBuffer(grad);
        using var mQuant = backend.AllocateByteBuffer(length);
        using var vQuant = backend.AllocateByteBuffer(length);
        using var mScales = backend.AllocateBuffer(new float[numBlocks]);
        using var vScales = backend.AllocateBuffer(new float[numBlocks]);
        for (int step = 1; step <= 4; step++)
        {
            float bc1 = 1f - MathF.Pow(beta1, step);
            float bc2 = 1f - MathF.Pow(beta2, step);
            compressed.Adam8BitUpdate(
                param, gradBuffer, mQuant, vQuant, mScales, vScales,
                lr, beta1, beta2, eps, 1f - beta1, 1f - beta2,
                bc1, bc2, blockSize, length, numBlocks);
        }

        AssertClose(SimulateInt8Adam(initial, grad, 4, lr, beta1, beta2, eps, blockSize),
            backend.DownloadBuffer(param), length, 2e-3f, $"{kind} int8 Adam");
    }

    private static AcquiredBackend TryCreate(BackendKind kind)
    {
        try
        {
            switch (kind)
            {
                case BackendKind.Cuda:
                {
                    var b = new CudaBackend();
                    return b.IsAvailable
                        ? new AcquiredBackend(b, b.Dispose, null)
                        : new AcquiredBackend(null, b.Dispose, null);
                }
                case BackendKind.Hip:
                {
                    var b = new HipBackend();
                    return b.IsAvailable
                        ? new AcquiredBackend(b, b.Dispose, null)
                        : new AcquiredBackend(null, b.Dispose, null);
                }
                case BackendKind.OpenCl:
                {
                    var b = new OpenClBackend();
                    return b.IsAvailable
                        ? new AcquiredBackend(b, b.Dispose, null)
                        : new AcquiredBackend(null, b.Dispose,
                            b.InitializationError is null ? null : new InvalidOperationException(b.InitializationError));
                }
                case BackendKind.Metal:
                {
                    var b = new MetalBackend();
                    return b.IsAvailable
                        ? new AcquiredBackend(b, b.Dispose, null)
                        : new AcquiredBackend(null, b.Dispose, null);
                }
                case BackendKind.Vulkan:
                {
                    var b = VulkanBackend.Instance;
                    return b.Initialize()
                        ? new AcquiredBackend(b, () => { }, null)
                        : new AcquiredBackend(null, () => { }, null);
                }
#if NET7_0_OR_GREATER
                case BackendKind.WebGpu:
                {
                    var b = new WebGpuBackend();
                    return b.IsAvailable
                        ? new AcquiredBackend(b, b.Dispose, null)
                        : new AcquiredBackend(null, b.Dispose, null);
                }
#endif
                default:
                    throw new ArgumentOutOfRangeException(nameof(kind), kind, null);
            }
        }
        catch (Exception ex)
        {
            return new AcquiredBackend(null, () => { }, ex);
        }
    }

    private static IDirectGpuBackend RequireReady(BackendKind kind, AcquiredBackend acquired)
    {
        if (acquired.Backend is not null)
            return acquired.Backend;

        if (RequireGpu)
            throw new InvalidOperationException(
                $"GPU tests were required (AIDOTNET_REQUIRE_GPU_TESTS=1) but the {kind} optimizer backend was unavailable.",
                acquired.Error);

        Skip.If(true, $"{kind} backend unavailable on this system.");
        throw new InvalidOperationException("Unreachable after skip.");
    }

    private static float[] Vector(int length, float start, float delta)
    {
        var data = new float[length];
        for (int i = 0; i < length; i++)
            data[i] = start + i * delta + ((i & 1) == 0 ? 0.003f : -0.002f);
        return data;
    }

    private static float[] Gradient(int length)
    {
        var data = new float[length];
        for (int i = 0; i < length; i++)
            data[i] = ((i % 7) - 3) * 0.0175f + ((i % 3) - 1) * 0.004f;
        return data;
    }

    private static float[] SimulateFp32Adam(
        float[] initialParam,
        float[] grad,
        int steps,
        float lr,
        float beta1,
        float beta2,
        float eps,
        float weightDecay,
        bool adamW)
    {
        var param = (float[])initialParam.Clone();
        var m = new float[param.Length];
        var v = new float[param.Length];

        for (int step = 1; step <= steps; step++)
        {
            float bc1 = 1f - MathF.Pow(beta1, step);
            float bc2 = 1f - MathF.Pow(beta2, step);
            for (int i = 0; i < param.Length; i++)
            {
                if (adamW && weightDecay > 0f)
                    param[i] *= 1f - lr * weightDecay;
                float g = adamW ? grad[i] : grad[i] + weightDecay * param[i];
                m[i] = beta1 * m[i] + (1f - beta1) * g;
                v[i] = beta2 * v[i] + (1f - beta2) * g * g;
                float update = lr * (m[i] / bc1) / (MathF.Sqrt(v[i] / bc2) + eps);
                param[i] -= update;
            }
        }

        return param;
    }

    private static float[] SimulateDenseOptimizer(
        OptimizerType optimizer,
        float[] initialParam,
        float[] grad,
        int steps,
        float lr,
        float beta1,
        float beta2,
        float eps,
        float weightDecay)
    {
        var param = (float[])initialParam.Clone();
        var state1 = new float[param.Length];
        var state2 = new float[param.Length];
        var state3 = new float[param.Length];

        for (int step = 1; step <= steps; step++)
        {
            float bc1 = 1f - MathF.Pow(beta1, step);
            float bc1Next = 1f - MathF.Pow(beta1, step + 1);
            float bc2 = 1f - MathF.Pow(beta2, step);
            for (int i = 0; i < param.Length; i++)
            {
                switch (optimizer)
                {
                    case OptimizerType.SGD:
                    {
                        float g = grad[i] + weightDecay * param[i];
                        param[i] -= lr * g;
                        break;
                    }
                    case OptimizerType.SGDMomentum:
                    {
                        float g = grad[i] + weightDecay * param[i];
                        state1[i] = beta1 * state1[i] + g;
                        param[i] -= lr * state1[i];
                        break;
                    }
                    case OptimizerType.Adam:
                    {
                        float g = grad[i] + weightDecay * param[i];
                        state1[i] = beta1 * state1[i] + (1f - beta1) * g;
                        state2[i] = beta2 * state2[i] + (1f - beta2) * g * g;
                        param[i] -= lr * (state1[i] / bc1) / (MathF.Sqrt(state2[i] / bc2) + eps);
                        break;
                    }
                    case OptimizerType.AdamW:
                    {
                        param[i] *= 1f - lr * weightDecay;
                        float g = grad[i];
                        state1[i] = beta1 * state1[i] + (1f - beta1) * g;
                        state2[i] = beta2 * state2[i] + (1f - beta2) * g * g;
                        param[i] -= lr * (state1[i] / bc1) / (MathF.Sqrt(state2[i] / bc2) + eps);
                        break;
                    }
                    case OptimizerType.AMSGrad:
                    {
                        float g = grad[i] + weightDecay * param[i];
                        state1[i] = beta1 * state1[i] + (1f - beta1) * g;
                        state2[i] = beta2 * state2[i] + (1f - beta2) * g * g;
                        state3[i] = MathF.Max(state3[i], state2[i]);
                        param[i] -= lr * (state1[i] / bc1) / (MathF.Sqrt(state3[i] / bc2) + eps);
                        break;
                    }
                    case OptimizerType.Nadam:
                    {
                        float g = grad[i] + weightDecay * param[i];
                        state1[i] = beta1 * state1[i] + (1f - beta1) * g;
                        state2[i] = beta2 * state2[i] + (1f - beta2) * g * g;
                        float mHat = state1[i] / bc1;
                        float vHat = state2[i] / bc2;
                        float mNesterov = beta1 * mHat + (1f - beta1) * g / bc1Next;
                        param[i] -= lr * mNesterov / (MathF.Sqrt(vHat) + eps);
                        break;
                    }
                    case OptimizerType.RMSprop:
                    {
                        float g = grad[i] + weightDecay * param[i];
                        state1[i] = beta2 * state1[i] + (1f - beta2) * g * g;
                        param[i] -= lr * g / (MathF.Sqrt(state1[i]) + eps);
                        break;
                    }
                    case OptimizerType.Adagrad:
                    {
                        float g = grad[i] + weightDecay * param[i];
                        state1[i] += g * g;
                        param[i] -= lr * g / (MathF.Sqrt(state1[i]) + eps);
                        break;
                    }
                    case OptimizerType.Lion:
                    {
                        float update = MathF.Sign(beta1 * state1[i] + (1f - beta1) * grad[i]);
                        param[i] -= lr * (update + weightDecay * param[i]);
                        state1[i] = beta2 * state1[i] + (1f - beta2) * grad[i];
                        break;
                    }
                    case OptimizerType.AdaMax:
                    {
                        float g = grad[i] + weightDecay * param[i];
                        state1[i] = beta1 * state1[i] + (1f - beta1) * g;
                        state2[i] = MathF.Max(beta2 * state2[i], MathF.Abs(g));
                        param[i] -= lr * (state1[i] / bc1) / (state2[i] + eps);
                        break;
                    }
                    default:
                        throw new ArgumentOutOfRangeException(nameof(optimizer), optimizer, null);
                }
            }
        }

        return param;
    }

    private static float[] SimulateBf16Adam(
        float[] initialParam,
        float[] grad,
        int steps,
        float lr,
        float beta1,
        float beta2,
        float eps,
        float weightDecay,
        bool adamW)
    {
        var param = (float[])initialParam.Clone();
        var m = new ushort[param.Length];
        var v = new ushort[param.Length];

        for (int step = 1; step <= steps; step++)
        {
            float bc1 = 1f - MathF.Pow(beta1, step);
            float bc2 = 1f - MathF.Pow(beta2, step);
            for (int i = 0; i < param.Length; i++)
            {
                if (adamW && weightDecay > 0f)
                    param[i] *= 1f - lr * weightDecay;
                float g = adamW ? grad[i] : grad[i] + weightDecay * param[i];
                float newM = beta1 * Bf16ToFloat(m[i]) + (1f - beta1) * g;
                float newV = beta2 * Bf16ToFloat(v[i]) + (1f - beta2) * g * g;
                m[i] = Bf16FromFloat(newM);
                v[i] = Bf16FromFloat(newV);
                float update = lr * (newM / bc1) / (MathF.Sqrt(newV / bc2) + eps);
                param[i] -= update;
            }
        }

        return param;
    }

    private static float[] SimulateInt8Adam(
        float[] initialParam,
        float[] grad,
        int steps,
        float lr,
        float beta1,
        float beta2,
        float eps,
        int blockSize)
    {
        var param = (float[])initialParam.Clone();
        var mQuant = new byte[param.Length];
        var vQuant = new byte[param.Length];
        int numBlocks = (param.Length + blockSize - 1) / blockSize;
        var mScales = new float[numBlocks];
        var vScales = new float[numBlocks];

        for (int step = 1; step <= steps; step++)
        {
            float oneMinusBeta1 = 1f - beta1;
            float oneMinusBeta2 = 1f - beta2;
            float bc1 = 1f - MathF.Pow(beta1, step);
            float bc2 = 1f - MathF.Pow(beta2, step);
            bool firstStep = step == 1;

            for (int block = 0; block < numBlocks; block++)
            {
                int start = block * blockSize;
                int end = Math.Min(start + blockSize, param.Length);
                float oldMScale = firstStep ? 0f : mScales[block];
                float oldVScale = firstStep ? 0f : vScales[block];
                float maxM = 0f;
                float maxV = 0f;

                for (int i = start; i < end; i++)
                {
                    float oldM = firstStep ? 0f : (mQuant[i] - 128) * oldMScale;
                    float oldV = firstStep ? 0f : vQuant[i] * oldVScale;
                    float newM = beta1 * oldM + oneMinusBeta1 * grad[i];
                    float newV = beta2 * oldV + oneMinusBeta2 * grad[i] * grad[i];
                    maxM = MathF.Max(maxM, MathF.Abs(newM));
                    maxV = MathF.Max(maxV, MathF.Abs(newV));
                }

                float newMScale = MathF.Max(maxM / 127f, 1e-10f);
                float newVScale = MathF.Max(maxV / 255f, 1e-10f);
                mScales[block] = newMScale;
                vScales[block] = newVScale;

                for (int i = start; i < end; i++)
                {
                    float oldM = firstStep ? 0f : (mQuant[i] - 128) * oldMScale;
                    float oldV = firstStep ? 0f : vQuant[i] * oldVScale;
                    float newM = beta1 * oldM + oneMinusBeta1 * grad[i];
                    float newV = beta2 * oldV + oneMinusBeta2 * grad[i] * grad[i];
                    param[i] -= lr * (newM / bc1) / (MathF.Sqrt(newV / bc2) + eps);

                    int qm = (int)Math.Round(newM / newMScale, MidpointRounding.ToEven);
                    if (qm < -127) qm = -127;
                    if (qm > 127) qm = 127;
                    mQuant[i] = (byte)(qm + 128);

                    int qv = (int)Math.Round(newV / newVScale, MidpointRounding.ToEven);
                    if (qv < 0) qv = 0;
                    if (qv > 255) qv = 255;
                    vQuant[i] = (byte)qv;
                }
            }
        }

        return param;
    }

    private static void AssertClose(float[] expected, float[] actual, int length, float tolerance, string label)
    {
        Assert.True(actual.Length >= length, $"{label}: downloaded {actual.Length} values, expected at least {length}.");
        for (int i = 0; i < length; i++)
            Assert.True(Math.Abs(actual[i] - expected[i]) <= tolerance,
                $"{label} mismatch at [{i}]: expected {expected[i]:R}, actual {actual[i]:R}, tolerance {tolerance:R}.");
    }

    private static unsafe float Bf16ToFloat(ushort value)
    {
        uint bits = (uint)value << 16;
        return *(float*)&bits;
    }

    private static unsafe ushort Bf16FromFloat(float value)
    {
        uint bits = *(uint*)&value;
        if ((bits & 0x7FFFFFFFu) > 0x7F800000u)
            return (ushort)((bits >> 16) | 0x0040u);

        uint rounding = 0x7FFFu + ((bits >> 16) & 1u);
        return (ushort)((bits + rounding) >> 16);
    }

    private sealed class AcquiredBackend : IDisposable
    {
        private readonly Action _dispose;

        public AcquiredBackend(IDirectGpuBackend? backend, Action dispose, Exception? error)
        {
            Backend = backend;
            _dispose = dispose;
            Error = error;
        }

        public IDirectGpuBackend? Backend { get; }

        public Exception? Error { get; }

        public void Dispose() => _dispose();
    }
}
