// Copyright (c) AiDotNet. All rights reserved.
// WebGPU backend implementation for browser-based GPU compute.
// Only available in .NET 7+ with Blazor WebAssembly.

#if NET7_0_OR_GREATER
using System;
using System.Collections.Concurrent;
using System.Threading.Tasks;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

/// <summary>
/// WebGPU backend for GPU-accelerated tensor operations in the browser.
/// </summary>
/// <remarks>
/// <para><b>Browser GPU Acceleration:</b></para>
/// <para>
/// This backend enables GPU compute in Blazor WebAssembly applications
/// using the WebGPU API. It provides significant performance improvements
/// for neural network inference and tensor operations.
/// </para>
/// <para><b>Supported Browsers:</b></para>
/// <list type="bullet">
/// <item>Chrome 113+ (recommended)</item>
/// <item>Edge 113+</item>
/// <item>Firefox (behind flag)</item>
/// <item>Safari (experimental)</item>
/// </list>
/// <para><b>Fallback:</b></para>
/// <para>
/// If WebGPU is not available, operations fall back to CPU computation.
/// Always check IsAvailable before using GPU operations.
/// </para>
/// </remarks>
public sealed partial class WebGpuBackend : IDirectGpuBackend
{
    private readonly WebGpuDevice _device;
    private readonly WebGpuShaderModule _shaderLibrary;
    private readonly ConcurrentDictionary<string, int> _pipelineCache = new();
    private bool _initialized;
    private bool _disposed;

    /// <inheritdoc/>
    public bool IsAvailable => _initialized && !_disposed;

    /// <inheritdoc/>
    public string BackendName => "WebGPU";

    /// <summary>
    /// Gets the backend type identifier.
    /// </summary>
    public string BackendType => "WebGPU";

    /// <inheritdoc/>
    public string DeviceName => _device.AdapterInfo;

    /// <inheritdoc/>
    public string DeviceVendor => _device.AdapterInfo ?? "Unknown";

    /// <inheritdoc/>
    public int ComputeUnits => 1;

    /// <inheritdoc/>
    public long GlobalMemoryBytes => _device.MaxBufferSize;

    /// <inheritdoc/>
    public long LocalMemoryBytes => _device.MaxWorkgroupSize * 4L;

    /// <summary>
    /// Gets the maximum buffer size in bytes.
    /// </summary>
    public long MaxBufferSize => _device.MaxBufferSize;

    /// <summary>
    /// Gets the maximum workgroup size.
    /// </summary>
    public int MaxWorkgroupSize => _device.MaxWorkgroupSize;

    /// <summary>
    /// Gets the device limits.
    /// </summary>
    public WebGpuDeviceLimits? Limits => _device.Limits;

    /// <summary>
    /// Creates a new WebGPU backend instance.
    /// </summary>
    public WebGpuBackend()
    {
        _device = WebGpuDevice.Instance;
        _shaderLibrary = new WebGpuShaderModule();
    }

    /// <summary>
    /// Initializes the WebGPU backend asynchronously.
    /// </summary>
    /// <returns>True if initialization succeeded.</returns>
    public async Task<bool> InitializeAsync()
    {
        if (_initialized)
        {
            return true;
        }

        if (!WebGpuDevice.IsSupported)
        {
            return false;
        }

        var success = await _device.InitializeAsync();
        if (!success)
        {
            return false;
        }

        _initialized = true;
        return true;
    }

    /// <summary>
    /// Allocates a GPU buffer with the specified element count.
    /// </summary>
    /// <param name="elementCount">Number of float elements.</param>
    /// <returns>A new GPU buffer.</returns>
    public IGpuBuffer AllocateBuffer(int elementCount)
    {
        ThrowIfNotInitialized();
        return new WebGpuBuffer(elementCount);
    }

    /// <summary>
    /// Allocates a GPU buffer and uploads data.
    /// </summary>
    /// <param name="data">Data to upload.</param>
    /// <returns>A new GPU buffer containing the data.</returns>
    public IGpuBuffer AllocateBuffer(float[] data)
    {
        ThrowIfNotInitialized();
        return new WebGpuBuffer(data);
    }

    #region Element-wise Operations

    /// <summary>
    /// Element-wise addition: C = A + B
    /// </summary>
    public async Task AddAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        await DispatchBinaryOpAsync("add", A, B, C, size);
    }

    /// <summary>
    /// Element-wise subtraction: C = A - B
    /// </summary>
    public async Task SubAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        await DispatchBinaryOpAsync("sub", A, B, C, size);
    }

    /// <summary>
    /// Element-wise multiplication: C = A * B
    /// </summary>
    public async Task MulAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        await DispatchBinaryOpAsync("mul", A, B, C, size);
    }

    /// <summary>
    /// Element-wise division: C = A / B
    /// </summary>
    public async Task DivAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        await DispatchBinaryOpAsync("div", A, B, C, size);
    }

    /// <summary>
    /// Element-wise maximum: C = max(A, B)
    /// </summary>
    public async Task MaximumAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        await DispatchBinaryOpAsync("maximum", A, B, C, size);
    }

    /// <summary>
    /// Element-wise minimum: C = min(A, B)
    /// </summary>
    public async Task MinimumAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        await DispatchBinaryOpAsync("minimum", A, B, C, size);
    }

    private async Task DispatchBinaryOpAsync(string kernelName, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        ThrowIfNotInitialized();

        var aBuffer = (WebGpuBuffer)A;
        var bBuffer = (WebGpuBuffer)B;
        var cBuffer = (WebGpuBuffer)C;

        var pipelineId = await GetOrCreatePipelineAsync("ElementWise", WebGpuKernels.ElementWiseSource, kernelName);

        // Create uniform buffer for params
        var paramsData = new float[] { BitConverter.Int32BitsToSingle(size), 0, 0, 0 };
        using var uniformBuffer = new WebGpuBuffer(paramsData, WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);

        using var bindGroup = new WebGpuBindGroup(pipelineId, aBuffer, bBuffer, cBuffer);

        var (workgroups, _) = _device.CalculateWorkgroups1D(size);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipelineId, bindGroup.BindGroupId, uniformBuffer.BufferId, workgroups, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    #endregion

    #region Scalar Operations

    /// <summary>
    /// Scalar addition: B = A + scalar
    /// </summary>
    public async Task AddScalarAsync(IGpuBuffer A, IGpuBuffer B, float scalar, int size)
    {
        await DispatchScalarOpAsync("add_scalar", A, B, scalar, size);
    }

    /// <summary>
    /// Scalar multiplication: B = A * scalar
    /// </summary>
    public async Task ScaleAsync(IGpuBuffer A, IGpuBuffer B, float scalar, int size)
    {
        await DispatchScalarOpAsync("mul_scalar", A, B, scalar, size);
    }

    /// <summary>
    /// Power operation: B = A ^ scalar
    /// </summary>
    public async Task PowerAsync(IGpuBuffer A, IGpuBuffer B, float scalar, int size)
    {
        await DispatchScalarOpAsync("pow_scalar", A, B, scalar, size);
    }

    private async Task DispatchScalarOpAsync(string kernelName, IGpuBuffer A, IGpuBuffer B, float scalar, int size)
    {
        ThrowIfNotInitialized();

        var aBuffer = (WebGpuBuffer)A;
        var bBuffer = (WebGpuBuffer)B;

        var pipelineId = await GetOrCreatePipelineAsync("ScalarOps", WebGpuKernels.ScalarOpsSource, kernelName);

        // Pack size and scalar into uniform buffer
        var paramsData = new float[] { BitConverter.Int32BitsToSingle(size), scalar, 0, 0 };
        using var uniformBuffer = new WebGpuBuffer(paramsData, WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);

        using var bindGroup = new WebGpuBindGroup(pipelineId, aBuffer, bBuffer);

        var (workgroups, _) = _device.CalculateWorkgroups1D(size);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipelineId, bindGroup.BindGroupId, uniformBuffer.BufferId, workgroups, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    #endregion

    #region Unary Operations

    /// <summary>
    /// Square root: B = sqrt(A)
    /// </summary>
    public async Task SqrtAsync(IGpuBuffer A, IGpuBuffer B, int size)
    {
        await DispatchUnaryOpAsync("sqrt_op", A, B, size);
    }

    /// <summary>
    /// Exponential: B = exp(A)
    /// </summary>
    public async Task ExpAsync(IGpuBuffer A, IGpuBuffer B, int size)
    {
        await DispatchUnaryOpAsync("exp_op", A, B, size);
    }

    /// <summary>
    /// Natural logarithm: B = log(A)
    /// </summary>
    public async Task LogAsync(IGpuBuffer A, IGpuBuffer B, int size)
    {
        await DispatchUnaryOpAsync("log_op", A, B, size);
    }

    /// <summary>
    /// Absolute value: B = abs(A)
    /// </summary>
    public async Task AbsAsync(IGpuBuffer A, IGpuBuffer B, int size)
    {
        await DispatchUnaryOpAsync("abs_op", A, B, size);
    }

    /// <summary>
    /// Negation: B = -A
    /// </summary>
    public async Task NegAsync(IGpuBuffer A, IGpuBuffer B, int size)
    {
        await DispatchUnaryOpAsync("neg_op", A, B, size);
    }

    /// <summary>
    /// Sign: B = sign(A)
    /// </summary>
    public async Task SignAsync(IGpuBuffer A, IGpuBuffer B, int size)
    {
        await DispatchUnaryOpAsync("sign_op", A, B, size);
    }

    /// <summary>
    /// Square: B = A * A
    /// </summary>
    public async Task SquareAsync(IGpuBuffer A, IGpuBuffer B, int size)
    {
        await DispatchUnaryOpAsync("square_op", A, B, size);
    }

    private async Task DispatchUnaryOpAsync(string kernelName, IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfNotInitialized();

        var aBuffer = (WebGpuBuffer)A;
        var bBuffer = (WebGpuBuffer)B;

        var pipelineId = await GetOrCreatePipelineAsync("UnaryMath", WebGpuKernels.UnaryMathSource, kernelName);

        var paramsData = new float[] { BitConverter.Int32BitsToSingle(size), 0, 0, 0 };
        using var uniformBuffer = new WebGpuBuffer(paramsData, WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);

        using var bindGroup = new WebGpuBindGroup(pipelineId, aBuffer, bBuffer);

        var (workgroups, _) = _device.CalculateWorkgroups1D(size);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipelineId, bindGroup.BindGroupId, uniformBuffer.BufferId, workgroups, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    #endregion

    #region Activation Functions

    /// <summary>
    /// ReLU activation: B = max(0, A)
    /// </summary>
    public async Task ReLUAsync(IGpuBuffer A, IGpuBuffer B, int size)
    {
        await DispatchActivationAsync("relu", A, B, size, 0);
    }

    /// <summary>
    /// Leaky ReLU activation: B = A > 0 ? A : alpha * A
    /// </summary>
    public async Task LeakyReLUAsync(IGpuBuffer A, IGpuBuffer B, int size, float alpha = 0.01f)
    {
        await DispatchActivationAsync("leaky_relu", A, B, size, alpha);
    }

    /// <summary>
    /// Sigmoid activation: B = 1 / (1 + exp(-A))
    /// </summary>
    public async Task SigmoidAsync(IGpuBuffer A, IGpuBuffer B, int size)
    {
        await DispatchActivationAsync("sigmoid", A, B, size, 0);
    }

    /// <summary>
    /// Tanh activation: B = tanh(A)
    /// </summary>
    public async Task TanhAsync(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfNotInitialized();

        var aBuffer = (WebGpuBuffer)A;
        var bBuffer = (WebGpuBuffer)B;

        var pipelineId = await GetOrCreatePipelineAsync("Trig", WebGpuKernels.TrigSource, "tanh_op");

        var paramsData = new float[] { BitConverter.Int32BitsToSingle(size), 0, 0, 0 };
        using var uniformBuffer = new WebGpuBuffer(paramsData, WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);

        using var bindGroup = new WebGpuBindGroup(pipelineId, aBuffer, bBuffer);

        var (workgroups, _) = _device.CalculateWorkgroups1D(size);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipelineId, bindGroup.BindGroupId, uniformBuffer.BufferId, workgroups, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    /// <summary>
    /// GELU activation: B = 0.5 * A * (1 + tanh(sqrt(2/pi) * (A + 0.044715 * A^3)))
    /// </summary>
    public async Task GeLUAsync(IGpuBuffer A, IGpuBuffer B, int size)
    {
        await DispatchActivationAsync("gelu_op", A, B, size, 0);
    }

    /// <summary>
    /// Swish/SiLU activation: B = A * sigmoid(A)
    /// </summary>
    public async Task SwishAsync(IGpuBuffer A, IGpuBuffer B, int size)
    {
        await DispatchActivationAsync("swish", A, B, size, 0);
    }

    /// <summary>
    /// ELU activation: B = A > 0 ? A : alpha * (exp(A) - 1)
    /// </summary>
    public async Task ELUAsync(IGpuBuffer A, IGpuBuffer B, int size, float alpha = 1.0f)
    {
        await DispatchActivationAsync("elu", A, B, size, alpha);
    }

    /// <summary>
    /// Softplus activation: B = log(1 + exp(A))
    /// </summary>
    public async Task SoftplusAsync(IGpuBuffer A, IGpuBuffer B, int size)
    {
        await DispatchActivationAsync("softplus", A, B, size, 0);
    }

    /// <summary>
    /// Mish activation: B = A * tanh(softplus(A))
    /// </summary>
    public async Task MishAsync(IGpuBuffer A, IGpuBuffer B, int size)
    {
        await DispatchActivationAsync("mish", A, B, size, 0);
    }

    private async Task DispatchActivationAsync(string kernelName, IGpuBuffer A, IGpuBuffer B, int size, float alpha)
    {
        ThrowIfNotInitialized();

        var aBuffer = (WebGpuBuffer)A;
        var bBuffer = (WebGpuBuffer)B;

        var pipelineId = await GetOrCreatePipelineAsync("Activation", WebGpuKernels.ActivationSource, kernelName);

        var paramsData = new float[] { BitConverter.Int32BitsToSingle(size), alpha, 0, 0 };
        using var uniformBuffer = new WebGpuBuffer(paramsData, WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);

        using var bindGroup = new WebGpuBindGroup(pipelineId, aBuffer, bBuffer);

        var (workgroups, _) = _device.CalculateWorkgroups1D(size);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipelineId, bindGroup.BindGroupId, uniformBuffer.BufferId, workgroups, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    #endregion

    #region Reduction Operations

    /// <summary>
    /// Sum reduction: returns sum of all elements.
    /// </summary>
    public async Task<float> SumAsync(IGpuBuffer A, int size)
    {
        return await ReduceAsync("sum_reduce", A, size, 0);
    }

    /// <summary>
    /// Max reduction: returns maximum element.
    /// </summary>
    public async Task<float> MaxAsync(IGpuBuffer A, int size)
    {
        return await ReduceAsync("max_reduce", A, size, float.NegativeInfinity);
    }

    /// <summary>
    /// Min reduction: returns minimum element.
    /// </summary>
    public async Task<float> MinAsync(IGpuBuffer A, int size)
    {
        return await ReduceAsync("min_reduce", A, size, float.PositiveInfinity);
    }

    private async Task<float> ReduceAsync(string kernelName, IGpuBuffer A, int size, float identity)
    {
        ThrowIfNotInitialized();

        var aBuffer = (WebGpuBuffer)A;

        const int workgroupSize = 256;
        int numWorkgroups = (size + workgroupSize - 1) / workgroupSize;

        var pipelineId = await GetOrCreatePipelineAsync("Reduction", WebGpuKernels.ReductionSource, kernelName);

        // First pass reduction
        using var partialResults = new WebGpuBuffer(numWorkgroups);

        var paramsData = new float[] { BitConverter.Int32BitsToSingle(size), 0, 0, 0 };
        using var uniformBuffer = new WebGpuBuffer(paramsData, WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);

        using var bindGroup = new WebGpuBindGroup(pipelineId, aBuffer, partialResults);

        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipelineId, bindGroup.BindGroupId, uniformBuffer.BufferId, numWorkgroups, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();

        // Continue reducing until we have a single value
        while (numWorkgroups > 1)
        {
            int prevSize = numWorkgroups;
            numWorkgroups = (numWorkgroups + workgroupSize - 1) / workgroupSize;

            using var newPartialResults = new WebGpuBuffer(numWorkgroups);

            paramsData[0] = BitConverter.Int32BitsToSingle(prevSize);
            uniformBuffer.CopyFrom(paramsData);

            using var bindGroup2 = new WebGpuBindGroup(pipelineId, partialResults, newPartialResults);

            await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipelineId, bindGroup2.BindGroupId, uniformBuffer.BufferId, numWorkgroups, 1, 1);
            await WebGpuNativeBindings.SubmitAndWaitAsync();

            if (numWorkgroups == 1)
            {
                var result = await newPartialResults.DownloadAsync();
                return result[0];
            }
        }

        var finalResult = await partialResults.DownloadAsync();
        return finalResult[0];
    }

    #endregion

    #region Matrix Operations

    /// <summary>
    /// General matrix multiplication: C = alpha * A * B + beta * C
    /// </summary>
    public async Task GemmAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
    {
        ThrowIfNotInitialized();

        var aBuffer = (WebGpuBuffer)A;
        var bBuffer = (WebGpuBuffer)B;
        var cBuffer = (WebGpuBuffer)C;

        string kernelName = (M * N) > 256 ? "gemm" : "gemm_simple";
        var pipelineId = await GetOrCreatePipelineAsync("MatMul", WebGpuKernels.MatMulSource, kernelName);

        // Pack matrix dimensions into uniform buffer
        var paramsData = new float[]
        {
            BitConverter.Int32BitsToSingle(M),
            BitConverter.Int32BitsToSingle(N),
            BitConverter.Int32BitsToSingle(K),
            alpha,
            beta, 0, 0, 0
        };
        using var uniformBuffer = new WebGpuBuffer(paramsData, WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);

        using var bindGroup = new WebGpuBindGroup(pipelineId, aBuffer, bBuffer, cBuffer);

        int workgroupsX, workgroupsY;
        if (kernelName == "gemm")
        {
            // Tiled GEMM uses 16x16 workgroups
            workgroupsX = (N + 15) / 16;
            workgroupsY = (M + 15) / 16;
        }
        else
        {
            // Simple GEMM uses 1D dispatch
            var (wg, _) = _device.CalculateWorkgroups1D(M * N);
            workgroupsX = wg;
            workgroupsY = 1;
        }

        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipelineId, bindGroup.BindGroupId, uniformBuffer.BufferId, workgroupsX, workgroupsY, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    /// <summary>
    /// Matrix transpose.
    /// </summary>
    public async Task TransposeAsync(IGpuBuffer input, IGpuBuffer output, int rows, int cols)
    {
        ThrowIfNotInitialized();

        var inputBuffer = (WebGpuBuffer)input;
        var outputBuffer = (WebGpuBuffer)output;

        string kernelName = (rows * cols) > 256 ? "transpose" : "transpose_simple";
        var pipelineId = await GetOrCreatePipelineAsync("Transpose", WebGpuKernels.TransposeSource, kernelName);

        var paramsData = new float[]
        {
            BitConverter.Int32BitsToSingle(rows),
            BitConverter.Int32BitsToSingle(cols),
            0, 0
        };
        using var uniformBuffer = new WebGpuBuffer(paramsData, WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);

        using var bindGroup = new WebGpuBindGroup(pipelineId, inputBuffer, outputBuffer);

        int workgroupsX, workgroupsY;
        if (kernelName == "transpose")
        {
            workgroupsX = (cols + 15) / 16;
            workgroupsY = (rows + 15) / 16;
        }
        else
        {
            var (wg, _) = _device.CalculateWorkgroups1D(rows * cols);
            workgroupsX = wg;
            workgroupsY = 1;
        }

        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipelineId, bindGroup.BindGroupId, uniformBuffer.BufferId, workgroupsX, workgroupsY, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    #endregion

    #region Softmax and Normalization

    /// <summary>
    /// Softmax: output = exp(input - max) / sum(exp(input - max))
    /// </summary>
    public async Task SoftmaxAsync(IGpuBuffer input, IGpuBuffer output, int batchSize, int classCount)
    {
        ThrowIfNotInitialized();

        var inputBuffer = (WebGpuBuffer)input;
        var outputBuffer = (WebGpuBuffer)output;

        var pipelineId = await GetOrCreatePipelineAsync("Softmax", WebGpuKernels.SoftmaxSource, "softmax");

        var paramsData = new float[]
        {
            BitConverter.Int32BitsToSingle(batchSize),
            BitConverter.Int32BitsToSingle(classCount),
            0, 0
        };
        using var uniformBuffer = new WebGpuBuffer(paramsData, WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);

        using var bindGroup = new WebGpuBindGroup(pipelineId, inputBuffer, outputBuffer);

        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipelineId, bindGroup.BindGroupId, uniformBuffer.BufferId, batchSize, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    /// <summary>
    /// Layer normalization.
    /// </summary>
    public async Task LayerNormAsync(IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer beta, IGpuBuffer output, int batchSize, int featureSize, float epsilon = 1e-5f)
    {
        ThrowIfNotInitialized();

        var inputBuffer = (WebGpuBuffer)input;
        var gammaBuffer = (WebGpuBuffer)gamma;
        var betaBuffer = (WebGpuBuffer)beta;
        var outputBuffer = (WebGpuBuffer)output;

        var pipelineId = await GetOrCreatePipelineAsync("LayerNorm", WebGpuKernels.LayerNormSource, "layer_norm");

        var paramsData = new float[]
        {
            BitConverter.Int32BitsToSingle(batchSize),
            BitConverter.Int32BitsToSingle(featureSize),
            epsilon, 0
        };
        using var uniformBuffer = new WebGpuBuffer(paramsData, WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);

        using var bindGroup = new WebGpuBindGroup(pipelineId, inputBuffer, gammaBuffer, betaBuffer, outputBuffer);

        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipelineId, bindGroup.BindGroupId, uniformBuffer.BufferId, batchSize, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    #endregion

    #region Generalized GPU Dispatch Helpers

    private static float[] MakeUniform1(int size) =>
        new float[] { BitConverter.Int32BitsToSingle(size), 0, 0, 0 };

    private static float[] MakeUniform2(int size, float p1) =>
        new float[] { BitConverter.Int32BitsToSingle(size), p1, 0, 0 };

    private static float[] MakeUniform3(int size, float p1, float p2) =>
        new float[] { BitConverter.Int32BitsToSingle(size), p1, p2, 0 };

    private static float[] MakeUniform4(int size, float p1, float p2, float p3) =>
        new float[] { BitConverter.Int32BitsToSingle(size), p1, p2, p3 };

    private static float[] MakeUniformInts2(int a, int b) =>
        new float[] { BitConverter.Int32BitsToSingle(a), BitConverter.Int32BitsToSingle(b), 0, 0 };

    internal async Task Dispatch1BufferAsync(string moduleName, string source, string kernelName,
        IGpuBuffer a, float[] uniformParams, int workSize)
    {
        ThrowIfNotInitialized();
        var aBuffer = (WebGpuBuffer)a;
        var pipelineId = await GetOrCreatePipelineAsync(moduleName, source, kernelName);
        using var uniformBuffer = new WebGpuBuffer(uniformParams, WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bindGroup = new WebGpuBindGroup(pipelineId, aBuffer);
        var (workgroups, _) = _device.CalculateWorkgroups1D(workSize);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(
            pipelineId, bindGroup.BindGroupId, uniformBuffer.BufferId, workgroups, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    internal async Task Dispatch2BufferAsync(string moduleName, string source, string kernelName,
        IGpuBuffer a, IGpuBuffer b, float[] uniformParams, int workSize)
    {
        ThrowIfNotInitialized();
        var aBuffer = (WebGpuBuffer)a;
        var bBuffer = (WebGpuBuffer)b;
        var pipelineId = await GetOrCreatePipelineAsync(moduleName, source, kernelName);
        using var uniformBuffer = new WebGpuBuffer(uniformParams, WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bindGroup = new WebGpuBindGroup(pipelineId, aBuffer, bBuffer);
        var (workgroups, _) = _device.CalculateWorkgroups1D(workSize);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(
            pipelineId, bindGroup.BindGroupId, uniformBuffer.BufferId, workgroups, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    internal async Task Dispatch3BufferAsync(string moduleName, string source, string kernelName,
        IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, float[] uniformParams, int workSize)
    {
        ThrowIfNotInitialized();
        var aBuffer = (WebGpuBuffer)a;
        var bBuffer = (WebGpuBuffer)b;
        var cBuffer = (WebGpuBuffer)c;
        var pipelineId = await GetOrCreatePipelineAsync(moduleName, source, kernelName);
        using var uniformBuffer = new WebGpuBuffer(uniformParams, WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bindGroup = new WebGpuBindGroup(pipelineId, aBuffer, bBuffer, cBuffer);
        var (workgroups, _) = _device.CalculateWorkgroups1D(workSize);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(
            pipelineId, bindGroup.BindGroupId, uniformBuffer.BufferId, workgroups, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    internal async Task Dispatch4BufferAsync(string moduleName, string source, string kernelName,
        IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, IGpuBuffer d, float[] uniformParams, int workSize)
    {
        ThrowIfNotInitialized();
        var aBuffer = (WebGpuBuffer)a;
        var bBuffer = (WebGpuBuffer)b;
        var cBuffer = (WebGpuBuffer)c;
        var dBuffer = (WebGpuBuffer)d;
        var pipelineId = await GetOrCreatePipelineAsync(moduleName, source, kernelName);
        using var uniformBuffer = new WebGpuBuffer(uniformParams, WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bindGroup = new WebGpuBindGroup(pipelineId, aBuffer, bBuffer, cBuffer, dBuffer);
        var (workgroups, _) = _device.CalculateWorkgroups1D(workSize);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(
            pipelineId, bindGroup.BindGroupId, uniformBuffer.BufferId, workgroups, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    internal async Task Dispatch5BufferAsync(string moduleName, string source, string kernelName,
        IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, IGpuBuffer d, IGpuBuffer e,
        float[] uniformParams, int workSize)
    {
        ThrowIfNotInitialized();
        var aBuffer = (WebGpuBuffer)a;
        var bBuffer = (WebGpuBuffer)b;
        var cBuffer = (WebGpuBuffer)c;
        var dBuffer = (WebGpuBuffer)d;
        var eBuffer = (WebGpuBuffer)e;
        var pipelineId = await GetOrCreatePipelineAsync(moduleName, source, kernelName);
        using var uniformBuffer = new WebGpuBuffer(uniformParams, WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bindGroup = new WebGpuBindGroup(pipelineId, aBuffer, bBuffer, cBuffer, dBuffer, eBuffer);
        var (workgroups, _) = _device.CalculateWorkgroups1D(workSize);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(
            pipelineId, bindGroup.BindGroupId, uniformBuffer.BufferId, workgroups, 1, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    internal async Task Dispatch3Buffer2DAsync(string moduleName, string source, string kernelName,
        IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, float[] uniformParams, int workgroupsX, int workgroupsY)
    {
        ThrowIfNotInitialized();
        var aBuffer = (WebGpuBuffer)a;
        var bBuffer = (WebGpuBuffer)b;
        var cBuffer = (WebGpuBuffer)c;
        var pipelineId = await GetOrCreatePipelineAsync(moduleName, source, kernelName);
        using var uniformBuffer = new WebGpuBuffer(uniformParams, WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        using var bindGroup = new WebGpuBindGroup(pipelineId, aBuffer, bBuffer, cBuffer);
        await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(
            pipelineId, bindGroup.BindGroupId, uniformBuffer.BufferId, workgroupsX, workgroupsY, 1);
        await WebGpuNativeBindings.SubmitAndWaitAsync();
    }

    #endregion

    #region Helper Methods

    private async Task<int> GetOrCreatePipelineAsync(string moduleName, string source, string entryPoint)
    {
        var key = $"{moduleName}::{entryPoint}";
        if (_pipelineCache.TryGetValue(key, out var pipelineId))
        {
            return pipelineId;
        }

        pipelineId = await _shaderLibrary.GetOrCreatePipelineAsync(moduleName, source, entryPoint);
        _pipelineCache[key] = pipelineId;
        return pipelineId;
    }

    private void ThrowIfNotInitialized()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(WebGpuBackend));
        }

        if (!_initialized)
        {
            throw new InvalidOperationException("WebGPU backend not initialized. Call InitializeAsync() first.");
        }
    }

    #endregion

    // Synchronous wrappers are defined in WebGpuBackend.GpuBackend.cs as part of IDirectGpuBackend implementation

    /// <summary>
    /// Disposes the WebGPU backend and releases resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _shaderLibrary.Dispose();
        _pipelineCache.Clear();
    }

    public override string ToString()
    {
        return $"WebGpuBackend[Available={IsAvailable}, Device={DeviceName}]";
    }
}
#endif
