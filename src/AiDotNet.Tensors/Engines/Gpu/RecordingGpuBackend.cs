using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu.Graph;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// GPU backend wrapper that records operations to an ExecutionGraphBuilder
/// instead of executing them immediately when in recording mode.
/// </summary>
/// <remarks>
/// <para>
/// This implements the Decorator pattern to intercept GPU operations and record them
/// to an execution graph for deferred, optimized execution.
/// </para>
/// <para><b>Usage:</b></para>
/// <code>
/// var recording = new RecordingGpuBackend(actualBackend);
/// recording.BeginRecording(graphBuilder);
///
/// // Operations are recorded, not executed
/// recording.Gemm(a, b, c, M, N, K);
/// recording.Relu(input, output, size);
///
/// recording.EndRecording();
/// // Graph can now be compiled and executed
/// </code>
/// <para><b>Buffer Lifetime Contract:</b></para>
/// <para>
/// When recording operations, buffer references are captured by the execution graph nodes.
/// Callers MUST NOT dispose or reuse buffers passed to recorded operations until after
/// the graph has been executed. Disposing buffers before execution will result in undefined
/// behavior when the graph runs.
/// </para>
/// <para>
/// This is the standard contract for deferred execution systems - buffers must remain valid
/// from recording time until execution completes. Use <see cref="DeferredScope"/> for
/// automatic lifetime management when possible.
/// </para>
/// </remarks>
public class RecordingGpuBackend : DelegatingGpuBackend
{
    private ExecutionGraphBuilder? _graphBuilder;
    private bool _isRecording;

    /// <summary>
    /// Gets whether the backend is currently recording operations.
    /// </summary>
    public bool IsRecording => _isRecording;

    /// <summary>
    /// Gets the current graph builder, if recording.
    /// </summary>
    public ExecutionGraphBuilder? GraphBuilder => _graphBuilder;

    /// <summary>
    /// Creates a new recording GPU backend wrapper.
    /// </summary>
    /// <param name="inner">The actual backend to delegate to when not recording.</param>
    public RecordingGpuBackend(IDirectGpuBackend inner) : base(inner)
    {
    }

    /// <summary>
    /// Begins recording operations to the specified graph builder.
    /// </summary>
    /// <param name="graphBuilder">The graph builder to record operations to.</param>
    public void BeginRecording(ExecutionGraphBuilder graphBuilder)
    {
        if (_isRecording)
        {
            throw new InvalidOperationException("Already recording. Call EndRecording() first.");
        }

        _graphBuilder = graphBuilder ?? throw new ArgumentNullException(nameof(graphBuilder));
        _isRecording = true;
    }

    /// <summary>
    /// Ends recording mode. Operations will execute immediately after this.
    /// </summary>
    public void EndRecording()
    {
        _isRecording = false;
        _graphBuilder = null;
    }

    #region Helper Methods for Recording

    /// <summary>
    /// Creates a GPU tensor wrapper for a buffer to use in graph nodes.
    /// </summary>
    private Tensor<float> CreateTensorWrapper(IGpuBuffer buffer, GpuTensorRole role = GpuTensorRole.Intermediate)
    {
        return Tensor<float>.FromGpuBuffer(Inner, buffer, new[] { buffer.Size }, role, ownsBuffer: false);
    }

    /// <summary>
    /// Records a kernel operation or executes it immediately based on recording state.
    /// </summary>
    private void RecordOrExecute(
        KernelType kernelType,
        IGpuBuffer[] inputs,
        IGpuBuffer[] outputs,
        Action executeAction,
        Dictionary<string, object>? parameters = null)
    {
        if (_isRecording && _graphBuilder != null)
        {
            var inputTensors = inputs.Select(b => CreateTensorWrapper(b, GpuTensorRole.Input)).ToArray();
            var outputTensors = outputs.Select(b => CreateTensorWrapper(b, GpuTensorRole.Output)).ToArray();

            // Capture the action to execute later
            Action<IDirectGpuBackend, IGpuStream?> deferredAction = (backend, stream) => executeAction();

            _graphBuilder.AddKernel(kernelType, inputTensors, outputTensors, deferredAction, parameters);
        }
        else
        {
            executeAction();
        }
    }

    #endregion

    #region GEMM Operations (Override to Record)

    /// <inheritdoc/>
    public override void Gemm(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
    {
        RecordOrExecute(
            KernelType.Gemm,
            new[] { A, B },
            new[] { C },
            () => Inner.Gemm(A, B, C, M, N, K, alpha, beta),
            new Dictionary<string, object>
            {
                ["M"] = M,
                ["N"] = N,
                ["K"] = K,
                ["alpha"] = alpha,
                ["beta"] = beta
            });
    }

    /// <inheritdoc/>
    public override IGpuBuffer MatMul(IGpuBuffer A, IGpuBuffer B, int M, int N, int K)
    {
        if (_isRecording && _graphBuilder != null)
        {
            // Allocate output buffer
            var output = Inner.AllocateBuffer(M * N);

            RecordOrExecute(
                KernelType.Gemm,
                new[] { A, B },
                new[] { output },
                () => Inner.Gemm(A, B, output, M, N, K, 1.0f, 0.0f),
                new Dictionary<string, object> { ["M"] = M, ["N"] = N, ["K"] = K });

            return output;
        }

        return Inner.MatMul(A, B, M, N, K);
    }

    /// <inheritdoc/>
    public override void BatchedGemm(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, int batchCount, float alpha = 1.0f, float beta = 0.0f)
    {
        RecordOrExecute(
            KernelType.BatchedGemm,
            new[] { A, B },
            new[] { C },
            () => Inner.BatchedGemm(A, B, C, M, N, K, batchCount, alpha, beta),
            new Dictionary<string, object>
            {
                ["M"] = M,
                ["N"] = N,
                ["K"] = K,
                ["batchCount"] = batchCount,
                ["alpha"] = alpha,
                ["beta"] = beta
            });
    }

    #endregion

    #region Fused Operations (Override to Record)

    /// <inheritdoc/>
    public override IGpuBuffer GemmBiasRelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        if (_isRecording && _graphBuilder != null)
        {
            var output = Inner.AllocateBuffer(M * N);

            var inputTensors = new[] { A, B, bias }.Select(b => CreateTensorWrapper(b, GpuTensorRole.Input)).ToArray();
            var outputTensor = CreateTensorWrapper(output, GpuTensorRole.Output);

            Action<IDirectGpuBackend, IGpuStream?> action = (backend, stream) =>
            {
                var result = backend.GemmBiasRelu(A, B, bias, M, N, K);
                backend.Copy(result, output, M * N);
                result.Dispose();
            };

            var node = FusedKernelNode.CreateGemmBiasActivation(
                inputTensors[0], inputTensors[1], inputTensors[2], outputTensor,
                M, N, K, FusedActivationType.ReLU, action);

            _graphBuilder.AddNode(node);
            return output;
        }

        return Inner.GemmBiasRelu(A, B, bias, M, N, K);
    }

    /// <inheritdoc/>
    public override IGpuBuffer GemmBiasGelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        if (_isRecording && _graphBuilder != null)
        {
            var output = Inner.AllocateBuffer(M * N);

            var inputTensors = new[] { A, B, bias }.Select(b => CreateTensorWrapper(b, GpuTensorRole.Input)).ToArray();
            var outputTensor = CreateTensorWrapper(output, GpuTensorRole.Output);

            Action<IDirectGpuBackend, IGpuStream?> action = (backend, stream) =>
            {
                var result = backend.GemmBiasGelu(A, B, bias, M, N, K);
                backend.Copy(result, output, M * N);
                result.Dispose();
            };

            var node = FusedKernelNode.CreateGemmBiasActivation(
                inputTensors[0], inputTensors[1], inputTensors[2], outputTensor,
                M, N, K, FusedActivationType.GELU, action);

            _graphBuilder.AddNode(node);
            return output;
        }

        return Inner.GemmBiasGelu(A, B, bias, M, N, K);
    }

    /// <inheritdoc/>
    public override IGpuBuffer GemmBiasSigmoid(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        if (_isRecording && _graphBuilder != null)
        {
            var output = Inner.AllocateBuffer(M * N);

            var inputTensors = new[] { A, B, bias }.Select(b => CreateTensorWrapper(b, GpuTensorRole.Input)).ToArray();
            var outputTensor = CreateTensorWrapper(output, GpuTensorRole.Output);

            Action<IDirectGpuBackend, IGpuStream?> action = (backend, stream) =>
            {
                var result = backend.GemmBiasSigmoid(A, B, bias, M, N, K);
                backend.Copy(result, output, M * N);
                result.Dispose();
            };

            var node = FusedKernelNode.CreateGemmBiasActivation(
                inputTensors[0], inputTensors[1], inputTensors[2], outputTensor,
                M, N, K, FusedActivationType.Sigmoid, action);

            _graphBuilder.AddNode(node);
            return output;
        }

        return Inner.GemmBiasSigmoid(A, B, bias, M, N, K);
    }

    /// <inheritdoc/>
    public override IGpuBuffer GemmBiasTanh(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        if (_isRecording && _graphBuilder != null)
        {
            var output = Inner.AllocateBuffer(M * N);

            var inputTensors = new[] { A, B, bias }.Select(b => CreateTensorWrapper(b, GpuTensorRole.Input)).ToArray();
            var outputTensor = CreateTensorWrapper(output, GpuTensorRole.Output);

            Action<IDirectGpuBackend, IGpuStream?> action = (backend, stream) =>
            {
                var result = backend.GemmBiasTanh(A, B, bias, M, N, K);
                backend.Copy(result, output, M * N);
                result.Dispose();
            };

            var node = FusedKernelNode.CreateGemmBiasActivation(
                inputTensors[0], inputTensors[1], inputTensors[2], outputTensor,
                M, N, K, FusedActivationType.Tanh, action);

            _graphBuilder.AddNode(node);
            return output;
        }

        return Inner.GemmBiasTanh(A, B, bias, M, N, K);
    }

    /// <inheritdoc/>
    public override IGpuBuffer GemmBias(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
    {
        if (_isRecording && _graphBuilder != null)
        {
            var output = Inner.AllocateBuffer(M * N);

            var inputTensors = new[] { A, B, bias }.Select(b => CreateTensorWrapper(b, GpuTensorRole.Input)).ToArray();
            var outputTensor = CreateTensorWrapper(output, GpuTensorRole.Output);

            Action<IDirectGpuBackend, IGpuStream?> action = (backend, stream) =>
            {
                var result = backend.GemmBias(A, B, bias, M, N, K);
                backend.Copy(result, output, M * N);
                result.Dispose();
            };

            var node = FusedKernelNode.CreateGemmBiasActivation(
                inputTensors[0], inputTensors[1], inputTensors[2], outputTensor,
                M, N, K, FusedActivationType.None, action);

            _graphBuilder.AddNode(node);
            return output;
        }

        return Inner.GemmBias(A, B, bias, M, N, K);
    }

    #endregion

    #region Element-wise Operations (Override to Record)

    /// <inheritdoc/>
    public override void Add(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        RecordOrExecute(
            KernelType.ElementWise,
            new[] { A, B },
            new[] { C },
            () => Inner.Add(A, B, C, size),
            new Dictionary<string, object> { ["op"] = "add", ["size"] = size });
    }

    /// <inheritdoc/>
    public override void Subtract(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        RecordOrExecute(
            KernelType.ElementWise,
            new[] { A, B },
            new[] { C },
            () => Inner.Subtract(A, B, C, size),
            new Dictionary<string, object> { ["op"] = "subtract", ["size"] = size });
    }

    /// <inheritdoc/>
    public override void Multiply(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        RecordOrExecute(
            KernelType.ElementWise,
            new[] { A, B },
            new[] { C },
            () => Inner.Multiply(A, B, C, size),
            new Dictionary<string, object> { ["op"] = "multiply", ["size"] = size });
    }

    /// <inheritdoc/>
    public override void Divide(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        RecordOrExecute(
            KernelType.ElementWise,
            new[] { A, B },
            new[] { C },
            () => Inner.Divide(A, B, C, size),
            new Dictionary<string, object> { ["op"] = "divide", ["size"] = size });
    }

    /// <inheritdoc/>
    public override void Scale(IGpuBuffer A, IGpuBuffer B, float scalar, int size)
    {
        RecordOrExecute(
            KernelType.ElementWise,
            new[] { A },
            new[] { B },
            () => Inner.Scale(A, B, scalar, size),
            new Dictionary<string, object> { ["op"] = "scale", ["scalar"] = scalar, ["size"] = size });
    }

    #endregion

    #region Activation Functions (Override to Record)

    /// <inheritdoc/>
    public override void Relu(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (_isRecording && _graphBuilder != null)
        {
            var inputTensor = CreateTensorWrapper(A, GpuTensorRole.Input);
            var outputTensor = CreateTensorWrapper(B, GpuTensorRole.Output);

            _graphBuilder.AddActivation(inputTensor, outputTensor, FusedActivationType.ReLU,
                (backend, stream) => backend.Relu(A, B, size));
        }
        else
        {
            Inner.Relu(A, B, size);
        }
    }

    /// <inheritdoc/>
    public override void Sigmoid(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (_isRecording && _graphBuilder != null)
        {
            var inputTensor = CreateTensorWrapper(A, GpuTensorRole.Input);
            var outputTensor = CreateTensorWrapper(B, GpuTensorRole.Output);

            _graphBuilder.AddActivation(inputTensor, outputTensor, FusedActivationType.Sigmoid,
                (backend, stream) => backend.Sigmoid(A, B, size));
        }
        else
        {
            Inner.Sigmoid(A, B, size);
        }
    }

    /// <inheritdoc/>
    public override void Tanh(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (_isRecording && _graphBuilder != null)
        {
            var inputTensor = CreateTensorWrapper(A, GpuTensorRole.Input);
            var outputTensor = CreateTensorWrapper(B, GpuTensorRole.Output);

            _graphBuilder.AddActivation(inputTensor, outputTensor, FusedActivationType.Tanh,
                (backend, stream) => backend.Tanh(A, B, size));
        }
        else
        {
            Inner.Tanh(A, B, size);
        }
    }

    /// <inheritdoc/>
    public override void Gelu(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (_isRecording && _graphBuilder != null)
        {
            var inputTensor = CreateTensorWrapper(A, GpuTensorRole.Input);
            var outputTensor = CreateTensorWrapper(B, GpuTensorRole.Output);

            _graphBuilder.AddActivation(inputTensor, outputTensor, FusedActivationType.GELU,
                (backend, stream) => backend.Gelu(A, B, size));
        }
        else
        {
            Inner.Gelu(A, B, size);
        }
    }

    /// <inheritdoc/>
    public override void Softmax(IGpuBuffer A, IGpuBuffer B, int batchSize, int features)
    {
        if (_isRecording && _graphBuilder != null)
        {
            var inputTensor = CreateTensorWrapper(A, GpuTensorRole.Input);
            var outputTensor = CreateTensorWrapper(B, GpuTensorRole.Output);

            _graphBuilder.AddActivation(inputTensor, outputTensor, FusedActivationType.Softmax,
                (backend, stream) => backend.Softmax(A, B, batchSize, features));
        }
        else
        {
            Inner.Softmax(A, B, batchSize, features);
        }
    }

    /// <inheritdoc/>
    public override void LeakyRelu(IGpuBuffer A, IGpuBuffer B, float alpha, int size)
    {
        if (_isRecording && _graphBuilder != null)
        {
            var inputTensor = CreateTensorWrapper(A, GpuTensorRole.Input);
            var outputTensor = CreateTensorWrapper(B, GpuTensorRole.Output);

            _graphBuilder.AddActivation(inputTensor, outputTensor, FusedActivationType.LeakyReLU,
                (backend, stream) => backend.LeakyRelu(A, B, alpha, size));
        }
        else
        {
            Inner.LeakyRelu(A, B, alpha, size);
        }
    }

    /// <inheritdoc/>
    public override void Swish(IGpuBuffer A, IGpuBuffer B, int size)
    {
        if (_isRecording && _graphBuilder != null)
        {
            var inputTensor = CreateTensorWrapper(A, GpuTensorRole.Input);
            var outputTensor = CreateTensorWrapper(B, GpuTensorRole.Output);

            _graphBuilder.AddActivation(inputTensor, outputTensor, FusedActivationType.Swish,
                (backend, stream) => backend.Swish(A, B, size));
        }
        else
        {
            Inner.Swish(A, B, size);
        }
    }

    #endregion

    #region Convolution Operations (Override to Record)

    /// <inheritdoc/>
    public override void Conv2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW)
    {
        RecordOrExecute(
            KernelType.Conv2D,
            new[] { input, kernel },
            new[] { output },
            () => Inner.Conv2D(input, kernel, output, batch, inChannels, inHeight, inWidth,
                outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW),
            new Dictionary<string, object>
            {
                ["batch"] = batch,
                ["inChannels"] = inChannels,
                ["inHeight"] = inHeight,
                ["inWidth"] = inWidth,
                ["outChannels"] = outChannels,
                ["outHeight"] = outHeight,
                ["outWidth"] = outWidth,
                ["kernelH"] = kernelH,
                ["kernelW"] = kernelW,
                ["strideH"] = strideH,
                ["strideW"] = strideW,
                ["padH"] = padH,
                ["padW"] = padW,
                ["dilationH"] = dilationH,
                ["dilationW"] = dilationW
            });
    }

    #endregion

    #region Normalization Operations (Override to Record)

    /// <inheritdoc/>
    public override void BatchNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer runningMean, IGpuBuffer runningVar, IGpuBuffer saveMean, IGpuBuffer saveInvVar,
        int batch, int channels, int spatialSize, float epsilon, float momentum, bool training)
    {
        RecordOrExecute(
            KernelType.BatchNorm,
            new[] { input, gamma, beta, runningMean, runningVar },
            new[] { output, saveMean, saveInvVar },
            () => Inner.BatchNorm(input, output, gamma, beta, runningMean, runningVar, saveMean, saveInvVar,
                batch, channels, spatialSize, epsilon, momentum, training),
            new Dictionary<string, object>
            {
                ["batch"] = batch,
                ["channels"] = channels,
                ["spatialSize"] = spatialSize,
                ["epsilon"] = epsilon,
                ["momentum"] = momentum,
                ["training"] = training
            });
    }

    /// <inheritdoc/>
    public override void LayerNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batchSize, int normalizedSize, float epsilon)
    {
        RecordOrExecute(
            KernelType.LayerNorm,
            new[] { input, gamma, beta },
            new[] { output, saveMean, saveInvVar },
            () => Inner.LayerNorm(input, output, gamma, beta, saveMean, saveInvVar, batchSize, normalizedSize, epsilon),
            new Dictionary<string, object>
            {
                ["batchSize"] = batchSize,
                ["normalizedSize"] = normalizedSize,
                ["epsilon"] = epsilon
            });
    }

    /// <inheritdoc/>
    public override void GroupNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batch, int numGroups, int channels, int spatialSize, float epsilon)
    {
        RecordOrExecute(
            KernelType.GroupNorm,
            new[] { input, gamma, beta },
            new[] { output, saveMean, saveInvVar },
            () => Inner.GroupNorm(input, output, gamma, beta, saveMean, saveInvVar, batch, numGroups, channels, spatialSize, epsilon),
            new Dictionary<string, object>
            {
                ["batch"] = batch,
                ["numGroups"] = numGroups,
                ["channels"] = channels,
                ["spatialSize"] = spatialSize,
                ["epsilon"] = epsilon
            });
    }

    /// <inheritdoc/>
    public override void InstanceNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batch, int channels, int spatialSize, float epsilon)
    {
        RecordOrExecute(
            KernelType.InstanceNorm,
            new[] { input, gamma, beta },
            new[] { output, saveMean, saveInvVar },
            () => Inner.InstanceNorm(input, output, gamma, beta, saveMean, saveInvVar, batch, channels, spatialSize, epsilon),
            new Dictionary<string, object>
            {
                ["batch"] = batch,
                ["channels"] = channels,
                ["spatialSize"] = spatialSize,
                ["epsilon"] = epsilon
            });
    }

    #endregion

    #region Attention Operations (Override to Record)

    /// <inheritdoc/>
    public override void FlashAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? mask, int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal)
    {
        var inputs = mask != null
            ? new[] { query, key, value, mask }
            : new[] { query, key, value };

        RecordOrExecute(
            KernelType.Attention,
            inputs,
            new[] { output },
            () => Inner.FlashAttention(query, key, value, output, mask, batch, numHeads, seqLen, headDim, scale, isCausal),
            new Dictionary<string, object>
            {
                ["batch"] = batch,
                ["numHeads"] = numHeads,
                ["seqLen"] = seqLen,
                ["headDim"] = headDim,
                ["scale"] = scale,
                ["isCausal"] = isCausal
            });
    }

    /// <inheritdoc/>
    public override void ScaledDotProductAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? attentionWeights, IGpuBuffer? mask,
        int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal)
    {
        var inputs = new List<IGpuBuffer> { query, key, value };
        if (mask != null) inputs.Add(mask);

        var outputs = new List<IGpuBuffer> { output };
        if (attentionWeights != null) outputs.Add(attentionWeights);

        RecordOrExecute(
            KernelType.Attention,
            inputs.ToArray(),
            outputs.ToArray(),
            () => Inner.ScaledDotProductAttention(query, key, value, output, attentionWeights, mask,
                batch, numHeads, seqLen, headDim, scale, isCausal),
            new Dictionary<string, object>
            {
                ["batch"] = batch,
                ["numHeads"] = numHeads,
                ["seqLen"] = seqLen,
                ["headDim"] = headDim,
                ["scale"] = scale,
                ["isCausal"] = isCausal
            });
    }

    #endregion

    #region Memory Operations (Override to Record Transfers)

    /// <inheritdoc/>
    public override IGpuBuffer AllocateBuffer(float[] data)
    {
        var buffer = Inner.AllocateBuffer(data);

        if (_isRecording && _graphBuilder != null)
        {
            // Record the upload operation
            _graphBuilder.AddUpload(data, buffer);
        }

        return buffer;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Note: Download operations execute immediately even during recording, as the caller
    /// needs the data synchronously. The download is also recorded for graph tracking purposes.
    /// For fully deferred downloads, use the execution graph's deferred download functionality.
    /// </remarks>
    public override float[] DownloadBuffer(IGpuBuffer buffer)
    {
        if (_isRecording && _graphBuilder != null)
        {
            // Record the download operation for tracking (but execute immediately since caller needs data)
            _graphBuilder.AddDownload(buffer, buffer.Size);
        }

        return Inner.DownloadBuffer(buffer);
    }

    /// <summary>
    /// Records a deferred download operation that will execute during graph execution.
    /// </summary>
    /// <param name="buffer">The GPU buffer to download.</param>
    /// <param name="size">The number of elements to download.</param>
    /// <returns>A deferred download handle to retrieve the data after execution.</returns>
    /// <exception cref="InvalidOperationException">Thrown when not in recording mode.</exception>
    /// <remarks>
    /// Unlike <see cref="DownloadBuffer"/>, this method does not execute immediately.
    /// The download is deferred until the execution graph is executed.
    /// Use <see cref="DeferredDownload.GetResult"/> after graph execution to retrieve the data.
    /// </remarks>
    public DeferredDownload DownloadBufferDeferred(IGpuBuffer buffer, int size)
    {
        if (!_isRecording || _graphBuilder == null)
        {
            throw new InvalidOperationException(
                "DownloadBufferDeferred can only be called during recording. " +
                "Use DownloadBuffer for immediate downloads outside of recording mode.");
        }

        var transferNode = _graphBuilder.AddDownloadWithHandle(buffer, size);
        return new DeferredDownload(transferNode);
    }

    /// <inheritdoc/>
    public override void Copy(IGpuBuffer source, IGpuBuffer destination, int size)
    {
        if (_isRecording && _graphBuilder != null)
        {
            _graphBuilder.AddCopy(source, destination, size);
        }
        else
        {
            Inner.Copy(source, destination, size);
        }
    }

    /// <inheritdoc/>
    public override void NearestNeighborUpsample(IGpuBuffer input, IGpuBuffer output, int batchChannels, int height, int width, int scaleFactor)
    {
        // #642: record UNet upsample so it replays in order in a deferred graph (was eager mid-record).
        RecordOrExecute(KernelType.ElementWise, new[] { input }, new[] { output },
            () => Inner.NearestNeighborUpsample(input, output, batchChannels, height, width, scaleFactor));
    }

    /// <inheritdoc/>
    public override void ConvTranspose2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int outputPadH, int outputPadW)
    {
        // #642: record transposed-conv (UNet decoder upsample) for deferred replay.
        RecordOrExecute(KernelType.Conv2D, new[] { input, kernel }, new[] { output },
            () => Inner.ConvTranspose2D(input, kernel, output, batch, inChannels, inHeight, inWidth,
                outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, outputPadH, outputPadW));
    }

    /// <inheritdoc/>
    public override void MaxPool2D(IGpuBuffer input, IGpuBuffer output, IGpuBuffer? indices,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth, int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW)
    {
        // #642: record downsample pool for deferred replay. `indices` (optional, used by backward) is
        // also an output so a later read of it is ordered after this node.
        var outputs = indices is null ? new[] { output } : new[] { output, indices };
        RecordOrExecute(KernelType.Pooling, new[] { input }, outputs,
            () => Inner.MaxPool2D(input, output, indices, batch, channels, inHeight, inWidth,
                outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW));
    }

    /// <inheritdoc/>
    public override void AvgPool2D(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth, int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW, bool countIncludePad)
    {
        // #642: record downsample pool for deferred replay. The earlier intermittent crash was a
        // CudaBackend.AvgPool2D launch bug (dropped the countIncludePad kernel arg) — now fixed.
        RecordOrExecute(KernelType.Pooling, new[] { input }, new[] { output },
            () => Inner.AvgPool2D(input, output, batch, channels, inHeight, inWidth,
                outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, countIncludePad));
    }

    /// <inheritdoc/>
    public override void Transpose(IGpuBuffer A, IGpuBuffer B, int rows, int cols)
    {
        // #642: record 2-D transpose (attention reshapes) for deferred replay.
        RecordOrExecute(KernelType.Transpose, new[] { A }, new[] { B },
            () => Inner.Transpose(A, B, rows, cols));
    }

    /// <inheritdoc/>
    public override void GlobalAvgPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int height, int width)
    {
        // #642: record global/adaptive avg-pool (SE blocks, attention pooling) for deferred replay.
        RecordOrExecute(KernelType.Pooling, new[] { input }, new[] { output },
            () => Inner.GlobalAvgPool2D(input, output, batch, channels, height, width));
    }

    /// <inheritdoc/>
    public override void GlobalMaxPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int height, int width)
    {
        // #642: record global max-pool for deferred replay.
        RecordOrExecute(KernelType.Pooling, new[] { input }, new[] { output },
            () => Inner.GlobalMaxPool2D(input, output, batch, channels, height, width));
    }

    /// <inheritdoc/>
    public override void AdaptiveAvgPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth)
    {
        // #642: record adaptive avg-pool for deferred replay.
        RecordOrExecute(KernelType.Pooling, new[] { input }, new[] { output },
            () => Inner.AdaptiveAvgPool2D(input, output, batch, channels, inHeight, inWidth, outHeight, outWidth));
    }

    /// <inheritdoc/>
    public override void Embedding(IGpuBuffer indices, IGpuBuffer embeddingTable, IGpuBuffer output, int numIndices, int embeddingDim)
    {
        // #642: record embedding gather (conditioning / token lookup) for deferred replay. The table
        // is an input; indices is a separate (int) input the kernel reads.
        RecordOrExecute(KernelType.ElementWise, new[] { indices, embeddingTable }, new[] { output },
            () => Inner.Embedding(indices, embeddingTable, output, numIndices, embeddingDim));
    }

    /// <inheritdoc/>
    public override void Conv2DBiasAdd(IGpuBuffer output, IGpuBuffer bias, int batch, int channels, int spatialSize)
    {
        // #642: in-place per-channel bias add after conv — `output` is both read and written, so it is
        // the node's input AND output (dependency tracker chains it after the conv that produced it).
        RecordOrExecute(KernelType.ElementWise, new[] { output, bias }, new[] { output },
            () => Inner.Conv2DBiasAdd(output, bias, batch, channels, spatialSize));
    }

    /// <inheritdoc/>
    public override void Copy(IGpuBuffer source, int sourceOffset, IGpuBuffer destination, int destinationOffset, int length)
    {
        // #642: record the offset device-to-device copy as a kernel node (input=source,
        // output=destination) so the dependency tracker orders it after the producer of `source`
        // and before any reader of `destination`. This is the primitive a deferred-correct Concat
        // composes from — ConcatAxis lives on IGpuBatchExecution, which the recording backend does
        // NOT wrap, so concat must route through IDirectGpuBackend.Copy (which it does). AddCopy's
        // TransferNode carries no offsets, hence RecordOrExecute with the offsets in the closure.
        RecordOrExecute(
            KernelType.ElementWise,
            new[] { source },
            new[] { destination },
            () => Inner.Copy(source, sourceOffset, destination, destinationOffset, length));
    }

    #endregion
}

