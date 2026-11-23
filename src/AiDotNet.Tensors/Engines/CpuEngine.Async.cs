// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

// The CPU computes on the calling thread, so each task is already complete when it is returned. The synchronous
// members are virtual: an engine derived from this one answers through its own overrides.
public partial class CpuEngine : IAsyncEngine
{
    /// <inheritdoc/>
    public ValueTask<Tensor<T>> TensorAddAsync<T>(Tensor<T> a, Tensor<T> b, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return new ValueTask<Tensor<T>>(TensorAdd(a, b));
    }

    /// <inheritdoc/>
    public ValueTask<Tensor<T>> TensorSubtractAsync<T>(Tensor<T> a, Tensor<T> b, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return new ValueTask<Tensor<T>>(TensorSubtract(a, b));
    }

    /// <inheritdoc/>
    public ValueTask<Tensor<T>> TensorMultiplyAsync<T>(Tensor<T> a, Tensor<T> b, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return new ValueTask<Tensor<T>>(TensorMultiply(a, b));
    }

    /// <inheritdoc/>
    public ValueTask<Tensor<T>> TensorDivideAsync<T>(Tensor<T> a, Tensor<T> b, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return new ValueTask<Tensor<T>>(TensorDivide(a, b));
    }

    /// <inheritdoc/>
    public ValueTask<Tensor<T>> TensorMatMulAsync<T>(Tensor<T> a, Tensor<T> b, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return new ValueTask<Tensor<T>>(TensorMatMul(a, b));
    }

    /// <inheritdoc/>
    public ValueTask<Tensor<T>> TensorExpAsync<T>(Tensor<T> tensor, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return new ValueTask<Tensor<T>>(TensorExp(tensor));
    }

    /// <inheritdoc/>
    public ValueTask<Tensor<T>> TensorLogAsync<T>(Tensor<T> tensor, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return new ValueTask<Tensor<T>>(TensorLog(tensor));
    }

    /// <inheritdoc/>
    public ValueTask<Tensor<T>> TensorSqrtAsync<T>(Tensor<T> tensor, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return new ValueTask<Tensor<T>>(TensorSqrt(tensor));
    }

    /// <inheritdoc/>
    public ValueTask<Tensor<T>> TensorAbsAsync<T>(Tensor<T> tensor, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return new ValueTask<Tensor<T>>(TensorAbs(tensor));
    }

    /// <inheritdoc/>
    public ValueTask<Tensor<T>> TensorNegateAsync<T>(Tensor<T> tensor, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return new ValueTask<Tensor<T>>(TensorNegate(tensor));
    }

    /// <inheritdoc/>
    public ValueTask<Tensor<T>> ReLUAsync<T>(Tensor<T> tensor, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return new ValueTask<Tensor<T>>(ReLU(tensor));
    }

    /// <inheritdoc/>
    public ValueTask<Tensor<T>> SigmoidAsync<T>(Tensor<T> tensor, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return new ValueTask<Tensor<T>>(Sigmoid(tensor));
    }

    /// <inheritdoc/>
    public ValueTask<Tensor<T>> TanhAsync<T>(Tensor<T> tensor, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return new ValueTask<Tensor<T>>(Tanh(tensor));
    }
}
