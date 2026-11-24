// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Tensor operations that return a task instead of blocking: the engine surface for a device that cannot be waited
/// on synchronously, such as WebGPU in a browser.
/// </summary>
/// <remarks>
/// <para>
/// This is a sibling of <see cref="IEngine"/>, not a part of it: an engine that computes synchronously has no reason
/// to return tasks, and one whose device completes work through an event loop cannot honour a synchronous contract.
/// <see cref="CpuEngine"/> implements both (its tasks are already complete), so code written against this interface
/// runs on any device. <see cref="AsyncEngine.CreateAsync"/> picks the best engine available.
/// </para>
/// <para>
/// Each operation means exactly what the <see cref="IEngine"/> member of the same name without the Async suffix
/// means, including broadcasting and gradient recording on an active <see cref="Autodiff.GradientTape{T}"/>. The
/// cancellation token is checked before work is queued; work already on the device runs to completion.
/// </para>
/// </remarks>
public interface IAsyncEngine
{
    /// <summary>Elementwise a + b (see <see cref="IEngine.TensorAdd{T}"/>).</summary>
    ValueTask<Tensor<T>> TensorAddAsync<T>(Tensor<T> a, Tensor<T> b, CancellationToken cancellationToken = default);

    /// <summary>Elementwise a - b (see <see cref="IEngine.TensorSubtract{T}"/>).</summary>
    ValueTask<Tensor<T>> TensorSubtractAsync<T>(Tensor<T> a, Tensor<T> b, CancellationToken cancellationToken = default);

    /// <summary>Elementwise a * b (see <see cref="IEngine.TensorMultiply{T}"/>).</summary>
    ValueTask<Tensor<T>> TensorMultiplyAsync<T>(Tensor<T> a, Tensor<T> b, CancellationToken cancellationToken = default);

    /// <summary>Elementwise a / b (see <see cref="IEngine.TensorDivide{T}"/>).</summary>
    ValueTask<Tensor<T>> TensorDivideAsync<T>(Tensor<T> a, Tensor<T> b, CancellationToken cancellationToken = default);

    /// <summary>Matrix product (see <see cref="IEngine.TensorMatMul{T}"/>).</summary>
    ValueTask<Tensor<T>> TensorMatMulAsync<T>(Tensor<T> a, Tensor<T> b, CancellationToken cancellationToken = default);

    /// <summary>Elementwise exp (see <see cref="IEngine.TensorExp{T}"/>).</summary>
    ValueTask<Tensor<T>> TensorExpAsync<T>(Tensor<T> tensor, CancellationToken cancellationToken = default);

    /// <summary>Elementwise natural log (see <see cref="IEngine.TensorLog{T}"/>).</summary>
    ValueTask<Tensor<T>> TensorLogAsync<T>(Tensor<T> tensor, CancellationToken cancellationToken = default);

    /// <summary>Elementwise square root (see <see cref="IEngine.TensorSqrt{T}"/>).</summary>
    ValueTask<Tensor<T>> TensorSqrtAsync<T>(Tensor<T> tensor, CancellationToken cancellationToken = default);

    /// <summary>Elementwise absolute value (see <see cref="IEngine.TensorAbs{T}"/>).</summary>
    ValueTask<Tensor<T>> TensorAbsAsync<T>(Tensor<T> tensor, CancellationToken cancellationToken = default);

    /// <summary>Elementwise negation (see <see cref="IEngine.TensorNegate{T}"/>).</summary>
    ValueTask<Tensor<T>> TensorNegateAsync<T>(Tensor<T> tensor, CancellationToken cancellationToken = default);

    /// <summary>Elementwise max(x, 0) (see <see cref="IEngine.ReLU{T}(Tensor{T})"/>).</summary>
    ValueTask<Tensor<T>> ReLUAsync<T>(Tensor<T> tensor, CancellationToken cancellationToken = default);

    /// <summary>Elementwise logistic (see <see cref="IEngine.Sigmoid{T}(Tensor{T})"/>).</summary>
    ValueTask<Tensor<T>> SigmoidAsync<T>(Tensor<T> tensor, CancellationToken cancellationToken = default);

    /// <summary>Elementwise tanh (see <see cref="IEngine.Tanh{T}(Tensor{T})"/>).</summary>
    ValueTask<Tensor<T>> TanhAsync<T>(Tensor<T> tensor, CancellationToken cancellationToken = default);
}
