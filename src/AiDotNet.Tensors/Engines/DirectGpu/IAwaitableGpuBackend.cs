// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System.Threading.Tasks;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// A GPU backend whose operations complete through a task instead of blocking the caller: the sibling of
/// <see cref="IDirectGpuBackend"/>, for a device that cannot be waited on synchronously.
/// </summary>
/// <remarks>
/// <para>
/// WebGPU in a browser hands every submission's completion to the page's event loop. On single-threaded WebAssembly
/// that loop runs on the same thread as the caller, so a synchronous wait for the device can never finish. A backend
/// for such a device implements this interface and not <see cref="IDirectGpuBackend"/>: no caller can then reach a
/// blocking wait through it.
/// </para>
/// <para>
/// Each operation's task completes once the device has run it, so a buffer an operation wrote can be read, or passed
/// to the next operation, after awaiting it. Arguments mean what they mean on the matching
/// <see cref="IDirectGpuBackend"/> member.
/// </para>
/// </remarks>
public interface IAwaitableGpuBackend : IGpuDevice
{
    /// <summary>Reads a buffer's float elements back to the host.</summary>
    Task<float[]> DownloadBufferAsync(IGpuBuffer buffer);

    /// <summary>C = A + B, elementwise over <paramref name="size"/> elements.</summary>
    Task AddAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size);

    /// <summary>C = A - B, elementwise.</summary>
    Task SubtractAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size);

    /// <summary>C = A * B, elementwise.</summary>
    Task MultiplyAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size);

    /// <summary>C = A / B, elementwise.</summary>
    Task DivideAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size);

    /// <summary>B = exp(A).</summary>
    Task ExpAsync(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>B = log(A).</summary>
    Task LogAsync(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>B = sqrt(A).</summary>
    Task SqrtAsync(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>B = |A|.</summary>
    Task AbsAsync(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>B = -A.</summary>
    Task NegateAsync(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>B = max(A, 0).</summary>
    Task ReluAsync(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>B = 1 / (1 + exp(-A)).</summary>
    Task SigmoidAsync(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>B = tanh(A).</summary>
    Task TanhAsync(IGpuBuffer A, IGpuBuffer B, int size);

    /// <summary>C = alpha * A[M,K] x B[K,N] + beta * C[M,N], row-major.</summary>
    Task GemmAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f);
}
