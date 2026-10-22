#if NET7_0_OR_GREATER
// Copyright (c) AiDotNet. All rights reserved.
using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// #1522: the multi-tensor sum of squares and the guarded optimizer updates on the device, so a compiled step's
// finiteness verdict needs no host read (WebGpuFiniteGuardKernels), and a readback that waits only for its own copy.
public sealed partial class WebGpuBackend
{
    private const string FiniteGuardModule = "FiniteGuard";
    private const string GuardedOptimizerModule = "OptimizerGuarded";

    private void GuardDispatch(string kernel, IGpuBuffer first, IGpuBuffer accumulator, int n, float maxNorm, int workSize)
        => DispatchNBufferAsync(FiniteGuardModule, WebGpuFiniteGuardKernels.Source, kernel, new[] { first, accumulator },
            MultiTensorParams(n, maxNorm), workSize).GetAwaiter().GetResult();

    // Turns the float sum and non-finite mark in the buffer's two slots into the double they stand for.
    internal void FinishSumOfSquaresDouble(IGpuBuffer accumulator)
        => GuardDispatch("guard_sum_squares_finish", SharedDummyBuffer, accumulator, 1, 0f, 1);

    private void AdamGuardedOnDevice(bool decoupledWeightDecay,
        IReadOnlyList<IGpuBuffer> parameters, IReadOnlyList<IGpuBuffer> gradients,
        IReadOnlyList<IGpuBuffer> firstMoments, IReadOnlyList<IGpuBuffer> secondMoments,
        IReadOnlyList<int> sizes, float learningRate, float beta1, float beta2, float epsilon,
        float weightDecay, int step, IGpuBuffer finiteGuard)
    {
        PerTensorMultiTensorOptimizer.ValidateGuard(finiteGuard);
        PerTensorMultiTensorOptimizer.ValidateAdam(parameters, gradients, firstMoments, secondMoments, sizes, epsilon, step);
        for (int t = 0; t < parameters.Count; t++)
            DispatchNBufferAsync(GuardedOptimizerModule, WebGpuFiniteGuardKernels.GuardedOptimizerSource,
                decoupledWeightDecay ? "adamw" : "adam",
                new[] { parameters[t], gradients[t], firstMoments[t], secondMoments[t], finiteGuard },
                MakeOptimizerUniforms(sizes[t], learningRate, beta1, beta2, epsilon, weightDecay, step), sizes[t])
                .GetAwaiter().GetResult();
    }

    private void SgdGuardedOnDevice(IReadOnlyList<IGpuBuffer> parameters, IReadOnlyList<IGpuBuffer> gradients,
        IReadOnlyList<int> sizes, float learningRate, float weightDecay, IGpuBuffer finiteGuard)
    {
        PerTensorMultiTensorOptimizer.ValidateGuard(finiteGuard);
        PerTensorMultiTensorOptimizer.ValidateSgd(parameters, gradients, sizes);
        var dummy = SharedDummyBuffer;
        for (int t = 0; t < parameters.Count; t++)
            DispatchNBufferAsync(GuardedOptimizerModule, WebGpuFiniteGuardKernels.GuardedOptimizerSource, "sgd",
                new[] { parameters[t], gradients[t], dummy, dummy, finiteGuard },
                MakeOptimizerUniforms(sizes[t], learningRate, 0, 0, 0, weightDecay, 0), sizes[t]).GetAwaiter().GetResult();
    }

    private sealed class WebGpuDeviceReadback : IDeviceReadback
    {
        private readonly WebGpuBackend _backend;
        private readonly int _capacity;
        private byte[] _host = Array.Empty<byte>();
        private int _token;
        private int _bytes;
        private bool _queued;
        private bool _unread;
        private bool _disposed;

        public WebGpuDeviceReadback(WebGpuBackend backend, int byteCapacity)
        {
            if (byteCapacity <= 0) throw new ArgumentOutOfRangeException(nameof(byteCapacity));
            _backend = backend ?? throw new ArgumentNullException(nameof(backend));
            _capacity = byteCapacity;
        }

        public bool HasUnreadValue => _unread;

        public void Enqueue(IGpuBuffer source, int byteCount)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(WebGpuDeviceReadback));
            if (source is null) throw new ArgumentNullException(nameof(source));
            if (byteCount <= 0 || byteCount > _capacity) throw new ArgumentOutOfRangeException(nameof(byteCount));
            var buffer = AsWebGpu(source);
            if ((long)buffer.Size * sizeof(float) < byteCount)
                throw new ArgumentOutOfRangeException(nameof(byteCount),
                    $"The buffer holds {buffer.Size} floats, fewer than {byteCount} bytes.");
            // A copy still unread is superseded: finish it so its staging buffer is released.
            Drain();
            int token = WebGpuNativeBindings.BeginReadback(buffer.BufferId, byteCount, buffer.OffsetBytes);
            if (token == 0) throw new InvalidOperationException("The WebGPU readback could not be started.");
            _token = token;
            _bytes = byteCount;
            _queued = true;
            _unread = true;
        }

        public ReadOnlySpan<byte> Read()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(WebGpuDeviceReadback));
            if (!_queued) throw new InvalidOperationException("Nothing was queued for readback.");
            Drain();
            _unread = false;
            return new ReadOnlySpan<byte>(_host, 0, Math.Min(_bytes, _host.Length));
        }

        private void Drain()
        {
            if (_token == 0) return;
            int token = _token;
            _token = 0;
            _host = WebGpuNativeBindings.FinishReadbackAsync(token).GetAwaiter().GetResult();
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (_backend.IsAvailable) Drain();
        }
    }
}
#endif
