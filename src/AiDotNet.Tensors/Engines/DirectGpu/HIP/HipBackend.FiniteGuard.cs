// Copyright (c) AiDotNet. All rights reserved.
// NOTE: validated only by an NVRTC compile of the kernels -- the dev box has no AMD GPU.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

// #1519: the guarded optimizer updates read the multi-tensor sum of squares (a double, written by
// multi_tensor_sum_squares) on the device, so a compiled step's finiteness verdict needs no host read; and a readback
// that waits only for its own copy.
public sealed partial class HipBackend
{
    private bool DeviceGuardReady =>
        _kernelCache.ContainsKey("multi_tensor_sum_squares") && _kernelCache.ContainsKey("sgd_update_guarded")
        && _kernelCache.ContainsKey("adam_update_guarded") && _kernelCache.ContainsKey("adamw_update_guarded");

    private unsafe void AdamGuardedOnDevice(bool decoupledWeightDecay,
        IReadOnlyList<IGpuBuffer> parameters, IReadOnlyList<IGpuBuffer> gradients,
        IReadOnlyList<IGpuBuffer> firstMoments, IReadOnlyList<IGpuBuffer> secondMoments,
        IReadOnlyList<int> sizes, float learningRate, float beta1, float beta2, float epsilon,
        float weightDecay, int step, IGpuBuffer finiteGuard)
    {
        PerTensorMultiTensorOptimizer.ValidateGuard(finiteGuard);
        PerTensorMultiTensorOptimizer.ValidateAdam(parameters, gradients, firstMoments, secondMoments, sizes, epsilon, step);
        var kernel = _kernelCache[decoupledWeightDecay ? "adamw_update_guarded" : "adam_update_guarded"];
        IntPtr guard = finiteGuard.Handle;
        void** args = stackalloc void*[12];
        for (int t = 0; t < parameters.Count; t++)
        {
            IntPtr p = parameters[t].Handle, g = gradients[t].Handle, m = firstMoments[t].Handle, v = secondMoments[t].Handle;
            int size = sizes[t];
            args[0] = &p; args[1] = &g; args[2] = &m; args[3] = &v;
            args[4] = &learningRate; args[5] = &beta1; args[6] = &beta2; args[7] = &epsilon;
            args[8] = &weightDecay; args[9] = &step; args[10] = &size; args[11] = &guard;
            LaunchKernel(kernel, (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
        }
    }

    private unsafe void SgdGuardedOnDevice(IReadOnlyList<IGpuBuffer> parameters, IReadOnlyList<IGpuBuffer> gradients,
        IReadOnlyList<int> sizes, float learningRate, float weightDecay, IGpuBuffer finiteGuard)
    {
        PerTensorMultiTensorOptimizer.ValidateGuard(finiteGuard);
        PerTensorMultiTensorOptimizer.ValidateSgd(parameters, gradients, sizes);
        var kernel = _kernelCache["sgd_update_guarded"];
        IntPtr guard = finiteGuard.Handle;
        void** args = stackalloc void*[6];
        for (int t = 0; t < parameters.Count; t++)
        {
            IntPtr p = parameters[t].Handle, g = gradients[t].Handle;
            int size = sizes[t];
            args[0] = &p; args[1] = &g; args[2] = &learningRate; args[3] = &weightDecay; args[4] = &size; args[5] = &guard;
            LaunchKernel(kernel, (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
        }
    }

    // A hipMemcpyAsync into pinned host memory behind the work already queued on the backend's stream, plus an event the
    // host waits on only in Read.
    private sealed class HipDeviceReadback : IDeviceReadback
    {
        private readonly HipBackend _backend;
        private readonly int _capacity;
        private readonly byte[] _host;
        private IntPtr _pinned;
        private IntPtr _event;
        private int _bytes;
        private bool _queued;
        private bool _unread;
        private bool _pending;

        public HipDeviceReadback(HipBackend backend, int byteCapacity)
        {
            if (byteCapacity <= 0) throw new ArgumentOutOfRangeException(nameof(byteCapacity));
            _backend = backend ?? throw new ArgumentNullException(nameof(backend));
            _capacity = byteCapacity;
            _host = new byte[byteCapacity];
            var err = HipNativeBindings.hipHostMalloc(ref _pinned, (UIntPtr)byteCapacity, 0);
            if (err != HipError.Success) throw new InvalidOperationException($"hipHostMalloc(readback) failed: {err}");
            err = HipNativeBindings.hipEventCreate(ref _event);
            if (err != HipError.Success)
            {
                HipNativeBindings.hipHostFree(_pinned);
                _pinned = IntPtr.Zero;
                throw new InvalidOperationException($"hipEventCreate(readback) failed: {err}");
            }
        }

        public bool HasUnreadValue => _unread;

        public void Enqueue(IGpuBuffer source, int byteCount)
        {
            if (_pinned == IntPtr.Zero) throw new ObjectDisposedException(nameof(HipDeviceReadback));
            if (source is null) throw new ArgumentNullException(nameof(source));
            if (byteCount <= 0 || byteCount > _capacity) throw new ArgumentOutOfRangeException(nameof(byteCount));
            if ((long)source.Size * sizeof(float) < byteCount)
                throw new ArgumentOutOfRangeException(nameof(byteCount),
                    $"The buffer holds {source.Size} floats, fewer than {byteCount} bytes.");
            // The previous copy into the same pinned bytes is ordered ahead of this one on the stream.
            var err = HipNativeBindings.hipMemcpyAsync(_pinned, source.Handle, (UIntPtr)byteCount,
                HipMemcpyKind.DeviceToHost, _backend._stream);
            if (err != HipError.Success) throw new InvalidOperationException($"hipMemcpyAsync(readback) failed: {err}");
            err = HipNativeBindings.hipEventRecord(_event, _backend._stream);
            if (err != HipError.Success) throw new InvalidOperationException($"hipEventRecord(readback) failed: {err}");
            _bytes = byteCount;
            _queued = true;
            _unread = true;
            _pending = true;
        }

        public ReadOnlySpan<byte> Read()
        {
            if (_pinned == IntPtr.Zero) throw new ObjectDisposedException(nameof(HipDeviceReadback));
            if (!_queued) throw new InvalidOperationException("Nothing was queued for readback.");
            if (_pending)
            {
                var err = HipNativeBindings.hipEventSynchronize(_event);
                if (err != HipError.Success) throw new InvalidOperationException($"hipEventSynchronize(readback) failed: {err}");
                _pending = false;
            }
            Marshal.Copy(_pinned, _host, 0, _bytes);
            _unread = false;
            return new ReadOnlySpan<byte>(_host, 0, _bytes);
        }

        public void Dispose()
        {
            if (_pinned == IntPtr.Zero) return;
            // The copy may still be writing the pinned bytes; they are freed only once it is done.
            if (_pending && _backend.IsAvailable) HipNativeBindings.hipEventSynchronize(_event);
            HipNativeBindings.hipEventDestroy(_event);
            HipNativeBindings.hipHostFree(_pinned);
            _event = IntPtr.Zero;
            _pinned = IntPtr.Zero;
        }
    }
}
