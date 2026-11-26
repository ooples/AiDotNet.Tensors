// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IGradientHostStagingSource
{
    IGradientHostStaging? IGradientHostStagingSource.TryCreateGradientHostStaging(int floats)
    {
        if (floats <= 0 || _disposed || !IsAvailable) return null;
        if ((long)floats * sizeof(float) > int.MaxValue) return null;
        return new PinnedGradientStaging(this, floats);
    }

    IGradientHostStaging? IGradientHostStagingSource.TryRegisterGradientHostStaging(IntPtr host, int floats,
        out string? failure)
    {
        failure = null;
        if (host == IntPtr.Zero || floats <= 0) { failure = "no host memory"; return null; }
        if (_disposed || !IsAvailable) { failure = "the CUDA backend is not available"; return null; }
        ulong bytes = (ulong)floats * sizeof(float);
        using var _ = PushContext();
        var result = CudaNativeBindings.cuMemHostRegister(host, bytes, 1u /* CU_MEMHOSTREGISTER_PORTABLE */);
        if (result != CudaResult.Success) { failure = "cuMemHostRegister returned " + result; return null; }
        return new PinnedGradientStaging(this, host, floats);
    }

    private sealed unsafe class PinnedGradientStaging : IGradientHostStaging
    {
        private readonly CudaBackend _owner;
        private readonly int _floats;
        private readonly int _bytes;
        private IntPtr _host;

        // Registered: the memory is the caller's (page-locked in place); disposing unregisters it instead of pooling it.
        private readonly bool _registered;

        internal PinnedGradientStaging(CudaBackend owner, int floats)
        {
            _owner = owner;
            _floats = floats;
            _bytes = floats * sizeof(float);
            _host = owner._pinnedPool.Rent(_bytes);
        }

        internal PinnedGradientStaging(CudaBackend owner, IntPtr registeredHost, int floats)
        {
            _owner = owner;
            _floats = floats;
            _bytes = floats * sizeof(float);
            _host = registeredHost;
            _registered = true;
        }

        public Span<float> Host
        {
            get
            {
                if (_host == IntPtr.Zero) throw new ObjectDisposedException(nameof(PinnedGradientStaging));
                return new Span<float>((void*)_host, _floats);
            }
        }

        private void Check(IGpuBuffer buffer, int offset, int count)
        {
            if (_host == IntPtr.Zero) throw new ObjectDisposedException(nameof(PinnedGradientStaging));
            if (buffer is null) throw new ArgumentNullException(nameof(buffer));
            if (buffer.Handle == IntPtr.Zero)
                throw new ObjectDisposedException(nameof(buffer), "GPU buffer was released before its staged transfer.");
            if (offset < 0 || count < 0 || (long)offset + count > _floats)
                throw new ArgumentOutOfRangeException(nameof(count), "The transfer does not fit the staging area.");
            if (count > buffer.Size)
                throw new ArgumentOutOfRangeException(nameof(count), "The transfer is longer than the device buffer.");
            GpuFloatLayout.Require(buffer, nameof(IGradientHostStaging));
        }

        public void EnqueueDownload(IGpuBuffer buffer, int offset, int count)
        {
            Check(buffer, offset, count);
            if (count == 0) return;
            using var _ = _owner.PushContext();
            CuBlasNative.CheckCudaResult(
                CudaNativeBindings.cuMemcpyDtoHAsync(_host + offset * sizeof(float), buffer.Handle,
                    (ulong)count * sizeof(float), _owner._stream),
                "cuMemcpyDtoHAsync(gradient staging)");
        }

        public void EnqueueUpload(IGpuBuffer buffer, int offset, int count) => EnqueueUploadAt(buffer, 0, offset, count);

        public void EnqueueUploadAt(IGpuBuffer buffer, int bufferOffset, int hostOffset, int count)
        {
            Check(buffer, hostOffset, count);
            if (bufferOffset < 0 || (long)bufferOffset + count > buffer.Size)
                throw new ArgumentOutOfRangeException(nameof(bufferOffset), "The transfer does not fit the device buffer.");
            if (count == 0) return;
            using var _ = _owner.PushContext();
            CuBlasNative.CheckCudaResult(
                CudaNativeBindings.cuMemcpyHtoDAsync(buffer.Handle + bufferOffset * sizeof(float),
                    _host + hostOffset * sizeof(float), (ulong)count * sizeof(float), _owner._stream),
                "cuMemcpyHtoDAsync(gradient staging)");
        }

        public void WaitForTransfers()
        {
            if (_host == IntPtr.Zero) throw new ObjectDisposedException(nameof(PinnedGradientStaging));
            using var _ = _owner.PushContext();
            CuBlasNative.CheckCudaResult(CudaNativeBindings.cuStreamSynchronize(_owner._stream),
                "cuStreamSynchronize(gradient staging)");
        }

        public void Dispose()
        {
            if (_host == IntPtr.Zero) return;
            try
            {
                if (!_owner._disposed)
                {
                    // In-flight uploads still read the area: it goes back to the pool only once they have finished.
                    using var _ = _owner.PushContext();
                    CudaNativeBindings.cuStreamSynchronize(_owner._stream);
                }
            }
            finally
            {
                if (_registered)
                {
                    if (!_owner._disposed)
                    {
                        using var _ = _owner.PushContext();
                        CudaNativeBindings.cuMemHostUnregister(_host);
                    }
                }
                else
                {
                    _owner._pinnedPool.Return(_host, _bytes);
                }
                _host = IntPtr.Zero;
            }
        }
    }
}
