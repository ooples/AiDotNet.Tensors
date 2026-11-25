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

    private sealed unsafe class PinnedGradientStaging : IGradientHostStaging
    {
        private readonly CudaBackend _owner;
        private readonly int _floats;
        private readonly int _bytes;
        private IntPtr _host;

        internal PinnedGradientStaging(CudaBackend owner, int floats)
        {
            _owner = owner;
            _floats = floats;
            _bytes = floats * sizeof(float);
            _host = owner._pinnedPool.Rent(_bytes);
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

        public void EnqueueUpload(IGpuBuffer buffer, int offset, int count)
        {
            Check(buffer, offset, count);
            if (count == 0) return;
            using var _ = _owner.PushContext();
            CuBlasNative.CheckCudaResult(
                CudaNativeBindings.cuMemcpyHtoDAsync(buffer.Handle, _host + offset * sizeof(float),
                    (ulong)count * sizeof(float), _owner._stream),
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
                _owner._pinnedPool.Return(_host, _bytes);
                _host = IntPtr.Zero;
            }
        }
    }
}
