// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// Device loss scaling (CudaGradScalerKernels) over a gradient table the scaler owns, so a captured graph that replays it
// is unaffected when another caller replaces the shared table.
public sealed partial class CudaBackend : IDeviceGradScalerFactory
{
    private readonly object _gradScalerModuleLock = new();
    private bool _gradScalerModuleTried;
    private bool _gradScalerModuleReady;

    private bool EnsureGradScalerModule()
    {
        lock (_gradScalerModuleLock)
        {
            if (_gradScalerModuleTried) return _gradScalerModuleReady;
            _gradScalerModuleTried = true;
            try
            {
                using var _ = PushContext();
                if (CudaNativeBindings.cuCtxGetDevice(out int device) != CudaResult.Success) return false;
                CompileKernelModule(device, Kernels.CudaGradScalerKernels.GetSource(), "grad_scaler_kernels",
                    Kernels.CudaGradScalerKernels.GetKernelNames());
                foreach (var name in Kernels.CudaGradScalerKernels.GetKernelNames())
                    if (!_kernelCache.ContainsKey(name)) return false;
                _gradScalerModuleReady = true;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.TraceWarning($"Device loss scaling unavailable: {ex.GetType().Name}: {ex.Message}");
            }
            return _gradScalerModuleReady;
        }
    }

    IDeviceGradScaler? IDeviceGradScalerFactory.CreateGradScaler(IReadOnlyList<IGpuBuffer> gradients, IReadOnlyList<int> sizes,
        DeviceGradScalerSettings settings, Compilation.LossScaleState initial)
    {
        if (gradients is null) throw new ArgumentNullException(nameof(gradients));
        if (sizes is null || sizes.Count != gradients.Count) throw new ArgumentException("One size per gradient.", nameof(sizes));
        if (!IsAvailable || gradients.Count == 0 || IsStreamCapturing() || !EnsureGradScalerModule()) return null;
        var table = BuildGradientTable(gradients, sizes);
        IGpuBuffer? state = null;
        try
        {
            state = AllocateBuffer(Kernels.CudaGradScalerKernels.StateWords);
            var scaler = new CudaDeviceGradScaler(this, table, state, settings);
            scaler.Write(initial);
            return scaler;
        }
        catch
        {
            state?.Dispose();
            table.Dispose();
            throw;
        }
    }

    private sealed unsafe class CudaDeviceGradScaler : IDeviceGradScaler
    {
        private readonly CudaBackend _backend;
        private MultiTensorTable? _table;
        private IGpuBuffer? _state;
        private readonly DeviceGradScalerSettings _settings;

        public CudaDeviceGradScaler(CudaBackend backend, MultiTensorTable table, IGpuBuffer state, DeviceGradScalerSettings settings)
        {
            _backend = backend;
            _table = table;
            _state = state;
            _settings = settings;
        }

        public bool Covers(IReadOnlyList<IGpuBuffer> gradients, IReadOnlyList<int> sizes)
            => _table is { } table && table.Matches(gradients, sizes);

        private IGpuBuffer State => _state ?? throw new ObjectDisposedException(nameof(CudaDeviceGradScaler));

        public void UnscaleAndUpdate(IGpuBuffer seed, int seedLength)
        {
            var table = _table ?? throw new ObjectDisposedException(nameof(CudaDeviceGradScaler));
            if (seed is null) throw new ArgumentNullException(nameof(seed));
            if (seedLength <= 0 || seed.Size < seedLength) throw new ArgumentOutOfRangeException(nameof(seedLength));
            if (!_backend._kernelCache.TryGetValue("gs_unscale", out var unscale)
                || !_backend._kernelCache.TryGetValue("gs_update", out var update))
                throw new GpuCapabilityUnavailableException("CUDA kernels not found: gs_unscale / gs_update");
            using var _ = _backend.PushContext();
            IntPtr pH = table.Pointers!.Handle, sH = table.SizesBuffer!.Handle, ctH = table.ChunkTensor!.Handle,
                csH = table.ChunkStart!.Handle, stH = State.Handle, seedH = seed.Handle;
            int total = table.TotalChunks, chunk = GradientTableChunk;
            void** a = stackalloc void*[7];
            a[0] = &pH; a[1] = &sH; a[2] = &ctH; a[3] = &csH; a[4] = &total; a[5] = &chunk; a[6] = &stH;
            _backend.LaunchKernel(unscale, (uint)Math.Max(1, Math.Min(total, MultiTensorScaleMaxBlocks)), 256, a);
            float growth = _settings.GrowthFactor, backoff = _settings.BackoffFactor, min = _settings.MinScale,
                max = _settings.MaxScale;
            int interval = _settings.GrowthInterval, dynamic = _settings.Dynamic ? 1 : 0, n = seedLength;
            void** u = stackalloc void*[9];
            u[0] = &stH; u[1] = &seedH; u[2] = &n; u[3] = &growth; u[4] = &backoff; u[5] = &interval; u[6] = &min;
            u[7] = &max; u[8] = &dynamic;
            _backend.LaunchKernel(update, 1, 32, u);
        }

        public void WriteSeed(IGpuBuffer seed, int seedLength)
        {
            if (seed is null) throw new ArgumentNullException(nameof(seed));
            if (seedLength <= 0 || seed.Size < seedLength) throw new ArgumentOutOfRangeException(nameof(seedLength));
            if (!_backend._kernelCache.TryGetValue("gs_seed", out var kernel))
                throw new GpuCapabilityUnavailableException("CUDA kernel not found: gs_seed");
            using var _ = _backend.PushContext();
            IntPtr stH = State.Handle, seedH = seed.Handle;
            int n = seedLength;
            void** a = stackalloc void*[3];
            a[0] = &stH; a[1] = &seedH; a[2] = &n;
            _backend.LaunchKernel(kernel, 1, 32, a);
        }

        public Compilation.LossScaleState Read()
        {
            float[] words = _backend.DownloadBuffer(State);
            uint Word(int i) => BitConverter.ToUInt32(BitConverter.GetBytes(words[i]), 0);
            return new Compilation.LossScaleState(words[0], (int)Word(1), Word(3), Word(4));
        }

        public void Write(Compilation.LossScaleState state)
        {
            if (_backend.IsStreamCapturing())
                throw new InvalidOperationException("The loss scale cannot be uploaded during stream capture.");
            var words = new float[Kernels.CudaGradScalerKernels.StateWords];
            static float Bits(uint v) => BitConverter.ToSingle(BitConverter.GetBytes(v), 0);
            words[0] = state.Scale;
            words[1] = Bits((uint)Math.Max(0, state.GrowthTracker));
            words[3] = Bits((uint)Math.Min(uint.MaxValue, Math.Max(0L, state.OverflowSteps)));
            words[4] = Bits((uint)Math.Min(uint.MaxValue, Math.Max(0L, state.Steps)));
            _backend.UploadBufferInPlace(words, State);
        }

        public void Dispose()
        {
            _state?.Dispose();
            _state = null;
            _table?.Dispose();
            _table = null;
        }
    }
}
