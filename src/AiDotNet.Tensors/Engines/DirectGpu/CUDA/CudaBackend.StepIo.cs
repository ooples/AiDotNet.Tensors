// Copyright (c) AiDotNet. All rights reserved.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// A captured step's inputs, loss and progress through page-locked memory (IGraphStepIo, CudaStepIoKernels). On this
// machine's WDDM driver every launch, copy or event record outside a graph cost the host 10-25 us; the parity LSTM's
// per-step input refresh alone was seven such calls per input.
public sealed partial class CudaBackend
{
    private const int StepIoMaxInputs = Kernels.CudaStepIoKernels.MaxInputs;

    private readonly object _stepIoModuleLock = new();
    private bool _stepIoModuleTried;
    private bool _stepIoModuleReady;

    private bool EnsureStepIoModule()
    {
        lock (_stepIoModuleLock)
        {
            if (_stepIoModuleTried) return _stepIoModuleReady;
            _stepIoModuleTried = true;
            try
            {
                using var _ = PushContext();
                if (CudaNativeBindings.cuCtxGetDevice(out int device) != CudaResult.Success) return false;
                CompileKernelModule(device, Kernels.CudaStepIoKernels.GetSource(), "step_io_kernels",
                    Kernels.CudaStepIoKernels.GetKernelNames());
                foreach (var name in Kernels.CudaStepIoKernels.GetKernelNames())
                    if (!_kernelCache.ContainsKey(name)) return false;
                _stepIoModuleReady = true;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.TraceWarning($"Graph step I/O unavailable: {ex.GetType().Name}: {ex.Message}");
            }
            return _stepIoModuleReady;
        }
    }

    IGraphStepIo? IGpuGraphCapture.CreateStepIo(IReadOnlyList<IGpuBuffer> inputs, IReadOnlyList<int> lengths)
    {
        if (inputs is null) throw new ArgumentNullException(nameof(inputs));
        if (lengths is null || lengths.Count != inputs.Count) throw new ArgumentException("One length per input.", nameof(lengths));
        if (!IsAvailable || inputs.Count > StepIoMaxInputs || Environment.GetEnvironmentVariable("AIDOTNET_GRAPH_STEP_IO") == "0"
            || !EnsureStepIoModule())
            return null;
        long total = 0;
        for (int i = 0; i < inputs.Count; i++)
        {
            if (inputs[i] is null || inputs[i].Handle == IntPtr.Zero || lengths[i] <= 0 || lengths[i] > inputs[i].Size) return null;
            total += lengths[i];
        }
        if (total > int.MaxValue) return null;
        return new CudaGraphStepIo(this, inputs, lengths);
    }

    [StructLayout(LayoutKind.Sequential)]
    private unsafe struct StepIoArgs
    {
        public fixed ulong Src0[StepIoMaxInputs];
        public fixed ulong Src1[StepIoMaxInputs];
        public fixed ulong Dst[StepIoMaxInputs];
        public fixed int Start[StepIoMaxInputs + 1];
        public int Count;
        public ulong LrTable0;
        public ulong LrTable1;
        public ulong DynScalars;
        public ulong State;
        public fixed int Len[StepIoMaxInputs];
        public int Vector;
        public ulong Guard;
        public ulong Masks;
        public ulong CopySrc;
        public ulong CopyDst;
        public int CopyCount;
    }

    // Learning rates a slot's table holds: one per step number a replay may run as while earlier replays' finiteness
    // verdicts are still unknown to the host (see CompiledTrainingPlan's captured optimizer).
    private const int StepIoLrCapacity = 8;

    // Must match StepIoResults in CudaStepIoKernels.
    private const int StepIoLossCapacity = 64;
    private const int StepIoResultsBytes = 16 + 2 * 8 + 2 * StepIoLossCapacity * sizeof(float);

    private sealed unsafe class CudaGraphStepIo : IGraphStepIo
    {
        private readonly CudaBackend _backend;
        private readonly IGpuBuffer[] _inputs;
        private readonly int[] _lengths;
        private readonly IntPtr[] _staging;   // [input * 2 + slot], host addresses
        private readonly IntPtr[] _stagingDevice;
        private IntPtr _results;              // page-locked StepIoResults
        private IntPtr _masks;                // page-locked uint[2][MaskCopies][MaskStrideWords]: each slot's copy mask
        private const int MaskWords = 2 * Kernels.CudaStepIoKernels.MaskCopies * Kernels.CudaStepIoKernels.MaskStrideWords;
        private IntPtr _masksDevice;
        private IntPtr _resultsDevice;
        private IGpuBuffer? _state;           // device uint[4]: replays, applied, fault, unused
        private IntPtr[] _lrTable = new IntPtr[2];        // page-locked [first step, n, rates...] per slot
        private IntPtr[] _lrTableDevice = new IntPtr[2];
        private IGpuBuffer? _dynamicScalars;  // device float[2]: the step's learning rate and step number
        private IGpuBuffer? _guard;           // device double: the gradients' sum of squares
        private uint _initialApplied;
        private MultiTensorTable? _gradientTable;   // private to this graph (see PrepareOptimizer)
        private MultiTensorTable? _adamTable;
        private readonly MultiTensorTable?[] _pingPong = new MultiTensorTable?[2];   // A -> B, B -> A (PrepareSgdPingPong)
        private IGpuBuffer? _pingPongPartials;   // a double per chunk
        private IGpuBuffer? _pingPongDone;       // the kernel's finished-block counter, zero between launches
        private IGpuBuffer? _copySource, _copyDestination;   // the prologue's per-replay device copy (TrySetPrologueCopy)
        private int _copyCount;
        private bool _disposed;

        // The prologue's grid covers its copy in one pass up to 1024 blocks of 256 threads; past that it strides.
        private const int PrologueCopyMaxFloats = 1 << 20;

        public CudaGraphStepIo(CudaBackend backend, IReadOnlyList<IGpuBuffer> inputs, IReadOnlyList<int> lengths)
        {
            _backend = backend;
            _inputs = new IGpuBuffer[inputs.Count];
            _lengths = new int[inputs.Count];
            _staging = new IntPtr[2 * inputs.Count];
            _stagingDevice = new IntPtr[2 * inputs.Count];
            using var _ = backend.PushContext();
            try
            {
                for (int i = 0; i < inputs.Count; i++)
                {
                    _inputs[i] = inputs[i];
                    _lengths[i] = lengths[i];
                    for (int s = 0; s < 2; s++)
                    {
                        ulong bytes = (ulong)_lengths[i] * sizeof(float);
                        CuBlasNative.CheckCudaResult(CuBlasNative.cuMemAllocHost(out _staging[2 * i + s], bytes), "cuMemAllocHost(step io)");
                        new Span<byte>((void*)_staging[2 * i + s], checked((int)bytes)).Clear();
                        CuBlasNative.CheckCudaResult(
                            CudaNativeBindings.cuMemHostGetDevicePointer(out _stagingDevice[2 * i + s], _staging[2 * i + s], 0),
                            "cuMemHostGetDevicePointer(step io)");
                    }
                }
                CuBlasNative.CheckCudaResult(CuBlasNative.cuMemAllocHost(out _results, StepIoResultsBytes), "cuMemAllocHost(step io results)");
                new Span<byte>((void*)_results, StepIoResultsBytes).Clear();
                CuBlasNative.CheckCudaResult(CudaNativeBindings.cuMemHostGetDevicePointer(out _resultsDevice, _results, 0),
                    "cuMemHostGetDevicePointer(step io results)");
                CuBlasNative.CheckCudaResult(CuBlasNative.cuMemAllocHost(out _masks, MaskWords * sizeof(uint)), "cuMemAllocHost(step io masks)");
                new Span<uint>((void*)_masks, MaskWords).Fill(uint.MaxValue);
                CuBlasNative.CheckCudaResult(CudaNativeBindings.cuMemHostGetDevicePointer(out _masksDevice, _masks, 0),
                    "cuMemHostGetDevicePointer(step io masks)");
                _state = backend.AllocateBuffer(4);
                ((IDeviceZeroFill)backend).ZeroBuffer(_state, 4);
            }
            catch
            {
                Dispose();
                throw;
            }
        }

        public int InputCount => _inputs.Length;

        public int InputLength(int input) => _lengths[input];

        public IntPtr InputStaging(int input, int slot) => _staging[2 * input + (slot & 1)];

        public void SetInputsToCopy(uint replay, uint mask)
        {
            if (_masks == IntPtr.Zero) throw new ObjectDisposedException(nameof(CudaGraphStepIo));
            const int copies = Kernels.CudaStepIoKernels.MaskCopies, stride = Kernels.CudaStepIoKernels.MaskStrideWords;
            uint* slot = (uint*)_masks + (replay & 1) * copies * stride;
            if (Volatile.Read(ref slot[0]) == mask) return;
            for (int c = 0; c < copies; c++) Volatile.Write(ref slot[c * stride], mask);
        }

        public uint CompletedReplays
        {
            get
            {
                if (_results == IntPtr.Zero) throw new ObjectDisposedException(nameof(CudaGraphStepIo));
                uint done = Volatile.Read(ref *(uint*)_results);
                if (Volatile.Read(ref *((uint*)_results + 2)) != 0)
                    throw new InvalidOperationException(
                        "A captured step's learning-rate table did not cover the device's step number (step I/O bookkeeping fault).");
                return done;
            }
        }

        public void WaitForReplays(uint replays)
        {
            if (CompletedReplays >= replays) return;
            // The wait is a replay or two of device time (tens to hundreds of microseconds): spin, then yield the core,
            // but never sleep. SpinWait.SpinOnce escalates to Thread.Sleep(1), which on Windows' default timer
            // resolution sleeps a whole ~15.6 ms tick (measured: 7.4 ms per step once the host ran ahead of the device).
            var waited = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 1; CompletedReplays < replays; i++)
            {
                if (i < 64) Thread.SpinWait(32);
                else Thread.Yield();
                if ((i & 1023) == 0 && waited.ElapsedMilliseconds > 10_000)
                {
                    // No progress for 10 s: surface the device's own error if it has one.
                    using (_backend.PushContext())
                        CuBlasNative.CheckCudaResult(CudaNativeBindings.cuStreamSynchronize(_backend._mainStream), "cuStreamSynchronize(step io)");
                    if (CompletedReplays >= replays) return;
                    throw new InvalidOperationException(
                        $"The captured step did not complete replay {replays} (completed {CompletedReplays}).");
                }
            }
        }

        public int LearningRateCapacity => StepIoLrCapacity;

        public void InitializeAppliedSteps(uint applied)
        {
            EnsureOptimizerBuffers();   // the prologue (recorded first) resolves the learning rate into them
            _initialApplied = applied;
            using var _ = _backend.PushContext();
            // state[0] = replays done (the caller waited for every launched replay), state[1] = applied.
            var bytes = new byte[16];
            BitConverter.GetBytes(CompletedReplays).CopyTo(bytes, 0);
            BitConverter.GetBytes(applied).CopyTo(bytes, 4);
            _backend.UploadByteBuffer(_state!, bytes);
            // The published count is the host's view until the next replay completes: it must say the same.
            Volatile.Write(ref *((uint*)_results + 1), applied);
        }

        public uint AppliedSteps
        {
            get
            {
                if (_results == IntPtr.Zero) throw new ObjectDisposedException(nameof(CudaGraphStepIo));
                if (Volatile.Read(ref *(uint*)_results) == 0) return _initialApplied;
                return Volatile.Read(ref *((uint*)_results + 1));
            }
        }

        public double SumOfSquares(uint replay)
        {
            if (_results == IntPtr.Zero) throw new ObjectDisposedException(nameof(CudaGraphStepIo));
            return Volatile.Read(ref *((double*)((byte*)_results + 16) + (replay & 1)));
        }

        public void WriteLearningRates(uint replay, int firstStep, ReadOnlySpan<float> rates)
        {
            if (rates.Length < 1 || rates.Length > StepIoLrCapacity) throw new ArgumentOutOfRangeException(nameof(rates));
            EnsureOptimizerBuffers();
            int* table = (int*)_lrTable[replay & 1];
            table[0] = firstStep;
            table[1] = rates.Length;
            rates.CopyTo(new Span<float>(table + 2, rates.Length));
        }

        private void EnsureOptimizerBuffers()
        {
            if (_dynamicScalars is not null) return;
            using var _ = _backend.PushContext();
            for (int s = 0; s < 2; s++)
            {
                ulong bytes = 8 + StepIoLrCapacity * sizeof(float);
                CuBlasNative.CheckCudaResult(CuBlasNative.cuMemAllocHost(out _lrTable[s], bytes), "cuMemAllocHost(step io lr)");
                new Span<byte>((void*)_lrTable[s], (int)bytes).Clear();
                ((int*)_lrTable[s])[1] = 1;
                CuBlasNative.CheckCudaResult(CudaNativeBindings.cuMemHostGetDevicePointer(out _lrTableDevice[s], _lrTable[s], 0),
                    "cuMemHostGetDevicePointer(step io lr)");
            }
            _dynamicScalars = _backend.AllocateBuffer(2);
            _guard = _backend.AllocateBuffer(2);
            ((IDeviceZeroFill)_backend).ZeroBuffer(_guard, 2);   // the captured reduction accumulates onto it
        }

        public void PrepareOptimizer(IReadOnlyList<IGpuBuffer> parameters, IReadOnlyList<IGpuBuffer> gradients,
            IReadOnlyList<IGpuBuffer>? firstMoments, IReadOnlyList<IGpuBuffer>? secondMoments, IReadOnlyList<int> sizes)
        {
            EnsureOptimizerBuffers();
            _gradientTable?.Dispose();
            _gradientTable = _backend.BuildGradientTable(gradients, sizes);
            _adamTable?.Dispose();
            _adamTable = firstMoments is not null && secondMoments is not null
                ? _backend.BuildAdamTable(parameters, gradients, firstMoments, secondMoments, sizes)
                : null;
        }

        public void CaptureSgdUpdate(IReadOnlyList<IGpuBuffer> parameters, IReadOnlyList<IGpuBuffer> gradients,
            IReadOnlyList<int> sizes, float weightDecay)
        {
            if (_gradientTable is null) throw new InvalidOperationException("PrepareOptimizer was not called before the capture.");
            _backend.MultiTensorSumOfSquares(_gradientTable, _guard!, zeroFirst: false);   // the epilogue re-zeroes it
            _backend.SgdMultiTensorImpl(parameters, gradients, sizes, 0f, weightDecay, _guard!.Handle, _dynamicScalars!.Handle);
        }

        public void PrepareSgdPingPong(IReadOnlyList<IGpuBuffer> first, IReadOnlyList<IGpuBuffer> second,
            IReadOnlyList<IGpuBuffer> gradients, IReadOnlyList<int> sizes)
        {
            EnsureOptimizerBuffers();
            DisposePingPong();
            _pingPong[0] = _backend.BuildPingPongTable(first, second, gradients, sizes);
            _pingPong[1] = _backend.BuildPingPongTable(second, first, gradients, sizes);
            _pingPongPartials = _backend.AllocateBuffer(Math.Max(2, 2 * _pingPong[0]!.TotalChunks));
            _pingPongDone = _backend.AllocateBuffer(2);
            ((IDeviceZeroFill)_backend).ZeroBuffer(_pingPongDone, 2);
        }

        public void CaptureSgdPingPongUpdate(int direction, float weightDecay)
        {
            var table = _pingPong[direction & 1]
                ?? throw new InvalidOperationException("PrepareSgdPingPong was not called before the capture.");
            // The update and the gradients' sum of squares in one pass, then the undo of a non-finite step.
            _backend.SgdPingPongUpdate(table, weightDecay, _dynamicScalars!.Handle, _pingPongPartials!, _pingPongDone!, _guard!.Handle);
            _backend.SgdPingPongRevert(table, _guard.Handle);
        }

        public bool CapturedGraphReads(IntPtr graphExec, IReadOnlyList<IGpuBuffer> buffers)
            => _backend.CapturedGraphUses(graphExec, buffers);

        private void DisposePingPong()
        {
            _pingPong[0]?.Dispose(); _pingPong[0] = null;
            _pingPong[1]?.Dispose(); _pingPong[1] = null;
            _pingPongPartials?.Dispose(); _pingPongPartials = null;
            _pingPongDone?.Dispose(); _pingPongDone = null;
        }

        public void CaptureAdamUpdate(bool decoupled, IReadOnlyList<IGpuBuffer> parameters, IReadOnlyList<IGpuBuffer> gradients,
            IReadOnlyList<IGpuBuffer> firstMoments, IReadOnlyList<IGpuBuffer> secondMoments, IReadOnlyList<int> sizes,
            float beta1, float beta2, float epsilon, float weightDecay)
        {
            if (_gradientTable is null || _adamTable is null)
                throw new InvalidOperationException("PrepareOptimizer was not called before the capture.");
            _backend.MultiTensorSumOfSquares(_gradientTable, _guard!, zeroFirst: false);   // the epilogue re-zeroes it
            // learningRate and step are placeholders: the kernel reads both from the dynamic scalars.
            _backend.MultiTensorAdamImpl(decoupled ? "adamw_multi_tensor_update" : "adam_multi_tensor_update",
                parameters, gradients, firstMoments, secondMoments, sizes, 0f, beta1, beta2, epsilon, weightDecay, 1,
                _guard, _dynamicScalars!.Handle, _adamTable);
        }

        public void Flush()
        {
            using var _ = _backend.PushContext();
            // cuStreamQuery submits the stream's pending work; its "not ready" result is the normal outcome here.
            CudaNativeBindings.cuStreamQuery(_backend._mainStream);
        }

        public bool TrySetPrologueCopy(IGpuBuffer? source, IGpuBuffer? destination, int count)
        {
            if (source is null || destination is null || count <= 0)
            {
                _copySource = _copyDestination = null;
                _copyCount = 0;
                return source is null && destination is null;
            }
            if (count > PrologueCopyMaxFloats || source.Size < count || destination.Size < count) return false;
            _copySource = source;
            _copyDestination = destination;
            _copyCount = count;
            return true;
        }

        public void CapturePrologue(int parity)
        {
            if (!_backend._kernelCache.TryGetValue("step_io_prologue", out var prologueKernel))
                throw new GpuCapabilityUnavailableException("CUDA kernel not found: step_io_prologue");
            var args = default(StepIoArgs);
            bool vector = true;
            for (int i = 0; i < _inputs.Length; i++)
                vector &= ((_stagingDevice[2 * i].ToInt64() | _stagingDevice[2 * i + 1].ToInt64() | _inputs[i].Handle.ToInt64()) & 15) == 0;
            int total = 0;
            for (int i = 0; i < _inputs.Length; i++)
            {
                args.Src0[i] = (ulong)_stagingDevice[2 * i].ToInt64();
                args.Src1[i] = (ulong)_stagingDevice[2 * i + 1].ToInt64();
                args.Dst[i] = (ulong)_inputs[i].Handle.ToInt64();
                args.Len[i] = _lengths[i];
                args.Start[i] = total;
                total += vector ? (_lengths[i] + 3) / 4 : _lengths[i];
            }
            args.Vector = vector ? 1 : 0;
            args.Start[_inputs.Length] = total;
            args.Count = _inputs.Length;
            args.State = (ulong)_state!.Handle.ToInt64();
            args.Masks = (ulong)_masksDevice.ToInt64();
            if (_dynamicScalars is not null)
            {
                args.LrTable0 = (ulong)_lrTableDevice[0].ToInt64();
                args.LrTable1 = (ulong)_lrTableDevice[1].ToInt64();
                args.DynScalars = (ulong)_dynamicScalars.Handle.ToInt64();
                args.Guard = (ulong)(_guard?.Handle.ToInt64() ?? 0);
            }
            if (_copySource is not null && _copyDestination is not null)
            {
                args.CopySrc = (ulong)_copySource.Handle.ToInt64();
                args.CopyDst = (ulong)_copyDestination.Handle.ToInt64();
                args.CopyCount = _copyCount;
            }
            void** launchArgs = stackalloc void*[1];
            launchArgs[0] = &args;
            uint blocks = (uint)Math.Max(1, Math.Min(1024, (Math.Max(total, _copyCount) + 255) / 256));
            _backend.LaunchKernel(prologueKernel, blocks, 256, launchArgs);
        }

        public void CaptureEpilogue(IGpuBuffer? loss, int lossCount)
        {
            if (!_backend._kernelCache.TryGetValue("step_io_epilogue", out var epilogueKernel))
                throw new GpuCapabilityUnavailableException("CUDA kernel not found: step_io_epilogue");
            if (lossCount < 0 || lossCount > StepIoLossCapacity) throw new ArgumentOutOfRangeException(nameof(lossCount));
            ulong stateAddress = (ulong)_state!.Handle.ToInt64(), resultsAddress = (ulong)_resultsDevice.ToInt64();
            IntPtr guard = _guard?.Handle ?? IntPtr.Zero, lossPtr = loss?.Handle ?? IntPtr.Zero;
            int count = loss is null ? 0 : lossCount;
            void** launchArgs = stackalloc void*[5];
            launchArgs[0] = &stateAddress; launchArgs[1] = &resultsAddress; launchArgs[2] = &guard;
            launchArgs[3] = &lossPtr; launchArgs[4] = &count;
            _backend.LaunchKernel(epilogueKernel, 1, 32, launchArgs);
        }

        public ReadOnlySpan<float> Loss(uint replay, int count)
        {
            if (count < 0 || count > StepIoLossCapacity) throw new ArgumentOutOfRangeException(nameof(count));
            if (_results == IntPtr.Zero) throw new ObjectDisposedException(nameof(CudaGraphStepIo));
            // StepIoResults: progress, applied, fault, unused (4 uints), sumSquares[2] (2 doubles), loss[2][64].
            float* loss = (float*)((byte*)_results + 16 + 16) + (replay & 1) * StepIoLossCapacity;
            return new ReadOnlySpan<float>(loss, count);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (!_backend.IsAvailable) return;   // the context took the page-locked memory with it
            using var _ = _backend.PushContext();
            // A replay may still read the staging or write the results.
            CudaNativeBindings.cuStreamSynchronize(_backend._mainStream);
            foreach (var p in _staging) if (p != IntPtr.Zero) CuBlasNative.cuMemFreeHost(p);
            foreach (var p in _lrTable) if (p != IntPtr.Zero) CuBlasNative.cuMemFreeHost(p);
            _dynamicScalars?.Dispose();
            _guard?.Dispose();
            _gradientTable?.Dispose();
            _adamTable?.Dispose();
            DisposePingPong();
            if (_results != IntPtr.Zero) CuBlasNative.cuMemFreeHost(_results);
            _results = IntPtr.Zero;
            if (_masks != IntPtr.Zero) CuBlasNative.cuMemFreeHost(_masks);
            _masks = IntPtr.Zero;
            _state?.Dispose();
            _state = null;
        }
    }
}
