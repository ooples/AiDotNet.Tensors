using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

internal sealed partial class CompiledTrainingPlan<T>
{
    private readonly bool _gradientBankShadowsEnabled =
        Environment.GetEnvironmentVariable("AIDOTNET_COMPILED_GRADIENT_BANK_SHADOWS") == "1";
    private readonly HashSet<Array> _shadowBankArrays = new(Autodiff.ReferenceEqualityComparer<Array>.Instance);
    private readonly Dictionary<Array, (IGpuBuffer Buffer, IDirectGpuBackend Backend, int Thread, long Key)> _shadowBankLoans =
        new(Autodiff.ReferenceEqualityComparer<Array>.Instance);
    private readonly List<Tensor<T>> _shadowBankBindings = new();
    private readonly HashSet<object> _shadowBankVectors = new(Autodiff.ReferenceEqualityComparer<object>.Instance);
    internal long GradientBankShadowBytes { get; private set; }
    internal long GradientBankShadowCompletedSteps { get; private set; }

    private bool TryBindShadowGradientBank(DirectGpuTensorEngine engine, IDirectGpuBackend backend,
        Tensor<T> gradient, out IGpuBuffer buffer)
    {
        buffer = null!;
        if (!_gradientBankShadowsEnabled || !CanUseGradientShadows(engine)
            || gradient.GraphBackingArrayIdentity is not T[] array || !_shadowBankArrays.Contains(array)
            || !ResultBufferTracker.IsUnexposedPrivateArray(array) || !engine.CanUseCompiledGradientShadow(gradient)) return false;
        int thread = Environment.CurrentManagedThreadId;
        if (_shadowBankLoans.TryGetValue(array, out var held))
        {
            if (!ReferenceEquals(held.Backend, backend) || held.Thread != thread || held.Buffer.Handle == IntPtr.Zero) return false;
            buffer = held.Buffer;
        }
        else
        {
            buffer = engine.CompiledWorkspaceBroker.TryRent(backend, array.Length, exactSize: true)
                ?? TryTakePhaseGradientBacking(engine, gradient, array.Length)
                ?? backend.AllocateBuffer(array.Length);
            var owner = _shadowLoanOwner ??= engine.CreateScratchPool();
            long key = ++_nextShadowLoan;
            owner.Track(buffer);
            owner.Slots.Add(key, buffer);
            _shadowBankLoans.Add(array, (buffer, backend, thread, key));
        }
        engine.BindResidentBuffer(gradient, buffer, backend);
        return true;
    }

    private long ShadowCompletedGradientBanks(DirectGpuTensorEngine engine, IDirectGpuBackend backend,
        HashSet<IGpuBuffer> protectedRoots)
    {
        if (!_gradientBankShadowsEnabled || _scratchPool is not { } pool) return 0;
        long bytes = 0;
        foreach (var gradient in _preAllocatedGrads)
        {
            if (gradient._gpuBuffer is not { } root || protectedRoots.Contains(root)
                || gradient.GraphBackingArrayIdentity is not T[] array
                || array is not float[] destination || !ResultBufferTracker.IsUnexposedPrivateArray(array)
                || ((Vector<T>)gradient.GraphVectorIdentity).ResultOwner is not { } owner) continue;
            bool loan = _shadowBankLoans.TryGetValue(array, out var held) && ReferenceEquals(held.Buffer, root);
            if (!loan && (StepScratchPool.IsPooled(root) || !_privateGradientAllocations.TryGetValue(root, out _))) continue;
            if (!ResidentHostMirror.TryCollectGradientBankBindings(array, root, backend, owner,
                    _shadowBankBindings, _shadowBankVectors)) continue;
            bool eligible = true;
            foreach (var binding in _shadowBankBindings)
                if (!engine.CanUseCompiledGradientShadow(binding)) { eligible = false; break; }
            if (!eligible) continue;
            try { backend.DownloadBuffer(root, destination); }
            catch (NotSupportedException capabilityGap)
            { GpuLaunchProbe.OnFallback("CompiledGradientBankShadow", capabilityGap); continue; }
            pool.Track(root);
            foreach (var binding in _preAllocatedGrads)
                if (ReferenceEquals(binding._gpuBuffer, root)) _shadowPreviousBuffers[binding] = root;
            foreach (var binding in _shadowBankBindings) engine.DetachHostCurrentCompiledGradient(binding);
            _pooledGradDeviceBuffers?.Remove(array);
            _privateGradientAllocations.Remove(root);
            if (loan)
            {
                _shadowLoanOwner!.Slots.Remove(held.Key);
                _shadowBankLoans.Remove(array);
            }
            _shadowBankArrays.Add(array);
            engine.CompiledWorkspaceBroker.Return(backend, root);
            bytes += root.SizeInBytes;
        }
        if (bytes != 0)
        {
            GradientBankShadowBytes += bytes;
            GradientBankShadowCompletedSteps++;
        }
        return bytes;
    }
}
