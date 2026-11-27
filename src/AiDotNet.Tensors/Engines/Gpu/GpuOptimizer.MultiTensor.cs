// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Whole-parameter-list ("foreach", as <c>torch.optim</c>'s <c>foreach=True</c>) forms of every dense GPU optimizer
/// step. On device-resident lists each one equals calling its per-tensor step (<see cref="GpuOptimizer.TryRmspropStep"/>
/// and so on) on every parameter in order, under one GPU operation lease. It is all-or-nothing: every parameter,
/// gradient and state tensor must already hold its current value on the device (a list step never uploads, as a
/// per-tensor step may), and all of them are resolved before any kernel runs, so <c>false</c> leaves every value in the
/// list unchanged for the caller's CPU path (a caller can never double-apply the step to the tensors before the one
/// that failed). As with
/// <see cref="GpuOptimizer.TrySgdStepMultiTensor"/>, null lists or entries and length mismatches are caller errors and
/// throw; <c>false</c> means only that the list is not device-resident.
/// </summary>
public static partial class GpuOptimizer
{
    private delegate void MultiTensorKernel(
        IDirectGpuBackend backend, IGpuBuffer parameter, IGpuBuffer gradient, IGpuBuffer[] states, int length);

    private static bool TryStepMultiTensor(
        string site,
        IReadOnlyList<Tensor<float>> parameters,
        IReadOnlyList<Tensor<float>> gradients,
        IReadOnlyList<Tensor<float>>[] states,
        MultiTensorKernel kernel)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        if (gradients is null) throw new ArgumentNullException(nameof(gradients));
        int n = parameters.Count;
        if (gradients.Count != n)
            throw new ArgumentException($"{site}: every parameter needs a gradient.", nameof(gradients));
        for (int s = 0; s < states.Length; s++)
        {
            if (states[s] is null) throw new ArgumentNullException(nameof(states));
            if (states[s].Count != n)
                throw new ArgumentException($"{site}: every parameter needs each optimizer state tensor.", nameof(states));
        }
        for (int i = 0; i < n; i++)
        {
            var pCheck = parameters[i] ?? throw new ArgumentNullException(nameof(parameters));
            var gCheck = gradients[i] ?? throw new ArgumentNullException(nameof(gradients));
            if (pCheck.Length != gCheck.Length)
                throw new ArgumentException($"{site}: parameter {i} and its gradient differ in length.", nameof(gradients));
            for (int s = 0; s < states.Length; s++)
            {
                var sCheck = states[s][i] ?? throw new ArgumentNullException(nameof(states));
                if (sCheck.Length != pCheck.Length)
                    throw new ArgumentException($"{site}: parameter {i} and its optimizer state differ in length.", nameof(states));
            }
        }
        if (!AiDotNetEngine.Current.TryGet<DirectGpuTensorEngine>(out var gpuEngine)) return false;
        using var __gpuOp = gpuEngine.EnterGpuOp();
        var backend = gpuEngine.GetBackend();
        if (backend is null) return false;
        if (n == 0) return true;

        // The list form is the all-resident fast path: one tensor with no current device copy refuses the whole list
        // before any kernel runs, so the caller's per-tensor path (which does move host tensors onto the device) takes
        // the step with nothing half-done. Same contract as TrySgdStepMultiTensor. Copy-on-write tensors detach first,
        // as the step writes them; a detached one is on the host and refuses the list. Gradients and states are
        // checked here, parameters as they resolve (TryResolveListParameterBuffer never uploads).
        for (int i = 0; i < n; i++)
        {
            if (parameters[i].Length == 0) continue;
            parameters[i].EnsureOwnedForDeviceWrite();
            for (int s = 0; s < states.Length; s++) states[s][i].EnsureOwnedForDeviceWrite();
        }
        for (int i = 0; i < n; i++)
        {
            int len = parameters[i].Length;
            if (len == 0) continue;
            if (!IsListTensorDeviceResident(gpuEngine, backend, gradients[i], len)) return false;
            for (int s = 0; s < states.Length; s++)
                if (!IsListTensorDeviceResident(gpuEngine, backend, states[s][i], len)) return false;
        }

        var parameterBuffers = new IGpuBuffer[n];
        var gradientBuffers = new IGpuBuffer[n];
        var stateBuffers = new IGpuBuffer[n][];
        var sizes = new int[n];
        for (int i = 0; i < n; i++)
        {
            var param = parameters[i];
            var grad = gradients[i];
            int len = param.Length;
            if (len == 0) continue; // nothing to update; sizes[i] stays 0 and the dispatch skips it

            var pBuf = TryResolveListParameterBuffer(gpuEngine, backend, param);
            var gBuf = TryGetCurrentGpuBuffer(gpuEngine, backend, grad);
            if (pBuf is null || pBuf.Size < len || gBuf is null || gBuf.Size < len)
                return false;

            var perTensorStates = new IGpuBuffer[states.Length];
            for (int s = 0; s < states.Length; s++)
            {
                var sBuf = TryGetCurrentGpuBuffer(gpuEngine, backend, states[s][i]);
                if (sBuf is null || sBuf.Size < len)
                    return false;
                perTensorStates[s] = sBuf;
            }

            parameterBuffers[i] = pBuf;
            gradientBuffers[i] = gBuf;
            stateBuffers[i] = perTensorStates;
            sizes[i] = len;
        }

        int active = 0;
        for (int i = 0; i < n; i++)
            if (sizes[i] != 0) active++;
        if (active == 0) return true;

        for (int i = 0; i < n; i++)
        {
            if (sizes[i] == 0) continue;
            kernel(backend, parameterBuffers[i], gradientBuffers[i], stateBuffers[i], sizes[i]);
        }

        int marks = 1 + states.Length;
        var syncLeases = GpuSyncPoint.CreateSharedLeaseGroup(CreateWriteSyncPoint(backend), marks * active);
        int lease = 0;
        for (int i = 0; i < n; i++)
        {
            if (sizes[i] == 0) continue;
            MarkParameterUpdatedWithSync(gpuEngine, backend, parameters[i], parameterBuffers[i], syncLeases[lease++]);
            for (int s = 0; s < states.Length; s++)
                MarkGpuUpdatedWithSync(states[s][i], syncLeases[lease++]);
        }
        return true;
    }

    // A gradient or optimizer state already on this backend's device, checked read-only: an offload buffer large enough,
    // or a current device buffer. This is the first branch of TryResolveStateBuffer and TryAcquireReadOnlyFloatBuffer,
    // so a tensor that passes is resolved there without an upload. Call it after EnsureOwnedForDeviceWrite: detaching a
    // copy-on-write tensor rebinds it to host storage.
    internal static bool IsListTensorDeviceResident(
        DirectGpuTensorEngine engine, IDirectGpuBackend backend, Tensor<float> tensor, int length)
    {
        if (tensor._gpuBuffer is null && tensor.OffloadDevicePointer != IntPtr.Zero)
            return tensor.TryGetGpuBuffer() is { } offload && offload.Handle != IntPtr.Zero && offload.Size >= length;
        var current = TryGetCurrentGpuBuffer(engine, backend, tensor);
        return current is not null && current.Size >= length;
    }

    // A list step's parameter buffer, never uploaded: a flat ParameterBuffer view from its flat array's device buffer, an
    // offload buffer, or what AcquireParameterBuffer serves without an upload. A parameter AcquireParameterBuffer cannot
    // take (strided, or offset into its storage) uses its current bound buffer. Null when its value is only on the host.
    internal static IGpuBuffer? TryResolveListParameterBuffer(
        DirectGpuTensorEngine engine, IDirectGpuBackend backend, Tensor<float> param)
    {
        if (DirectGpuTensorEngine.IsFlatParameterView(param))
            return engine.AcquireParameterBuffer(param, out _, allowUpload: false);
        if (param._gpuBuffer is null && param.OffloadDevicePointer != IntPtr.Zero)
            return param.TryGetGpuBuffer();
        if (param.IsContiguous && param._storageOffset == 0)
            return engine.AcquireParameterBuffer(param, out _, allowUpload: false);
        return TryGetCurrentGpuBuffer(engine, backend, param);
    }

    /// <summary>SGD with momentum over a parameter list; equals <see cref="TrySgdMomentumStep"/> per tensor.</summary>
    public static bool TrySgdMomentumStepMultiTensor(
        IReadOnlyList<Tensor<float>> parameters, IReadOnlyList<Tensor<float>> gradients,
        IReadOnlyList<Tensor<float>> velocities, float lr, float momentum, float weightDecay)
        => TryStepMultiTensor(nameof(TrySgdMomentumStepMultiTensor), parameters, gradients, new[] { velocities },
            (b, p, g, s, len) => b.SgdMomentumUpdate(p, g, s[0], lr, momentum, weightDecay, len));

    /// <summary>Nesterov momentum over a parameter list; equals <see cref="TryNagStep"/> per tensor.</summary>
    public static bool TryNagStepMultiTensor(
        IReadOnlyList<Tensor<float>> parameters, IReadOnlyList<Tensor<float>> gradients,
        IReadOnlyList<Tensor<float>> velocities, float lr, float momentum, float weightDecay)
        => TryStepMultiTensor(nameof(TryNagStepMultiTensor), parameters, gradients, new[] { velocities },
            (b, p, g, s, len) => b.NagUpdate(p, g, s[0], lr, momentum, weightDecay, len));

    /// <summary>RMSprop over a parameter list; equals <see cref="TryRmspropStep"/> per tensor.</summary>
    public static bool TryRmspropStepMultiTensor(
        IReadOnlyList<Tensor<float>> parameters, IReadOnlyList<Tensor<float>> gradients,
        IReadOnlyList<Tensor<float>> squaredAverages, float lr, float rho, float epsilon, float weightDecay)
        => TryStepMultiTensor(nameof(TryRmspropStepMultiTensor), parameters, gradients, new[] { squaredAverages },
            (b, p, g, s, len) => b.RmspropUpdate(p, g, s[0], lr, rho, epsilon, weightDecay, len));

    /// <summary>Adagrad over a parameter list; equals <see cref="TryAdagradStep"/> per tensor.</summary>
    public static bool TryAdagradStepMultiTensor(
        IReadOnlyList<Tensor<float>> parameters, IReadOnlyList<Tensor<float>> gradients,
        IReadOnlyList<Tensor<float>> accumulators, float lr, float epsilon, float weightDecay)
        => TryStepMultiTensor(nameof(TryAdagradStepMultiTensor), parameters, gradients, new[] { accumulators },
            (b, p, g, s, len) => b.AdagradUpdate(p, g, s[0], lr, epsilon, weightDecay, len));

    /// <summary>Adadelta over a parameter list; equals <see cref="TryAdadeltaStep"/> per tensor.</summary>
    public static bool TryAdadeltaStepMultiTensor(
        IReadOnlyList<Tensor<float>> parameters, IReadOnlyList<Tensor<float>> gradients,
        IReadOnlyList<Tensor<float>> accumulatedGradients, IReadOnlyList<Tensor<float>> accumulatedUpdates,
        float rho, float epsilon, float weightDecay)
        => TryStepMultiTensor(nameof(TryAdadeltaStepMultiTensor), parameters, gradients,
            new[] { accumulatedGradients, accumulatedUpdates },
            (b, p, g, s, len) => b.AdadeltaUpdate(p, g, s[0], s[1], rho, epsilon, weightDecay, len));

    /// <summary>AMSGrad over a parameter list; equals <see cref="TryAmsgradStep"/> per tensor.</summary>
    public static bool TryAmsgradStepMultiTensor(
        IReadOnlyList<Tensor<float>> parameters, IReadOnlyList<Tensor<float>> gradients,
        IReadOnlyList<Tensor<float>> firstMoments, IReadOnlyList<Tensor<float>> secondMoments,
        IReadOnlyList<Tensor<float>> maxSecondMoments,
        float lr, float beta1, float beta2, float epsilon, float weightDecay, int step)
        => TryStepMultiTensor(nameof(TryAmsgradStepMultiTensor), parameters, gradients,
            new[] { firstMoments, secondMoments, maxSecondMoments },
            (b, p, g, s, len) => b.AmsgradUpdate(p, g, s[0], s[1], s[2], lr, beta1, beta2, epsilon, weightDecay, step, len));

    /// <summary>Adamax over a parameter list; equals <see cref="TryAdamaxStep"/> per tensor.</summary>
    public static bool TryAdamaxStepMultiTensor(
        IReadOnlyList<Tensor<float>> parameters, IReadOnlyList<Tensor<float>> gradients,
        IReadOnlyList<Tensor<float>> firstMoments, IReadOnlyList<Tensor<float>> infinityNorms,
        float lr, float beta1, float beta2, float epsilon, float weightDecay, int step)
        => TryStepMultiTensor(nameof(TryAdamaxStepMultiTensor), parameters, gradients,
            new[] { firstMoments, infinityNorms },
            (b, p, g, s, len) => b.AdamaxUpdate(p, g, s[0], s[1], lr, beta1, beta2, epsilon, weightDecay, step, len));

    /// <summary>Lion over a parameter list; equals <see cref="TryLionStep"/> per tensor.</summary>
    public static bool TryLionStepMultiTensor(
        IReadOnlyList<Tensor<float>> parameters, IReadOnlyList<Tensor<float>> gradients,
        IReadOnlyList<Tensor<float>> moments, float lr, float beta1, float beta2, float weightDecay)
        => TryStepMultiTensor(nameof(TryLionStepMultiTensor), parameters, gradients, new[] { moments },
            (b, p, g, s, len) => b.LionUpdate(p, g, s[0], lr, beta1, beta2, weightDecay, len));

    /// <summary>Nadam over a parameter list; equals <see cref="TryNadamStep"/> per tensor.</summary>
    public static bool TryNadamStepMultiTensor(
        IReadOnlyList<Tensor<float>> parameters, IReadOnlyList<Tensor<float>> gradients,
        IReadOnlyList<Tensor<float>> firstMoments, IReadOnlyList<Tensor<float>> secondMoments,
        float lr, float beta1, float beta2, float epsilon, float weightDecay, int step)
        => TryStepMultiTensor(nameof(TryNadamStepMultiTensor), parameters, gradients,
            new[] { firstMoments, secondMoments },
            (b, p, g, s, len) => b.NadamUpdate(p, g, s[0], s[1], lr, beta1, beta2, epsilon, weightDecay, step, len));

    /// <summary>FTRL-Proximal over a parameter list; equals <see cref="TryFtrlStep"/> per tensor.</summary>
    public static bool TryFtrlStepMultiTensor(
        IReadOnlyList<Tensor<float>> parameters, IReadOnlyList<Tensor<float>> gradients,
        IReadOnlyList<Tensor<float>> zAccumulators, IReadOnlyList<Tensor<float>> nAccumulators,
        float lr, float l1Reg, float l2Reg, float beta)
        => TryStepMultiTensor(nameof(TryFtrlStepMultiTensor), parameters, gradients,
            new[] { zAccumulators, nAccumulators },
            (b, p, g, s, len) => b.FtrlUpdate(p, g, s[0], s[1], lr, l1Reg, l2Reg, beta, len));

    /// <summary>
    /// LAMB over a parameter list; equals <see cref="TryLambStep"/> per tensor (each tensor keeps its own layer-wise
    /// trust ratio).
    /// </summary>
    public static bool TryLambStepMultiTensor(
        IReadOnlyList<Tensor<float>> parameters, IReadOnlyList<Tensor<float>> gradients,
        IReadOnlyList<Tensor<float>> firstMoments, IReadOnlyList<Tensor<float>> secondMoments,
        float lr, float beta1, float beta2, float epsilon, float weightDecay, int step,
        float maxTrustRatio = 0f, bool biasCorrection = true)
        => TryStepMultiTensor(nameof(TryLambStepMultiTensor), parameters, gradients,
            new[] { firstMoments, secondMoments },
            (b, p, g, s, len) => b.LambUpdate(p, g, s[0], s[1], lr, beta1, beta2, epsilon, weightDecay, step, len,
                maxTrustRatio, biasCorrection));

    /// <summary>
    /// LARS over a parameter list; equals <see cref="TryLarsStep"/> per tensor (each tensor keeps its own layer-wise
    /// trust ratio).
    /// </summary>
    public static bool TryLarsStepMultiTensor(
        IReadOnlyList<Tensor<float>> parameters, IReadOnlyList<Tensor<float>> gradients,
        IReadOnlyList<Tensor<float>> velocities, float lr, float momentum, float weightDecay, float trustCoeff, float epsilon)
        => TryStepMultiTensor(nameof(TryLarsStepMultiTensor), parameters, gradients, new[] { velocities },
            (b, p, g, s, len) => b.LarsUpdate(p, g, s[0], lr, momentum, weightDecay, trustCoeff, epsilon, len));

    /// <summary>Proximal gradient (L1) over a parameter list; equals <see cref="TryProximalL1Step"/> per tensor.</summary>
    public static bool TryProximalL1StepMultiTensor(
        IReadOnlyList<Tensor<float>> parameters, IReadOnlyList<Tensor<float>> gradients, float lr, float l1Strength)
        => TryStepMultiTensor(nameof(TryProximalL1StepMultiTensor), parameters, gradients,
            Array.Empty<IReadOnlyList<Tensor<float>>>(),
            (b, p, g, s, len) => b.ProximalL1Update(p, g, lr, l1Strength, len));
}
