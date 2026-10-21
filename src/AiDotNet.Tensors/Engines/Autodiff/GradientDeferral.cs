// Copyright (c) AiDotNet. All rights reserved.
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Gradient contributions a compiled GPU step defers instead of adding: for a tensor whose gradient is consumed only by a
/// backward that can sum two output gradients as it loads them (the fused BatchNorm+ReLU backwards), the second
/// contribution is parked here rather than added into the first (an add_vectors pass over the activation), and that
/// backward takes it. The sum is the same, gradOutput + parked, in the in-place accumulation's order.
/// </summary>
internal sealed class GradientDeferral
{
    private readonly HashSet<object> _deferrable;
    private readonly Dictionary<object, object> _parked = new(ReferenceEqualityComparer.Instance);

    /// <param name="deferrable">The tensors (by reference) whose second contribution may be parked.</param>
    internal GradientDeferral(IEnumerable<object> deferrable)
    {
        _deferrable = new HashSet<object>(deferrable, ReferenceEqualityComparer.Instance);
    }

    /// <summary>Kill switch: AIDOTNET_GRAD_DEFER=0 adds every contribution as it arrives.</summary>
    internal static bool Enabled { get; set; } = Environment.GetEnvironmentVariable("AIDOTNET_GRAD_DEFER") != "0";

    /// <summary>Contributions parked so far in this process (tests).</summary>
    internal static long ParkedCount;

    /// <summary>
    /// The deferral of the compiled GPU step body running on this thread, or null where there is none and every
    /// contribution is added as it arrives. Thread-static, not engine state: one engine serves every plan compiled on
    /// it while <c>CompiledTrainingPlan._stepSync</c> serialises one plan only, so held on the engine a second plan's
    /// step would overwrite this deferral and its finally would clear it - and a contribution parked before that whose
    /// consuming backward then takes nothing is dropped, which is a wrong gradient rather than a slower step. A step
    /// body parks and takes on the single thread that runs it (Step takes the lock, StepCore stays on the thread, and
    /// the backward walk is sequential over nodes), so the thread is the scope that isolates concurrent plans.
    /// </summary>
    internal static GradientDeferral? Current
    {
        get => s_current;
        set => s_current = value;
    }

    [ThreadStatic]
    private static GradientDeferral? s_current;

    internal int DeferrableCount => _deferrable.Count;

    internal bool Accepts(object tensor) => _deferrable.Contains(tensor);

    /// <summary>Parks <paramref name="contribution"/> for <paramref name="tensor"/>; false when one is already parked.</summary>
    internal bool TryPark(object tensor, object contribution)
    {
        if (_parked.ContainsKey(tensor)) return false;
        _parked[tensor] = contribution;
        System.Threading.Interlocked.Increment(ref ParkedCount);
        return true;
    }

    /// <summary>Removes and returns the contribution parked for <paramref name="tensor"/>, or null.</summary>
    internal Tensor<T>? Take<T>(Tensor<T> tensor)
    {
        if (!_parked.TryGetValue(tensor, out var parked)) return null;
        _parked.Remove(tensor);
        return (Tensor<T>)parked;
    }

    /// <summary>Drops anything a previous run left parked (a step that threw part way).</summary>
    internal void Reset() => _parked.Clear();

    private sealed class ReferenceEqualityComparer : IEqualityComparer<object>
    {
        internal static readonly ReferenceEqualityComparer Instance = new();
        public new bool Equals(object? x, object? y) => ReferenceEquals(x, y);
        public int GetHashCode(object obj) => RuntimeHelpers.GetHashCode(obj);
    }
}
