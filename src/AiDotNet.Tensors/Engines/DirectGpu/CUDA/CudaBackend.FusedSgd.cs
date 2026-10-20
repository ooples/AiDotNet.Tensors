// Copyright (c) AiDotNet. All rights reserved.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// The ping-pong SGD update fused into the weight-gradient kernels that write a parameter's gradient last. A captured
// step's ping-pong update (sgd_pingpong_update) reads every gradient back from memory once the backward has written
// it; a dW kernel that already holds the gradient in registers can apply the step there instead. It still writes dW,
// since the plan publishes its gradients, but the update no longer reads them back, and sgd_pingpong_update skips that
// tensor (it still undoes it when the step's gradients are not finite).
public sealed partial class CudaBackend
{
    /// <summary>The device layout of <c>PpFused</c> (CudaOptimizerKernels.PingPongDevice).</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct PpFusedArgs
    {
        public ulong Src, Dst, Dyn, Partials;
        public float WeightDecay;
        public int Slot;
    }

    /// <summary>The device layout of <c>PpSkip</c>: one bit per tensor index below <see cref="FusedSgdMaxTensors"/>.</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct PpSkipArgs
    {
        public fixed uint Bits[8];
    }

    /// <summary>Tensors past this index are never fused: the skip mask the update kernel takes by value has 256 bits.</summary>
    internal const int FusedSgdMaxTensors = 256;

    /// <summary>
    /// One step graph's fused updates, armed by the step I/O for the graph it is capturing (its ping-pong direction):
    /// which gradient buffers a dW kernel may update through, and the per-block partial sums handed out so far.
    /// </summary>
    internal sealed class FusedSgdCapture
    {
        private readonly object _sync = new();
        private readonly Dictionary<long, int> _byGradient = new();
        private readonly ulong[] _src, _dst;
        private readonly int[] _sizes;
        private readonly ulong _dyn, _partials;
        private readonly float _weightDecay;
        private readonly int _capacity;
        private readonly uint[] _skip = new uint[8];
        private int _used;

        internal FusedSgdCapture(IReadOnlyList<IntPtr> sources, IReadOnlyList<IntPtr> destinations,
            IReadOnlyList<IntPtr> gradients, IReadOnlyList<int> sizes, IReadOnlyList<bool> fusable, IntPtr dynamicScalars,
            IGpuBuffer partials, float weightDecay)
        {
            int n = sizes.Count;
            _src = new ulong[n]; _dst = new ulong[n]; _sizes = new int[n];
            for (int t = 0; t < n; t++)
            {
                _src[t] = (ulong)sources[t].ToInt64();
                _dst[t] = (ulong)destinations[t].ToInt64();
                _sizes[t] = sizes[t];
                // A gradient two tensors share (or that appears twice) is never fused: neither write would be the last.
                if (t < FusedSgdMaxTensors && fusable[t])
                {
                    long g = gradients[t].ToInt64();
                    if (_byGradient.ContainsKey(g)) _byGradient[g] = -1; else _byGradient[g] = t;
                }
            }
            _dyn = (ulong)dynamicScalars.ToInt64();
            _partials = (ulong)partials.Handle.ToInt64();
            _capacity = (int)Math.Min(int.MaxValue, partials.Size / 2);   // doubles in a float-counted buffer
            _weightDecay = weightDecay;
        }

        /// <summary>
        /// Claims <paramref name="output"/>'s update for a dW kernel writing all <paramref name="count"/> of its elements
        /// over <paramref name="blocks"/> blocks: false when it is not a fusable gradient of exactly that size, was
        /// already claimed in this graph, or the partials are full.
        /// </summary>
        internal bool TryClaim(IGpuBuffer output, int count, int blocks, out PpFusedArgs args)
        {
            args = default;
            lock (_sync)
            {
                if (!_byGradient.TryGetValue(output.Handle.ToInt64(), out int t) || t < 0 || _sizes[t] != count) return false;
                if ((_skip[t >> 5] & (1u << (t & 31))) != 0 || blocks <= 0 || _used + blocks > _capacity) return false;
                args = new PpFusedArgs
                {
                    Src = _src[t], Dst = _dst[t], Dyn = _dyn, Partials = _partials, WeightDecay = _weightDecay, Slot = _used,
                };
                _used += blocks;
                _skip[t >> 5] |= 1u << (t & 31);
                return true;
            }
        }

        /// <summary>The tensors claimed so far and the partial sums their kernels write (for sgd_pingpong_update).</summary>
        internal unsafe (PpSkipArgs Skip, int PartialCount) Snapshot()
        {
            lock (_sync)
            {
                var skip = default(PpSkipArgs);
                for (int i = 0; i < 8; i++) skip.Bits[i] = _skip[i];
                return (skip, _used);
            }
        }

        /// <summary>How many tensors a dW kernel updates in this graph (tests).</summary>
        internal int ClaimedCount
        {
            get
            {
                lock (_sync)
                {
                    int c = 0;
                    foreach (var word in _skip) for (uint w = word; w != 0; w &= w - 1) c++;
                    return c;
                }
            }
        }
    }

    // Armed by CudaGraphStepIo.CapturePrologue for the step graph being captured, disarmed when it records the update.
    private FusedSgdCapture? _fusedSgdCapture;

    /// <summary>Kill switch for A/B and tests: AIDOTNET_FUSED_SGD=0 keeps every update in sgd_pingpong_update.</summary>
    internal static bool FusedSgdEnabled { get; set; } =
        Environment.GetEnvironmentVariable("AIDOTNET_FUSED_SGD") != "0";

    /// <summary>
    /// Claims the fused update of <paramref name="output"/> for a dW kernel being captured into a step graph (see
    /// <see cref="FusedSgdCapture.TryClaim"/>). The caller has already checked its fused kernel is loaded, since a claim
    /// takes the tensor out of sgd_pingpong_update.
    /// </summary>
    private bool TryClaimFusedSgd(IGpuBuffer output, int count, int blocks, out PpFusedArgs args)
    {
        args = default;
        var capture = _fusedSgdCapture;
        return capture is not null && IsStreamCapturing() && capture.TryClaim(output, count, blocks, out args);
    }
}
