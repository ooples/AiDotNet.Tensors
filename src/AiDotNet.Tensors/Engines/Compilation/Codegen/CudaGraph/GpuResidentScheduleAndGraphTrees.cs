// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.CudaGraph;

/// <summary>
/// Fixed-address GPU-resident scalar schedule buffer for full-step training CUDA Graphs
/// (<c>Forward + Loss + Backward + Multi-Tensor GradClip + Optimizer.Step</c> in one <c>cuGraphLaunch</c>).
/// <para>
/// Instead of passing learning rate, bias-correction <c>1 - beta^t</c>, weight decay, and gradient-clipping
/// thresholds as host-by-value kernel launch arguments (which freeze at stream-capture time and force CUDA Graph
/// recapture on every cosine/warmup LR step), this buffer keeps a fixed 8-float device allocation whose address
/// (<see cref="Buffer"/>.<c>Handle</c>) never changes across training steps. Before each <c>cuGraphLaunch</c>,
/// <see cref="UpdateSchedule"/> refreshes the 32-byte device payload on the compute stream.
/// </para>
/// </summary>
public sealed class GpuResidentScheduleScalars : IDisposable
{
    /// <summary>Number of float32 slots in the resident schedule buffer.</summary>
    public const int SlotCount = 8;

    public const int LearningRateSlot = 0;
    public const int WeightDecaySlot = 1;
    public const int Beta1Slot = 2;
    public const int Beta2Slot = 3;
    public const int BiasCorrection1Slot = 4;
    public const int BiasCorrection2Slot = 5;
    public const int StepNumberSlot = 6;
    public const int MaxGradNormSlot = 7;

    private readonly IDirectGpuBackend _backend;
    private readonly float[] _hostStaging = new float[SlotCount];
    private readonly Action<IGpuBuffer, float[]>? _customUploader;
    private IGpuBuffer? _deviceBuffer;
    private bool _disposed;

    /// <summary>Creates and allocates a fixed-address 8-float GPU schedule buffer.</summary>
    public GpuResidentScheduleScalars(
        IDirectGpuBackend backend,
        Action<IGpuBuffer, float[]>? customUploader = null)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _customUploader = customUploader;
        _hostStaging[Beta1Slot] = 0.9f;
        _hostStaging[Beta2Slot] = 0.999f;
        _hostStaging[BiasCorrection1Slot] = 1.0f;
        _hostStaging[BiasCorrection2Slot] = 1.0f;
        _deviceBuffer = _backend.AllocateBuffer(_hostStaging);
    }

    /// <summary>Fixed-address GPU buffer bound into captured optimizer and gradient-clipping kernels.</summary>
    public IGpuBuffer Buffer
    {
        get
        {
            if (_disposed || _deviceBuffer is null)
                throw new ObjectDisposedException(nameof(GpuResidentScheduleScalars));
            return _deviceBuffer;
        }
    }

    /// <summary>Current host mirror of a schedule slot (for diagnostics and verification).</summary>
    public float GetSlotValue(int slot)
    {
        if ((uint)slot >= SlotCount)
            throw new ArgumentOutOfRangeException(nameof(slot));
        return _hostStaging[slot];
    }

    /// <summary>
    /// Updates the resident GPU scalar slots in place without changing <see cref="Buffer"/>'s device pointer,
    /// keeping any captured full-step CUDA Graph 100% valid across learning-rate schedules and step increments.
    /// </summary>
    public void UpdateSchedule(
        float learningRate,
        float weightDecay,
        float beta1,
        float beta2,
        int step,
        float maxGradNorm = 0.0f)
    {
        if (_disposed || _deviceBuffer is null)
            throw new ObjectDisposedException(nameof(GpuResidentScheduleScalars));
        if (step < 1)
            throw new ArgumentOutOfRangeException(nameof(step), "Optimizer step index must be >= 1.");

        float bc1 = 1.0f - (float)Math.Pow(beta1, step);
        float bc2 = 1.0f - (float)Math.Pow(beta2, step);
        if (bc1 <= 1e-12f) bc1 = 1.0f;
        if (bc2 <= 1e-12f) bc2 = 1.0f;

        _hostStaging[LearningRateSlot] = learningRate;
        _hostStaging[WeightDecaySlot] = weightDecay;
        _hostStaging[Beta1Slot] = beta1;
        _hostStaging[Beta2Slot] = beta2;
        _hostStaging[BiasCorrection1Slot] = bc1;
        _hostStaging[BiasCorrection2Slot] = bc2;
        _hostStaging[StepNumberSlot] = step;
        _hostStaging[MaxGradNormSlot] = maxGradNorm;

        IntPtr pointerBefore = _deviceBuffer.Handle;
        if (_customUploader is not null)
        {
            _customUploader(_deviceBuffer, _hostStaging);
        }
        else if (_backend is IOverlappedUploadBackend overlapped && overlapped.TryUploadOverlapped(_hostStaging, _deviceBuffer))
        {
            // Overlapped copy-stream refresh succeeded without reallocating _deviceBuffer.
        }
        else
        {
            _backend.UploadBufferInPlace(_hostStaging, _deviceBuffer);
        }

        if (_deviceBuffer.Handle != pointerBefore)
        {
            throw new InvalidOperationException(
                "Schedule scalar buffer upload mutated its device pointer, violating the static CUDA Graph contract.");
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _deviceBuffer?.Dispose();
        _deviceBuffer = null;
    }
}

/// <summary>
/// Multi-bucket <c>CUDAGraphTrees</c> manager sharing a SINGLE physical <see cref="IGpuBuffer"/> memory arena
/// across 4–6 SizeClass sequence/batch buckets so variable-length workloads achieve zero-launch-overhead CUDA Graph
/// replay without multiplying workspace VRAM by the number of buckets.
/// </summary>
public sealed class CudaGraphTreePool : IDisposable
{
    private sealed class BucketEntry : IDisposable
    {
        public int BucketCapacity { get; }
        public IGpuBuffer ArenaView { get; }
        public StaticShapeCudaGraphScope Scope { get; }

        public BucketEntry(int bucketCapacity, IGpuBuffer arenaView, StaticShapeCudaGraphScope scope)
        {
            BucketCapacity = bucketCapacity;
            ArenaView = arenaView;
            Scope = scope;
        }

        public void Dispose()
        {
            Scope.Dispose();
            ArenaView.Dispose();
        }
    }

    private sealed class OffsetBufferView : IGpuBuffer, IGpuBufferViewSource
    {
        private readonly IGpuBuffer _parent;
        private readonly int _elementOffset;

        public OffsetBufferView(IGpuBuffer parent, int elementOffset, int elementCount)
        {
            _parent = parent;
            _elementOffset = elementOffset;
            Size = elementCount;
        }

        public IGpuBuffer? ViewParent => _parent;
        public GpuElementType ElementType => _parent.ElementType;
        public int Size { get; }
        public long SizeInBytes => (long)Size * sizeof(float);
        public IntPtr Handle => _parent.Handle == IntPtr.Zero
            ? IntPtr.Zero
            : new IntPtr(_parent.Handle.ToInt64() + (long)_elementOffset * sizeof(float));

        public void Dispose()
        {
            // Non-owning view into the shared graph arena.
        }
    }

    private readonly IDirectGpuBackend _backend;
    private readonly int[] _sortedBuckets;
    private readonly int _elementsPerUnit;
    private readonly IGpuBuffer _sharedArena;
    private readonly Dictionary<int, BucketEntry> _entries;
    private readonly object _gate = new();
    private bool _disposed;

    /// <summary>
    /// Creates a multi-bucket CUDA Graph tree pool allocating ONE shared arena of
    /// <c>max(bucketSizes) * workspaceElementsPerUnit</c> floats and carving zero-overhead views for every bucket.
    /// </summary>
    public CudaGraphTreePool(
        IDirectGpuBackend backend,
        IReadOnlyList<int> bucketSizes,
        int workspaceElementsPerUnit,
        CudaGraphExecutionScopeKind scopeKind = CudaGraphExecutionScopeKind.InferenceForward,
        StaticShapeCudaGraphOptions? options = null)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        if (bucketSizes is null || bucketSizes.Count == 0)
            throw new ArgumentException("At least one bucket size must be specified.", nameof(bucketSizes));
        if (workspaceElementsPerUnit <= 0)
            throw new ArgumentOutOfRangeException(nameof(workspaceElementsPerUnit));

        var distinct = new SortedSet<int>();
        for (int i = 0; i < bucketSizes.Count; i++)
        {
            if (bucketSizes[i] <= 0)
                throw new ArgumentOutOfRangeException(nameof(bucketSizes), "Bucket sizes must be positive.");
            distinct.Add(bucketSizes[i]);
        }

        _sortedBuckets = new int[distinct.Count];
        distinct.CopyTo(_sortedBuckets);
        _elementsPerUnit = workspaceElementsPerUnit;

        int maxBucket = _sortedBuckets[_sortedBuckets.Length - 1];
        int maxArenaElements = checked(maxBucket * _elementsPerUnit);
        _sharedArena = _backend.AllocateBuffer(maxArenaElements);

        _entries = new Dictionary<int, BucketEntry>(_sortedBuckets.Length);
        for (int i = 0; i < _sortedBuckets.Length; i++)
        {
            int cap = _sortedBuckets[i];
            int viewLen = checked(cap * _elementsPerUnit);
            IGpuBuffer view = (_backend is IGpuBufferViews viewBackend
                ? viewBackend.TryCreateView(_sharedArena, 0, viewLen)
                : null) ?? new OffsetBufferView(_sharedArena, 0, viewLen);

            var scope = new StaticShapeCudaGraphScope(_backend, scopeKind, options);
            _entries[cap] = new BucketEntry(cap, view, scope);
        }
    }

    /// <summary>Physical element capacity of the single shared arena backing all graph buckets.</summary>
    public int SharedArenaCapacityElements => _sharedArena.Size;

    /// <summary>Shared root arena pointer shared identically across every bucket's workspace view.</summary>
    public IntPtr SharedArenaHandle => _sharedArena.Handle;

    /// <summary>Number of configured sequence/batch SizeClass buckets.</summary>
    public int BucketCount => _sortedBuckets.Length;

    /// <summary>Resolves the smallest configured bucket capacity <c>&gt;= requestedUnits</c>, or -1 if larger than max bucket.</summary>
    public int ResolveBucket(int requestedUnits)
    {
        if (requestedUnits <= 0) return -1;
        int lo = 0;
        int hi = _sortedBuckets.Length - 1;
        int found = -1;
        while (lo <= hi)
        {
            int mid = lo + ((hi - lo) >> 1);
            if (_sortedBuckets[mid] >= requestedUnits)
            {
                found = _sortedBuckets[mid];
                hi = mid - 1;
            }
            else
            {
                lo = mid + 1;
            }
        }
        return found;
    }

    /// <summary>
    /// Routes <paramref name="requestedUnits"/> to its smallest fitting bucket, passing that bucket's shared-arena
    /// workspace view to <paramref name="bucketStep"/> and replaying its captured CUDA Graph.
    /// </summary>
    public int ExecuteForSize(
        int requestedUnits,
        Action<int, IGpuBuffer> bucketStep,
        IReadOnlyList<IGpuBuffer>? externalWatchedBuffers = null,
        long scalarSignature = 0)
    {
        if (bucketStep is null) throw new ArgumentNullException(nameof(bucketStep));
        lock (_gate)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(CudaGraphTreePool));
            int bucket = ResolveBucket(requestedUnits);
            if (bucket < 0)
                throw new ArgumentOutOfRangeException(nameof(requestedUnits),
                    $"Requested size {requestedUnits} exceeds maximum configured CUDAGraphTree bucket {_sortedBuckets[_sortedBuckets.Length - 1]}.");

            var entry = _entries[bucket];
            int extraCount = externalWatchedBuffers?.Count ?? 0;
            var watched = new IGpuBuffer[1 + extraCount];
            watched[0] = entry.ArenaView;
            if (externalWatchedBuffers is not null)
            {
                for (int i = 0; i < extraCount; i++)
                {
                    watched[1 + i] = externalWatchedBuffers[i];
                }
            }

            long combinedSig = unchecked((scalarSignature * 1099511628211L) ^ bucket);
            entry.Scope.Execute(() => bucketStep(bucket, entry.ArenaView), watched, combinedSig);
            return bucket;
        }
    }

    /// <summary>Inspects the <see cref="StaticShapeCudaGraphScope"/> of a specific bucket.</summary>
    public StaticShapeCudaGraphScope GetBucketScope(int bucketCapacity)
    {
        if (!_entries.TryGetValue(bucketCapacity, out var entry))
            throw new KeyNotFoundException($"No CUDAGraphTree bucket configured for capacity {bucketCapacity}.");
        return entry.Scope;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed) return;
            _disposed = true;
            foreach (var entry in _entries.Values)
            {
                entry.Dispose();
            }
            _entries.Clear();
            _sharedArena.Dispose();
        }
    }
}
