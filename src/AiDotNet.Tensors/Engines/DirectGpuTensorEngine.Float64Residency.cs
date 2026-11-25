// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

// Device residency for typed tensors: double, and the integer types (DirectGpuTensorEngine.Int32Residency).
//
// The registry, the pool and the budget below are one set for every element type; a resident records which type its
// buffer holds. The routes in this file are the double ones.
//
// A double op's result stays on the device as a Float64 buffer; its host array is filled on the first host read
// (HostSync), exactly as a float result's is. The next double op that takes the result as an operand reads the device
// buffer, so a chain of double ops downloads only what the host reads.
//
// These buffers are kept apart from the float activation cache on purpose: every reader of that cache treats its
// buffers as float (or FP16), and a Float64 buffer handed to a float kernel would be read as garbage.
//
// What is resident: op results, and tensors registered with RegisterPersistentTensor (uploaded once, on first use, and
// reused until a host write or InvalidatePersistentTensor). Any other host operand is uploaded for the one op and not
// cached: a caller can write the array it built the tensor over without the tensor seeing it, and a cached upload
// would then answer with stale values. Registration is the caller accepting that contract, as it is for float weights.
//
// A result is a host tensor with no host array (it is allocated by the first host read) and is held here through a
// weak reference to its data vector. Once nothing can read the tensor, its device buffer goes back to a pool without
// being downloaded, and the next result of that size reuses it: an intermediate of a chain costs no transfer, no host
// memory and, in steady state, no device allocation.
//
// Routing counts transfers, because they dominate. An op goes to the device only when that needs no more host<->device
// copies than the CPU engine would (which has to bring down every operand whose current values are on the device);
// two plain host operands always take the CPU.
public partial class DirectGpuTensorEngine
{
    private sealed class TypedResident
    {
        public TypedResident(object key, IGpuBuffer buffer, IDirectGpuBackend backend, GpuElementType elementType, int length,
            int hostVersion, long stamp, object? registration = null)
        {
            ElementType = elementType;
            Key = new WeakReference<object>(key);
            Registration = registration is null ? null : new WeakReference<object>(registration);
            Buffer = buffer;
            Backend = backend;
            Length = length;
            HostVersion = hostVersion;
            Stamp = stamp;
        }

        /// <summary>The tensor's data vector, which every view of it shares.</summary>
        public WeakReference<object> Key { get; }
        /// <summary>The registered persistent array this is a kept upload of (null for a device result): what
        /// invalidation names, and what any number of tensors over that array share.</summary>
        public WeakReference<object>? Registration { get; }
        public IGpuBuffer Buffer { get; }
        public IDirectGpuBackend Backend { get; }
        public int Length { get; }
        /// <summary>The storage's cache version when the device copy was written: a later host write advances it.</summary>
        public int HostVersion { get; }
        public long Stamp { get; set; }
        public bool Released { get; set; }
        /// <summary>What the buffer holds; a tensor's element type never changes, so neither does this.</summary>
        public GpuElementType ElementType { get; }
        public long Bytes => (long)Length * GpuElementTypes.SizeInBytes(ElementType);
    }

    /// <summary>
    /// Device bytes this engine's double buffers (resident tensors and the pool) may hold before unreachable results
    /// are collected and, failing that, the least recently used are brought back to the host. Null: a quarter of the
    /// device's memory, and at least 256 MB.
    /// </summary>
    internal static long? TypedResidentBudgetOverride;

    private static long TypedResidentBudget(IDirectGpuBackend backend)
        => TypedResidentBudgetOverride ?? Math.Max(256L * 1024 * 1024, backend.GlobalMemoryBytes / 4);

    /// <summary>
    /// The shortest double tensor an element-wise op or a resident reduction runs on the device. A kernel launch has a
    /// fixed cost the CPU engine beats on small tensors even when nothing has to be copied.
    /// </summary>
    internal const int Float64ElementwiseMinLength = 1 << 17;

    // Held for a whole double op: operands resolved under it stay allocated until the op has launched.
    private readonly object _typedLock = new();
    // By data-vector identity, without keeping the vector alive; `_typedAll` is the same set, for sweeping and trimming.
    private readonly ConditionalWeakTable<object, TypedResident> _typedByVector = new();
    private readonly List<TypedResident> _typedAll = new();
    // Buffers of results nothing can read any more, by element count, ready for the next result of that size.
    private readonly GpuTypedBufferPool _typedPool = new(IsLiveDeviceBuffer);
    private long _typedResidentBytes;
    private long _typedStamp;

    /// <summary>Device bytes currently held by resident double tensors.</summary>
    internal long TypedResidentBytes
    {
        get { lock (_typedLock) return _typedResidentBytes; }
    }

    /// <summary>Device bytes held by pooled buffers of results nothing can read any more.</summary>
    internal long TypedPooledBytes
    {
        get { lock (_typedLock) return _typedPool.Bytes; }
    }

    /// <summary>Whether <paramref name="tensor"/>'s current values are in a Float64 buffer on the device.</summary>
    internal bool IsTypedResident<T>(Tensor<T> tensor)
    {
        lock (_typedLock) return FindResident(tensor) is not null;
    }

    private static GpuFloat64BinaryOperation? Float64BinaryOperationFor(string callerName) => callerName switch
    {
        nameof(TensorAdd) => GpuFloat64BinaryOperation.Add,
        nameof(TensorSubtract) => GpuFloat64BinaryOperation.Subtract,
        nameof(TensorMultiply) => GpuFloat64BinaryOperation.Multiply,
        nameof(TensorDivide) => GpuFloat64BinaryOperation.Divide,
        _ => null,
    };

    /// <summary>
    /// Runs a double element-wise op on the device and leaves its result there. Null when this op, backend, policy or
    /// scope has no double route, or the CPU engine is the cheaper one for these operands; the caller then takes it,
    /// and it is exact too.
    /// </summary>
    private Tensor<double>? TryBinaryFloat64(Tensor<double> left, Tensor<double> right, string callerName)
    {
        if (Float64BinaryOperationFor(callerName) is not { } operation) return null;
        int length = left.Length;
        if (length < Float64ElementwiseMinLength || right.Length != length) return null;
        // Element i of one operand meets element i of the other in storage order: operands laid out differently are the
        // CPU engine's, which reorders them.
        if (left.Layout != right.Layout) return null;
        // A compiled step replays launches through baked buffer addresses; these buffers are not part of its plan.
        if (ResidentStepActive || !IsGpuPrecisionSafe(left, right) || !TryGetBackend(out var backend)) return null;
        if (backend is not IGpuFloat64Elementwise { SupportsFloat64Elementwise: true } elementwise) return null;
        if (!backend.SupportsElementType(GpuElementType.Float64)) return null;
        // The policy decides: the default for double preserves it (this route), a speed-first scope narrows to float.
        var plan = GpuPrecisionPlanner.CreatePlan<double>(backend, PrecisionOperationForBinary(callerName), callerName);
        if (plan.Route != GpuExecutionRoute.Gpu || plan.InputStorage != GpuScalarType.Float64) return null;

        lock (_typedLock)
        {
            CountResidentTransfers(left, out int uploads, out int downloads);
            if (!ReferenceEquals(left, right))
            {
                CountResidentTransfers(right, out int rightUploads, out int rightDownloads);
                uploads += rightUploads;
                downloads += rightDownloads;
            }

            if (uploads > downloads) return null;

            using var a = ResolveResidentOperand(backend, left, GpuElementType.Float64);
            using var b = ReferenceEquals(left, right)
                ? new OwnedBuffer(a.Buffer, ownsBuffer: false)
                : ResolveResidentOperand(backend, right, GpuElementType.Float64, left.DataVector);
            var output = RentResidentBuffer(backend, GpuElementType.Float64, length, left.DataVector, right.DataVector);
            try
            {
                elementwise.BinaryFloat64(operation, a.Buffer, b.Buffer, output, length);
                var result = RegisterResidentResult<double>(backend, output, length, left.Shape.ToArray(), GpuElementType.Float64);
                result.Layout = left.Layout;
                GpuPrecisionDiagnostics.Publish(plan);
                return result;
            }
            catch (NotSupportedException capabilityGap)
            {
                output.Dispose();
                GpuLaunchProbe.OnFallback(callerName, capabilityGap);
                return null;
            }
            catch
            {
                output.Dispose();
                throw;
            }
        }
    }

    private static GpuFloat64UnaryOperation? Float64UnaryOperationFor(string callerName) => callerName switch
    {
        nameof(TensorNegate) => GpuFloat64UnaryOperation.Negate,
        nameof(TensorAbs) => GpuFloat64UnaryOperation.Abs,
        nameof(TensorSqrt) => GpuFloat64UnaryOperation.Sqrt,
        nameof(TensorExp) => GpuFloat64UnaryOperation.Exp,
        nameof(TensorLog) => GpuFloat64UnaryOperation.Log,
        nameof(TensorTanh) => GpuFloat64UnaryOperation.Tanh,
        nameof(TensorSigmoid) or nameof(Sigmoid) => GpuFloat64UnaryOperation.Sigmoid,
        nameof(TensorReLU) => GpuFloat64UnaryOperation.Relu,
        _ => null,
    };

    /// <summary>
    /// A double unary op on the device, the result left there. Null when the op has no double kernel or the CPU engine
    /// is the cheaper route (the operand would have to be uploaded).
    /// </summary>
    private Tensor<double>? TryUnaryFloat64(Tensor<double> input, string callerName)
        => Float64UnaryOperationFor(callerName) is { } operation
            ? TryMapFloat64(input, callerName, (elementwise, a, output, size) => elementwise.UnaryFloat64(operation, a, output, size),
                elementwise => elementwise is not IGpuFloat64UnaryAvailability availability || availability.SupportsFloat64Unary(operation))
            : null;

    /// <summary>
    /// <c>input (op) scalar</c> in double on the device, the result left there. Null when the CPU engine is the cheaper
    /// route, or under a graph trace or anomaly mode (which keep the CPU engine's path, as the float route does).
    /// </summary>
    private Tensor<double>? TryScalarFloat64(Tensor<double> input, double scalar, GpuFloat64BinaryOperation operation, string callerName)
        => Compilation.GraphMode.IsActive || Autodiff.AnomalyModeScope.IsActive
            ? null
            : TryMapFloat64(input, callerName, (elementwise, a, output, size) => elementwise.ScalarFloat64(operation, a, scalar, output, size));

    private Tensor<double>? TryMapFloat64(Tensor<double> input, string callerName, Action<IGpuFloat64Elementwise, IGpuBuffer, IGpuBuffer, int> launch,
        Func<IGpuFloat64Elementwise, bool>? available = null)
    {
        int length = input.Length;
        if (length < Float64ElementwiseMinLength) return null;
        if (ResidentStepActive || !IsGpuPrecisionSafe(input) || !TryGetBackend(out var backend)) return null;
        if (backend is not IGpuFloat64Elementwise { SupportsFloat64Elementwise: true } elementwise) return null;
        // A kernel the backend already knows it cannot build: answer on the CPU before any operand is uploaded.
        if (available is not null && !available(elementwise)) return null;
        if (!backend.SupportsElementType(GpuElementType.Float64) || !GpuPrecisionPlanner.Float64DeviceRouteAllowed(backend)) return null;
        // Only the type-preserving policy takes this route; a speed-first scope narrows to float on its own path.
        if (AutocastScope.IsEnabled
            || GpuExecutionPolicyScope.EffectivePolicyFor(typeof(double)).AccuracyMode != GpuAccuracyMode.PreserveInputType)
            return null;

        lock (_typedLock)
        {
            CountResidentTransfers(input, out int uploads, out int downloads);
            if (uploads > downloads) return null;

            using var a = ResolveResidentOperand(backend, input, GpuElementType.Float64);
            var output = RentResidentBuffer(backend, GpuElementType.Float64, length, input.DataVector, input.DataVector);
            try
            {
                launch(elementwise, a.Buffer, output, length);
                var result = RegisterResidentResult<double>(backend, output, length, input.Shape.ToArray(), GpuElementType.Float64);
                // The kernel maps the elements in storage order, so the result is laid out as its input is (a packed
                // NCHWc tensor stays packed), as the CPU engine's elementwise ops leave it.
                result.Layout = input.Layout;
                return result;
            }
            catch (NotSupportedException capabilityGap)
            {
                output.Dispose();
                GpuLaunchProbe.OnFallback(callerName, capabilityGap);
                return null;
            }
            catch
            {
                output.Dispose();
                throw;
            }
        }
    }

    /// <summary>
    /// The least work (m x n x k) at which a double matrix multiply is considered for the device at all. Below it the
    /// launch alone costs more than the CPU engine's multiply (measured: 256 cubed takes 0.17 ms on the CPU engine and
    /// 0.36 ms on an OpenCL device with the operands already resident).
    /// </summary>
    internal const long Float64MatMulMinWork = 1L << 25;

    /// <summary>Test override: true sends every eligible double matrix multiply to the device, false to the CPU.</summary>
    internal static bool? Float64MatMulForceDevice;

    // Whether the device won, per (log2 of the work, operands need uploading). Above the floor neither side wins
    // everywhere: a tuned many-core CPU multiply beats a mid-range GPU's double kernel (measured here: 56 ms against
    // 110 ms at 2048 cubed) while a vendor dgemm on a workstation GPU beats the CPU several times over. So the first
    // multiply of each size class runs both and the faster one is used from then on.
    private readonly Dictionary<(int WorkLog2, bool Uploads), bool> _float64MatMulDeviceWins = new();

    /// <summary>How many size classes this engine has measured (both routes timed) so far. Diagnostic.</summary>
    internal long Float64MatMulMeasurements => Interlocked.Read(ref _float64MatMulMeasurements);
    private long _float64MatMulMeasurements;

    /// <summary>Forgets every measured matrix-multiply route, so the next multiply of each size class measures again.</summary>
    internal void ResetFloat64MatMulDecisions()
    {
        lock (_typedLock) _float64MatMulDeviceWins.Clear();
    }

    /// <summary>
    /// <c>a · b</c> for double operands, computed by whichever of the device and the CPU engine is faster for this size
    /// class. A device product stays on the device. Null when there is no double routine for these operands (the caller
    /// then takes the CPU engine).
    /// </summary>
    private Tensor<T>? TryMatMulFloat64<T>(IDirectGpuBackend backend, Tensor<T> a, Tensor<T> b, int m, int n, int k, int[] shape)
    {
        if (a is not Tensor<double> left || b is not Tensor<double> right) return null;
        if (ResidentStepActive || !IsGpuPrecisionSafe(a, b)) return null;
        if (backend is not IGpuFloat64MatrixMultiply { SupportsFloat64MatrixMultiply: true } multiply) return null;
        if (!backend.SupportsElementType(GpuElementType.Float64) || !GpuPrecisionPlanner.Float64DeviceRouteAllowed(backend)) return null;
        if (m <= 0 || n <= 0 || k <= 0 || left.Length != (long)m * k || right.Length != (long)k * n) return null;
        long work = (long)m * n * k;
        long productLength = (long)m * n;
        if (productLength > int.MaxValue) return null;
        if (Float64MatMulForceDevice == false || (Float64MatMulForceDevice is null && work < Float64MatMulMinWork)) return null;
        try
        {
            lock (_typedLock)
            {
                CountResidentTransfers(left, out int uploads, out _);
                if (!ReferenceEquals(left, right))
                {
                    CountResidentTransfers(right, out int rightUploads, out _);
                    uploads += rightUploads;
                }

                if (Float64MatMulForceDevice == true)
                    return DoublesAs<T>(MatMulFloat64OnDevice(backend, multiply, left, right, m, n, k, shape));

                int workLog2 = 0;
                for (long remaining = work; remaining > 1; remaining >>= 1) workLog2++;
                var sizeClass = (workLog2, uploads > 0);
                if (_float64MatMulDeviceWins.TryGetValue(sizeClass, out bool deviceWins))
                    return deviceWins ? DoublesAs<T>(MatMulFloat64OnDevice(backend, multiply, left, right, m, n, k, shape)) : null;

                // First multiply of this size class: run both, twice each (the first run of either pays one-time
                // costs), and keep the faster. Its product is returned, so nothing is computed a third time.
                Interlocked.Increment(ref _float64MatMulMeasurements);
                var clock = new System.Diagnostics.Stopwatch();
                var warmUp = MatMulFloat64OnDevice(backend, multiply, left, right, m, n, k, shape);
                backend.Synchronize();
                // Nothing reads the warm-up product: its buffer goes back for the timed run.
                PoolFloat64Result(warmUp);
                clock.Restart();
                var deviceProduct = MatMulFloat64OnDevice(backend, multiply, left, right, m, n, k, shape);
                backend.Synchronize();
                long deviceTicks = clock.ElapsedTicks;

                Tensor<double> hostProduct;
                long hostTicks;
                // Not recorded: the caller records the one product it returns.
                using (Autodiff.GradientTape<double>.NoGrad())
                {
                    _ = base.TensorMatMul(left, right);
                    clock.Restart();
                    hostProduct = base.TensorMatMul(left, right);
                    hostTicks = clock.ElapsedTicks;
                }

                deviceWins = deviceTicks < hostTicks;
                _float64MatMulDeviceWins[sizeClass] = deviceWins;
                if (deviceWins) return DoublesAs<T>(deviceProduct);
                PoolFloat64Result(deviceProduct);
                return DoublesAs<T>(hostProduct);
            }
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("TensorMatMulFloat64", capabilityGap);
            return null;
        }
    }

    /// <summary>A double tensor as the <c>Tensor&lt;T&gt;</c> it is when <typeparamref name="T"/> is double.</summary>
    internal static Tensor<T> DoublesAs<T>(Tensor<double> tensor)
        => tensor as Tensor<T> ?? throw new InvalidCastException($"A double tensor is not a Tensor<{typeof(T).Name}>.");

    // Callers hold _typedLock. Returns the buffer of a device result that no caller will ever see to the pool.
    private void PoolFloat64Result(Tensor<double> unseen)
    {
        if (_typedByVector.TryGetValue(unseen.DataVector, out var resident))
            ReleaseResident(resident, bringToHost: false, toPool: true);
    }

    // Callers hold _typedLock.
    private Tensor<double> MatMulFloat64OnDevice(IDirectGpuBackend backend, IGpuFloat64MatrixMultiply multiply,
        Tensor<double> left, Tensor<double> right, int m, int n, int k, int[] shape)
    {
        using var operandA = ResolveResidentOperand(backend, left, GpuElementType.Float64);
        using var operandB = ReferenceEquals(left, right)
            ? new OwnedBuffer(operandA.Buffer, ownsBuffer: false)
            : ResolveResidentOperand(backend, right, GpuElementType.Float64, left.DataVector);
        var output = RentResidentBuffer(backend, GpuElementType.Float64, m * n, left.DataVector, right.DataVector);
        try
        {
            multiply.MatrixMultiplyFloat64(operandA.Buffer, operandB.Buffer, output, m, n, k);
        }
        catch
        {
            output.Dispose();
            throw;
        }

        return RegisterResidentResult<double>(backend, output, m * n, shape, GpuElementType.Float64);
    }

    /// <summary>The sum of a double tensor computed on the device, when that is the cheaper route.</summary>
    private bool TrySumFloat64Tensor(Tensor<double> tensor, out double sum, GpuFloat64Reduction reduction = GpuFloat64Reduction.Sum)
    {
        sum = 0.0;
        int length = tensor.Length;
        if (length == 0 || ResidentStepActive || !IsGpuPrecisionSafe(tensor) || !TryGetBackend(out var backend)) return false;
        if (backend is not IGpuFloat64Reductions { SupportsFloat64Reductions: true } reductions
            || !backend.SupportsElementType(GpuElementType.Float64) || !GpuPrecisionPlanner.Float64DeviceRouteAllowed(backend))
            return false;
        try
        {
            lock (_typedLock)
            {
                // A tensor already on the device is reduced there at any size the kernel launch pays for; a host
                // tensor pays an upload, which only the vector threshold's sizes recover.
                CountResidentTransfers(tensor, out int uploads, out _);
                if (length < (uploads == 0 ? Float64ElementwiseMinLength : Float64ReductionMinLength)) return false;
                using var operand = ResolveResidentOperand(backend, tensor, GpuElementType.Float64);
                sum = reduction switch
                {
                    GpuFloat64Reduction.Max => reductions.MaxFloat64(operand.Buffer, length),
                    GpuFloat64Reduction.Min => reductions.MinFloat64(operand.Buffer, length),
                    _ => reductions.SumFloat64(operand.Buffer, length),
                };
                return true;
            }
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("TensorSumFloat64", capabilityGap);
            return false;
        }
        catch (GpuBufferTooLargeException tooLarge)
        {
            // The CPU engine's under the default chunking policy; rethrown under fail-fast, as the float path does.
            HandleBufferTooLarge<double>(tooLarge, "TensorSumFloat64");
            return false;
        }
    }

    // Callers hold _typedLock. The host<->device copies `tensor` costs each route: an upload when the device route
    // would have to send it (not resident, and not a registered persistent tensor whose one upload is amortized), a
    // download when the CPU route would have to bring it down (its current values are only on the device).
    private void CountResidentTransfers<T>(Tensor<T> tensor, out int uploads, out int downloads)
    {
        uploads = 0;
        downloads = 0;
        if (FindResident(tensor) is not null)
        {
            if (Helpers.HostSync.IsPending(tensor.DataVector)) downloads = 1;
            return;
        }

        if (!IsPersistentResident(tensor)) uploads = 1;
    }

    private bool IsPersistentResident<T>(Tensor<T> tensor)
        => tensor.CoversWholeVector
           && tensor.GetBackingArrayForCacheLookupUnsafe() is { } key
           && key.Length == tensor.Length
           && _pendingPersistentRoles.TryGetValue(key, out _);

    /// <summary>Frees the device copy of a persistent double tensor whose host values changed or that was unregistered.</summary>
    private void DropResident<T>(Tensor<T> tensor)
    {
        lock (_typedLock)
        {
            if (_typedByVector.TryGetValue(tensor.DataVector, out var resident)) ReleaseResident(resident, bringToHost: true);
            // The registration is the ARRAY: another tensor wrapping the same array through its own vector has its own
            // kept upload of it, which the host write made just as stale.
            if (tensor.GetBackingArrayForCacheLookupUnsafe() is not { } registered) return;
            var sameArray = _typedAll
                .Where(other => other.Registration is { } weak && weak.TryGetTarget(out var array) && ReferenceEquals(array, registered))
                .ToList();
            foreach (var other in sameArray) ReleaseResident(other, bringToHost: false);
        }
    }

    // Callers hold _typedLock.
    private TypedResident? FindResident<T>(Tensor<T> tensor)
    {
        if (_typedAll.Count == 0 || !tensor.IsContiguous || tensor._storageOffset != 0) return null;
        object key = tensor.DataVector;
        if (!_typedByVector.TryGetValue(key, out var resident)) return null;
        bool current = resident.HostVersion == tensor.GpuCacheVersion && IsLiveDeviceBuffer(resident.Buffer);
        if (current && resident.Length == tensor.Length)
        {
            resident.Stamp = ++_typedStamp;
            return resident;
        }

        // A view of another length reads the same values: it is uploaded, and the resident copy stays valid.
        if (current) return null;
        // The host wrote the tensor, so the device copy is stale and the host copy is the current one: a download still
        // pending must not overwrite it.
        Helpers.HostSync.Remove(key);
        ReleaseResident(resident, bringToHost: false);
        return null;
    }

    // Callers hold _typedLock.
    // `alsoKeep` is the storage of an operand already resolved for the same launch: making room for this one must
    // not evict it, or the launch would read a buffer the eviction disposed.
    private OwnedBuffer ResolveResidentOperand<T>(IDirectGpuBackend backend, Tensor<T> tensor, GpuElementType type, object? alsoKeep = null)
        where T : unmanaged
    {
        if (FindResident(tensor) is { } resident && ReferenceEquals(resident.Backend, backend))
            return new OwnedBuffer(resident.Buffer, ownsBuffer: false);

        var contiguous = !tensor.IsContiguous || tensor._storageOffset != 0 ? tensor.Contiguous() : tensor;
        // The read-only accessor brings a pending device result down first, so the upload sees current values.
        var host = contiguous.GetReadOnlyDataArray();
        bool keep = ReferenceEquals(contiguous, tensor) && IsPersistentResident(tensor) && !_typedByVector.TryGetValue(tensor.DataVector, out _);
        var upload = keep
            ? RentResidentBuffer(backend, type, contiguous.Length, tensor.DataVector, alsoKeep ?? tensor.DataVector)
            : backend.AllocateBuffer(contiguous.Length, type);
        try
        {
            backend.UploadBuffer<T>(upload, new ReadOnlySpan<T>(host, 0, contiguous.Length));
        }
        catch
        {
            upload.Dispose();
            throw;
        }

        if (keep)
        {
            // Registered persistent: this upload is kept and reused until the host writes the tensor.
            AddResident(new TypedResident(tensor.DataVector, upload, backend, type, tensor.Length, tensor.GpuCacheVersion, ++_typedStamp,
                registration: tensor.GetBackingArrayForCacheLookupUnsafe()));
            return new OwnedBuffer(upload, ownsBuffer: false);
        }

        return new OwnedBuffer(upload, ownsBuffer: true);
    }

    // Callers hold _typedLock.
    private Tensor<T> RegisterResidentResult<T>(IDirectGpuBackend backend, IGpuBuffer output, int length, int[] shape, GpuElementType type)
        where T : unmanaged
    {
        // No host array: the first host read allocates one and fills it from the device; a chain's intermediates never
        // get one.
        var result = Tensor<T>.CreateHostDeferred(shape);
        var vector = result.DataVector;
        // Only the buffer and backend are captured: the registration must not keep the tensor alive.
        var capturedBackend = backend;
        var capturedBuffer = output;
        Helpers.HostSync.Register(vector, owner =>
        {
            var destination = new T[length];
            capturedBackend.DownloadBuffer<T>(capturedBuffer, destination);
            ((VectorBase<T>)owner).MaterializeBacking(destination);
        });
        // Read after the registration: a host write from here on advances the version, and the device copy is then
        // recognised as stale.
        AddResident(new TypedResident(vector, output, backend, type, length, result.GpuCacheVersion, ++_typedStamp));
        return result;
    }

    // Callers hold _typedLock.
    private void AddResident(TypedResident resident)
    {
        if (!resident.Key.TryGetTarget(out var key)) return;
        _typedByVector.Add(key, resident);
        _typedAll.Add(resident);
        _typedResidentBytes += resident.Bytes;
    }

    // Callers hold _typedLock. Pools the buffers of results whose tensor is unreachable: nothing can read them.
    private void SweepResidents()
    {
        for (int i = _typedAll.Count - 1; i >= 0; i--)
        {
            var resident = _typedAll[i];
            if (!resident.Key.TryGetTarget(out _)) ReleaseResident(resident, bringToHost: false, toPool: true);
        }
    }

    // Callers hold _typedLock. A buffer for `length` doubles: a pooled one when there is one, a new one within the
    // budget, and otherwise whatever room reclaiming makes, never taking this op's operands.
    private IGpuBuffer RentResidentBuffer(IDirectGpuBackend backend, GpuElementType type, int length, object keepLeft, object keepRight)
    {
        long bytes = (long)length * GpuElementTypes.SizeInBytes(type);
        long budget = TypedResidentBudget(backend);
        // The fast path: a pooled buffer, while the results on the device fit the budget with it. Past the budget the
        // dropped results are reclaimed first, pooled buffer or not; they would otherwise be held until the pool ran dry.
        if (_typedResidentBytes + bytes <= budget && _typedPool.TryTake(backend, type, length, out var ready)) return ready;
        // Results a chain has dropped are reclaimed only once the collector has noticed them, and it
        // cannot see the device memory they hold: after enough new device memory, ask the cheap generations (a result
        // tensor is a small young object), so a loop reuses its buffers long before it reaches the budget.
        SweepResidents();
        if (_typedAllocatedSinceCollect + bytes > Math.Max(64L * 1024 * 1024, budget / 16))
        {
            GC.Collect(1, GCCollectionMode.Forced, blocking: true);
            SweepResidents();
            _typedAllocatedSinceCollect = 0;
        }

        for (int attempt = 0; ; attempt++)
        {
            if (_typedPool.TryTake(backend, type, length, out var pooled))
            {
                // Still over with every dropped result reclaimed: the rest are reachable, so the oldest come down.
                if (_typedResidentBytes + bytes > budget)
                {
                    // The buffer is this method's until it is returned: nothing else can free it if making room throws.
                    try
                    {
                        _typedPool.Trim();
                        EvictResidents(budget - bytes, keepLeft, keepRight);
                    }
                    catch
                    {
                        pooled.Dispose();
                        throw;
                    }
                }

                return pooled;
            }

            if (_typedResidentBytes + _typedPool.Bytes + bytes <= budget) break;
            if (attempt == 0)
            {
                // Pooled buffers of other sizes are only memory now.
                _typedPool.Trim();
            }
            else if (attempt == 1)
            {
                // Over budget: a full collection finds the dropped results the cheap ones could not (old tensors).
                GC.Collect();
                SweepResidents();
                _typedPool.Trim(type, length);
            }
            else
            {
                // Every remaining result is reachable: bring the least recently used down to make room.
                EvictResidents(budget - bytes, keepLeft, keepRight);
                break;
            }
        }

        _typedAllocatedSinceCollect += bytes;
        return backend.AllocateBuffer(length, type);
    }

    private long _typedAllocatedSinceCollect;

    // Callers hold _typedLock.
    private void EvictResidents(long targetBytes, object keepLeft, object keepRight)
    {
        var oldestFirst = new List<TypedResident>(_typedAll);
        oldestFirst.Sort(static (x, y) => x.Stamp.CompareTo(y.Stamp));
        foreach (var resident in oldestFirst)
        {
            if (_typedResidentBytes + _typedPool.Bytes <= targetBytes) break;
            if (resident.Key.TryGetTarget(out var key) && (ReferenceEquals(key, keepLeft) || ReferenceEquals(key, keepRight)))
                continue;
            try
            {
                ReleaseResident(resident, bringToHost: true);
            }
            catch (InvalidOperationException downloadFailure)
            {
                // Its values could not be brought down, so it keeps its buffer (a later read retries the download)
                // and the room is made from the next oldest instead.
                System.Diagnostics.Trace.TraceWarning(
                    "[AiDotNet] A resident double tensor could not be brought back to the host and stays on the device: {0}",
                    downloadFailure);
            }
        }
    }

    // Callers hold _typedLock. With `bringToHost`, a result the host has not read yet is downloaded first, so no
    // reachable tensor loses its values. With `toPool`, the buffer is kept for the next result of its size.
    private void ReleaseResident(TypedResident resident, bool bringToHost, bool toPool = false)
    {
        if (resident.Released) return;
        resident.Key.TryGetTarget(out var key);
        // Before anything is given up: the pending download reads this buffer, and a failed one stays pending for a
        // retry, so a throw here leaves the resident registered with its buffer. Decided under the storage's lock,
        // which also waits for a download another thread is running on this buffer.
        if (bringToHost && key is not null) Helpers.HostSync.Settle(key);
        resident.Released = true;
        _typedAll.Remove(resident);
        _typedResidentBytes -= resident.Bytes;
        bool pooled = false;
        try
        {
            if (key is not null) _typedByVector.Remove(key);

            pooled = toPool && _typedPool.TryReturn(resident.Backend, resident.Buffer);
        }
        finally
        {
            if (!pooled) resident.Buffer.Dispose();
        }
    }

    /// <summary>Brings every resident double tensor back to the host and frees its device buffer.</summary>
    internal void ClearTypedResidents()
    {
        lock (_typedLock)
        {
            foreach (var resident in new List<TypedResident>(_typedAll))
            {
                try
                {
                    ReleaseResident(resident, bringToHost: true);
                }
                catch (InvalidOperationException downloadFailure)
                {
                    // The device is already gone (teardown after a lost context): the values cannot be recovered, and
                    // the buffer still has to be freed. The result is marked unavailable, so a later host read reports
                    // that instead of running a download against the freed buffer.
                    System.Diagnostics.Trace.TraceWarning(
                        "[AiDotNet] A resident double tensor could not be brought back to the host: {0}", downloadFailure);
                    if (resident.Key.TryGetTarget(out var lost))
                        Helpers.HostSync.Release(lost,
                            "This tensor's values were on a GPU device that was released before they could be downloaded.");
                    ReleaseResident(resident, bringToHost: false);
                }
            }

            _typedPool.Clear();
        }
    }
}
