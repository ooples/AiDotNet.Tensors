// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

// Device residency for double tensors.
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
    private sealed class Float64Resident
    {
        public Float64Resident(object key, IGpuBuffer buffer, IDirectGpuBackend backend, int length, int hostVersion, long stamp)
        {
            Key = new WeakReference<object>(key);
            Buffer = buffer;
            Backend = backend;
            Length = length;
            HostVersion = hostVersion;
            Stamp = stamp;
        }

        /// <summary>The tensor's data vector, which every view of it shares.</summary>
        public WeakReference<object> Key { get; }
        public IGpuBuffer Buffer { get; }
        public IDirectGpuBackend Backend { get; }
        public int Length { get; }
        /// <summary>The storage's cache version when the device copy was written: a later host write advances it.</summary>
        public int HostVersion { get; }
        public long Stamp { get; set; }
        public bool Released { get; set; }
        public long Bytes => (long)Length * sizeof(double);
    }

    /// <summary>
    /// Device bytes this engine's double buffers (resident tensors and the pool) may hold before unreachable results
    /// are collected and, failing that, the least recently used are brought back to the host. Null: a quarter of the
    /// device's memory, and at least 256 MB.
    /// </summary>
    internal static long? Float64ResidentBudgetOverride;

    private static long Float64ResidentBudget(IDirectGpuBackend backend)
        => Float64ResidentBudgetOverride ?? Math.Max(256L * 1024 * 1024, backend.GlobalMemoryBytes / 4);

    /// <summary>
    /// The shortest double tensor an element-wise op or a resident reduction runs on the device. A kernel launch has a
    /// fixed cost the CPU engine beats on small tensors even when nothing has to be copied.
    /// </summary>
    internal const int Float64ElementwiseMinLength = 1 << 17;

    // Held for a whole double op: operands resolved under it stay allocated until the op has launched.
    private readonly object _float64Lock = new();
    // By data-vector identity, without keeping the vector alive; `_float64All` is the same set, for sweeping and trimming.
    private readonly ConditionalWeakTable<object, Float64Resident> _float64ByVector = new();
    private readonly List<Float64Resident> _float64All = new();
    // Buffers of results nothing can read any more, by element count, ready for the next result of that size.
    private readonly Dictionary<int, Stack<IGpuBuffer>> _float64Pool = new();
    private long _float64ResidentBytes;
    private long _float64PoolBytes;
    private long _float64Stamp;

    /// <summary>Device bytes currently held by resident double tensors.</summary>
    internal long Float64ResidentBytes
    {
        get { lock (_float64Lock) return _float64ResidentBytes; }
    }

    /// <summary>Device bytes held by pooled buffers of results nothing can read any more.</summary>
    internal long Float64PooledBytes
    {
        get { lock (_float64Lock) return _float64PoolBytes; }
    }

    /// <summary>Whether <paramref name="tensor"/>'s current values are in a Float64 buffer on the device.</summary>
    internal bool IsFloat64Resident(Tensor<double> tensor)
    {
        lock (_float64Lock) return FindFloat64Resident(tensor) is not null;
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
        // A compiled step replays launches through baked buffer addresses; these buffers are not part of its plan.
        if (ResidentStepActive || !IsGpuPrecisionSafe(left, right) || !TryGetBackend(out var backend)) return null;
        if (backend is not IGpuFloat64Elementwise { SupportsFloat64Elementwise: true } elementwise) return null;
        if (!backend.SupportsElementType(GpuElementType.Float64)) return null;
        // The policy decides: the default for double preserves it (this route), a speed-first scope narrows to float.
        var plan = GpuPrecisionPlanner.CreatePlan<double>(backend, PrecisionOperationForBinary(callerName), callerName);
        if (plan.Route != GpuExecutionRoute.Gpu || plan.InputStorage != GpuScalarType.Float64) return null;

        lock (_float64Lock)
        {
            CountFloat64Transfers(left, out int uploads, out int downloads);
            if (!ReferenceEquals(left, right))
            {
                CountFloat64Transfers(right, out int rightUploads, out int rightDownloads);
                uploads += rightUploads;
                downloads += rightDownloads;
            }

            if (uploads > downloads) return null;

            using var a = ResolveFloat64Operand(backend, left);
            using var b = ReferenceEquals(left, right) ? new OwnedBuffer(a.Buffer, ownsBuffer: false) : ResolveFloat64Operand(backend, right);
            var output = RentFloat64Buffer(backend, length, left, right);
            try
            {
                elementwise.BinaryFloat64(operation, a.Buffer, b.Buffer, output, length);
                var result = RegisterFloat64Result(backend, output, length, left.Shape.ToArray());
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
            ? TryMapFloat64(input, callerName, (elementwise, a, output, size) => elementwise.UnaryFloat64(operation, a, output, size))
            : null;

    /// <summary>
    /// <c>input (op) scalar</c> in double on the device, the result left there. Null when the CPU engine is the cheaper
    /// route, or under a graph trace or anomaly mode (which keep the CPU engine's path, as the float route does).
    /// </summary>
    private Tensor<double>? TryScalarFloat64(Tensor<double> input, double scalar, GpuFloat64BinaryOperation operation, string callerName)
        => Compilation.GraphMode.IsActive || Autodiff.AnomalyModeScope.IsActive
            ? null
            : TryMapFloat64(input, callerName, (elementwise, a, output, size) => elementwise.ScalarFloat64(operation, a, scalar, output, size));

    private Tensor<double>? TryMapFloat64(Tensor<double> input, string callerName, Action<IGpuFloat64Elementwise, IGpuBuffer, IGpuBuffer, int> launch)
    {
        int length = input.Length;
        if (length < Float64ElementwiseMinLength) return null;
        if (ResidentStepActive || !IsGpuPrecisionSafe(input) || !TryGetBackend(out var backend)) return null;
        if (backend is not IGpuFloat64Elementwise { SupportsFloat64Elementwise: true } elementwise) return null;
        if (!backend.SupportsElementType(GpuElementType.Float64)) return null;
        // Only the type-preserving policy takes this route; a speed-first scope narrows to float on its own path.
        if (AutocastScope.IsEnabled
            || GpuExecutionPolicyScope.EffectivePolicyFor(typeof(double)).AccuracyMode != GpuAccuracyMode.PreserveInputType)
            return null;

        lock (_float64Lock)
        {
            CountFloat64Transfers(input, out int uploads, out int downloads);
            if (uploads > downloads) return null;

            using var a = ResolveFloat64Operand(backend, input);
            var output = RentFloat64Buffer(backend, length, input, input);
            try
            {
                launch(elementwise, a.Buffer, output, length);
                return RegisterFloat64Result(backend, output, length, input.Shape.ToArray());
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
        if (!backend.SupportsElementType(GpuElementType.Float64)) return null;
        if (m <= 0 || n <= 0 || k <= 0 || left.Length != (long)m * k || right.Length != (long)k * n) return null;
        long work = (long)m * n * k;
        long productLength = (long)m * n;
        if (productLength > int.MaxValue) return null;
        if (Float64MatMulForceDevice == false || (Float64MatMulForceDevice is null && work < Float64MatMulMinWork)) return null;
        try
        {
            lock (_float64Lock)
            {
                CountFloat64Transfers(left, out int uploads, out _);
                if (!ReferenceEquals(left, right))
                {
                    CountFloat64Transfers(right, out int rightUploads, out _);
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

    // Callers hold _float64Lock. Returns the buffer of a device result that no caller will ever see to the pool.
    private void PoolFloat64Result(Tensor<double> unseen)
    {
        if (_float64ByVector.TryGetValue(unseen.DataVector, out var resident))
            ReleaseFloat64Resident(resident, bringToHost: false, toPool: true);
    }

    // Callers hold _float64Lock.
    private Tensor<double> MatMulFloat64OnDevice(IDirectGpuBackend backend, IGpuFloat64MatrixMultiply multiply,
        Tensor<double> left, Tensor<double> right, int m, int n, int k, int[] shape)
    {
        using var operandA = ResolveFloat64Operand(backend, left);
        using var operandB = ReferenceEquals(left, right) ? new OwnedBuffer(operandA.Buffer, ownsBuffer: false) : ResolveFloat64Operand(backend, right);
        var output = RentFloat64Buffer(backend, m * n, left, right);
        try
        {
            multiply.MatrixMultiplyFloat64(operandA.Buffer, operandB.Buffer, output, m, n, k);
        }
        catch
        {
            output.Dispose();
            throw;
        }

        return RegisterFloat64Result(backend, output, m * n, shape);
    }

    /// <summary>The sum of a double tensor computed on the device, when that is the cheaper route.</summary>
    private bool TrySumFloat64Tensor(Tensor<double> tensor, out double sum)
    {
        sum = 0.0;
        int length = tensor.Length;
        if (length == 0 || ResidentStepActive || !TryGetBackend(out var backend)) return false;
        if (backend is not IGpuFloat64Reductions { SupportsFloat64Reductions: true } reductions
            || !backend.SupportsElementType(GpuElementType.Float64))
            return false;
        try
        {
            lock (_float64Lock)
            {
                // A tensor already on the device is reduced there at any size the kernel launch pays for; a host
                // tensor pays an upload, which only the vector threshold's sizes recover.
                CountFloat64Transfers(tensor, out int uploads, out _);
                if (length < (uploads == 0 ? Float64ElementwiseMinLength : Float64ReductionMinLength)) return false;
                using var operand = ResolveFloat64Operand(backend, tensor);
                sum = reductions.SumFloat64(operand.Buffer, length);
                return true;
            }
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("TensorSumFloat64", capabilityGap);
            return false;
        }
    }

    // Callers hold _float64Lock. The host<->device copies `tensor` costs each route: an upload when the device route
    // would have to send it (not resident, and not a registered persistent tensor whose one upload is amortized), a
    // download when the CPU route would have to bring it down (its current values are only on the device).
    private void CountFloat64Transfers(Tensor<double> tensor, out int uploads, out int downloads)
    {
        uploads = 0;
        downloads = 0;
        if (FindFloat64Resident(tensor) is not null)
        {
            if (Helpers.HostSync.IsPending(tensor.DataVector)) downloads = 1;
            return;
        }

        if (!IsPersistentFloat64(tensor)) uploads = 1;
    }

    private bool IsPersistentFloat64(Tensor<double> tensor)
        => tensor.CoversWholeVector
           && tensor.GetBackingArrayForCacheLookupUnsafe() is { } key
           && key.Length == tensor.Length
           && _pendingPersistentRoles.TryGetValue(key, out _);

    /// <summary>Frees the device copy of a persistent double tensor whose host values changed or that was unregistered.</summary>
    private void DropFloat64Resident(Tensor<double> tensor)
    {
        lock (_float64Lock)
        {
            if (_float64ByVector.TryGetValue(tensor.DataVector, out var resident)) ReleaseFloat64Resident(resident, bringToHost: true);
        }
    }

    // Callers hold _float64Lock.
    private Float64Resident? FindFloat64Resident(Tensor<double> tensor)
    {
        if (_float64All.Count == 0 || !tensor.IsContiguous || tensor._storageOffset != 0) return null;
        object key = tensor.DataVector;
        if (!_float64ByVector.TryGetValue(key, out var resident)) return null;
        bool current = resident.HostVersion == tensor.GpuCacheVersion && IsLiveDeviceBuffer(resident.Buffer);
        if (current && resident.Length == tensor.Length)
        {
            resident.Stamp = ++_float64Stamp;
            return resident;
        }

        // A view of another length reads the same values: it is uploaded, and the resident copy stays valid.
        if (current) return null;
        // The host wrote the tensor, so the device copy is stale and the host copy is the current one: a download still
        // pending must not overwrite it.
        Helpers.HostSync.Remove(key);
        ReleaseFloat64Resident(resident, bringToHost: false);
        return null;
    }

    // Callers hold _float64Lock.
    private OwnedBuffer ResolveFloat64Operand(IDirectGpuBackend backend, Tensor<double> tensor)
    {
        if (FindFloat64Resident(tensor) is { } resident && ReferenceEquals(resident.Backend, backend))
            return new OwnedBuffer(resident.Buffer, ownsBuffer: false);

        var contiguous = !tensor.IsContiguous || tensor._storageOffset != 0 ? tensor.Contiguous() : tensor;
        // The read-only accessor brings a pending device result down first, so the upload sees current values.
        var host = contiguous.GetReadOnlyDataArray();
        bool keep = ReferenceEquals(contiguous, tensor) && IsPersistentFloat64(tensor) && !_float64ByVector.TryGetValue(tensor.DataVector, out _);
        var upload = keep ? RentFloat64Buffer(backend, contiguous.Length, tensor, tensor) : backend.AllocateBuffer(contiguous.Length, GpuElementType.Float64);
        try
        {
            backend.UploadBuffer<double>(upload, new ReadOnlySpan<double>(host, 0, contiguous.Length));
        }
        catch
        {
            upload.Dispose();
            throw;
        }

        if (keep)
        {
            // Registered persistent: this upload is kept and reused until the host writes the tensor.
            AddFloat64Resident(new Float64Resident(tensor.DataVector, upload, backend, tensor.Length, tensor.GpuCacheVersion, ++_float64Stamp));
            return new OwnedBuffer(upload, ownsBuffer: false);
        }

        return new OwnedBuffer(upload, ownsBuffer: true);
    }

    // Callers hold _float64Lock.
    private Tensor<double> RegisterFloat64Result(IDirectGpuBackend backend, IGpuBuffer output, int length, int[] shape)
    {
        // No host array: the first host read allocates one and fills it from the device; a chain's intermediates never
        // get one.
        var result = Tensor<double>.CreateHostDeferred(shape);
        var vector = result.DataVector;
        // Only the buffer and backend are captured: the registration must not keep the tensor alive.
        var capturedBackend = backend;
        var capturedBuffer = output;
        Helpers.HostSync.Register(vector, owner =>
        {
            var destination = new double[length];
            capturedBackend.DownloadBuffer<double>(capturedBuffer, destination);
            ((VectorBase<double>)owner).MaterializeBacking(destination);
        });
        // Read after the registration: a host write from here on advances the version, and the device copy is then
        // recognised as stale.
        AddFloat64Resident(new Float64Resident(vector, output, backend, length, result.GpuCacheVersion, ++_float64Stamp));
        return result;
    }

    // Callers hold _float64Lock.
    private void AddFloat64Resident(Float64Resident resident)
    {
        if (!resident.Key.TryGetTarget(out var key)) return;
        _float64ByVector.Add(key, resident);
        _float64All.Add(resident);
        _float64ResidentBytes += resident.Bytes;
    }

    // Callers hold _float64Lock. Pools the buffers of results whose tensor is unreachable: nothing can read them.
    private void SweepFloat64Residents()
    {
        for (int i = _float64All.Count - 1; i >= 0; i--)
        {
            var resident = _float64All[i];
            if (!resident.Key.TryGetTarget(out _)) ReleaseFloat64Resident(resident, bringToHost: false, toPool: true);
        }
    }

    // Callers hold _float64Lock. A buffer for `length` doubles: a pooled one when there is one, a new one within the
    // budget, and otherwise whatever room reclaiming makes, never taking this op's operands.
    private IGpuBuffer RentFloat64Buffer(IDirectGpuBackend backend, int length, Tensor<double> left, Tensor<double> right)
    {
        long bytes = (long)length * sizeof(double);
        long budget = Float64ResidentBudget(backend);
        // The fast path: a pooled buffer, while the results on the device fit the budget with it. Past the budget the
        // dropped results are reclaimed first, pooled buffer or not; they would otherwise be held until the pool ran dry.
        if (_float64ResidentBytes + bytes <= budget && TryTakePooledFloat64(backend, length, out var ready)) return ready;
        // Results a chain has dropped are reclaimed only once the collector has noticed them, and it
        // cannot see the device memory they hold: after enough new device memory, ask the cheap generations (a result
        // tensor is a small young object), so a loop reuses its buffers long before it reaches the budget.
        SweepFloat64Residents();
        if (_float64AllocatedSinceCollect + bytes > Math.Max(64L * 1024 * 1024, budget / 16))
        {
            GC.Collect(1, GCCollectionMode.Forced, blocking: true);
            SweepFloat64Residents();
            _float64AllocatedSinceCollect = 0;
        }

        for (int attempt = 0; ; attempt++)
        {
            if (TryTakePooledFloat64(backend, length, out var pooled))
            {
                // Still over with every dropped result reclaimed: the rest are reachable, so the oldest come down.
                if (_float64ResidentBytes + bytes > budget)
                {
                    DisposeFloat64Pool();
                    EvictFloat64Residents(budget - bytes, left, right);
                }

                return pooled;
            }

            if (_float64ResidentBytes + _float64PoolBytes + bytes <= budget) break;
            if (attempt == 0)
            {
                // Pooled buffers of other sizes are only memory now.
                DisposeFloat64Pool();
            }
            else if (attempt == 1)
            {
                // Over budget: a full collection finds the dropped results the cheap ones could not (old tensors).
                GC.Collect();
                SweepFloat64Residents();
                DisposeFloat64Pool(exceptLength: length);
            }
            else
            {
                // Every remaining result is reachable: bring the least recently used down to make room.
                EvictFloat64Residents(budget - bytes, left, right);
                break;
            }
        }

        _float64AllocatedSinceCollect += bytes;
        return backend.AllocateBuffer(length, GpuElementType.Float64);
    }

    private long _float64AllocatedSinceCollect;

    // Callers hold _float64Lock.
    private bool TryTakePooledFloat64(IDirectGpuBackend backend, int length, out IGpuBuffer buffer)
    {
        buffer = null!;
        if (!_float64Pool.TryGetValue(length, out var stack)) return false;
        while (stack.Count > 0)
        {
            var candidate = stack.Pop();
            bool taken = false;
            try
            {
                _float64PoolBytes -= (long)length * sizeof(double);
                taken = IsLiveDeviceBuffer(candidate) && ReferenceEquals(_float64PoolBackend, backend);
            }
            finally
            {
                if (!taken) candidate.Dispose();
            }

            if (taken)
            {
                buffer = candidate;
                return true;
            }
        }

        return false;
    }

    private IDirectGpuBackend? _float64PoolBackend;

    // Callers hold _float64Lock.
    private void DisposeFloat64Pool(int exceptLength = -1)
    {
        foreach (var entry in _float64Pool.Where(pooled => pooled.Key != exceptLength))
        {
            while (entry.Value.Count > 0)
            {
                entry.Value.Pop().Dispose();
                _float64PoolBytes -= (long)entry.Key * sizeof(double);
            }
        }
    }

    // Callers hold _float64Lock.
    private void EvictFloat64Residents(long targetBytes, Tensor<double> left, Tensor<double> right)
    {
        object keepLeft = left.DataVector;
        object keepRight = right.DataVector;
        var oldestFirst = new List<Float64Resident>(_float64All);
        oldestFirst.Sort(static (x, y) => x.Stamp.CompareTo(y.Stamp));
        foreach (var resident in oldestFirst)
        {
            if (_float64ResidentBytes + _float64PoolBytes <= targetBytes) break;
            if (resident.Key.TryGetTarget(out var key) && (ReferenceEquals(key, keepLeft) || ReferenceEquals(key, keepRight)))
                continue;
            ReleaseFloat64Resident(resident, bringToHost: true);
        }
    }

    // Callers hold _float64Lock. With `bringToHost`, a result the host has not read yet is downloaded first, so no
    // reachable tensor loses its values. With `toPool`, the buffer is kept for the next result of its size.
    private void ReleaseFloat64Resident(Float64Resident resident, bool bringToHost, bool toPool = false)
    {
        if (resident.Released) return;
        resident.Released = true;
        _float64All.Remove(resident);
        _float64ResidentBytes -= resident.Bytes;
        bool pooled = false;
        try
        {
            if (resident.Key.TryGetTarget(out var key))
            {
                _float64ByVector.Remove(key);
                // Also waits for a download another thread is running on this buffer.
                if (bringToHost && Helpers.HostSync.IsPending(key)) Helpers.HostSync.TryMaterialize(key);
            }

            if (toPool && IsLiveDeviceBuffer(resident.Buffer)
                && (_float64PoolBackend is null || ReferenceEquals(_float64PoolBackend, resident.Backend)))
            {
                _float64PoolBackend = resident.Backend;
                if (!_float64Pool.TryGetValue(resident.Length, out var stack))
                    _float64Pool[resident.Length] = stack = new Stack<IGpuBuffer>();
                stack.Push(resident.Buffer);
                _float64PoolBytes += resident.Bytes;
                pooled = true;
            }
        }
        finally
        {
            if (!pooled) resident.Buffer.Dispose();
        }
    }

    /// <summary>Brings every resident double tensor back to the host and frees its device buffer.</summary>
    internal void ClearFloat64Residents()
    {
        lock (_float64Lock)
        {
            foreach (var resident in new List<Float64Resident>(_float64All))
            {
                try
                {
                    ReleaseFloat64Resident(resident, bringToHost: true);
                }
                catch (InvalidOperationException downloadFailure)
                {
                    // The device is already gone (teardown after a lost context): the values cannot be recovered, and
                    // the remaining buffers still have to be freed.
                    System.Diagnostics.Trace.TraceWarning(
                        "[AiDotNet] A resident double tensor could not be brought back to the host: {0}", downloadFailure);
                }
            }

            DisposeFloat64Pool();
            _float64PoolBackend = null;
        }
    }
}
