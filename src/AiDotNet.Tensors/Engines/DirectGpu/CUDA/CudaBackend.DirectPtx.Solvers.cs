using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IExtendedLinalgBackend
{
    private readonly DirectPtxKernelCache<DirectPtxSolver4x4Key, PtxRegisterSolver4x4F32Kernel>
        _directPtxSolver4x4Kernels = new(192);
    private readonly DirectPtxPlanCache<DirectPtxSolver4x4PlanKey, int>
        _directPtxSolver4x4Plans = new(64);
    private long _directPtxSolver4x4DispatchCount;

    internal long DirectPtxSolver4x4DispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxSolver4x4DispatchCount);
    internal int DirectPtxSolver4x4KernelCapacity => _directPtxSolver4x4Kernels.Capacity;
    internal int DirectPtxSolver4x4PinnedKernelCount
    {
        get { lock (_directPtxLock) return _directPtxSolver4x4Kernels.PinnedCount; }
    }

    internal bool IsDirectPtxSolver4x4Enabled(DirectPtxSolver4x4Operation operation) =>
        SolverGate(operation) && IsAvailable &&
        DirectPtxArchitecture.IsSolver4x4ExperimentArchitecture(_ccMajor, _ccMinor);

    internal bool TryDirectPtxLuFactor4x4(
        IGpuBuffer input, IGpuBuffer output, IGpuBuffer pivots,
        int batchCount, int m, int n)
    {
        const DirectPtxSolver4x4Operation operation = DirectPtxSolver4x4Operation.LuFactor;
        if (!ValidateOperation(operation, batchCount, m, n, extra: 4)) return false;
        if (!ValidateThreeBuffers(operation, input, MatrixBytes(batchCount), 16,
            output, MatrixBytes(batchCount), 16, pivots, VectorBytes(batchCount), 16)) return false;
        return Launch3(operation, batchCount, input, output, pivots);
    }

    internal bool TryDirectPtxQrReduced4x4(
        IGpuBuffer input, IGpuBuffer q, IGpuBuffer r,
        int batchCount, int m, int n)
    {
        const DirectPtxSolver4x4Operation operation = DirectPtxSolver4x4Operation.QrReduced;
        if (!ValidateOperation(operation, batchCount, m, n, extra: 4)) return false;
        long bytes = MatrixBytes(batchCount);
        if (!ValidateThreeBuffers(operation, input, bytes, 16, q, bytes, 16, r, bytes, 16)) return false;
        return Launch3(operation, batchCount, input, q, r);
    }

    internal bool TryDirectPtxEighUpper4x4(
        IGpuBuffer input, IGpuBuffer eigenvalues, IGpuBuffer eigenvectors,
        int batchCount, int n)
    {
        const DirectPtxSolver4x4Operation operation = DirectPtxSolver4x4Operation.EighUpper;
        if (!ValidateOperation(operation, batchCount, n, n, extra: 4)) return false;
        if (!ValidateThreeBuffers(operation, input, MatrixBytes(batchCount), 16,
            eigenvalues, VectorBytes(batchCount), 16,
            eigenvectors, MatrixBytes(batchCount), 16)) return false;
        return Launch3(operation, batchCount, input, eigenvalues, eigenvectors);
    }

    internal bool TryDirectPtxEigh4x4(
        IGpuBuffer input, IGpuBuffer eigenvalues, IGpuBuffer eigenvectors,
        int batchCount, int n, bool upper)
    {
        DirectPtxSolver4x4Operation operation = upper
            ? DirectPtxSolver4x4Operation.EighUpper
            : DirectPtxSolver4x4Operation.EighLower;
        if (!ValidateOperation(operation, batchCount, n, n, extra: 4)) return false;
        if (!ValidateThreeBuffers(operation, input, MatrixBytes(batchCount), 16,
            eigenvalues, VectorBytes(batchCount), 16,
            eigenvectors, MatrixBytes(batchCount), 16)) return false;
        return Launch3(operation, batchCount, input, eigenvalues, eigenvectors);
    }

    internal bool TryDirectPtxSvdReduced4x4(
        IGpuBuffer input, IGpuBuffer u, IGpuBuffer singularValues, IGpuBuffer vh,
        int batchCount, int m, int n)
    {
        const DirectPtxSolver4x4Operation operation = DirectPtxSolver4x4Operation.SvdReduced;
        if (!ValidateOperation(operation, batchCount, m, n, extra: 4)) return false;
        long matrixBytes = MatrixBytes(batchCount);
        if (!ValidateFourBuffers(operation,
            input, matrixBytes, 16, u, matrixBytes, 16,
            singularValues, VectorBytes(batchCount), 16, vh, matrixBytes, 16)) return false;
        return Launch4(operation, batchCount, input, u, singularValues, vh);
    }

    internal bool TryDirectPtxLuSolveVector4x4(
        IGpuBuffer lu, IGpuBuffer pivots, IGpuBuffer rhs, IGpuBuffer solution,
        int batchCount, int n, int rightHandSides, bool rhsIsVector)
    {
        const DirectPtxSolver4x4Operation operation = DirectPtxSolver4x4Operation.LuSolveVector;
        if (!ValidateOperation(operation, batchCount, n, n, extra: rightHandSides)) return false;
        if (!rhsIsVector || rightHandSides != 1)
        {
            DirectPtxLastError = "lu-solve-4x4-rhs-layout-not-implemented";
            return false;
        }
        if (!ValidateFourBuffers(operation,
            lu, MatrixBytes(batchCount), 16, pivots, VectorBytes(batchCount), 16,
            rhs, VectorBytes(batchCount), 16, solution, VectorBytes(batchCount), 16)) return false;
        return Launch4(operation, batchCount, lu, pivots, rhs, solution);
    }

    internal bool TryDirectPtxLdlFactor4x4(
        IGpuBuffer input, IGpuBuffer ld, IGpuBuffer pivots,
        int batchCount, int n, bool upper)
    {
        const DirectPtxSolver4x4Operation operation = DirectPtxSolver4x4Operation.LdlFactorLower;
        if (upper)
        {
            DirectPtxLastError = "ldl-factor-upper-4x4-semantics-not-implemented";
            return false;
        }
        if (!ValidateOperation(operation, batchCount, n, n, extra: 4)) return false;
        if (!ValidateThreeBuffers(operation, input, MatrixBytes(batchCount), 16,
            ld, MatrixBytes(batchCount), 16, pivots, VectorBytes(batchCount), 16)) return false;
        return Launch3(operation, batchCount, input, ld, pivots);
    }

    internal bool TryDirectPtxLdlSolveVector4x4(
        IGpuBuffer ld, IGpuBuffer pivots, IGpuBuffer rhs, IGpuBuffer solution,
        int batchCount, int n, bool upper, bool rhsIsVector)
    {
        const DirectPtxSolver4x4Operation operation = DirectPtxSolver4x4Operation.LdlSolveVectorLower;
        if (upper || !rhsIsVector)
        {
            DirectPtxLastError = "ldl-solve-lower-vector-4x4-semantics-not-implemented";
            return false;
        }
        if (!ValidateOperation(operation, batchCount, n, n, extra: 4)) return false;
        if (!ValidateFourBuffers(operation, ld, MatrixBytes(batchCount), 16,
            pivots, VectorBytes(batchCount), 16, rhs, VectorBytes(batchCount), 16,
            solution, VectorBytes(batchCount), 16)) return false;
        return Launch4(operation, batchCount, ld, pivots, rhs, solution);
    }

    internal bool TryDirectPtxGeneralSolveVector4x4(
        IGpuBuffer input, IGpuBuffer rhs, IGpuBuffer solution, IGpuBuffer info,
        int batchCount, int n)
    {
        const DirectPtxSolver4x4Operation operation = DirectPtxSolver4x4Operation.GeneralSolveVector;
        if (!ValidateOperation(operation, batchCount, n, n, extra: 4)) return false;
        if (!ValidateFourBuffers(operation, input, MatrixBytes(batchCount), 16,
            rhs, VectorBytes(batchCount), 16, solution, VectorBytes(batchCount), 16,
            info, ScalarBytes(batchCount), 16)) return false;
        return Launch4(operation, batchCount, input, rhs, solution, info);
    }

    internal bool TryDirectPtxTriangularSolveVector4x4(
        IGpuBuffer input, IGpuBuffer rhs, IGpuBuffer solution,
        int batchCount, int n, bool upper, bool unitDiagonal)
    {
        DirectPtxSolver4x4Operation operation = upper
            ? DirectPtxSolver4x4Operation.TriangularSolveVectorUpper
            : DirectPtxSolver4x4Operation.TriangularSolveVectorLower;
        if (unitDiagonal)
        {
            DirectPtxLastError = $"{OperationSlug(operation)}-unit-diagonal-not-implemented";
            return false;
        }
        if (!ValidateOperation(operation, batchCount, n, n, extra: 4)) return false;
        if (!ValidateThreeBuffers(operation, input, MatrixBytes(batchCount), 16,
            rhs, VectorBytes(batchCount), 16, solution, VectorBytes(batchCount), 16)) return false;
        return Launch3(operation, batchCount, input, rhs, solution);
    }

    internal bool TryDirectPtxCholeskyBackwardLower4x4(
        IGpuBuffer factor, IGpuBuffer gradOutput, IGpuBuffer gradInput,
        int batchCount, int n)
    {
        const DirectPtxSolver4x4Operation operation = DirectPtxSolver4x4Operation.CholeskyBackwardLower;
        if (!ValidateOperation(operation, batchCount, n, n, extra: 4)) return false;
        long bytes = MatrixBytes(batchCount);
        if (!ValidateThreeBuffers(operation, factor, bytes, 16,
            gradOutput, bytes, 16, gradInput, bytes, 16)) return false;
        return Launch3(operation, batchCount, factor, gradOutput, gradInput);
    }

    internal bool TryDirectPtxSolveBackwardVector4x4(
        IGpuBuffer input, IGpuBuffer solution, IGpuBuffer gradOutput,
        IGpuBuffer gradInput, IGpuBuffer gradRhs, int batchCount, int n)
    {
        const DirectPtxSolver4x4Operation operation = DirectPtxSolver4x4Operation.SolveBackwardVector;
        if (!ValidateOperation(operation, batchCount, n, n, extra: 4)) return false;
        if (!ValidateFiveBuffers(operation,
            input, MatrixBytes(batchCount), solution, VectorBytes(batchCount),
            gradOutput, VectorBytes(batchCount), gradInput, MatrixBytes(batchCount),
            gradRhs, VectorBytes(batchCount))) return false;
        return Launch5(operation, batchCount, input, solution, gradOutput, gradInput, gradRhs);
    }

    void IExtendedLinalgBackend.LinalgSvdReduced(
        IGpuBuffer input, IGpuBuffer u, IGpuBuffer singularValues, IGpuBuffer vh,
        int batchCount, int m, int n)
    {
        if (!TryDirectPtxSvdReduced4x4(input, u, singularValues, vh, batchCount, m, n))
            throw new InvalidOperationException(
                $"Direct PTX SVD specialization rejected the request: {DirectPtxLastError}");
    }

    void IExtendedLinalgBackend.LinalgQrReducedDirect(
        IGpuBuffer input, IGpuBuffer q, IGpuBuffer r, int batchCount, int m, int n)
    {
        if (!TryDirectPtxQrReduced4x4(input, q, r, batchCount, m, n))
            throw new InvalidOperationException($"Direct PTX QR specialization rejected the request: {DirectPtxLastError}");
    }

    void IExtendedLinalgBackend.LinalgLuSolve(
        IGpuBuffer lu, IGpuBuffer pivots, IGpuBuffer rhs, IGpuBuffer solution,
        int batchCount, int n, int rightHandSides, bool rhsIsVector)
    {
        if (!TryDirectPtxLuSolveVector4x4(
            lu, pivots, rhs, solution, batchCount, n, rightHandSides, rhsIsVector))
            throw new InvalidOperationException(
                $"Direct PTX LU-solve specialization rejected the request: {DirectPtxLastError}");
    }

    void IExtendedLinalgBackend.LinalgEighSymmetric(
        IGpuBuffer input, IGpuBuffer eigenvalues, IGpuBuffer eigenvectors,
        int batchCount, int n, bool upper)
    {
        if (TryDirectPtxEigh4x4(input, eigenvalues, eigenvectors, batchCount, n, upper)) return;
        if (upper)
        {
            // Preserve the established CUDA/NVRTC route when the experimental
            // upper specialization is disabled or rejects its exact contract.
            LinalgEigh(input, eigenvalues, eigenvectors, batchCount, n);
            return;
        }
        throw new InvalidOperationException($"Direct PTX lower Eigh specialization rejected the request: {DirectPtxLastError}");
    }

    void IExtendedLinalgBackend.LinalgLdlFactor(
        IGpuBuffer input, IGpuBuffer ld, IGpuBuffer pivots, int batchCount, int n, bool upper)
    {
        if (!TryDirectPtxLdlFactor4x4(input, ld, pivots, batchCount, n, upper))
            throw new InvalidOperationException($"Direct PTX LDL factor specialization rejected the request: {DirectPtxLastError}");
    }

    void IExtendedLinalgBackend.LinalgLdlSolve(
        IGpuBuffer ld, IGpuBuffer pivots, IGpuBuffer rhs, IGpuBuffer solution,
        int batchCount, int n, bool upper, bool rhsIsVector)
    {
        if (!TryDirectPtxLdlSolveVector4x4(ld, pivots, rhs, solution, batchCount, n, upper, rhsIsVector))
            throw new InvalidOperationException($"Direct PTX LDL solve specialization rejected the request: {DirectPtxLastError}");
    }

    void IExtendedLinalgBackend.LinalgSolveVector(
        IGpuBuffer input, IGpuBuffer rhs, IGpuBuffer solution, IGpuBuffer info,
        int batchCount, int n)
    {
        if (!TryDirectPtxGeneralSolveVector4x4(input, rhs, solution, info, batchCount, n))
            throw new InvalidOperationException($"Direct PTX general solve specialization rejected the request: {DirectPtxLastError}");
    }

    void IExtendedLinalgBackend.LinalgTriangularSolveVector(
        IGpuBuffer input, IGpuBuffer rhs, IGpuBuffer solution,
        int batchCount, int n, bool upper, bool unitDiagonal)
    {
        if (!TryDirectPtxTriangularSolveVector4x4(input, rhs, solution, batchCount, n, upper, unitDiagonal))
            throw new InvalidOperationException($"Direct PTX triangular solve specialization rejected the request: {DirectPtxLastError}");
    }

    void IExtendedLinalgBackend.LinalgCholeskyBackwardLower(
        IGpuBuffer factor, IGpuBuffer gradOutput, IGpuBuffer gradInput, int batchCount, int n)
    {
        if (!TryDirectPtxCholeskyBackwardLower4x4(factor, gradOutput, gradInput, batchCount, n))
            throw new InvalidOperationException($"Direct PTX Cholesky backward specialization rejected the request: {DirectPtxLastError}");
    }

    void IExtendedLinalgBackend.LinalgSolveBackwardVector(
        IGpuBuffer input, IGpuBuffer solution, IGpuBuffer gradOutput,
        IGpuBuffer gradInput, IGpuBuffer gradRhs, int batchCount, int n)
    {
        if (!TryDirectPtxSolveBackwardVector4x4(
            input, solution, gradOutput, gradInput, gradRhs, batchCount, n))
            throw new InvalidOperationException($"Direct PTX solve backward specialization rejected the request: {DirectPtxLastError}");
    }

    internal bool PrewarmDirectPtxSolver4x4(
        DirectPtxSolver4x4Operation operation,
        int batchCount)
    {
        if (!IsDirectPtxSolver4x4Enabled(operation)) return false;
        if (!PtxRegisterSolver4x4F32Kernel.IsSupportedBatchCount(batchCount))
        {
            DirectPtxLastError = $"{OperationSlug(operation)}-batch-not-implemented";
            return false;
        }
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = $"Direct PTX {OperationSlug(operation)} prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                var planKey = new DirectPtxSolver4x4PlanKey(operation, batchCount);
                if (!_directPtxSolver4x4Plans.TryGetValue(planKey, out int blockThreads))
                {
                    blockThreads = DirectPtxSolver4x4Autotuner.DefaultBlockThreads;
                    _directPtxSolver4x4Plans.Set(planKey, blockThreads);
                }
                _ = GetOrCreateSolver4x4Kernel(
                    new DirectPtxSolver4x4Key(operation, batchCount, blockThreads));
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryGetDirectPtxSolver4x4Audit(
        DirectPtxSolver4x4Operation operation,
        int batchCount,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            if (_directPtxSolver4x4Plans.TryGetValue(
                new DirectPtxSolver4x4PlanKey(operation, batchCount), out int blockThreads) &&
                _directPtxSolver4x4Kernels.TryGetValue(
                    new DirectPtxSolver4x4Key(operation, batchCount, blockThreads), out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    private bool Launch3(
        DirectPtxSolver4x4Operation operation,
        int batchCount,
        IGpuBuffer first,
        IGpuBuffer second,
        IGpuBuffer third)
    {
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var planKey = new DirectPtxSolver4x4PlanKey(operation, batchCount);
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxSolver4x4Plans.TryGetValue(planKey, out _))
                {
                    DirectPtxLastError = $"Direct PTX {OperationSlug(operation)} must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                int blockThreads;
                if (!_directPtxSolver4x4Plans.TryGetValue(planKey, out blockThreads))
                {
                    blockThreads = DirectPtxFeatureGate.IsAutotuneEnabled
                        ? DirectPtxSolver4x4Autotuner.Select(candidate =>
                        {
                            var candidateKey = new DirectPtxSolver4x4Key(operation, batchCount, candidate);
                            PtxRegisterSolver4x4F32Kernel candidateKernel = GetOrCreateSolver4x4Kernel(candidateKey);
                            var firstView = DirectPtxTensorView.Create(first, candidateKernel.Blueprint.Tensors[0]);
                            var secondView = DirectPtxTensorView.Create(second, candidateKernel.Blueprint.Tensors[1]);
                            var thirdView = DirectPtxTensorView.Create(third, candidateKernel.Blueprint.Tensors[2]);
                            return _directPtxRuntime.MeasureKernelSamples(
                                () => { lock (GpuDispatchLock) candidateKernel.Launch3(firstView, secondView, thirdView); },
                                warmup: 2, samples: 5, launchesPerSample: 1);
                        })
                        : DirectPtxSolver4x4Autotuner.DefaultBlockThreads;
                    _directPtxSolver4x4Plans.Set(planKey, blockThreads);
                }
                var key = new DirectPtxSolver4x4Key(operation, batchCount, blockThreads);
                if (capturing && !_directPtxSolver4x4Kernels.TryGetValue(key, out _))
                {
                    DirectPtxLastError = $"Direct PTX {OperationSlug(operation)} selected module is not resident for capture.";
                    return false;
                }
                PtxRegisterSolver4x4F32Kernel kernel = GetOrCreateSolver4x4Kernel(key);
                if (capturing && !_directPtxSolver4x4Kernels.Pin(key))
                {
                    DirectPtxLastError = $"{OperationSlug(operation)}-capture-pin-failed";
                    return false;
                }
                lock (GpuDispatchLock)
                    kernel.Launch3(
                        DirectPtxTensorView.Create(first, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(second, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(third, kernel.Blueprint.Tensors[2]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxSolver4x4DispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private bool Launch4(
        DirectPtxSolver4x4Operation operation,
        int batchCount,
        IGpuBuffer first,
        IGpuBuffer second,
        IGpuBuffer third,
        IGpuBuffer fourth)
    {
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var planKey = new DirectPtxSolver4x4PlanKey(operation, batchCount);
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxSolver4x4Plans.TryGetValue(planKey, out _))
                {
                    DirectPtxLastError = $"Direct PTX {OperationSlug(operation)} must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                int blockThreads;
                if (!_directPtxSolver4x4Plans.TryGetValue(planKey, out blockThreads))
                {
                    blockThreads = DirectPtxFeatureGate.IsAutotuneEnabled
                        ? DirectPtxSolver4x4Autotuner.Select(candidate =>
                        {
                            var candidateKey = new DirectPtxSolver4x4Key(operation, batchCount, candidate);
                            PtxRegisterSolver4x4F32Kernel candidateKernel = GetOrCreateSolver4x4Kernel(candidateKey);
                            var firstView = DirectPtxTensorView.Create(first, candidateKernel.Blueprint.Tensors[0]);
                            var secondView = DirectPtxTensorView.Create(second, candidateKernel.Blueprint.Tensors[1]);
                            var thirdView = DirectPtxTensorView.Create(third, candidateKernel.Blueprint.Tensors[2]);
                            var fourthView = DirectPtxTensorView.Create(fourth, candidateKernel.Blueprint.Tensors[3]);
                            return _directPtxRuntime.MeasureKernelSamples(
                                () => { lock (GpuDispatchLock) candidateKernel.Launch4(firstView, secondView, thirdView, fourthView); },
                                warmup: 2, samples: 5, launchesPerSample: 1);
                        })
                        : DirectPtxSolver4x4Autotuner.DefaultBlockThreads;
                    _directPtxSolver4x4Plans.Set(planKey, blockThreads);
                }
                var key = new DirectPtxSolver4x4Key(operation, batchCount, blockThreads);
                if (capturing && !_directPtxSolver4x4Kernels.TryGetValue(key, out _))
                {
                    DirectPtxLastError = $"Direct PTX {OperationSlug(operation)} selected module is not resident for capture.";
                    return false;
                }
                PtxRegisterSolver4x4F32Kernel kernel = GetOrCreateSolver4x4Kernel(key);
                if (capturing && !_directPtxSolver4x4Kernels.Pin(key))
                {
                    DirectPtxLastError = $"{OperationSlug(operation)}-capture-pin-failed";
                    return false;
                }
                lock (GpuDispatchLock)
                    kernel.Launch4(
                        DirectPtxTensorView.Create(first, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(second, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(third, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(fourth, kernel.Blueprint.Tensors[3]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxSolver4x4DispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private bool Launch5(
        DirectPtxSolver4x4Operation operation,
        int batchCount,
        IGpuBuffer first,
        IGpuBuffer second,
        IGpuBuffer third,
        IGpuBuffer fourth,
        IGpuBuffer fifth)
    {
        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var planKey = new DirectPtxSolver4x4PlanKey(operation, batchCount);
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxSolver4x4Plans.TryGetValue(planKey, out _))
                {
                    DirectPtxLastError = $"Direct PTX {OperationSlug(operation)} must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                int blockThreads;
                if (!_directPtxSolver4x4Plans.TryGetValue(planKey, out blockThreads))
                {
                    blockThreads = DirectPtxFeatureGate.IsAutotuneEnabled
                        ? DirectPtxSolver4x4Autotuner.Select(candidate =>
                        {
                            var candidateKey = new DirectPtxSolver4x4Key(operation, batchCount, candidate);
                            PtxRegisterSolver4x4F32Kernel candidateKernel = GetOrCreateSolver4x4Kernel(candidateKey);
                            var firstView = DirectPtxTensorView.Create(first, candidateKernel.Blueprint.Tensors[0]);
                            var secondView = DirectPtxTensorView.Create(second, candidateKernel.Blueprint.Tensors[1]);
                            var thirdView = DirectPtxTensorView.Create(third, candidateKernel.Blueprint.Tensors[2]);
                            var fourthView = DirectPtxTensorView.Create(fourth, candidateKernel.Blueprint.Tensors[3]);
                            var fifthView = DirectPtxTensorView.Create(fifth, candidateKernel.Blueprint.Tensors[4]);
                            return _directPtxRuntime.MeasureKernelSamples(
                                () =>
                                {
                                    lock (GpuDispatchLock)
                                        candidateKernel.Launch5(firstView, secondView, thirdView, fourthView, fifthView);
                                },
                                warmup: 2, samples: 5, launchesPerSample: 1);
                        })
                        : DirectPtxSolver4x4Autotuner.DefaultBlockThreads;
                    _directPtxSolver4x4Plans.Set(planKey, blockThreads);
                }
                var key = new DirectPtxSolver4x4Key(operation, batchCount, blockThreads);
                if (capturing && !_directPtxSolver4x4Kernels.TryGetValue(key, out _))
                {
                    DirectPtxLastError = $"Direct PTX {OperationSlug(operation)} selected module is not resident for capture.";
                    return false;
                }
                PtxRegisterSolver4x4F32Kernel kernel = GetOrCreateSolver4x4Kernel(key);
                if (capturing && !_directPtxSolver4x4Kernels.Pin(key))
                {
                    DirectPtxLastError = $"{OperationSlug(operation)}-capture-pin-failed";
                    return false;
                }
                lock (GpuDispatchLock)
                    kernel.Launch5(
                        DirectPtxTensorView.Create(first, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(second, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(third, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(fourth, kernel.Blueprint.Tensors[3]),
                        DirectPtxTensorView.Create(fifth, kernel.Blueprint.Tensors[4]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxSolver4x4DispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private bool ValidateOperation(
        DirectPtxSolver4x4Operation operation,
        int batchCount,
        int m,
        int n,
        int extra)
    {
        string slug = OperationSlug(operation);
        if (!SolverGate(operation))
        {
            DirectPtxLastError = $"{slug}-feature-disabled";
            return false;
        }
        if (!IsAvailable)
        {
            DirectPtxLastError = $"{slug}-backend-unavailable";
            return false;
        }
        if (!DirectPtxArchitecture.IsSolver4x4ExperimentArchitecture(_ccMajor, _ccMinor))
        {
            DirectPtxLastError = $"{slug}-architecture-not-implemented";
            return false;
        }
        if (m != 4 || n != 4)
        {
            DirectPtxLastError = $"{slug}-shape-not-implemented";
            return false;
        }
        if (!PtxRegisterSolver4x4F32Kernel.IsSupportedBatchCount(batchCount))
        {
            DirectPtxLastError = $"{slug}-batch-not-implemented";
            return false;
        }
        if (extra != 4 && operation != DirectPtxSolver4x4Operation.LuSolveVector)
        {
            DirectPtxLastError = $"{slug}-semantics-not-implemented";
            return false;
        }
        return true;
    }

    private bool ValidateThreeBuffers(
        DirectPtxSolver4x4Operation operation,
        IGpuBuffer first, long firstBytes, int firstAlignment,
        IGpuBuffer second, long secondBytes, int secondAlignment,
        IGpuBuffer third, long thirdBytes, int thirdAlignment)
    {
        string slug = OperationSlug(operation);
        if (first is null || second is null || third is null)
        {
            DirectPtxLastError = $"{slug}-null-buffer";
            return false;
        }
        if (first.SizeInBytes != firstBytes || second.SizeInBytes != secondBytes ||
            third.SizeInBytes != thirdBytes)
        {
            DirectPtxLastError = $"{slug}-physical-extent-mismatch";
            return false;
        }
        if (!ValidPointer(first, firstAlignment) || !ValidPointer(second, secondAlignment) ||
            !ValidPointer(third, thirdAlignment))
        {
            DirectPtxLastError = $"{slug}-pointer-or-alignment-mismatch";
            return false;
        }
        if (DirectPtxRangesOverlap(first, second) || DirectPtxRangesOverlap(first, third) ||
            DirectPtxRangesOverlap(second, third))
        {
            DirectPtxLastError = $"{slug}-alias-not-supported";
            return false;
        }
        return true;
    }

    private bool ValidateFourBuffers(
        DirectPtxSolver4x4Operation operation,
        IGpuBuffer first, long firstBytes, int firstAlignment,
        IGpuBuffer second, long secondBytes, int secondAlignment,
        IGpuBuffer third, long thirdBytes, int thirdAlignment,
        IGpuBuffer fourth, long fourthBytes, int fourthAlignment)
    {
        string slug = OperationSlug(operation);
        if (first is null || second is null || third is null || fourth is null)
        {
            DirectPtxLastError = $"{slug}-null-buffer";
            return false;
        }
        if (first.SizeInBytes != firstBytes || second.SizeInBytes != secondBytes ||
            third.SizeInBytes != thirdBytes || fourth.SizeInBytes != fourthBytes)
        {
            DirectPtxLastError = $"{slug}-physical-extent-mismatch";
            return false;
        }
        if (!ValidPointer(first, firstAlignment) || !ValidPointer(second, secondAlignment) ||
            !ValidPointer(third, thirdAlignment) || !ValidPointer(fourth, fourthAlignment))
        {
            DirectPtxLastError = $"{slug}-pointer-or-alignment-mismatch";
            return false;
        }
        if (DirectPtxRangesOverlap(first, second) || DirectPtxRangesOverlap(first, third) ||
            DirectPtxRangesOverlap(first, fourth) || DirectPtxRangesOverlap(second, third) ||
            DirectPtxRangesOverlap(second, fourth) || DirectPtxRangesOverlap(third, fourth))
        {
            DirectPtxLastError = $"{slug}-alias-not-supported";
            return false;
        }
        return true;
    }

    private bool ValidateFiveBuffers(
        DirectPtxSolver4x4Operation operation,
        IGpuBuffer first, long firstBytes,
        IGpuBuffer second, long secondBytes,
        IGpuBuffer third, long thirdBytes,
        IGpuBuffer fourth, long fourthBytes,
        IGpuBuffer fifth, long fifthBytes)
    {
        string slug = OperationSlug(operation);
        IGpuBuffer[] buffers = [first, second, third, fourth, fifth];
        long[] bytes = [firstBytes, secondBytes, thirdBytes, fourthBytes, fifthBytes];
        for (int i = 0; i < buffers.Length; i++)
        {
            if (buffers[i] is null)
            {
                DirectPtxLastError = $"{slug}-null-buffer";
                return false;
            }
            if (buffers[i].SizeInBytes != bytes[i])
            {
                DirectPtxLastError = $"{slug}-physical-extent-mismatch";
                return false;
            }
            if (!ValidPointer(buffers[i], 16))
            {
                DirectPtxLastError = $"{slug}-pointer-or-alignment-mismatch";
                return false;
            }
        }
        for (int i = 0; i < buffers.Length; i++)
        for (int j = i + 1; j < buffers.Length; j++)
            if (DirectPtxRangesOverlap(buffers[i], buffers[j]))
            {
                DirectPtxLastError = $"{slug}-alias-not-supported";
                return false;
            }
        return true;
    }

    private static bool ValidPointer(IGpuBuffer buffer, int alignment) =>
        buffer.Handle != IntPtr.Zero &&
        (PtxCompat.ToNuint(buffer.Handle) & (nuint)(alignment - 1)) == 0;

    private static long MatrixBytes(int batchCount) => checked((long)batchCount * 16 * sizeof(float));
    private static long VectorBytes(int batchCount) => checked((long)batchCount * 4 * sizeof(float));
    private static long ScalarBytes(int batchCount) => checked((long)batchCount * sizeof(int));

    private PtxRegisterSolver4x4F32Kernel GetOrCreateSolver4x4Kernel(DirectPtxSolver4x4Key key)
    {
        if (_directPtxSolver4x4Kernels.TryGetValue(key, out var existing)) return existing;
        var created = new PtxRegisterSolver4x4F32Kernel(
            _directPtxRuntime!, key.Operation, key.BatchCount, key.BlockThreads);
        return _directPtxSolver4x4Kernels.AddOrGetExisting(key, created);
    }

    private static bool SolverGate(DirectPtxSolver4x4Operation operation) => operation switch
    {
        DirectPtxSolver4x4Operation.LuFactor => DirectPtxFeatureGate.IsLuFactor4x4Enabled,
        DirectPtxSolver4x4Operation.QrReduced => DirectPtxFeatureGate.IsQr4x4Enabled,
        DirectPtxSolver4x4Operation.EighUpper or DirectPtxSolver4x4Operation.EighLower =>
            DirectPtxFeatureGate.IsEigh4x4Enabled,
        DirectPtxSolver4x4Operation.SvdReduced => DirectPtxFeatureGate.IsSvd4x4Enabled,
        DirectPtxSolver4x4Operation.LuSolveVector => DirectPtxFeatureGate.IsLuSolve4x4Enabled,
        DirectPtxSolver4x4Operation.LdlFactorLower => DirectPtxFeatureGate.IsLdlFactor4x4Enabled,
        DirectPtxSolver4x4Operation.LdlSolveVectorLower => DirectPtxFeatureGate.IsLdlSolve4x4Enabled,
        DirectPtxSolver4x4Operation.GeneralSolveVector => DirectPtxFeatureGate.IsSolve4x4Enabled,
        DirectPtxSolver4x4Operation.TriangularSolveVectorLower or
        DirectPtxSolver4x4Operation.TriangularSolveVectorUpper =>
            DirectPtxFeatureGate.IsTriangularSolve4x4Enabled,
        DirectPtxSolver4x4Operation.CholeskyBackwardLower or
        DirectPtxSolver4x4Operation.SolveBackwardVector => DirectPtxFeatureGate.IsSolverBackward4x4Enabled,
        _ => false
    };

    private static string OperationSlug(DirectPtxSolver4x4Operation operation) => operation switch
    {
        DirectPtxSolver4x4Operation.LuFactor => "lu-factor-4x4",
        DirectPtxSolver4x4Operation.QrReduced => "qr-reduced-4x4",
        DirectPtxSolver4x4Operation.EighUpper => "eigh-upper-4x4",
        DirectPtxSolver4x4Operation.EighLower => "eigh-lower-4x4",
        DirectPtxSolver4x4Operation.SvdReduced => "svd-reduced-4x4",
        DirectPtxSolver4x4Operation.LuSolveVector => "lu-solve-vector-4x4",
        DirectPtxSolver4x4Operation.LdlFactorLower => "ldl-factor-lower-4x4",
        DirectPtxSolver4x4Operation.LdlSolveVectorLower => "ldl-solve-lower-vector-4x4",
        DirectPtxSolver4x4Operation.GeneralSolveVector => "solve-vector-4x4",
        DirectPtxSolver4x4Operation.TriangularSolveVectorLower => "triangular-solve-lower-vector-4x4",
        DirectPtxSolver4x4Operation.TriangularSolveVectorUpper => "triangular-solve-upper-vector-4x4",
        DirectPtxSolver4x4Operation.CholeskyBackwardLower => "cholesky-backward-lower-4x4",
        DirectPtxSolver4x4Operation.SolveBackwardVector => "solve-backward-vector-4x4",
        _ => "solver-4x4"
    };

    private void DisposeDirectPtxSolver4x4Kernels()
    {
        _directPtxSolver4x4Kernels.Dispose();
        _directPtxSolver4x4Plans.Clear();
    }

    private readonly record struct DirectPtxSolver4x4PlanKey(
        DirectPtxSolver4x4Operation Operation,
        int BatchCount);
    private readonly record struct DirectPtxSolver4x4Key(
        DirectPtxSolver4x4Operation Operation,
        int BatchCount,
        int BlockThreads);
}
