// Copyright (c) CrowdTrainer LLC. All rights reserved.
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    // FP16 weight caches identify the underlying allocation, which a workspace
    // intentionally reuses for different kernels. Those plans keep private weights.
    internal bool SupportsWeightNormWorkspace => !Fp16ActEnabled && !Gpu.AutocastScope.IsEnabled;

    public override Tensor<T> TensorWeightNorm<T>(Tensor<T> direction, Tensor<T> gain, double epsilon = 1e-24)
    {
        var (rows, columns) = ValidateWeightNorm(direction, gain, epsilon);
        using var operation = EnterGpuOp();
        if (GraphMode.IsActive || DirectGpuEngine.ShouldFallbackForPrecision<T>()
            || rows == 0 || columns == 0 || !TryGetBackend(out var backend) || backend is not IGpuBatchExecution native)
            return base.TensorWeightNorm(direction, gain, epsilon);

        var output = CreateResidentOutput<T>(backend, direction._shape);
        ExecuteWeightNormInto(output, direction, gain, rows, columns, (float)epsilon, backend, native, callerOwned: false);
        DifferentiableOps.RecordBinary("TensorWeightNorm", output, direction, gain,
            BackwardFunctions<T>.WeightNormBackward, new object[] { epsilon });
        return output;
    }

    public override void TensorWeightNormInto<T>(Tensor<T> output, Tensor<T> direction, Tensor<T> gain, double epsilon = 1e-24)
    {
        var (rows, columns) = ValidateWeightNorm(direction, gain, epsilon);
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (!ShapesMatch(output._shape, direction._shape))
            throw new ArgumentException("The weight-norm output must have the direction shape.", nameof(output));
        using var operation = EnterGpuOp();
        // Noncanonical write destinations retain the existing offset/strided CPU
        // contract. Native input views are resolved by GetResidentOrPersistentInputBuffer.
        if (DirectGpuEngine.ShouldFallbackForPrecision<T>() || rows == 0 || columns == 0
            || !IsCanonicalDenseAllocation(output) || !TryGetBackend(out var backend) || backend is not IGpuBatchExecution native)
        {
            base.TensorWeightNormInto(output, direction, gain, epsilon);
            return;
        }
        ExecuteWeightNormInto(output, direction, gain, rows, columns, (float)epsilon, backend, native, callerOwned: true);
    }

    private void ExecuteWeightNormInto<T>(Tensor<T> output, Tensor<T> direction, Tensor<T> gain, int rows, int columns,
        float epsilon, IDirectGpuBackend backend, IGpuBatchExecution native, bool callerOwned)
    {
        using var sourceDirection = GetResidentOrPersistentInputBuffer(backend, direction);
        using var sourceGain = GetResidentOrPersistentInputBuffer(backend, gain);
        var destination = GetOrCreateResidentBuffer(backend, output, output.Length, fullyWritten: true);
        // A writable alias may straddle several rows. Snapshot only actual native
        // allocation overlap; ordinary normalized weights allocate no such scratch.
        using var savedDirection = SnapshotWeightNormAlias(backend, sourceDirection.Buffer, direction.Length, destination);
        using var savedGain = SnapshotWeightNormAlias(backend, sourceGain.Buffer, gain.Length, destination);
        native.WeightNorm(savedDirection.Buffer, savedGain.Buffer, destination, rows, columns, epsilon);
        ResidentSyncCheck("WeightNorm");
        output.IncrementVersion();
        BindResidentBuffer(output, destination, backend);
        if (callerOwned) MarkCallerOwnedDestination(output);
    }

    public override void TensorWeightNormBackwardInto<T>(Tensor<T> gradOutput, Tensor<T> direction, Tensor<T> gain,
        Tensor<T>? gradDirection, Tensor<T>? gradGain, double epsilon = 1e-24,
        bool accumulateDirection = false, bool accumulateGain = false)
    {
        var (rows, columns) = ValidateWeightNormBackward(gradOutput, direction, gain, gradDirection, gradGain, epsilon);
        if (gradDirection is null && gradGain is null) return;
        using var operation = EnterGpuOp();
        if (DirectGpuEngine.ShouldFallbackForPrecision<T>() || rows == 0 || columns == 0
            || (gradDirection is not null && !IsCanonicalDenseAllocation(gradDirection))
            || (gradGain is not null && !IsCanonicalDenseAllocation(gradGain))
            || !TryGetBackend(out var backend) || backend is not IGpuBatchExecution native)
        {
            base.TensorWeightNormBackwardInto(gradOutput, direction, gain, gradDirection, gradGain, epsilon,
                accumulateDirection, accumulateGain);
            return;
        }

        using var sourceOutput = GetResidentOrPersistentInputBuffer(backend, gradOutput);
        using var sourceDirection = GetResidentOrPersistentInputBuffer(backend, direction);
        using var sourceGain = GetResidentOrPersistentInputBuffer(backend, gain);
        bool gainFollowsDirection = gradDirection is not null && gradGain is not null
            && (ReferenceEquals(gradDirection, gradGain) || gradDirection.SharesStorageWith(gradGain));
        using var oldDirection = gradDirection is not null && accumulateDirection
            ? GetResidentOrPersistentInputBuffer(backend, gradDirection) : default;
        using var oldGain = gradGain is not null && accumulateGain && !gainFollowsDirection
            ? GetResidentOrPersistentInputBuffer(backend, gradGain) : default;
        var targetDirection = gradDirection is null ? null : accumulateDirection ? oldDirection.Buffer
            : GetOrCreateResidentBuffer(backend, gradDirection, gradDirection.Length, fullyWritten: true);
        var targetGain = gainFollowsDirection ? targetDirection : gradGain is null ? null : accumulateGain ? oldGain.Buffer
            : GetOrCreateResidentBuffer(backend, gradGain, gradGain.Length, fullyWritten: true);

        using var savedOutput = SnapshotWeightNormAlias(backend, sourceOutput.Buffer, gradOutput.Length, targetDirection, targetGain);
        using var savedDirection = SnapshotWeightNormAlias(backend, sourceDirection.Buffer, direction.Length, targetDirection, targetGain);
        using var savedGain = SnapshotWeightNormAlias(backend, sourceGain.Buffer, gain.Length, targetDirection, targetGain);
        using var stats = AllocateTransientScratch(backend, checked(rows * 2), fullyWritten: true);
        native.WeightNormStats(savedOutput.Buffer, savedDirection.Buffer, stats.Buffer, rows, columns, (float)epsilon);
        if (targetDirection is not null)
        {
            native.WeightNormDirectionGradient(savedOutput.Buffer, savedDirection.Buffer, savedGain.Buffer,
                stats.Buffer, targetDirection, rows, columns, accumulateDirection);
            gradDirection!.IncrementVersion();
            BindResidentBuffer(gradDirection, targetDirection, backend);
            MarkCallerOwnedDestination(gradDirection);
        }
        // Complete dV before dG. For a tied target, accumulation observes the dV
        // just committed, while all primal values were protected before either write.
        if (targetGain is not null)
        {
            native.WeightNormGainGradient(stats.Buffer, targetGain, rows, accumulateGain);
            gradGain!.IncrementVersion();
            BindResidentBuffer(gradGain, targetGain, backend);
            MarkCallerOwnedDestination(gradGain);
        }
        ResidentSyncCheck("WeightNormBackward");
    }

    private static OwnedBuffer SnapshotWeightNormAlias(IDirectGpuBackend backend, IGpuBuffer input, int length,
        IGpuBuffer? firstDestination, IGpuBuffer? secondDestination = null)
    {
        var root = GpuBufferViewRoot.Of(input);
        if ((firstDestination is null || !ReferenceEquals(root, GpuBufferViewRoot.Of(firstDestination)))
            && (secondDestination is null || !ReferenceEquals(root, GpuBufferViewRoot.Of(secondDestination))))
            return new OwnedBuffer(input, ownsBuffer: false);
        var snapshot = AllocateTransientScratch(backend, length, fullyWritten: true);
        backend.Copy(input, snapshot.Buffer, length);
        return snapshot;
    }

    internal Tensor<T> CreateWeightNormGradient<T>(Tensor<T> source)
    {
        using var operation = EnterGpuOp();
        if (typeof(T) == typeof(float) && source.Length > 0
            && TryGetBackend(out var backend) && backend is IGpuBatchExecution)
            return CreateResidentOutput<T>(backend, source._shape);
        return Helpers.TensorAllocator.RentUninitialized<T>(source._shape);
    }
}
