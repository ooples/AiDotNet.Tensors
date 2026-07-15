// Copyright (c) AiDotNet. All rights reserved.
// Metal GPU backend - Hyperbolic Geometry, Octonion Algebra, and Quantum Computing operations.
// All hyperbolic/octonion ops dispatch to MSL compute kernels on the GPU.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend
{
    private void DispatchResidentMetal(string kernelName, int threads, IGpuBuffer[] buffers, params uint[] values)
    {
        if (threads <= 0) return;
        if (_residentLibrary == IntPtr.Zero)
            throw new InvalidOperationException("Metal resident kernels are unavailable.");

        var pipeline = GetPipeline("Resident", _residentLibrary, kernelName);
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(threads);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        for (int i = 0; i < buffers.Length; ++i)
        {
            if (buffers[i] is not MetalGpuBuffer buffer)
                throw new ArgumentException("Buffers must be MetalGpuBuffer.", nameof(buffers));
            encoder.SetBuffer(buffer, i);
        }
        for (int i = 0; i < values.Length; ++i)
            encoder.SetBytes(values[i], (uint)(buffers.Length + i));
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    #region Hyperbolic Geometry Operations (Poincare Ball Model)

    private static void ValidateCurvature(float curvature)
    {
        if (curvature <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(curvature), curvature,
                "Curvature must be positive for hyperbolic operations");
        }
    }

    public void PoincareProject(IGpuBuffer input, IGpuBuffer output, int batchSize, int dim, float curvature, float epsilon = 1e-5f)
    {
        ThrowIfDisposed();
        ValidateCurvature(curvature);
        if (input is not MetalGpuBuffer ib || output is not MetalGpuBuffer ob)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetPipeline("Hyperbolic", _hyperbolicLibrary, "poincare_project");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(batchSize);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(ib, 0);
        encoder.SetBuffer(ob, 1);
        encoder.SetBytes((uint)batchSize, 2);
        encoder.SetBytes((uint)dim, 3);
        encoder.SetBytes(curvature, 4);
        encoder.SetBytes(epsilon, 5);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void MobiusAdd(IGpuBuffer x, IGpuBuffer y, IGpuBuffer output, int batchSize, int dim, float curvature)
    {
        ThrowIfDisposed();
        ValidateCurvature(curvature);
        if (x is not MetalGpuBuffer xb || y is not MetalGpuBuffer yb || output is not MetalGpuBuffer ob)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetPipeline("Hyperbolic", _hyperbolicLibrary, "mobius_add");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(batchSize);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(xb, 0);
        encoder.SetBuffer(yb, 1);
        encoder.SetBuffer(ob, 2);
        encoder.SetBytes((uint)batchSize, 3);
        encoder.SetBytes((uint)dim, 4);
        encoder.SetBytes(curvature, 5);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void PoincareExpMap(IGpuBuffer basePoint, IGpuBuffer tangentVec, IGpuBuffer output, int batchSize, int dim, float curvature)
    {
        ThrowIfDisposed();
        ValidateCurvature(curvature);
        if (basePoint is not MetalGpuBuffer bp || tangentVec is not MetalGpuBuffer tv || output is not MetalGpuBuffer ob)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetPipeline("Hyperbolic", _hyperbolicLibrary, "poincare_exp_map");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(batchSize);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(bp, 0);
        encoder.SetBuffer(tv, 1);
        encoder.SetBuffer(ob, 2);
        encoder.SetBytes((uint)batchSize, 3);
        encoder.SetBytes((uint)dim, 4);
        encoder.SetBytes(curvature, 5);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void PoincareDistance(IGpuBuffer x, IGpuBuffer y, IGpuBuffer output, int batchSize, int dim, float curvature)
    {
        ThrowIfDisposed();
        ValidateCurvature(curvature);
        if (x is not MetalGpuBuffer xb || y is not MetalGpuBuffer yb || output is not MetalGpuBuffer ob)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetPipeline("Hyperbolic", _hyperbolicLibrary, "poincare_distance");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(batchSize);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(xb, 0);
        encoder.SetBuffer(yb, 1);
        encoder.SetBuffer(ob, 2);
        encoder.SetBytes((uint)batchSize, 3);
        encoder.SetBytes((uint)dim, 4);
        encoder.SetBytes(curvature, 5);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void HyperbolicLinearForward(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer biases, IGpuBuffer output,
        int batchSize, int inputFeatures, int outputFeatures, float curvature, float epsilon)
    {
        ThrowIfDisposed();
        DispatchResidentMetal("hyperbolic_linear_forward", checked(batchSize * outputFeatures),
            new[] { input, weights, biases, output },
            (uint)batchSize, (uint)inputFeatures, (uint)outputFeatures);
        PoincareProject(output, output, batchSize, outputFeatures, curvature, epsilon);
    }

    public void HyperbolicLinearBackwardInput(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer gradInput,
        int batchSize, int inputFeatures, int outputFeatures, float curvature)
    {
        ThrowIfDisposed();
        DispatchResidentMetal("hyperbolic_backward_input", checked(batchSize * inputFeatures),
            new[] { gradOutput, weights, gradInput },
            (uint)batchSize, (uint)inputFeatures, (uint)outputFeatures);
    }

    public void HyperbolicLinearBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int batchSize, int inputFeatures, int outputFeatures, float curvature)
    {
        ThrowIfDisposed();
        DispatchResidentMetal("hyperbolic_backward_weights", checked(outputFeatures * inputFeatures),
            new[] { gradOutput, input, gradWeights },
            (uint)batchSize, (uint)inputFeatures, (uint)outputFeatures);
    }

    public void HyperbolicLinearBackwardBiases(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradBiases,
        int batchSize, int inputFeatures, int outputFeatures, float curvature)
    {
        ThrowIfDisposed();
        DispatchResidentMetal("hyperbolic_backward_biases", outputFeatures,
            new[] { gradOutput, gradBiases }, (uint)batchSize, (uint)outputFeatures);
    }

    #endregion

    #region Octonion Algebra Operations

    public void OctonionMultiply(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int count)
    {
        ThrowIfDisposed();
        if (a is not MetalGpuBuffer ab || b is not MetalGpuBuffer bb || output is not MetalGpuBuffer ob)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetPipeline("Octonion", _octonionLibrary, "octonion_multiply");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(count);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(ab, 0);
        encoder.SetBuffer(bb, 1);
        encoder.SetBuffer(ob, 2);
        encoder.SetBytes((uint)count, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void OctonionAdd(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int count)
    {
        // Element-wise addition of 8-component octonions = standard vector add
        Add(a, b, output, count * 8);
    }

    public void OctonionLinearForward(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer biases, IGpuBuffer output,
        int batchSize, int inputFeatures, int outputFeatures)
    {
        ThrowIfDisposed();
        if (input is not MetalGpuBuffer ib || weights is not MetalGpuBuffer wb ||
            biases is not MetalGpuBuffer bb || output is not MetalGpuBuffer ob)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        int totalPairs = batchSize * outputFeatures;
        var pipeline = GetPipeline("Octonion", _octonionLibrary, "octonion_linear_forward");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(totalPairs);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(ib, 0);
        encoder.SetBuffer(wb, 1);
        encoder.SetBuffer(bb, 2);
        encoder.SetBuffer(ob, 3);
        encoder.SetBytes((uint)batchSize, 4);
        encoder.SetBytes((uint)inputFeatures, 5);
        encoder.SetBytes((uint)outputFeatures, 6);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void OctonionLinearBackwardInput(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer gradInput,
        int batchSize, int inputFeatures, int outputFeatures)
    {
        ThrowIfDisposed();
        if (gradOutput is not MetalGpuBuffer gob || weights is not MetalGpuBuffer wb || gradInput is not MetalGpuBuffer gib)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        int totalPairs = batchSize * inputFeatures;
        var pipeline = GetPipeline("Octonion", _octonionLibrary, "octonion_linear_backward_input");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(totalPairs);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(gob, 0);
        encoder.SetBuffer(wb, 1);
        encoder.SetBuffer(gib, 2);
        encoder.SetBytes((uint)batchSize, 3);
        encoder.SetBytes((uint)inputFeatures, 4);
        encoder.SetBytes((uint)outputFeatures, 5);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void OctonionLinearBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int batchSize, int inputFeatures, int outputFeatures)
    {
        ThrowIfDisposed();
        if (gradOutput is not MetalGpuBuffer gob || input is not MetalGpuBuffer ib || gradWeights is not MetalGpuBuffer gwb)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        int totalPairs = outputFeatures * inputFeatures;
        var pipeline = GetPipeline("Octonion", _octonionLibrary, "octonion_linear_backward_weights");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(totalPairs);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(gob, 0);
        encoder.SetBuffer(ib, 1);
        encoder.SetBuffer(gwb, 2);
        encoder.SetBytes((uint)batchSize, 3);
        encoder.SetBytes((uint)inputFeatures, 4);
        encoder.SetBytes((uint)outputFeatures, 5);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void OctonionLinearBackwardBiases(IGpuBuffer gradOutput, IGpuBuffer gradBiases, int batchSize, int outputFeatures)
    {
        ThrowIfDisposed();
        if (gradOutput is not MetalGpuBuffer gob || gradBiases is not MetalGpuBuffer gbb)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = GetPipeline("Octonion", _octonionLibrary, "octonion_linear_backward_biases");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(outputFeatures);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(gob, 0);
        encoder.SetBuffer(gbb, 1);
        encoder.SetBytes((uint)batchSize, 2);
        encoder.SetBytes((uint)outputFeatures, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    #endregion

    #region Fused Kernel Operations

    public void HyperbolicLinearForwardFused(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer biases, IGpuBuffer output,
        int batchSize, int inputFeatures, int outputFeatures, float curvature, float epsilon)
    {
        HyperbolicLinearForward(input, weights, biases, output, batchSize, inputFeatures, outputFeatures, curvature, epsilon);
    }

    public void OctonionLinearForwardFusedReLU(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer biases, IGpuBuffer output,
        int batchSize, int inputFeatures, int outputFeatures)
    {
        ThrowIfDisposed();
        if (input is not MetalGpuBuffer ib || weights is not MetalGpuBuffer wb ||
            biases is not MetalGpuBuffer bb || output is not MetalGpuBuffer ob)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        int totalPairs = batchSize * outputFeatures;
        var pipeline = GetPipeline("Octonion", _octonionLibrary, "octonion_linear_forward_fused_relu");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(totalPairs);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(ib, 0);
        encoder.SetBuffer(wb, 1);
        encoder.SetBuffer(bb, 2);
        encoder.SetBuffer(ob, 3);
        encoder.SetBytes((uint)batchSize, 4);
        encoder.SetBytes((uint)inputFeatures, 5);
        encoder.SetBytes((uint)outputFeatures, 6);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    #endregion

    #region Complex Tensor Operations

    public void ComplexMultiply(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int numPairs)
    {
        ThrowIfDisposed();
        DispatchResidentMetal("complex_multiply_interleaved", numPairs,
            new[] { a, b, output }, (uint)numPairs);
    }

    public void ComplexConjugate(IGpuBuffer input, IGpuBuffer output, int numPairs)
    {
        ThrowIfDisposed();
        DispatchResidentMetal("complex_conjugate_interleaved", numPairs,
            new[] { input, output }, (uint)numPairs);
    }

    public void ComplexMagnitude(IGpuBuffer input, IGpuBuffer output, int numPairs)
    {
        ThrowIfDisposed();
        bool aliasesInput = ReferenceEquals(input, output);
        using var temporary = aliasesInput ? AllocateBuffer(numPairs) : null;
        IGpuBuffer target = temporary ?? output;
        DispatchResidentMetal("complex_magnitude_interleaved", numPairs,
            new[] { input, target }, (uint)numPairs);
        if (temporary is not null)
            Copy(temporary, output, numPairs);
    }

    #endregion

    #region Quantum Computing Operations

    public void QuantumMeasurement(IGpuBuffer realPart, IGpuBuffer imagPart, IGpuBuffer probabilities, int batchSize, int stateSize)
    {
        ThrowIfDisposed();
        int count = checked(batchSize * stateSize);
        DispatchResidentMetal("quantum_measurement", count,
            new[] { realPart, imagPart, probabilities }, (uint)count);
    }

    public void NormalizeProbabilities(IGpuBuffer probabilities, int batchSize, int stateSize)
    {
        ThrowIfDisposed();
        DispatchResidentMetal("normalize_probabilities", batchSize,
            new[] { probabilities }, (uint)batchSize, (uint)stateSize);
    }

    public void ComplexMatVec(IGpuBuffer matReal, IGpuBuffer matImag, IGpuBuffer vecReal, IGpuBuffer vecImag,
        IGpuBuffer outReal, IGpuBuffer outImag, int batchSize, int dim)
    {
        ThrowIfDisposed();
        int count = checked(batchSize * dim);
        bool aliasesInput = ReferenceEquals(outReal, matReal) || ReferenceEquals(outReal, matImag)
            || ReferenceEquals(outReal, vecReal) || ReferenceEquals(outReal, vecImag)
            || ReferenceEquals(outImag, matReal) || ReferenceEquals(outImag, matImag)
            || ReferenceEquals(outImag, vecReal) || ReferenceEquals(outImag, vecImag)
            || ReferenceEquals(outReal, outImag);
        using var temporaryReal = aliasesInput ? AllocateBuffer(count) : null;
        using var temporaryImag = aliasesInput ? AllocateBuffer(count) : null;
        IGpuBuffer realTarget = temporaryReal ?? outReal;
        IGpuBuffer imagTarget = temporaryImag ?? outImag;
        DispatchResidentMetal("complex_matvec", count,
            new[] { matReal, matImag, vecReal, vecImag, realTarget, imagTarget },
            (uint)batchSize, (uint)dim);
        if (temporaryReal is not null)
        {
            Copy(temporaryReal, outReal, count);
            Copy(temporaryImag!, outImag, count);
        }
    }

    public void MeasurementForward(IGpuBuffer input, IGpuBuffer output, int batchSize, int stateSize)
    {
        ThrowIfDisposed();
        int count = checked(batchSize * stateSize);
        bool aliasesInput = ReferenceEquals(input, output);
        using var temporary = aliasesInput ? AllocateBuffer(count) : null;
        IGpuBuffer target = temporary ?? output;
        DispatchResidentMetal("measurement_forward", count,
            new[] { input, target }, (uint)count);
        if (temporary is not null)
            Copy(temporary, output, count);
    }

    public void QuantumRotation(IGpuBuffer stateReal, IGpuBuffer stateImag, IGpuBuffer outReal, IGpuBuffer outImag,
        IGpuBuffer angles, int numQubits, int batchSize)
    {
        ThrowIfDisposed();
        if ((uint)numQubits >= 31u)
            throw new ArgumentOutOfRangeException(nameof(numQubits), "Qubit count must be between 0 and 30.");
        int stateSize = 1 << numQubits;
        int count = checked(batchSize * stateSize);
        bool outputsAlias = ReferenceEquals(outReal, outImag);
        using var temporaryReal = outputsAlias ? AllocateBuffer(count) : null;
        using var temporaryImag = outputsAlias ? AllocateBuffer(count) : null;
        IGpuBuffer realTarget = temporaryReal ?? outReal;
        IGpuBuffer imagTarget = temporaryImag ?? outImag;
        DispatchResidentMetal("quantum_rotation", count,
            new[] { stateReal, stateImag, angles, realTarget, imagTarget },
            (uint)stateSize, (uint)numQubits, (uint)batchSize);
        if (temporaryReal is not null)
        {
            Copy(temporaryReal, outReal, count);
            Copy(temporaryImag!, outImag, count);
        }
    }

    #endregion
}
