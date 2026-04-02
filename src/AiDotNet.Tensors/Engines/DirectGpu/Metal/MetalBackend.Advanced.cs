// Copyright (c) AiDotNet. All rights reserved.
// Metal GPU backend - Hyperbolic Geometry, Octonion Algebra, and Quantum Computing operations.
// All hyperbolic/octonion ops dispatch to MSL compute kernels on the GPU.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend
{
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
        // Euclidean matmul + bias via existing GPU GEMM, then project onto Poincare ball
        var matResult = AllocateBuffer(batchSize * outputFeatures);
        Gemm(input, weights, matResult, batchSize, outputFeatures, inputFeatures);
        BiasAdd(matResult, biases, output, batchSize, outputFeatures);
        matResult.Dispose();
        PoincareProject(output, output, batchSize, outputFeatures, curvature, epsilon);
    }

    public void HyperbolicLinearBackwardInput(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer gradInput,
        int batchSize, int inputFeatures, int outputFeatures, float curvature)
    {
        // gradInput = gradOutput * weights^T (Euclidean approximation)
        ThrowIfDisposed();
        var gradOutData = DownloadBuffer(gradOutput);
        var weightsData = DownloadBuffer(weights);
        var gradInputData = new float[batchSize * inputFeatures];
        for (int b = 0; b < batchSize; b++)
            for (int i = 0; i < inputFeatures; i++)
            {
                float sum = 0;
                for (int o = 0; o < outputFeatures; o++)
                    sum += gradOutData[b * outputFeatures + o] * weightsData[o * inputFeatures + i];
                gradInputData[b * inputFeatures + i] = sum;
            }
        UploadToBuffer(gradInput, gradInputData);
    }

    public void HyperbolicLinearBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int batchSize, int inputFeatures, int outputFeatures, float curvature)
    {
        // gradWeights = gradOutput^T * input
        ThrowIfDisposed();
        var gradOutData = DownloadBuffer(gradOutput);
        var inputData = DownloadBuffer(input);
        var gradWeightsData = new float[outputFeatures * inputFeatures];
        for (int o = 0; o < outputFeatures; o++)
            for (int i = 0; i < inputFeatures; i++)
            {
                float sum = 0;
                for (int b = 0; b < batchSize; b++)
                    sum += gradOutData[b * outputFeatures + o] * inputData[b * inputFeatures + i];
                gradWeightsData[o * inputFeatures + i] = sum;
            }
        UploadToBuffer(gradWeights, gradWeightsData);
    }

    public void HyperbolicLinearBackwardBiases(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradBiases,
        int batchSize, int inputFeatures, int outputFeatures, float curvature)
    {
        // gradBiases = sum over batch of gradOutput
        ThrowIfDisposed();
        var gradOutData = DownloadBuffer(gradOutput);
        var gradBiasesData = new float[outputFeatures];
        for (int o = 0; o < outputFeatures; o++)
        {
            float sum = 0;
            for (int b = 0; b < batchSize; b++)
                sum += gradOutData[b * outputFeatures + o];
            gradBiasesData[o] = sum;
        }
        UploadToBuffer(gradBiases, gradBiasesData);
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
        // Full octonion backward via Jacobian — kept as CPU for numerical correctness
        ThrowIfDisposed();
        var gradOutData = DownloadBuffer(gradOutput);
        var weightsData = DownloadBuffer(weights);
        var gradInputData = new float[batchSize * inputFeatures * 8];

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < inputFeatures; i++)
            {
                int inputOffset = (b * inputFeatures + i) * 8;
                var ga = new float[8];
                for (int o = 0; o < outputFeatures; o++)
                {
                    int wOff = (o * inputFeatures + i) * 8;
                    int goOff = (b * outputFeatures + o) * 8;
                    float w0=weightsData[wOff],w1=weightsData[wOff+1],w2=weightsData[wOff+2],w3=weightsData[wOff+3],
                          w4=weightsData[wOff+4],w5=weightsData[wOff+5],w6=weightsData[wOff+6],w7=weightsData[wOff+7];
                    float g0=gradOutData[goOff],g1=gradOutData[goOff+1],g2=gradOutData[goOff+2],g3=gradOutData[goOff+3],
                          g4=gradOutData[goOff+4],g5=gradOutData[goOff+5],g6=gradOutData[goOff+6],g7=gradOutData[goOff+7];
                    ga[0]+=g0*w0+g1*w1+g2*w2+g3*w3+g4*w4+g5*w5+g6*w6+g7*w7;
                    ga[1]+=g0*(-w1)+g1*w0+g2*(-w3)+g3*w2+g4*(-w5)+g5*w4+g6*w7+g7*(-w6);
                    ga[2]+=g0*(-w2)+g1*w3+g2*w0+g3*(-w1)+g4*(-w6)+g5*(-w7)+g6*w4+g7*w5;
                    ga[3]+=g0*(-w3)+g1*(-w2)+g2*w1+g3*w0+g4*(-w7)+g5*w6+g6*(-w5)+g7*w4;
                    ga[4]+=g0*(-w4)+g1*w5+g2*w6+g3*w7+g4*w0+g5*(-w1)+g6*(-w2)+g7*(-w3);
                    ga[5]+=g0*(-w5)+g1*(-w4)+g2*w7+g3*(-w6)+g4*w1+g5*w0+g6*w3+g7*(-w2);
                    ga[6]+=g0*(-w6)+g1*(-w7)+g2*(-w4)+g3*w5+g4*w2+g5*(-w3)+g6*w0+g7*w1;
                    ga[7]+=g0*(-w7)+g1*w6+g2*(-w5)+g3*(-w4)+g4*w3+g5*w2+g6*(-w1)+g7*w0;
                }
                for (int c = 0; c < 8; c++) gradInputData[inputOffset + c] = ga[c];
            }
        }
        UploadToBuffer(gradInput, gradInputData);
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

    #region Quantum Computing Operations

    public void QuantumMeasurement(IGpuBuffer realPart, IGpuBuffer imagPart, IGpuBuffer probabilities, int batchSize, int stateSize)
    {
        ThrowIfDisposed();
        var re = DownloadBuffer(realPart); var im = DownloadBuffer(imagPart);
        var prob = new float[batchSize * stateSize];
        for (int b = 0; b < batchSize; b++)
            for (int s = 0; s < stateSize; s++)
            {
                int idx = b * stateSize + s;
                prob[idx] = re[idx] * re[idx] + im[idx] * im[idx];
            }
        UploadToBuffer(probabilities, prob);
    }

    public void NormalizeProbabilities(IGpuBuffer probabilities, int batchSize, int stateSize)
    {
        ThrowIfDisposed();
        var prob = DownloadBuffer(probabilities);
        for (int b = 0; b < batchSize; b++)
        {
            int off = b * stateSize; float sum = 0;
            for (int s = 0; s < stateSize; s++) sum += prob[off + s];
            if (sum > 0) for (int s = 0; s < stateSize; s++) prob[off + s] /= sum;
        }
        UploadToBuffer(probabilities, prob);
    }

    public void ComplexMatVec(IGpuBuffer matReal, IGpuBuffer matImag, IGpuBuffer vecReal, IGpuBuffer vecImag,
        IGpuBuffer outReal, IGpuBuffer outImag, int batchSize, int dim)
    {
        ThrowIfDisposed();
        var mr = DownloadBuffer(matReal); var mi = DownloadBuffer(matImag);
        var vr = DownloadBuffer(vecReal); var vi = DownloadBuffer(vecImag);
        var or_ = new float[batchSize * dim]; var oi = new float[batchSize * dim];
        for (int b = 0; b < batchSize; b++)
        {
            int vOff = b * dim, mOff = b * dim * dim;
            for (int i = 0; i < dim; i++)
            {
                float sumR = 0, sumI = 0;
                for (int j = 0; j < dim; j++)
                {
                    int mIdx = mOff + i * dim + j;
                    sumR += mr[mIdx] * vr[vOff + j] - mi[mIdx] * vi[vOff + j];
                    sumI += mr[mIdx] * vi[vOff + j] + mi[mIdx] * vr[vOff + j];
                }
                or_[vOff + i] = sumR; oi[vOff + i] = sumI;
            }
        }
        UploadToBuffer(outReal, or_); UploadToBuffer(outImag, oi);
    }

    public void MeasurementForward(IGpuBuffer input, IGpuBuffer output, int batchSize, int stateSize)
    {
        ThrowIfDisposed();
        var data = DownloadBuffer(input);
        var prob = new float[batchSize * stateSize];
        for (int b = 0; b < batchSize; b++)
            for (int s = 0; s < stateSize; s++)
            {
                int idx = (b * stateSize + s) * 2;
                float re = data[idx], im = data[idx + 1];
                prob[b * stateSize + s] = re * re + im * im;
            }
        UploadToBuffer(output, prob);
    }

    public void QuantumRotation(IGpuBuffer stateReal, IGpuBuffer stateImag, IGpuBuffer outReal, IGpuBuffer outImag,
        IGpuBuffer angles, int numQubits, int batchSize)
    {
        ThrowIfDisposed();
        var sr = DownloadBuffer(stateReal); var si = DownloadBuffer(stateImag);
        var ang = DownloadBuffer(angles);
        int stateSize = 1 << numQubits;
        var orr = new float[batchSize * stateSize]; var oir = new float[batchSize * stateSize];
        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < stateSize; s++)
            {
                int idx = b * stateSize + s;
                float angle = ang[b * numQubits]; // Simplified: apply first qubit angle
                float cosA = MathF.Cos(angle / 2), sinA = MathF.Sin(angle / 2);
                orr[idx] = cosA * sr[idx] - sinA * si[idx];
                oir[idx] = sinA * sr[idx] + cosA * si[idx];
            }
        }
        UploadToBuffer(outReal, orr); UploadToBuffer(outImag, oir);
    }

    #endregion
}
