// Copyright (c) AiDotNet. All rights reserved.
// Secondary interface for backends that support the parity-210 hot-path
// kernels. DirectGpuTensorEngine type-tests against this interface when
// dispatching, so backends that don't implement it transparently fall
// through to the CpuEngine path via inheritance.

namespace AiDotNet.Tensors.Engines.DirectGpu
{
    /// <summary>
    /// Optional capability interface for GPU backends that ship native
    /// kernels for the 40 hot-path ops in Issue #210's op surface.
    /// Methods are synchronous per convention; WebGPU's async variants
    /// are available via concrete-type dispatch.
    /// </summary>
    public interface IParity210Backend
    {
        // Movement
        void Parity210Roll1D(IGpuBuffer input, IGpuBuffer output, int outerSize, int axisSize, int innerSize, int shift);
        void Parity210FlipAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int axisSize, int innerSize);
        void Parity210Triu(IGpuBuffer input, IGpuBuffer output, int batchSize, int rows, int cols, int diagonal);
        void Parity210Tril(IGpuBuffer input, IGpuBuffer output, int batchSize, int rows, int cols, int diagonal);
        void Parity210DiagEmbed(IGpuBuffer input, IGpuBuffer output, int batchSize, int diagLen, int matSize, int offset);

        // Cumulative
        void Parity210CumSum(IGpuBuffer input, IGpuBuffer output, int outerSize, int axisSize, int innerSize);
        void Parity210CumProd(IGpuBuffer input, IGpuBuffer output, int outerSize, int axisSize, int innerSize);
        void Parity210CumMax(IGpuBuffer input, IGpuBuffer output, int outerSize, int axisSize, int innerSize);
        void Parity210CumMin(IGpuBuffer input, IGpuBuffer output, int outerSize, int axisSize, int innerSize);
        void Parity210LogCumSumExp(IGpuBuffer input, IGpuBuffer output, int outerSize, int axisSize, int innerSize);

        // Element-wise binary special
        void Parity210Hypot(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size);
        void Parity210Copysign(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size);
        void Parity210Fmod(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size);
        void Parity210Remainder(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size);
        void Parity210FloatPower(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size);
        void Parity210LogAddExp(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size);
        void Parity210LogAddExp2(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size);
        void Parity210Xlogy(IGpuBuffer x, IGpuBuffer y, IGpuBuffer output, int size);
        void Parity210Xlog1py(IGpuBuffer x, IGpuBuffer y, IGpuBuffer output, int size);

        // Element-wise unary special
        void Parity210Erfc(IGpuBuffer input, IGpuBuffer output, int size);
        void Parity210Erfinv(IGpuBuffer input, IGpuBuffer output, int size);
        void Parity210Lgamma(IGpuBuffer input, IGpuBuffer output, int size);
        void Parity210Digamma(IGpuBuffer input, IGpuBuffer output, int size);
        void Parity210I0(IGpuBuffer input, IGpuBuffer output, int size);
        void Parity210I1(IGpuBuffer input, IGpuBuffer output, int size);
        void Parity210I0e(IGpuBuffer input, IGpuBuffer output, int size);
        void Parity210I1e(IGpuBuffer input, IGpuBuffer output, int size);
        void Parity210IsFinite(IGpuBuffer input, IGpuBuffer output, int size);
        void Parity210IsNan(IGpuBuffer input, IGpuBuffer output, int size);
        void Parity210IsInf(IGpuBuffer input, IGpuBuffer output, int size);
        void Parity210NanToNum(IGpuBuffer input, IGpuBuffer output, int size, float nanVal, float posInfVal, float negInfVal);

        // Pairwise
        void Parity210CosineSimilarityLast(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int n, int d, float eps);
        void Parity210CdistL2(IGpuBuffer x1, IGpuBuffer x2, IGpuBuffer output, int n, int m, int d);
    }
}
