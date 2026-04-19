// Copyright (c) AiDotNet. All rights reserved.
// Vulkan launcher shims for the parity-210 kernels. Uses VulkanBackend's
// existing GlslUnaryOp / GlslBinaryOp helpers, which cache compiled SPIR-V
// pipelines by source hash so repeated dispatches are O(1) after first call.

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed unsafe partial class VulkanBackend
{
    // Push-constant layouts — sizeof in bytes. GLSL struct alignment rules
    // round each uint/int/float to 4 bytes.
    private const uint PushCstU1 = 4u;
    private const uint PushCstU3 = 12u;
    private const uint PushCstU4 = 16u;
    private const uint PushCstU5 = 20u;
    private const uint PushCstU6 = 24u;
    private const uint PushCstU7 = 28u;

    // -----------------------------------------------------------------------
    // MOVEMENT
    // -----------------------------------------------------------------------

    public void Parity210Roll1D(IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize, int shift)
    {
        int total = outerSize * axisSize * innerSize;
        var pc = new uint[] { (uint)outerSize, (uint)axisSize, (uint)innerSize, unchecked((uint)shift) };
        GlslUnaryOp(VulkanParity210Kernels.Roll1D, input, output, total, pc, PushCstU4);
    }

    public void Parity210FlipAxis(IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
    {
        int total = outerSize * axisSize * innerSize;
        var pc = new uint[] { (uint)outerSize, (uint)axisSize, (uint)innerSize };
        GlslUnaryOp(VulkanParity210Kernels.FlipAxis, input, output, total, pc, PushCstU3);
    }

    public void Parity210Triu(IGpuBuffer input, IGpuBuffer output,
        int batchSize, int rows, int cols, int diagonal)
    {
        int total = batchSize * rows * cols;
        var pc = new uint[] { (uint)batchSize, (uint)rows, (uint)cols, unchecked((uint)diagonal) };
        GlslUnaryOp(VulkanParity210Kernels.Triu, input, output, total, pc, PushCstU4);
    }

    public void Parity210Tril(IGpuBuffer input, IGpuBuffer output,
        int batchSize, int rows, int cols, int diagonal)
    {
        int total = batchSize * rows * cols;
        var pc = new uint[] { (uint)batchSize, (uint)rows, (uint)cols, unchecked((uint)diagonal) };
        GlslUnaryOp(VulkanParity210Kernels.Tril, input, output, total, pc, PushCstU4);
    }

    public void Parity210DiagEmbed(IGpuBuffer input, IGpuBuffer output,
        int batchSize, int diagLen, int matSize, int offset)
    {
        int total = batchSize * matSize * matSize;
        var pc = new uint[] { (uint)batchSize, (uint)diagLen, (uint)matSize, unchecked((uint)offset) };
        GlslUnaryOp(VulkanParity210Kernels.DiagEmbed, input, output, total, pc, PushCstU4);
    }

    // -----------------------------------------------------------------------
    // CUMULATIVE
    // -----------------------------------------------------------------------

    public void Parity210CumSum(IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
        => DispatchCumulativeP210(VulkanParity210Kernels.CumSumAxis, input, output, outerSize, axisSize, innerSize);

    public void Parity210CumProd(IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
        => DispatchCumulativeP210(VulkanParity210Kernels.CumProdAxis, input, output, outerSize, axisSize, innerSize);

    public void Parity210CumMax(IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
        => DispatchCumulativeP210(VulkanParity210Kernels.CumMaxAxis, input, output, outerSize, axisSize, innerSize);

    public void Parity210CumMin(IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
        => DispatchCumulativeP210(VulkanParity210Kernels.CumMinAxis, input, output, outerSize, axisSize, innerSize);

    public void Parity210LogCumSumExp(IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
        => DispatchCumulativeP210(VulkanParity210Kernels.LogCumSumExpAxis, input, output, outerSize, axisSize, innerSize);

    private void DispatchCumulativeP210(string shader,
        IGpuBuffer input, IGpuBuffer output,
        int outerSize, int axisSize, int innerSize)
    {
        int total = outerSize * innerSize;
        var pc = new uint[] { (uint)outerSize, (uint)axisSize, (uint)innerSize };
        GlslUnaryOp(shader, input, output, total, pc, PushCstU3);
    }

    // -----------------------------------------------------------------------
    // ELEMENT-WISE BINARY
    // -----------------------------------------------------------------------

    public void Parity210Hypot(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int size)
        => GlslBinaryOp(VulkanParity210Kernels.Hypot, a, b, o, size);

    public void Parity210Copysign(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int size)
        => GlslBinaryOp(VulkanParity210Kernels.Copysign, a, b, o, size);

    public void Parity210Fmod(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int size)
        => GlslBinaryOp(VulkanParity210Kernels.Fmod, a, b, o, size);

    public void Parity210Remainder(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int size)
        => GlslBinaryOp(VulkanParity210Kernels.Remainder, a, b, o, size);

    public void Parity210FloatPower(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int size)
        => GlslBinaryOp(VulkanParity210Kernels.FloatPower, a, b, o, size);

    public void Parity210LogAddExp(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int size)
        => GlslBinaryOp(VulkanParity210Kernels.LogAddExp, a, b, o, size);

    public void Parity210LogAddExp2(IGpuBuffer a, IGpuBuffer b, IGpuBuffer o, int size)
        => GlslBinaryOp(VulkanParity210Kernels.LogAddExp2, a, b, o, size);

    public void Parity210Xlogy(IGpuBuffer x, IGpuBuffer y, IGpuBuffer o, int size)
        => GlslBinaryOp(VulkanParity210Kernels.Xlogy, x, y, o, size);

    public void Parity210Xlog1py(IGpuBuffer x, IGpuBuffer y, IGpuBuffer o, int size)
        => GlslBinaryOp(VulkanParity210Kernels.Xlog1py, x, y, o, size);

    // -----------------------------------------------------------------------
    // ELEMENT-WISE UNARY SPECIAL
    // -----------------------------------------------------------------------

    public void Parity210Erfc(IGpuBuffer input, IGpuBuffer output, int size)
        => GlslUnaryOp(VulkanParity210Kernels.Erfc, input, output, size);

    public void Parity210Erfinv(IGpuBuffer input, IGpuBuffer output, int size)
        => GlslUnaryOp(VulkanParity210Kernels.Erfinv, input, output, size);

    public void Parity210Lgamma(IGpuBuffer input, IGpuBuffer output, int size)
        => GlslUnaryOp(VulkanParity210Kernels.LgammaApprox, input, output, size);

    public void Parity210Digamma(IGpuBuffer input, IGpuBuffer output, int size)
        => GlslUnaryOp(VulkanParity210Kernels.Digamma, input, output, size);

    public void Parity210I0(IGpuBuffer input, IGpuBuffer output, int size)
        => GlslUnaryOp(VulkanParity210Kernels.I0, input, output, size);

    public void Parity210I1(IGpuBuffer input, IGpuBuffer output, int size)
        => GlslUnaryOp(VulkanParity210Kernels.I1, input, output, size);

    public void Parity210I0e(IGpuBuffer input, IGpuBuffer output, int size)
        => GlslUnaryOp(VulkanParity210Kernels.I0e, input, output, size);

    public void Parity210I1e(IGpuBuffer input, IGpuBuffer output, int size)
        => GlslUnaryOp(VulkanParity210Kernels.I1e, input, output, size);

    public void Parity210IsFinite(IGpuBuffer input, IGpuBuffer output, int size)
        => GlslUnaryOp(VulkanParity210Kernels.IsFinite, input, output, size);

    public void Parity210IsNan(IGpuBuffer input, IGpuBuffer output, int size)
        => GlslUnaryOp(VulkanParity210Kernels.IsNan, input, output, size);

    public void Parity210IsInf(IGpuBuffer input, IGpuBuffer output, int size)
        => GlslUnaryOp(VulkanParity210Kernels.IsInf, input, output, size);

    public void Parity210NanToNum(IGpuBuffer input, IGpuBuffer output, int size,
        float nanVal, float posInfVal, float negInfVal)
    {
        var pc = new uint[] {
            (uint)size,
            System.BitConverter.ToUInt32(System.BitConverter.GetBytes(nanVal), 0),
            System.BitConverter.ToUInt32(System.BitConverter.GetBytes(posInfVal), 0),
            System.BitConverter.ToUInt32(System.BitConverter.GetBytes(negInfVal), 0)
        };
        GlslUnaryOp(VulkanParity210Kernels.NanToNum, input, output, size, pc, PushCstU4);
    }

    // -----------------------------------------------------------------------
    // PAIRWISE
    // -----------------------------------------------------------------------

    public void Parity210CosineSimilarityLast(
        IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int n, int d, float eps)
    {
        var pc = new uint[] {
            (uint)n, (uint)d,
            System.BitConverter.ToUInt32(System.BitConverter.GetBytes(eps), 0)
        };
        GlslBinaryOp(VulkanParity210Kernels.CosineSimilarityLast, a, b, output, n, pc, PushCstU3);
    }

    public void Parity210CdistL2(IGpuBuffer x1, IGpuBuffer x2, IGpuBuffer output, int n, int m, int d)
    {
        int total = n * m;
        var pc = new uint[] { (uint)n, (uint)m, (uint)d };
        GlslBinaryOp(VulkanParity210Kernels.CdistL2, x1, x2, output, total, pc, PushCstU3);
    }
}
