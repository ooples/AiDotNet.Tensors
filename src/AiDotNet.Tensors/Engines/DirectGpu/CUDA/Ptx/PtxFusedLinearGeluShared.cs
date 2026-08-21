using System;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>Shared ABI validation and byte-stable PTX emitters for decode GELU kernels.</summary>
internal static class PtxFusedLinearGeluShared
{
    internal static void Require(
        DirectPtxTensorView view,
        DirectPtxTensorContract contract,
        string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent || view.ByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
    }

    internal static bool Overlaps(DirectPtxTensorView left, DirectPtxTensorView right)
    {
        nuint leftStart = PtxCompat.ToNuint(left.Pointer);
        nuint rightStart = PtxCompat.ToNuint(right.Pointer);
        nuint leftEnd = checked(leftStart + left.ByteLength);
        nuint rightEnd = checked(rightStart + right.ByteLength);
        return leftStart < rightEnd && rightStart < leftEnd;
    }

    internal static void EmitFp32WarpButterflyReduction(
        StringBuilder ptx,
        string accumulator,
        string accumulatorBits,
        string shuffledBits,
        string shuffledValue)
    {
        foreach (int delta in new[] { 16, 8, 4, 2, 1 })
        {
            ptx.AppendLine($"    mov.b32 {accumulatorBits}, {accumulator};");
            ptx.AppendLine(
                $"    shfl.sync.bfly.b32 {shuffledBits}, {accumulatorBits}, {delta}, 31, 0xffffffff;");
            ptx.AppendLine($"    mov.b32 {shuffledValue}, {shuffledBits};");
            ptx.AppendLine($"    add.rn.f32 {accumulator}, {accumulator}, {shuffledValue};");
        }
    }

    internal static void EmitInt32WarpButterflyReduction(
        StringBuilder ptx,
        string accumulator,
        string shuffledValue)
    {
        foreach (int delta in new[] { 16, 8, 4, 2, 1 })
        {
            ptx.AppendLine(
                $"    shfl.sync.bfly.b32 {shuffledValue}, {accumulator}, {delta}, 31, 0xffffffff;");
            ptx.AppendLine($"    add.s32 {accumulator}, {accumulator}, {shuffledValue};");
        }
    }

    internal static void EmitTanhGeluEpilogue(
        StringBuilder ptx,
        string value,
        string temporary,
        string outputAddress)
    {
        ptx.AppendLine($"    mul.rn.f32 {temporary}, {value}, {value};");
        ptx.AppendLine($"    mul.rn.f32 {temporary}, {temporary}, {value};");
        ptx.AppendLine($"    fma.rn.f32 {temporary}, {temporary}, 0f3D372713, {value};");
        ptx.AppendLine($"    mul.rn.f32 {temporary}, {temporary}, 0f3F4C422A;");
        ptx.AppendLine($"    tanh.approx.f32 {temporary}, {temporary};");
        ptx.AppendLine($"    add.rn.f32 {temporary}, {temporary}, 0f3F800000;");
        ptx.AppendLine($"    mul.rn.f32 {temporary}, {temporary}, {value};");
        ptx.AppendLine($"    mul.rn.f32 {temporary}, {temporary}, 0f3F000000;");
        ptx.AppendLine($"    st.global.f32 [{outputAddress}], {temporary};");
    }
}
