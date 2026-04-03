using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Functional tests verifying that differentiable ops actually record to the tape
/// when a GradientTape is active. Catches cases where an op is classified as
/// differentiable in OpRegistry but the DifferentiableOps.Record* call is missing.
/// </summary>
public class TapeRecordingTests
{
    private readonly IEngine _engine = new CpuEngine();

    private void AssertRecords(string opName, Action<IEngine> runOp)
    {
        using var tape = new GradientTape<float>();
        int before = tape.EntryCount;
        runOp(_engine);
        Assert.True(tape.EntryCount > before,
            $"Op '{opName}' did not record to the tape. Add DifferentiableOps.Record* in CpuEngine.");
    }

    // ── Arithmetic ──────────────────────────────────────────────

    [Fact] public void TensorAdd_Records() => AssertRecords("TensorAdd", e =>
        e.TensorAdd(new Tensor<float>([2, 3]), new Tensor<float>([2, 3])));

    [Fact] public void TensorSubtract_Records() => AssertRecords("TensorSubtract", e =>
        e.TensorSubtract(new Tensor<float>([2, 3]), new Tensor<float>([2, 3])));

    [Fact] public void TensorMultiply_Records() => AssertRecords("TensorMultiply", e =>
        e.TensorMultiply(new Tensor<float>([2, 3]), new Tensor<float>([2, 3])));

    [Fact] public void TensorDivide_Records() => AssertRecords("TensorDivide", e =>
    {
        var ones = new Tensor<float>([2, 3]);
        for (int i = 0; i < ones.Length; i++) ones[i] = 1.0f;
        e.TensorDivide(new Tensor<float>([2, 3]), ones);
    });

    [Fact] public void TensorNegate_Records() => AssertRecords("TensorNegate", e =>
        e.TensorNegate(new Tensor<float>([2, 3])));

    [Fact] public void TensorMultiplyScalar_Records() => AssertRecords("TensorMultiplyScalar", e =>
        e.TensorMultiplyScalar(new Tensor<float>([2, 3]), 2.0f));

    [Fact] public void TensorAddMany_Records() => AssertRecords("TensorAddMany", e =>
        e.TensorAddMany(new Tensor<float>([4]), new Tensor<float>([4]), new Tensor<float>([4])));

    [Fact] public void TensorMultiplyMany_Records() => AssertRecords("TensorMultiplyMany", e =>
    {
        var a = Tensor<float>.CreateRandom([4]);
        var b = Tensor<float>.CreateRandom([4]);
        e.TensorMultiplyMany(a, b);
    });

    [Fact] public void TensorAddScaled_Records() => AssertRecords("TensorAddScaled", e =>
        e.TensorAddScaled(new Tensor<float>([4]), new Tensor<float>([4]), 1.0f, 2.0f));

    // ── Math ────────────────────────────────────────────────────

    [Fact] public void TensorExp_Records() => AssertRecords("TensorExp", e =>
        e.TensorExp(new Tensor<float>([4])));

    [Fact] public void TensorLog_Records() => AssertRecords("TensorLog", e =>
    {
        var t = new Tensor<float>([4]);
        for (int i = 0; i < t.Length; i++) t[i] = 1.0f;
        e.TensorLog(t);
    });

    [Fact] public void TensorSqrt_Records() => AssertRecords("TensorSqrt", e =>
    {
        var t = new Tensor<float>([4]);
        for (int i = 0; i < t.Length; i++) t[i] = 1.0f;
        e.TensorSqrt(t);
    });

    [Fact] public void TensorCosh_Records() => AssertRecords("TensorCosh", e =>
        e.TensorCosh(new Tensor<float>([4])));

    [Fact] public void TensorSinh_Records() => AssertRecords("TensorSinh", e =>
        e.TensorSinh(new Tensor<float>([4])));

    [Fact] public void TensorFrac_Records() => AssertRecords("TensorFrac", e =>
        e.TensorFrac(new Tensor<float>([4])));

    [Fact] public void TensorPow_Records() => AssertRecords("TensorPow", e =>
        e.TensorPow(Tensor<float>.CreateRandom([4]), 2.0f));

    [Fact] public void TensorAbs_Records() => AssertRecords("TensorAbs", e =>
        e.TensorAbs(new Tensor<float>([4])));

    // ── Matrix ──────────────────────────────────────────────────

    [Fact] public void TensorMatMul_Records() => AssertRecords("TensorMatMul", e =>
        e.TensorMatMul(Tensor<float>.CreateRandom([2, 3]), Tensor<float>.CreateRandom([3, 4])));

    [Fact] public void BatchMatMul_Records() => AssertRecords("BatchMatMul", e =>
        e.BatchMatMul(Tensor<float>.CreateRandom([2, 3, 4]), Tensor<float>.CreateRandom([2, 4, 5])));

    [Fact] public void TensorOuterProduct_Records() => AssertRecords("TensorOuterProduct", e =>
        e.TensorOuterProduct(Tensor<float>.CreateRandom([3]), Tensor<float>.CreateRandom([4])));

    [Fact] public void TensorOuter_Records() => AssertRecords("TensorOuter", e =>
        e.TensorOuter(Tensor<float>.CreateRandom([3]), Tensor<float>.CreateRandom([4])));

    // ── Activations ─────────────────────────────────────────────

    [Fact] public void ReLU_Records() => AssertRecords("ReLU", e =>
        e.ReLU(Tensor<float>.CreateRandom([4])));

    [Fact] public void Sigmoid_Records() => AssertRecords("Sigmoid", e =>
        e.Sigmoid(Tensor<float>.CreateRandom([4])));

    [Fact] public void Tanh_Records() => AssertRecords("Tanh", e =>
        e.Tanh(Tensor<float>.CreateRandom([4])));

    [Fact] public void Softmax_Records() => AssertRecords("Softmax", e =>
        e.Softmax(Tensor<float>.CreateRandom([2, 4])));

    [Fact] public void GLU_Records() => AssertRecords("GLU", e =>
        e.GLU(Tensor<float>.CreateRandom([2, 8])));

    [Fact] public void Sparsemax_Records() => AssertRecords("Sparsemax", e =>
        e.Sparsemax(Tensor<float>.CreateRandom([2, 4])));

    // ── Shape ───────────────────────────────────────────────────

    [Fact] public void Reshape_Records() => AssertRecords("Reshape", e =>
        e.Reshape(Tensor<float>.CreateRandom([2, 3]), [6]));

    [Fact] public void TensorSliceAxis_Records() => AssertRecords("TensorSliceAxis", e =>
        e.TensorSliceAxis(Tensor<float>.CreateRandom([3, 4, 5]), 0, 1));

    [Fact] public void Concat_Records() => AssertRecords("Concat", e =>
        e.Concat(new[] { Tensor<float>.CreateRandom([2, 3]), Tensor<float>.CreateRandom([2, 3]) }, 0));

    [Fact] public void TensorStack_Records() => AssertRecords("TensorStack", e =>
        e.TensorStack(new[] { Tensor<float>.CreateRandom([3]), Tensor<float>.CreateRandom([3]) }, 0));

    [Fact] public void TensorDiagonal_Records() => AssertRecords("TensorDiagonal", e =>
        e.TensorDiagonal(Tensor<float>.CreateRandom([3, 3])));

    // ── Reduction ───────────────────────────────────────────────

    [Fact] public void ReduceSum_Records() => AssertRecords("ReduceSum", e =>
        e.ReduceSum(Tensor<float>.CreateRandom([2, 3]), new[] { 1 }));

    [Fact] public void ReduceMean_Records() => AssertRecords("ReduceMean", e =>
        e.ReduceMean(Tensor<float>.CreateRandom([2, 3]), new[] { 1 }, false));

    [Fact] public void ReduceMax_Records() => AssertRecords("ReduceMax", e =>
        e.ReduceMax(Tensor<float>.CreateRandom([2, 3]), new[] { 1 }, false, out _));

    // ── Convolution / Pooling ───────────────────────────────────

    [Fact] public void Conv2D_Records() => AssertRecords("Conv2D", e =>
        e.Conv2D(Tensor<float>.CreateRandom([1, 1, 4, 4]), Tensor<float>.CreateRandom([1, 1, 3, 3])));

    [Fact] public void MaxPool2D_Records() => AssertRecords("MaxPool2D", e =>
        e.MaxPool2D(Tensor<float>.CreateRandom([1, 1, 4, 4]), 2));

    [Fact] public void DepthwiseConv2D_Records() => AssertRecords("DepthwiseConv2D", e =>
        e.DepthwiseConv2D(Tensor<float>.CreateRandom([1, 2, 4, 4]), Tensor<float>.CreateRandom([2, 1, 3, 3]), [1, 1], [1, 1]));

    [Fact] public void PixelShuffle_Records() => AssertRecords("PixelShuffle", e =>
        e.PixelShuffle(Tensor<float>.CreateRandom([1, 4, 2, 2]), 2));

    // ── Spatial ─────────────────────────────────────────────────

    [Fact] public void Crop_Records() => AssertRecords("Crop", e =>
        e.Crop(Tensor<float>.CreateRandom([1, 1, 4, 4]), 0, 0, 2, 2));

    [Fact] public void Pad_Records() => AssertRecords("Pad", e =>
        e.Pad(Tensor<float>.CreateRandom([1, 1, 4, 4]), 1, 1, 1, 1, 0f));

    [Fact] public void TensorCumSum_Records() => AssertRecords("TensorCumSum", e =>
        e.TensorCumSum(Tensor<float>.CreateRandom([3, 4]), 1));

    [Fact] public void TensorRepeatElements_Records() => AssertRecords("TensorRepeatElements", e =>
        e.TensorRepeatElements(Tensor<float>.CreateRandom([2, 3]), 2, 0));

    // ── Scatter / Gather ────────────────────────────────────────

    [Fact] public void Gather_Records() => AssertRecords("Gather", e =>
        e.Gather(Tensor<float>.CreateRandom([3, 4]), new Tensor<int>([2], new Vector<int>([0, 2])), 0));

    [Fact] public void TensorIndexSelect_Records() => AssertRecords("TensorIndexSelect", e =>
        e.TensorIndexSelect(Tensor<float>.CreateRandom([3, 4]), new Tensor<int>([2], new Vector<int>([0, 2])), 0));

    [Fact] public void TensorMaskedFill_Records() => AssertRecords("TensorMaskedFill", e =>
        e.TensorMaskedFill(Tensor<float>.CreateRandom([4]), new bool[] { true, false, true, false }, -999f));

    // ── Normalization ───────────────────────────────────────────

    [Fact] public void BatchNorm_Records() => AssertRecords("BatchNorm", e =>
        e.BatchNorm(Tensor<float>.CreateRandom([2, 3, 4, 4]),
            Tensor<float>.CreateRandom([3]), Tensor<float>.CreateRandom([3]),
            1e-5, out _, out _));

    [Fact] public void LayerNorm_Records() => AssertRecords("LayerNorm", e =>
    {
        e.LayerNorm(Tensor<float>.CreateRandom([2, 4]),
            Tensor<float>.CreateRandom([4]), Tensor<float>.CreateRandom([4]),
            1e-5, out _, out _);
    });

    // ── Signal ──────────────────────────────────────────────────

    [Fact] public void RFFT_Records() => AssertRecords("RFFT", e =>
        e.RFFT(Tensor<float>.CreateRandom([8])));

    // ── Loss ────────────────────────────────────────────────────

    [Fact] public void TensorBinaryCrossEntropy_Records() => AssertRecords("TensorBinaryCrossEntropy", e =>
    {
        var pred = new Tensor<float>([4]);
        for (int i = 0; i < 4; i++) pred[i] = 0.5f;
        var target = new Tensor<float>([4]);
        for (int i = 0; i < 4; i++) target[i] = i % 2 == 0 ? 1f : 0f;
        e.TensorBinaryCrossEntropy(pred, target, 1e-7f);
    });

    // ── Complex ─────────────────────────────────────────────────

    [Fact] public void ComplexMagnitudeSquared_Records() => AssertRecords("ComplexMagnitudeSquared", e =>
        e.ComplexMagnitudeSquared(Tensor<float>.CreateRandom([4]), Tensor<float>.CreateRandom([4])));

    // ── Other ───────────────────────────────────────────────────

    [Fact] public void OctonionMatMulTensor_Records() => AssertRecords("OctonionMatMulTensor", e =>
        e.OctonionMatMulTensor(Tensor<float>.CreateRandom([2, 3, 8]), Tensor<float>.CreateRandom([4, 3, 8])));

    [Fact] public void ScatterAdd_Records() => AssertRecords("ScatterAdd", e =>
    {
        var source = Tensor<float>.CreateRandom([4]);
        var indices = new Tensor<int>([4], new Vector<int>([0, 1, 0, 1]));
        e.ScatterAdd(source, indices);
    });
}
