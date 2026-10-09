using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>Activations, fused add-multiply ops, index-map ops and the Linalg additions against PyTorch float64.</summary>
[Collection("EngineCurrentGlobalState")]
public class TorchMiscOpsTests
{
    private readonly CpuEngine _engine = new CpuEngine();

    private static Tensor<double> T(int[] shape, params double[] values) => new Tensor<double>(values, shape);

    private static void Close(double[] expected, Tensor<double> actual, string what, double tolerance = 1e-12)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual.GetFlat(i)) <= tolerance * Math.Max(1, Math.Abs(expected[i])),
                $"{what}[{i}]: torch {expected[i]:R}, ours {actual.GetFlat(i):R}");
    }

    private static Tensor<double> X() => T(new[] { 7 }, -3.0, -1.2, -0.4, 0.0, 0.3, 1.1, 2.5);
    private static Tensor<double> M() => T(new[] { 3, 3 }, 3, 1, 4, 1, 5, 9, 2, 6, 5);
    private static Tensor<double> V() => T(new[] { 3 }, 1, -2, 0.5);
    private static Tensor<double> W() => T(new[] { 3 }, 0.3, 2, -1);

    [Fact]
    public void Activations_MatchTorch()
    {
        Close(new[] { -0.6903653492868647, -0.5739353814964333, -0.3046973145945685, 0.0, 0.3, 1.1, 2.5 }, _engine.TensorCelu(X(), 0.7), "celu");
        Close(new[] { -0.5, -0.5, -0.4, 0.0, 0.3, 1.1, 1.2 }, _engine.TensorHardtanh(X(), -0.5, 1.2), "hardtanh");
        Close(new[] { -3.048587351573742, -1.4632824673380311, -0.9130152523999526, -0.6931471805599453, -0.5543552444685271, -0.2873353251154308, -0.07888973429254963 }, _engine.TensorLogSigmoid(X()), "logsigmoid");
        Close(new[] { -0.75, -0.5454545454545454, -0.28571428571428575, 0.0, 0.23076923076923075, 0.5238095238095238, 0.7142857142857143 }, _engine.TensorSoftsign(X()), "softsign");
        Close(new[] { -0.6000000000000001, -0.24, -0.08000000000000002, 0.0, 0.3, 1.1, 2.5 }, _engine.TensorRrelu(X(), 0.1, 0.3, training: false), "rrelu eval");
        Close(new[] { 0.11419519938459449, 0.8437947344813395, 0.04201006613406605, 0.9816903928255046, 0.017980286735531543, 0.00032932043896389293, 0.9362395518765058, 0.017147825545520388, 0.0466126225779739 }, _engine.TensorSoftmin(M(), 1), "softmin");
        var noisy = _engine.TensorRrelu(X(), 0.1, 0.3, training: true, seed: 3).ToArray();
        for (int i = 0; i < 3; i++) Assert.InRange(noisy[i] / X().GetFlat(i), 0.1, 0.3);
    }

    [Fact]
    public void FusedAddMultiply_MatchTorch()
    {
        Close(new[] { 1.15, -4.0, 0.25 }, _engine.TensorAddcmul(V(), W(), V(), 0.5), "addcmul");
        Close(new[] { 1.45, -0.5, 0.875 }, _engine.TensorAddcdiv(V(), W(), T(new[] { 3 }, 2, 4, -8), 3), "addcdiv");
        Close(new[] { -1.7000000000000002, 1.6000000000000014, 15.45 }, _engine.TensorAddmv(V(), M(), W(), 0.5, 2), "addmv");
        Close(new[] { 5.7, 0.0, 9.0, 2.6, 14.0, 16.0, 3.85, 11.0, 10.5 }, _engine.TensorAddr(M(), V(), W(), 2, -1), "addr");
        var b1 = T(new[] { 2, 2, 3 }, Enumerable.Range(0, 12).Select(i => i / 5.0).ToArray());
        var b2 = T(new[] { 2, 3, 2 }, Enumerable.Range(0, 12).Select(i => i / 7.0).ToArray());
        Close(new[] { 1.0714285714285714, 1.2428571428571429, 2.0999999999999996, 2.7857142857142856, 10.32857142857143, 11.528571428571428, 14.442857142857143, 16.15714285714286 }, _engine.TensorBaddbmm(T(new[] { 2, 2, 2 }, Enumerable.Repeat(1.0, 8).ToArray()), b1, b2, 0.5, 2), "baddbmm");
        Close(new[] { 10.899999999999999, 12.271428571428572, 16.042857142857144, 18.442857142857143 }, _engine.TensorAddbmm(T(new[] { 2, 2 }, 1, 1, 1, 1), b1, b2, 0.5, 2), "addbmm");
        Close(new[] { -1.7, 6.0, -2.0 }, _engine.TensorRsub(V(), W(), 2), "rsub");
    }

    [Fact]
    public void IndexMapOps_MatchTorch_AndRouteGradients()
    {
        Close(new[] { 1.0, 1.0, 4.0, 2.0, 5.0, 5.0, 3.0, 6.0, 9.0 }, _engine.TensorMsort(M()), "msort");
        Close(new[] { 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0 }, _engine.TensorDiagflat(T(new[] { 2 }, 1, 2), -1), "diagflat");
        Close(new[] { 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 9.0 }, _engine.TensorDiagonalScatter(new Tensor<double>(new[] { 3, 4 }), T(new[] { 3 }, 7, 8, 9), 1), "diagonal_scatter");

        var m = M();
        using var tape = new GradientTape<double>();
        var weights = T(new[] { 3, 3 }, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        var loss = _engine.ReduceSum(_engine.TensorMultiply(_engine.TensorMsort(m), weights), null, false);
        // Each element's gradient is the weight of the position it was sorted to (column 0 sorts 3,1,2 -> 1,2,3).
        Close(new[] { 7.0, 2, 3, 1, 5, 9, 4, 8, 6 }, tape.ComputeGradients(loss, new[] { m })[m], "msort gradient");
    }

    [Fact]
    public void LinalgAdditions_MatchTorch()
    {
        var spd = T(new[] { 3, 3 }, 4, 2, 0.6, 2, 5, 1, 0.6, 1, 3);
        var factor = Linalg.Cholesky(spd);
        var b = T(new[] { 3, 2 }, 1, 2, 0, 1, 3, -1);
        Close(new[] { 0.24663677130044845, 0.5291479820627802, -0.3094170403587444, 0.08161434977578474, 1.053811659192825, -0.4663677130044843 }, Linalg.CholeskySolve(b, factor), "cholesky_solve", 1e-10);
        Close(new[] { 0.31390134529147984, -0.1210762331838565, -0.022421524663677125, -0.1210762331838565, 0.2609865470852018, -0.06278026905829596, -0.022421524663677125, -0.06278026905829596, 0.3587443946188341 }, Linalg.CholeskyInverse(factor), "cholesky_inverse", 1e-10);
        Close(new[] { 3.7977338590260183 }, Linalg.Logdet(spd), "logdet", 1e-10);

        var a = T(new[] { 4, 3 }, 2, -1, 0.5, 1, 3, 2, 0, 1, 4, 1, 1, 1);
        var (packed, tau) = Linalg.Geqrf(a);
        Close(new[] { -2.449489742783178, -0.816496580927726, -1.632993161855452, 0.22474487139158905, -3.366501646120693, -2.722905743185854, 0.0, 0.15606118794672874, -3.3420229872128084, 0.22474487139158905, 0.16249737798832942, -0.02529145726061859 }, packed, "geqrf a", 1e-10);
        Close(new[] { 1.816496580927726, 1.9033833254838346, 1.9987215021803844 }, tau, "geqrf tau", 1e-10);
        var c = T(new[] { 4, 2 }, 1, 0, 2, 1, 0, 3, 1, 1);
        Close(new[] { -0.08034929195143281, -0.22095731023731693, -2.3734847402271133, -0.433869091021474, -0.5940885257860046, -3.161635394789695, 0.0846935813468008, 0.8757837114961079 }, Linalg.Ormqr(packed, tau, c), "ormqr", 1e-10);
        Close(new[] { -2.041241452319315, -0.816496580927726, -1.2871918058696765, -1.8812803316556812, 0.4004267173619032, -2.5565705800798435, -0.12700012700019042, 0.5080005080007619 }, Linalg.Ormqr(packed, tau, c, transpose: true), "ormqr transposed", 1e-10);
        Close(new[] { -1.224744871391589, -0.29704426289300234, -2.0021335868095163, 0.6350006350009524, -1.632993161855452, -2.871427874632355, -0.1540102759084243, -0.25400025400038106 }, Linalg.Ormqr(packed, tau, T(new[] { 2, 4 }, 1, 0, 2, 1, 0, 3, 1, 1), left: false), "ormqr right", 1e-10);

        var square = T(new[] { 3, 3 }, 0, 2, 1, 4, 1, 3, 2, 5, 7);
        var (lu, pivots) = Linalg.LuFactor(square);
        var (p, l, u) = Linalg.LuUnpack(lu, pivots);
        Close(square.ToArray(), _engine.TensorMatMul(p, _engine.TensorMatMul(l, u)), "P·L·U", 1e-12);
    }

    [Fact]
    public void Predicates_AndBroadcastShapes_AndStorageOffset()
    {
        Assert.True(_engine.TensorIsFloatingPoint(V()));
        Assert.False(_engine.TensorIsFloatingPoint(new Tensor<int>(new[] { 1 })));
        Assert.False(_engine.TensorIsSigned(new Tensor<byte>(new[] { 1 })));
        Assert.True(_engine.TensorIsSigned(V()));
        Assert.False(_engine.TensorIsComplex(V()));
        Assert.True(_engine.TensorIsNonzero(T(new[] { 1 }, 2)));
        Assert.Throws<InvalidOperationException>(() => _engine.TensorIsNonzero(V()));
        Assert.True(_engine.TensorIsSameSize(V(), W()));
        Assert.Equal(new[] { 4, 3, 5 }, _engine.BroadcastShapes(new[] { 3, 1 }, new[] { 4, 1, 5 }, new[] { 5 }));
        Assert.Throws<ArgumentException>(() => _engine.BroadcastShapes(new[] { 3 }, new[] { 4 }));
        Assert.Equal(0, M().StorageOffset);
        Assert.Equal(3, M().Slice(0, 1, 2).StorageOffset);
    }
}
