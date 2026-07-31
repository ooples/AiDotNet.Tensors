using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// An engine must compute on ITSELF: invoking an op on an explicit <see cref="CpuEngine"/>
/// must never dispatch to the process-global <see cref="AiDotNetEngine.Current"/>.
/// </summary>
/// <remarks>
/// <para>
/// Found via the gradcheck sweep. <c>TensorBroadcastSubtract&lt;double&gt;</c> produced an analytical
/// gradient of exactly 1 (correct — d(a-b)/da is unambiguously 1) against central finite differences
/// of 1.0133. The gradient was right and the FORWARD was wrong: for <c>a[0]-b[0]</c> the engine
/// returned <c>-0.2453465461730957</c> where the exact double result is <c>-0.24534654647360865</c> —
/// agreement to only ~8 significant digits, i.e. float32. Subtraction is exact in floating point, so
/// any error above rounding is a precision defect.
/// </para>
/// <para>
/// Root cause: <c>CpuEngine.TensorBroadcastSubtract</c>'s generic fallback calls
/// <c>Tensor&lt;T&gt;.BroadcastSubtract</c>, whose equal-shape shortcut delegated to
/// <c>Subtract(other)</c> → <c>AiDotNetEngine.Current.TensorSubtract</c>. On this GPU-auto-detect host
/// <c>AiDotNetEngine.Current</c> is <c>DirectGpuTensorEngine</c>, which computes elementwise
/// arithmetic in single precision — so an explicit <c>CpuEngine</c> silently returned float-precision
/// values in a <c>Tensor&lt;double&gt;</c>. <c>BroadcastAdd</c> had the identical shortcut;
/// <c>BroadcastMultiply</c>/<c>BroadcastDivide</c> already computed locally, which is why only these
/// two ops were affected.
/// </para>
/// <para>
/// These tests pin the contract at the two levels that matter: the engine op agrees with the exact
/// scalar result, and it agrees with the same engine's own non-broadcast kernel.
/// </para>
/// </remarks>
public class BroadcastEngineIsolationTests : IDisposable
{
    private readonly ITestOutputHelper _out;
    private readonly CpuEngine _engine = new();
    private readonly IEngine _originalCurrent;
    private readonly TripwireEngine _tripwire = new();

    /// <summary>
    /// A CpuEngine that records the moment anything dispatches to it.
    /// </summary>
    /// <remarks>
    /// Without this the precision assertions are HOST-DEPENDENT. On a CPU-only machine
    /// AiDotNetEngine.Current is already a CpuEngine, so an op that wrongly consults Current still
    /// computes in double and every assertion passes — the defect this file exists to pin would be
    /// invisible on exactly the hosts CI runs on. Installing an observable engine as Current turns
    /// "did it consult the global engine" into a direct observation instead of an inference from
    /// numerical precision.
    /// </remarks>
    private sealed class TripwireEngine : CpuEngine
    {
        public int Dispatches;
        public override Tensor<T> TensorSubtract<T>(Tensor<T> a, Tensor<T> b)
        {
            Dispatches++;
            return base.TensorSubtract(a, b);
        }
        public override Tensor<T> TensorAdd<T>(Tensor<T> a, Tensor<T> b)
        {
            Dispatches++;
            return base.TensorAdd(a, b);
        }
    }

    public BroadcastEngineIsolationTests(ITestOutputHelper o)
    {
        _out = o;
        _originalCurrent = AiDotNetEngine.Current;
        AiDotNetEngine.Current = _tripwire;
    }

    public void Dispose()
    {
        AiDotNetEngine.Current = _originalCurrent;
        GC.SuppressFinalize(this);
    }

    /// <summary>Asserts the process-global engine was never consulted.</summary>
    private void AssertGlobalEngineUntouched(string op)
    {
        _out.WriteLine($"global-engine dispatches during {op}: {_tripwire.Dispatches}");
        Assert.True(_tripwire.Dispatches == 0,
            $"{op} dispatched to AiDotNetEngine.Current {_tripwire.Dispatches} time(s). The op must " +
            $"compute on the engine it was invoked on. This passes on a CPU-only host by accident when " +
            $"only precision is checked, because Current is a CpuEngine there too.");
    }

    private static (Tensor<double> a, Tensor<double> b) Operands(int n, int seed)
    {
        var rng = new Random(seed);
        var a = new Tensor<double>([n]);
        var b = new Tensor<double>([n]);
        // Values near each other so the difference cancels leading digits — this is what
        // makes a float-precision intermediate visible in the relative error.
        for (int i = 0; i < n; i++) a[i] = 0.35 + rng.NextDouble() * 0.6;
        for (int i = 0; i < n; i++) b[i] = 0.35 + rng.NextDouble() * 0.6;
        return (a, b);
    }

    [Fact]
    public void TensorBroadcastSubtract_Double_KeepsDoublePrecision()
    {
        var (a, b) = Operands(6, seed: 1234);
        _out.WriteLine($"AiDotNetEngine.Current = {AiDotNetEngine.Current.GetType().Name}, op engine = {_engine.GetType().Name}");

        var r = _engine.TensorBroadcastSubtract(a, b);

        double worst = 0;
        for (int i = 0; i < r.Length; i++)
        {
            double exact = a[i] - b[i];
            double rel = Math.Abs(r[i] - exact) / Math.Max(1e-300, Math.Abs(exact));
            worst = Math.Max(worst, rel);
            _out.WriteLine($"r[{i}]={r[i]:G17}  exact={exact:G17}  rel={rel:E3}");
        }

        Assert.True(worst < 1e-15,
            $"TensorBroadcastSubtract<double> worst relative error {worst:E3} — subtraction is exact in " +
            "floating point, so any error above rounding means the double path computed at float precision.");
        AssertGlobalEngineUntouched(nameof(CpuEngine.TensorBroadcastSubtract));
    }

    [Fact]
    public void TensorBroadcastAdd_Double_KeepsDoublePrecision()
    {
        var (a, b) = Operands(6, seed: 4321);

        var r = _engine.TensorBroadcastAdd(a, b);

        double worst = 0;
        for (int i = 0; i < r.Length; i++)
        {
            double exact = a[i] + b[i];
            double rel = Math.Abs(r[i] - exact) / Math.Max(1e-300, Math.Abs(exact));
            worst = Math.Max(worst, rel);
            _out.WriteLine($"r[{i}]={r[i]:G17}  exact={exact:G17}  rel={rel:E3}");
        }

        Assert.True(worst < 1e-15,
            $"TensorBroadcastAdd<double> worst relative error {worst:E3} — addition of same-magnitude " +
            "doubles is correctly rounded, so error above 1 ULP means the double path computed at float precision.");
        AssertGlobalEngineUntouched(nameof(CpuEngine.TensorBroadcastAdd));
    }

    /// <summary>
    /// The broadcast op and the engine's own non-broadcast kernel must agree BIT-EXACTLY on
    /// equal-shaped operands — both are plain elementwise arithmetic on the same engine.
    /// This is the assertion that fails the instant an op starts dispatching elsewhere.
    /// </summary>
    [Theory]
    [InlineData(6)]
    [InlineData(64)]
    [InlineData(1000)]
    public void BroadcastOps_AgreeBitExactly_WithSameEngineElementwiseKernel(int n)
    {
        var (a, b) = Operands(n, seed: 909 + n);

        var subBroadcast = _engine.TensorBroadcastSubtract(a, b);
        var subDirect = _engine.TensorSubtract(a, b);
        var addBroadcast = _engine.TensorBroadcastAdd(a, b);
        var addDirect = _engine.TensorAdd(a, b);

        for (int i = 0; i < n; i++)
        {
            Assert.Equal(subDirect[i], subBroadcast[i]);
            Assert.Equal(addDirect[i], addBroadcast[i]);
        }
    }

    /// <summary>
    /// Genuinely-broadcasting shapes were never affected (they always used the local stride path),
    /// so this guards against the fix regressing the shape-expanding case.
    /// </summary>
    [Fact]
    public void BroadcastOps_StillBroadcast_AndStayExact()
    {
        var a = new Tensor<double>([2, 3]);
        for (int i = 0; i < 6; i++) a[i] = 0.35 + i * 0.11;
        var row = new Tensor<double>([3]);
        for (int i = 0; i < 3; i++) row[i] = 0.37 + i * 0.13;

        var diff = _engine.TensorBroadcastSubtract(a, row);
        var sum = _engine.TensorBroadcastAdd(a, row);

        Assert.Equal(new[] { 2, 3 }, diff.Shape.ToArray());
        Assert.Equal(new[] { 2, 3 }, sum.Shape.ToArray());
        for (int r = 0; r < 2; r++)
        {
            for (int c = 0; c < 3; c++)
            {
                Assert.Equal(a[r, c] - row[c], diff[r, c]);
                Assert.Equal(a[r, c] + row[c], sum[r, c]);
            }
        }
    }
}
