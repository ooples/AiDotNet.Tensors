using System.Threading;
using AiDotNet.Tensors.Engines.Optimization;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Optimization;

/// <summary>
/// "TensorCodecOptions.Current.EnableCompilation = false" is the documented way to opt out of compilation. Current
/// returned a brand-new default whenever no options had been set on the thread, so that assignment configured a
/// throwaway object and had no effect - AiDotNet's fused-vs-eager parity tests silently compared fused with fused.
/// </summary>
public class TensorCodecOptionsCurrentTests
{
    [Fact]
    public void An_assignment_through_Current_is_seen_by_the_next_read_on_the_same_thread()
    {
        var prior = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(null);   // a thread with nothing configured
            TensorCodecOptions.Current.EnableCompilation = false;
            Assert.False(TensorCodecOptions.Current.EnableCompilation);
        }
        finally
        {
            TensorCodecOptions.SetCurrent(prior);
        }
    }

    [Fact]
    public void An_assignment_through_Current_stays_on_its_own_thread()
    {
        var prior = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(null);
            TensorCodecOptions.Current.EnableCompilation = false;
            bool other = false;
            var thread = new Thread(() => other = TensorCodecOptions.Current.EnableCompilation);
            thread.Start();
            thread.Join();
            Assert.True(other);
        }
        finally
        {
            TensorCodecOptions.SetCurrent(prior);
        }
    }
}
