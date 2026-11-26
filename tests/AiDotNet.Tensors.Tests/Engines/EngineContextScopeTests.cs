using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// <see cref="AiDotNetEngine.UseInCurrentContext"/>: the calling context, and work it starts, see the scoped engine; a
/// thread outside it keeps the process-wide engine; disposal (nested in reverse order) restores what the context saw.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class EngineContextScopeTests
{
    [Fact]
    public async Task A_context_scope_applies_to_its_own_context_only_and_restores_on_dispose()
    {
        var global = AiDotNetEngine.Current;
        var teacher = new CpuEngine();
        var inner = new CpuEngine();
        Assert.NotSame(global, teacher);

        IEngine? outside = null;
        using var started = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        // A thread started before the scope, running while the scope is open.
        var other = new Thread(() => { started.Set(); release.Wait(); outside = AiDotNetEngine.Current; });
        other.Start();
        started.Wait();

        using (AiDotNetEngine.UseInCurrentContext(teacher))
        {
            Assert.Same(teacher, AiDotNetEngine.Current);
            // Work the scoped context starts inherits it.
            Assert.Same(teacher, await Task.Run(() => AiDotNetEngine.Current));
            release.Set();
            other.Join();
            Assert.Same(global, outside);

            using (AiDotNetEngine.UseInCurrentContext(inner))
                Assert.Same(inner, AiDotNetEngine.Current);
            Assert.Same(teacher, AiDotNetEngine.Current);
        }
        Assert.Same(global, AiDotNetEngine.Current);
    }
}
