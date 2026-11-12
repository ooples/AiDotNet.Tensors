using System.Threading;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

public class TensorPoolFreshScopeTests
{
    [Fact]
    public void Scope_AppliesToTheCallingThreadOnly_AndRestoresOnDispose()
    {
        bool processDefault = TensorPool.ForceFreshAllocations;
        bool? seenElsewhere = null;
        using (TensorPool.ForceFreshAllocationsScope(!processDefault))
        {
            Assert.Equal(!processDefault, TensorPool.ForceFreshAllocations);
            var other = new Thread(() => seenElsewhere = TensorPool.ForceFreshAllocations);
            other.Start();
            other.Join();

            using (TensorPool.ForceFreshAllocationsScope(processDefault))
                Assert.Equal(processDefault, TensorPool.ForceFreshAllocations);
            Assert.Equal(!processDefault, TensorPool.ForceFreshAllocations);
        }
        Assert.Equal(processDefault, seenElsewhere);
        Assert.Equal(processDefault, TensorPool.ForceFreshAllocations);
    }
}
