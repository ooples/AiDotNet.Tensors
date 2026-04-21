using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Collection fixture that serializes every test class that touches the
/// process-wide <c>VulkanBackend.Instance</c> singleton.
/// <para>
/// The Vulkan device (<see cref="AiDotNet.Tensors.Engines.DirectGpu.Vulkan.VulkanDevice"/>)
/// keeps per-thread <c>VkCommandPool</c> + <c>VkCommandBuffer</c> + <c>VkFence</c>
/// resources in a <see cref="System.Collections.Concurrent.ConcurrentDictionary{TKey, TValue}"/>
/// keyed by <see cref="System.Environment.CurrentManagedThreadId"/>. The pool is
/// never trimmed — every distinct thread that ever touches the device leaves an
/// entry behind for the lifetime of the process. xUnit's default "parallel
/// across collections" scheduler picks fresh threads for each test class, and
/// when several Vulkan-direct test classes overlap, the cumulative Vulkan
/// host-memory footprint exceeds the driver's budget mid-run and the next
/// <c>vkQueueSubmit</c> fails with <c>VK_ERROR_OUT_OF_HOST_MEMORY (-2)</c>.
/// That cascade then fails every subsequent Vulkan op in the run.
/// </para>
/// <para>
/// Pinning these classes into a single non-parallel collection forces the
/// per-thread cache to converge to a small steady-state — the runner reuses
/// the same handful of threads, so each thread's Vulkan resources get reused
/// rather than re-allocated. Mirror of <c>BlasGlobalStateCollection</c> for
/// the same reason on a different driver surface.
/// </para>
/// </summary>
[CollectionDefinition("VulkanGlobalState", DisableParallelization = true)]
public sealed class VulkanGlobalStateCollection { }
