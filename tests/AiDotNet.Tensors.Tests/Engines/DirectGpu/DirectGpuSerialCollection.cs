using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>Serializes all DirectGpu test classes so they don't share an OpenCL context concurrently
/// (AMD RDNA1 drivers crash / the device wedges on multi-threaded shared-context use — observed as a
/// multi-hour hang when the GPU parity suites run in parallel). <c>DisableParallelization = true</c> is
/// REQUIRED: without it this collection still runs concurrently with the other GPU collections and the
/// ungrouped GPU classes, defeating the serialization this type exists to provide.
/// Defined on every target framework: it used to live inside GpuCpuAutoDifferentialTests.cs, which is
/// compiled only off .NET Framework, so on net471 the collection had no definition and its classes ran
/// in parallel with everything else (a concurrent test's readback then failed AbcScan's zero-readback check).</summary>
[CollectionDefinition("DirectGpuSerial", DisableParallelization = true)]
public sealed class DirectGpuSerialCollection { }
