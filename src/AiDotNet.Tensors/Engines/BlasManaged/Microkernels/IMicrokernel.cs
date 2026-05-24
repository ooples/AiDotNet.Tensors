namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Marker interface for microkernel delegate signatures. Each microkernel is
/// invoked via an internal delegate matching this shape: the dispatcher binds
/// the concrete kernel based on (arch, precision, packing, trans). Concrete
/// delegates per arch are defined alongside their kernel files (added in
/// Phases B–E).
/// </summary>
internal interface IMicrokernel { }
