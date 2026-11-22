# 16-bit channels-last activations under autocast

## Why

Under BF16/FP16 autocast, a DCGAN training step on an RTX 3090 spends 8.26 ms in kernels. About 4.5 ms of that is
cuDNN's tensor-core convolutions. About 2.4 ms is the casts around them: every convolution pass narrows its FP32 NCHW
operands to 16-bit NHWC and widens its result back (72 narrows and 40 widens per step, measured with nsys). PyTorch's
autocast keeps a convolution's output 16-bit, and the next op reads it as it is. We will do the same, but with the
channels-last layout cuDNN's 16-bit kernels read natively, so neither the cast nor the transpose happens between ops.

## Representation

The tensor keeps its logical shape `[n, c, h, w]` and its element type `T = float`. What changes is what its device
state holds:

- **The 16-bit slot.** It holds a 16-bit buffer in NHWC order, the element type (BF16 or FP16), and the tensor version
  it was produced at.
- **No FP32 binding at first.** `_gpuBuffer` is null until something asks for FP32. This is deliberate: about 190 sites
  read `_gpuBuffer` directly, and handing them a 16-bit buffer would make them read garbage. A null binding makes them
  take their non-resident path. That path is correct, merely slower, and it is counted (see below).
- **A host materializer.** It is registered through `HostSync`. A host read widens the slot on the device and downloads
  the result.

## Consumers

1. **16-bit-aware ops** read the slot directly:
   - the convolution passes (forward, data gradient, filter gradient, and the transposed convolution);
   - BatchNorm forward and backward, which accumulate their statistics in FP32;
   - the fused bias, activation and BatchNorm epilogues;
   - ReLU, LeakyReLU, Tanh, Sigmoid;
   - Concat;
   - BCE-with-logits.

   Where their result is an activation, it goes into a new 16-bit NHWC slot.
2. **The device gap guard.** The input resolvers that unaware ops use are `GetOrAllocateBuffer` (1,106 call sites),
   `ResolveResidentBufferNoUpload`, `GetResidentOrPersistentInputBuffer` and `GetWeightBufferPreferResident`. When one
   of them meets a tensor with a 16-bit slot, it widens the slot on the device into an FP32 NCHW buffer. That buffer
   comes from the step scratch pool inside a compiled step, so it is safe under capture. The guard binds it as
   `_gpuBuffer`, and from then on the tensor has both forms. Activation memory grows only for tensors an unaware op
   actually reads.
3. **Everything else** reaches the host materializer. That is correct, but it costs a round trip and breaks a graph
   capture, so it is reported through `GpuLaunchProbe.OnFallback`. A regression test holds the BF16 DCGAN step at zero
   such fallbacks and zero device-to-host reads.

## Validity

A 16-bit slot is only valid while the tensor's values are the ones it was produced from:

- **Host write:** the version increases and the slot is dropped.
- **Device write through FP32:** an in-place op or a resident binding that writes `_gpuBuffer` drops the slot. These
  writes go through `BindResidentBuffer` and `MarkModified`, which are the hooks.
- **In a compiled step,** versions do not move. An action that writes a tensor in place is recorded at trace time,
  and the tracer drops the slot then, so replays never read a stale one.

Parameters never get a slot: they are FP32 master weights. Each convolution narrows its filter once per pass, as it
does today. Caching that narrow per step is a separate, smaller change.

## Precision

The rounding is the same as the current autocast route. Today a convolution's output is rounded to 16 bits inside
cuDNN and then widened. Keeping it 16-bit only moves the rounding point to the consumer, and the consumer would have
rounded it on its own narrow anyway.

The one real difference is element-wise ops between convolutions, which now see 16-bit inputs. PyTorch's autocast does
the same: its convolution outputs are 16-bit tensors. BatchNorm statistics, the loss and every reduction accumulate in
FP32, as in PyTorch.

## Backends

- **CUDA:** native kernels.
- **HIP, Metal, OpenCL, Vulkan, WebGPU:** these do not stage convolutions channels-last (they report no
  `SupportsChannelsLastMixedConv`), so they never create a slot and every op on them keeps the current FP32 path. The
  new interfaces still go on all six backends, as `TENS0001` requires, with real implementations rather than stubs, so
  that a backend which gains a 16-bit channels-last convolution also gets the slot path.

## Order of work, each step measured

1. **The slot, the host materializer, the gap guard and the validity hooks.** Convolution outputs stay 16-bit, and
   every other op widens at the gap. Expected effect on time: neutral (one widen where there is one today).
   Correctness tests: a host read, an unaware op, an in-place write, capture.
2. **Convolution inputs read the slot.** This removes the narrow of an activation that a 16-bit op produced.
3. **BatchNorm, activation and epilogue kernels** read and write 16-bit NHWC.
4. **The backward chain.** Gradients flowing into a convolution's backward stay 16-bit NHWC.

Each step is A/B'd with nsys against the previous build. The final comparison against torch is a quiet, interleaved
run.
