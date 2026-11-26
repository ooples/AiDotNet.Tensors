# Changelog

## [0.154.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.153.0...v0.154.0) (2026-11-26)


### Features

* checkpointed deterministic gated deltanet backward on every non-cuda backend ([97293ce](https://github.com/ooples/AiDotNet.Tensors/commit/97293ce7fdc924cf8c42ed0e6806d727d95aba96))


### Bug Fixes

* rename GLSL/WGSL reserved Householder identifiers, guard net471 module-init test, and sync 0.153.0 suppressions ([dc78072](https://github.com/ooples/AiDotNet.Tensors/commit/dc7807225acbbe4ff22f1c7ae5bc0b60097d7a15))


### Performance

* **gpu:** fused Householder/sequence-mixer ops, multi-tensor checkpoint flush, and in-place pointer refresh (2.66x faster than Compiled PyTorch + Triton) ([4cf0768](https://github.com/ooples/AiDotNet.Tensors/commit/4cf076818ea6312d1ad0b465bc5bb59507735a0b))
* **gpu:** GPU-resident tape broadcast/slice views, O(n) CudaStepArena planner, and warp-register GLA kernels ([0963db6](https://github.com/ooples/AiDotNet.Tensors/commit/0963db6016d71ddf84cafc1399e8718fa7b2b9c8))

## [0.153.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.152.1...v0.153.0) (2026-11-26)


### Features

* **compilation:** activation checkpointing inside the compiled GPU training plan ([5f10c35](https://github.com/ooples/AiDotNet.Tensors/commit/5f10c3557c4cb40fd6084234674fb757f3f13a0e))
* **compilation:** capture the optimizer update of a fused cross-entropy step ([7ddd75c](https://github.com/ooples/AiDotNet.Tensors/commit/7ddd75cd33a688f7853a37c5f81335e5f4e2e4f1))
* **compilation:** expose a closure guard's traced leaves to its caller ([49450c9](https://github.com/ooples/AiDotNet.Tensors/commit/49450c92bb04cd3c979dcb0a6abd70e763dff56e))
* **compilation:** expose a closure guard's traced leaves to its caller ([80c5d75](https://github.com/ooples/AiDotNet.Tensors/commit/80c5d75a035cb17fd93afe610569e873a7994156))
* **compilation:** gradient exchange between the backward and the update ([5fab0b0](https://github.com/ooples/AiDotNet.Tensors/commit/5fab0b0f6b0731d53c8c75f139244579a2171c71))
* **compilation:** in-plan loss scaling and cuBLAS matmul autocast for mixed precision ([041c64e](https://github.com/ooples/AiDotNet.Tensors/commit/041c64e014caddfb11eaed1a96be2179cfb649e8))
* **compilation:** mark activation-checkpoint segments in a training trace ([daa537f](https://github.com/ooples/AiDotNet.Tensors/commit/daa537f5f6b2d78b1d178c7f99b30bf2983a4acc))
* **compilation:** weight-tied loop with per-iteration checkpointing and a graph-safe adaptive exit ([df78260](https://github.com/ooples/AiDotNet.Tensors/commit/df78260cfcfba36c34ea4719eb2c108922fdbe06))
* **cuda:** matmul autocast casts the operands to 16 bits by default, as torch.autocast does ([28c55fd](https://github.com/ooples/AiDotNet.Tensors/commit/28c55fd1c88ee4aad79b8944dbba921cae818ed1))
* **cuda:** serve allocations between captured steps from the plan's idle step memory ([6a1745e](https://github.com/ooples/AiDotNet.Tensors/commit/6a1745e948f22df08bdf3134e9530f3b6e1fea83))
* **engines:** AiDotNetEngine.UseInCurrentContext -- a public, context-scoped engine ([71a3201](https://github.com/ooples/AiDotNet.Tensors/commit/71a3201091622e1709a4637abe280edfc5b08364))
* **engines:** device byte budget with LRU eviction for bf16 weight copies ([4fe5fd1](https://github.com/ooples/AiDotNet.Tensors/commit/4fe5fd1367e92e9243ee846841710028b390e5d2))
* **gpu:** a gradient exchange can stage through caller-owned host memory registered in place ([7d984c6](https://github.com/ooples/AiDotNet.Tensors/commit/7d984c6ef9444b31e1094f49ff2a5484b82c4348))
* **gpu:** a typed host-wait policy; stream waits block once they outlast a short poll ([4886465](https://github.com/ooples/AiDotNet.Tensors/commit/48864658720abcb2b63eb698cc6c9d90e950df1e))
* **gpu:** byte negate, abs and sign on the device ([a75610c](https://github.com/ooples/AiDotNet.Tensors/commit/a75610c5e8bfe138ff4c91bacd88cb2e8d831ab5))
* **gpu:** compute doubles on metal by integer emulation, opt-in ([549cf05](https://github.com/ooples/AiDotNet.Tensors/commit/549cf05e9da35a229652ba21049eed705b97503b))
* **gpu:** compute doubles on webgpu by integer emulation, opt-in ([1e27b18](https://github.com/ooples/AiDotNet.Tensors/commit/1e27b18efddb7fc68a084285d1bbfe55f2f710c8))
* **gpu:** compute integer and bool tensors on the device in their own type ([4392a06](https://github.com/ooples/AiDotNet.Tensors/commit/4392a0697c3ce46f76824feb02bc0786b6d8971c))
* **gpu:** divide, abs, sqrt, exp, log, tanh and sigmoid on 16-bit float buffers ([65572f9](https://github.com/ooples/AiDotNet.Tensors/commit/65572f9139c29c8b331b61e997237216886f634c))
* **gpu:** divide, functions, scalar forms and sum on 16-bit float buffers ([928d21e](https://github.com/ooples/AiDotNet.Tensors/commit/928d21e8e7300613b839e6576e247024669214e2))
* **gpu:** double-precision max, min and mean on the device ([b8a35e5](https://github.com/ooples/AiDotNet.Tensors/commit/b8a35e597ac62dce4587c5473a629651c2984177))
* **gpu:** double-precision max, min and mean reductions ([2b20c8d](https://github.com/ooples/AiDotNet.Tensors/commit/2b20c8d708fd2008a5a7942b0955021483c02153))
* **gpu:** half tensors take the 16-bit float device routes ([3afec92](https://github.com/ooples/AiDotNet.Tensors/commit/3afec92c629a73a3675cf07c83611d31c10aa4e7))
* **gpu:** integer divide and abs on the device behind device-side checks ([48dbfa5](https://github.com/ooples/AiDotNet.Tensors/commit/48dbfa53429e384d58ab8337640ead6b3f200ab5))
* **gpu:** integer divide, remainder, comparisons, abs and sign kernels ([cd70f03](https://github.com/ooples/AiDotNet.Tensors/commit/cd70f03f8a0b92e0dc988348afc8cb9f6f4ceea8))
* **gpu:** integer divide, remainder, comparisons, abs, sign and min/max reductions ([b1ac555](https://github.com/ooples/AiDotNet.Tensors/commit/b1ac5552b2c3987aaabc4fbebb59c1f8b44b8ddc))
* **gpu:** integer min and max reductions on the device ([8f711d1](https://github.com/ooples/AiDotNet.Tensors/commit/8f711d15da0fd4d19bbadd2da9ed0290b60bb195))
* **gpu:** integer, bool and 16-bit float compute on every backend through typed interfaces ([bf34152](https://github.com/ooples/AiDotNet.Tensors/commit/bf3415262eb14d10d3746c3c259ecd232ebd5552))
* **gpu:** keep bfloat16 tensors on the device through the typed route ([4f1774c](https://github.com/ooples/AiDotNet.Tensors/commit/4f1774c2d16754cee4a3eb5203c8f9b3391312e7))
* **gpu:** keep integer remainder, comparisons, abs, sign and scalar divide on the device ([e6a9091](https://github.com/ooples/AiDotNet.Tensors/commit/e6a9091f5a7e4c5486c62c71877d33ee0114d08e))
* **gpu:** matmul autocast on every backend, and opencl's 16-bit kernels compile again ([65e23ca](https://github.com/ooples/AiDotNet.Tensors/commit/65e23ca61a32f02120bf1c5cb1ff3f123bf5f893))
* **gpu:** narrow integer, bool, int64 and 16-bit float kernels on vulkan, metal and webgpu ([6f62c6b](https://github.com/ooples/AiDotNet.Tensors/commit/6f62c6ba34284081a5b7b8f7aea0a14f3cb691f7))
* **gpu:** packed kernels for the narrow integer, bool, int64 and 16-bit float types on vulkan ([ce62295](https://github.com/ooples/AiDotNet.Tensors/commit/ce622959e68d40e27e78e1a87a44e5c45cfb2fba))
* **gpu:** packed kernels for the narrow types on metal, with narrow element copies ([6c4af77](https://github.com/ooples/AiDotNet.Tensors/commit/6c4af77ad4446de9c4fa1a135f88c09c210f867e))
* **gpu:** packed kernels for the narrow types on webgpu, with narrow element copies ([ad51481](https://github.com/ooples/AiDotNet.Tensors/commit/ad51481a096edd2aceead45b26b3063fd5ea90e1))
* **gpu:** report half rows in the capability table and cover typed half buffers ([3cede47](https://github.com/ooples/AiDotNet.Tensors/commit/3cede47af121b49884dcd1d386f7d088e6085bb5))
* **gpu:** route integer tensor max and min to the device ([99f830e](https://github.com/ooples/AiDotNet.Tensors/commit/99f830eb5f14278b6f3f02fc26c8cb2321dcb22b))
* **gpu:** scalar forms of the 16-bit float operations, and divide on the bits ([6901692](https://github.com/ooples/AiDotNet.Tensors/commit/69016923df885ce789aaa7b5da57d4d5679a0b0e))
* **gpu:** sum of 16-bit float buffers on the device ([d3b81fb](https://github.com/ooples/AiDotNet.Tensors/commit/d3b81fb2ee906b6e7a3285d33c52a0cb1c827ddf))
* **gpu:** typed 16-bit float interfaces replace the fp16 side interfaces ([612f757](https://github.com/ooples/AiDotNet.Tensors/commit/612f75797275f8e62783839bcba3e6d34475eba5))
* **gpu:** typed-buffer safety, capability table, typed pool and doubles on every backend ([16708f4](https://github.com/ooples/AiDotNet.Tensors/commit/16708f46050c6430226477ae97bb305facffd720))
* per-process GPU memory budget and CPU core sets for sharing a machine ([283e30e](https://github.com/ooples/AiDotNet.Tensors/commit/283e30e4d9cbe83df1b4e09057c7ebd42f126888))
* per-process GPU memory budget and CPU core sets for sharing a machine ([b420281](https://github.com/ooples/AiDotNet.Tensors/commit/b420281a5aba0cebe95a779ae3b30045c5680500))
* **streaming:** the store's training/inference hint can be read back ([e967965](https://github.com/ooples/AiDotNet.Tensors/commit/e9679651b441d5efa7652c8775156c1b5fa5be07))
* **telemetry:** device memory, utilization and the compiled plan's clip norm for training monitors ([467b1a3](https://github.com/ooples/AiDotNet.Tensors/commit/467b1a34a6cdeb62a67c2fbef44970ef8cd11543))
* **telemetry:** the device's memory in use as NVML reports it (nvidia-smi's memory.used) ([79aebeb](https://github.com/ooples/AiDotNet.Tensors/commit/79aebeb0802c135338b2493836d7908e2638adf4))
* **webgpu:** awaitable backend interface and async engine ([#1811](https://github.com/ooples/AiDotNet.Tensors/issues/1811), phase 1) ([dbfc582](https://github.com/ooples/AiDotNet.Tensors/commit/dbfc58274979911673d106336545131ddeeb9ef1))
* **webgpu:** awaitable backend interface and async engine so browser builds never block on the device ([de8e518](https://github.com/ooples/AiDotNet.Tensors/commit/de8e518fc7d1509d0219b3aa57a36240359341fc))


### Bug Fixes

* a captured gather or index-copy recomputes indices its trace produced ([74a3a92](https://github.com/ooples/AiDotNet.Tensors/commit/74a3a92835c9dae6b50ee90b9447ec838e9ac6f2))
* a max over an axis is nan for a range that holds one, on the cpu and every device ([563c812](https://github.com/ooples/AiDotNet.Tensors/commit/563c81268cb419f710ef910f23949095887f3380))
* **autodiff:** the fused conv-bias-ReLU backward reads device-resident tensors on a CPU engine ([3d3725b](https://github.com/ooples/AiDotNet.Tensors/commit/3d3725b0af74ec3938a5569334cbd8088ec7365d))
* bf16 weight uploads make no host copy of the weight ([045b620](https://github.com/ooples/AiDotNet.Tensors/commit/045b620c87581c79b9b0a509a3a3dbce3518dd23))
* bf16 weight uploads make no host copy of the weight ([d25fe9e](https://github.com/ooples/AiDotNet.Tensors/commit/d25fe9e3df3417e567539ada9ee45b5bc4c22848))
* **broadcast:** expand a size-1 axis with a tile, not an add of a target-sized zero tensor ([70d7696](https://github.com/ooples/AiDotNet.Tensors/commit/70d769612aab87294b341e2635621de538754006))
* **budget:** address codeql notes on the admission hold and the sysfs path ([c388c10](https://github.com/ooples/AiDotNet.Tensors/commit/c388c10d701c46043584c1580d0dca8073126a73))
* **budget:** bound the cuda pool backing memory its reserved counter does not report ([7f5294a](https://github.com/ooples/AiDotNet.Tensors/commit/7f5294accbbed600b95094faf3c96801ecf414ec))
* **budget:** charge stream-ordered CUDA allocations only for pool growth ([95fa0e1](https://github.com/ooples/AiDotNet.Tensors/commit/95fa0e1e089fa8b81910bccb0ddf6b200c51ba0f))
* **budget:** confine every linux thread, exact reserve hand-over, shared-context accounting ([2cc3632](https://github.com/ooples/AiDotNet.Tensors/commit/2cc3632710f71eaf294ded21a86a683898aea09a))
* **budget:** count cuda arena blocks and scratch memory, enforce during capture ([377af06](https://github.com/ooples/AiDotNet.Tensors/commit/377af0650e14f1226f09b5906345a40aa0a32cee))
* **budget:** count the cuda pool memory in use when the driver reports less backing memory ([9583a7f](https://github.com/ooples/AiDotNet.Tensors/commit/9583a7f2e3659362018623901621f18d6418e45a))
* **budget:** lock-free thread-count property and explicit thread filter ([9ff9b62](https://github.com/ooples/AiDotNet.Tensors/commit/9ff9b621351e7298b1550dae0e9ece4288462570))
* **budget:** make admission exact and keep a zero budget a budget ([103dc65](https://github.com/ooples/AiDotNet.Tensors/commit/103dc6520c947ce973de72b487bafc5054caeba7))
* captured gather and index-copy recompute traced integer indices; vulkan spiral ring order ([f02840e](https://github.com/ooples/AiDotNet.Tensors/commit/f02840e53999a8e37d4784fcc6fb1171d40fd9b0))
* CI failures and CodeQL findings on the DCGAN branch ([e465e54](https://github.com/ooples/AiDotNet.Tensors/commit/e465e5459d131519f0c4eb2c81557a099d9aacb1))
* close the re-review findings on releasing, required backends, seeds and coverage ([4e243a0](https://github.com/ooples/AiDotNet.Tensors/commit/4e243a015e3b457419787bffbe471a397bb0fc34))
* **compilation,vulkan:** second review round on the merged branch ([b54f8f1](https://github.com/ooples/AiDotNet.Tensors/commit/b54f8f1ef9c82a4c19896006c308f9555215cf75))
* **compilation:** a capture never bakes a lent address into its step I/O tables ([dcd8938](https://github.com/ooples/AiDotNet.Tensors/commit/dcd8938ace216a2ba4fcb4bb3594e5bb52acb87d))
* **compilation:** a checkpoint recompute draws the dropout mask its forward drew ([418e35f](https://github.com/ooples/AiDotNet.Tensors/commit/418e35fe40632b1d6570109ed6be4d228e147da3))
* **compilation:** a checkpoint recompute replays seeds on its own thread only ([a414c6f](https://github.com/ooples/AiDotNet.Tensors/commit/a414c6fc6d0891d2f570d5c5d1e472c6d4f80fd7))
* **compilation:** a failing stage of releasing a training plan no longer leaks the rest ([b484bc8](https://github.com/ooples/AiDotNet.Tensors/commit/b484bc8261b2d9e4c9bb7348a77f62d8ed46b279))
* **compilation:** a host boolean mask no longer stops a training step's graph capture ([012ce2b](https://github.com/ooples/AiDotNet.Tensors/commit/012ce2bcfb3c7f459b9888ebf4f4ac64e1a30826))
* **compilation:** a host write to device-resident in-place state is what the next step continues from ([694be85](https://github.com/ooples/AiDotNet.Tensors/commit/694be85673b5bf64889f415d3e2393ef22893221))
* **compilation:** a plan's capture preparation never records a borrowed address ([740380d](https://github.com/ooples/AiDotNet.Tensors/commit/740380da6c3f1b79f31ae20d2fa1bf8f3e378dab))
* **compilation:** check a split step's loss-scale overflow after the exchange ([af05560](https://github.com/ooples/AiDotNet.Tensors/commit/af05560f3d2042a501e4ed2acdd93657f529fb40))
* **compilation:** idle step memory never lends scratch another plan co-owns ([eab0c69](https://github.com/ooples/AiDotNet.Tensors/commit/eab0c6947d88549c746c7039a0baae5ee6932ab5))
* **compilation:** lease the engine for the in-place state upload; hold the tests to the expected backend ([10c35e1](https://github.com/ooples/AiDotNet.Tensors/commit/10c35e167dfea0cecbfaf234ab9d4d123e62068c))
* **compilation:** re-arm host download of device-resident in-place state after graph replay ([72e46b1](https://github.com/ooples/AiDotNet.Tensors/commit/72e46b1faa46056d04b741e24d60e3c0a7554576))
* **compilation:** re-arm the host download of device-resident in-place state after every step ([82ec30e](https://github.com/ooples/AiDotNet.Tensors/commit/82ec30efd6358df24774d034428db1706d0765e9))
* **compilation:** retained outputs work in the mixed-precision compiled plan ([b0b4a2d](https://github.com/ooples/AiDotNet.Tensors/commit/b0b4a2d2751d4f2ee59118b63d3d222b60ce8136))
* **compilation:** typed diagnostics switches, a per-thread mixed-precision trace switch ([3a3e27e](https://github.com/ooples/AiDotNet.Tensors/commit/3a3e27e154691c9c499e3cd0672d2c89e33f5119))
* **compiled-plan:** free the unowned device values a trace left on forward outputs ([26f5bfe](https://github.com/ooples/AiDotNet.Tensors/commit/26f5bfe3810764987d9b9e05e129ed34823c5c86))
* **conv:** keep the rgb stem input gradient on the small-output kernel at any worker count ([6b52d3c](https://github.com/ooples/AiDotNet.Tensors/commit/6b52d3cffac27fa995f5226e1d9b64b16eaa5144))
* **conv:** tune a convolution by its geometry, not its epilogue ([eaca9a2](https://github.com/ooples/AiDotNet.Tensors/commit/eaca9a291319cdbbc7e4565f821eeae133519bc7))
* **cpu:** 16-bit float gelu computed in fp32 and rounded once ([8f9d19a](https://github.com/ooples/AiDotNet.Tensors/commit/8f9d19a8503370626ea0e0e399ee68819fb9c9f2))
* **cpu:** confine the process on linux and fail loudly when confinement is refused ([cbfcd4c](https://github.com/ooples/AiDotNet.Tensors/commit/cbfcd4c25abc5724d04ff04b4729637687627ffc))
* **cpu:** double max and min propagate a nan on every framework ([b104134](https://github.com/ooples/AiDotNet.Tensors/commit/b1041347b153593c39feabcfd9f8ad8bf98f5b8b))
* **cpu:** intra-op workers run a thread's chunks under its fresh-allocation scope ([fb6d0a5](https://github.com/ooples/AiDotNet.Tensors/commit/fb6d0a595c00533baa395f78ff10068dcdff2b2a))
* **cpu:** the float32 max and min are nan when any element is ([4189ea8](https://github.com/ooples/AiDotNet.Tensors/commit/4189ea8940b1060bd104c9f8e06c9a1eb72bdefb))
* **cuda-graphs:** reuse memory a capture releases for its later allocations ([1504c92](https://github.com/ooples/AiDotNet.Tensors/commit/1504c92b92f9c10cb1364bb0893f0489ebf83ba1))
* **cuda,compilation:** review fixes for autocast, loss scaling, idle loans and the in-place fused CE ([b1903dc](https://github.com/ooples/AiDotNet.Tensors/commit/b1903dce2826cc8f944cf93214e59fb19d270ae3))
* **cuda:** bound default WDDM mempool threshold and drain cached pools in ReclaimUnderPressure ([dbe6223](https://github.com/ooples/AiDotNet.Tensors/commit/dbe622348aa065316128b38555b611cfcb323897))
* **cuda:** CublasComputeType.Float32Fast16F/FastBF16 values were swapped ([7de17c0](https://github.com/ooples/AiDotNet.Tensors/commit/7de17c01c50382ded7002b38c5ef6b104a4c97b2))
* **cuda:** keep the channels-last BatchNorm and epilogue kernels out of the user-string heap ([680a7f5](https://github.com/ooples/AiDotNet.Tensors/commit/680a7f58cf9f5ef68a794b0a8e6f2acc33bc2612))
* **cuda:** keep the cublas warmup handle until the backend is disposed ([ed4e671](https://github.com/ooples/AiDotNet.Tensors/commit/ed4e6718d19dbb27b3fe6c2cc9d7a99f51a28fd2))
* **cuda:** lend idle step memory past fused attention and readout saved state; return loans in stream order ([5191528](https://github.com/ooples/AiDotNet.Tensors/commit/51915286fc1656cb9cddf91f52a095afd3c3eee1))
* **cuda:** log the already-disposed per-thread cublas store instead of swallowing it ([6098e48](https://github.com/ooples/AiDotNet.Tensors/commit/6098e480bac1f4739c03dac9aedd192bf55b7dd1))
* **cuda:** matmul autocast runs where cuBLASLt cannot be loaded ([322128d](https://github.com/ooples/AiDotNet.Tensors/commit/322128de8819976a155079205c81b7066ffe8ede))
* **cuda:** resolve cuFFT 12 (CUDA 13) ([d05b8ed](https://github.com/ooples/AiDotNet.Tensors/commit/d05b8eda58bc4d28fea3b67edd82d571473e234b))
* **cuda:** review findings on the cast-operands autocast ([66270bc](https://github.com/ooples/AiDotNet.Tensors/commit/66270bc8138ec18f8b125ecca4c64550ef580b36))
* **cuda:** second review round on the cast-operands autocast ([e78ec00](https://github.com/ooples/AiDotNet.Tensors/commit/e78ec00049580b3b88ef8bdeab8f7fee1b2aa910))
* **cuda:** use RuntimeInformation.IsOSPlatform and explicit null guard for net471 compatibility ([c7853b5](https://github.com/ooples/AiDotNet.Tensors/commit/c7853b52e78a11338e474a1d31dde2d80a25478a))
* eighth review round ([fb963be](https://github.com/ooples/AiDotNet.Tensors/commit/fb963be95dbaf4ab11aa856513bf7a49e8df288f))
* **engines:** a required GPU backend is enforced on every route ([68f0648](https://github.com/ooples/AiDotNet.Tensors/commit/68f06480cf155625f2c3aa6e829c45966d41dc51))
* **engines:** bf16-streamed weights run on CPU and GPU; bound the packed-weight cache ([e64cd8e](https://github.com/ooples/AiDotNet.Tensors/commit/e64cd8e96e02f367f260863d3106701e8fb4ceb4))
* **engines:** find the engine that owns a wrapped backend ([de81fc6](https://github.com/ooples/AiDotNet.Tensors/commit/de81fc6f4bc06a288e1ec47920a5a855224103f4))
* **engines:** release a collected weight's bf16 copy at the next upload ([fb06773](https://github.com/ooples/AiDotNet.Tensors/commit/fb067730884c34a49fe91f58238803e6f4fa8875))
* every captured indexed op recomputes integer indices its trace produced ([3b537ee](https://github.com/ooples/AiDotNet.Tensors/commit/3b537eeb30f171eac22508befb3d43cba14708ab))
* fifth review round ([94bed01](https://github.com/ooples/AiDotNet.Tensors/commit/94bed0135c95f3f79cc46fe5a0fd7d69f5d82feb))
* fourth review round ([966a2c3](https://github.com/ooples/AiDotNet.Tensors/commit/966a2c3cf16bb65ea2e2f536ab1a31e8885df48a))
* **fused-ce:** 16M-element row-chunk blocks for the chunked fused cross-entropy ([42e6879](https://github.com/ooples/AiDotNet.Tensors/commit/42e6879a133eae4efc9d624d06fa08a191276879))
* **fused-ce:** subtract the one-hot without a scratch sized to the chunk ([ed7acaa](https://github.com/ooples/AiDotNet.Tensors/commit/ed7acaa39637bb06d82e27e668d28750b9d1c27e))
* GPU dropout, AMD OpenCL, WDDM pool threshold and sgemm pack sizing (combines [#1826](https://github.com/ooples/AiDotNet.Tensors/issues/1826) [#1827](https://github.com/ooples/AiDotNet.Tensors/issues/1827) [#1829](https://github.com/ooples/AiDotNet.Tensors/issues/1829) [#1830](https://github.com/ooples/AiDotNet.Tensors/issues/1830)) ([6b0a029](https://github.com/ooples/AiDotNet.Tensors/commit/6b0a02983a7340f058e63842487b9e2d18782040))
* gpu matrix and strided vector ops no longer compute doubles in float ([b9ae264](https://github.com/ooples/AiDotNet.Tensors/commit/b9ae2645bedb851269447e2c26d36b88f5768845))
* gpu matrix and strided vector ops no longer compute doubles in float ([930c3e9](https://github.com/ooples/AiDotNet.Tensors/commit/930c3e9859fce856473c85a21ff04479dfde8db1))
* **gpu:** a CUDA backend that cannot be created is never a silent fallback ([a96ac09](https://github.com/ooples/AiDotNet.Tensors/commit/a96ac097763d2564b6303e0252dbfc521ac18c64))
* **gpu:** a failed idle-loan move frees its fresh memory, and a failing lender no longer breaks the others ([20969c0](https://github.com/ooples/AiDotNet.Tensors/commit/20969c0a8ddcdf737c9500cb41a62080a1639da9))
* **gpu:** a PollThenBlock wait still lets an engine reset free device memory ([c783c7a](https://github.com/ooples/AiDotNet.Tensors/commit/c783c7a3c14094baf9be508c03f610b75629d0b8))
* **gpu:** a prefix narrow view no longer poisons its source's device buffer ([0f99ae3](https://github.com/ooples/AiDotNet.Tensors/commit/0f99ae32d7bf52cc051ecf632333dce7c145cabf))
* **gpu:** a single-element max or min returns the canonical nan ([952adde](https://github.com/ooples/AiDotNet.Tensors/commit/952addef48a6d2bf398951c5aa1d476427750a3c))
* **gpu:** a view input operand no longer fails destination-aware ops ([13eac6b](https://github.com/ooples/AiDotNet.Tensors/commit/13eac6b6f67ca5ddd0cd06f47ab542efcdb04987))
* **gpu:** avoid two AMD Vulkan compiler miscompiles ([5c7441b](https://github.com/ooples/AiDotNet.Tensors/commit/5c7441b8ed8ce67446cf1f1c6cb7bcd1ee6e3f5a))
* **gpu:** bit-exact narrow float matmul and gelu tails, no-op self copies, metal library gap ([0e2312a](https://github.com/ooples/AiDotNet.Tensors/commit/0e2312aa4d913381e394d7c5f1026eb6b0a36032))
* **gpu:** bound typed copies and check metal launches ([472262d](https://github.com/ooples/AiDotNet.Tensors/commit/472262dc2a7e016b36e568ca783af089afaf1de8))
* **gpu:** capture the BF16 DCGAN step: device zeros for the absorbed conv-bias gradient ([72ee950](https://github.com/ooples/AiDotNet.Tensors/commit/72ee950226da55b245bec137221ac485fb618e8c))
* **gpu:** compiled steps under autocast keep elementwise activations on the device ([74ea9d5](https://github.com/ooples/AiDotNet.Tensors/commit/74ea9d58778471bbe1e61c49d9052a7aafe71951))
* **gpu:** copy overlapping ranges of one buffer through scratch, test copies of every type ([06f4260](https://github.com/ooples/AiDotNet.Tensors/commit/06f4260ca210dc6b96032d8ca4e25dd733c23487))
* **gpu:** decide subnormal operands and results of the 16-bit float functions on the bits ([0e14257](https://github.com/ooples/AiDotNet.Tensors/commit/0e14257049d77efe5bfccdedde146a8678741a45))
* **gpu:** disposing the engine frees its bf16 weight device copies ([d5420b6](https://github.com/ooples/AiDotNet.Tensors/commit/d5420b60df8660b4e7eb3b1d1521896b566cc6d4))
* **gpu:** dropout masks are independent across elements and seeds ([f918abb](https://github.com/ooples/AiDotNet.Tensors/commit/f918abbbe5e3cfc2a0b1cfbe196619cbfae03020))
* **gpu:** every GEMM route honors the matmul autocast scope ([77de205](https://github.com/ooples/AiDotNet.Tensors/commit/77de205f8f79bb9da56915293fbde4dcd8b81e28))
* **gpu:** float64 review follow-ups: correctly rounded vulkan sqrt, sigmoid tail, metal aliasing and compile gaps ([a33ce11](https://github.com/ooples/AiDotNet.Tensors/commit/a33ce118f13513c384482b9cd73de6bdcfc9d6e1))
* **gpu:** float64 review follow-ups: vulkan denormals, metal copies and kernel gaps, same-buffer copies ([553652f](https://github.com/ooples/AiDotNet.Tensors/commit/553652fe44a21e42432f41cec94e1d7ebabb8f62))
* **gpu:** gradient staging owns its page-locked memory, and its failures are visible ([59c3378](https://github.com/ooples/AiDotNet.Tensors/commit/59c337859ea7c9247fc91ff25b219eab387b110e))
* **gpu:** harden the integer kernels and restore the half gemm capability ([5057b97](https://github.com/ooples/AiDotNet.Tensors/commit/5057b9739e9d4191624061a3bca0c77bd1a2c0b9))
* **gpu:** keep a compiled plan's input binding when an eager op reads it ([8d65bbc](https://github.com/ooples/AiDotNet.Tensors/commit/8d65bbc7a4998478a32804df7c4d3f74c4c05a31))
* **gpu:** keep a view of fp16-resident data off the float permute on a shared backend ([d83891d](https://github.com/ooples/AiDotNet.Tensors/commit/d83891da25bfa936c43872339f307481d298c3f7))
* **gpu:** make the float64 planner and vulkan gate follow what is measured ([f87329a](https://github.com/ooples/AiDotNet.Tensors/commit/f87329a9b204d4b61a0acdac3c8ad5ead7f41787))
* **gpu:** matrix multiply output guard covers shared device memory, sum bound cannot overflow ([526ec92](https://github.com/ooples/AiDotNet.Tensors/commit/526ec925d9ac95f15fe9d8a3cebf4892a47f6766))
* **gpu:** measure vulkan double denormals when the mode is not offered ([4e60cd6](https://github.com/ooples/AiDotNet.Tensors/commit/4e60cd6ce9198c5d0afbefbb876a2711ffec50a9))
* **gpu:** nan results keep the cpu bits and cuda row max leaves the ptx fold ([95a9921](https://github.com/ooples/AiDotNet.Tensors/commit/95a992120564b763c5cc645d2ce06e25c31a3483))
* **gpu:** page-locked staging and pool blocks are released only in their live context lifetime ([aa82176](https://github.com/ooples/AiDotNet.Tensors/commit/aa82176b715915e674355cc5e97e600d4d158955))
* **gpu:** run float16 add, relu and gelu on the kernels that round like the host ([476349e](https://github.com/ooples/AiDotNet.Tensors/commit/476349ebd360b5e5212c0e23253f930eaf8858e6))
* **gpu:** the device max and min are nan when any element is, and webgpu reduces past 65536 ([461e6df](https://github.com/ooples/AiDotNet.Tensors/commit/461e6df46ad2c8e47e217775ac9c6228f1a5938e))
* **gpu:** the typed host-wait policy wins over the deprecated environment variable ([93204fe](https://github.com/ooples/AiDotNet.Tensors/commit/93204feb299aff76121e1c1e6bee7a3ed231060e))
* **gpu:** webgpu integer kernels follow the synchronous route's availability ([85f263e](https://github.com/ooples/AiDotNet.Tensors/commit/85f263e94bab5411955eda562c1015ebdb2370df))
* **gpu:** webgpu snapshot of a double buffer bound twice, sync surface off in browsers ([20c09b6](https://github.com/ooples/AiDotNet.Tensors/commit/20c09b63a87aecb28e147b0708824fdbe2ca9fae))
* indices that are another op auxiliary output are left to that op to refresh ([26b3d57](https://github.com/ooples/AiDotNet.Tensors/commit/26b3d571a0324ae43ef7665d5a97ea40979860cc))
* **linalg:** francis double-shift qr for eig, replacing unconverged unshifted qr ([a4721f8](https://github.com/ooples/AiDotNet.Tensors/commit/a4721f84bab20ab55848e59b09bcb89ede249732))
* **linalg:** francis double-shift qr for eig, replacing unconverged unshifted qr ([67bc324](https://github.com/ooples/AiDotNet.Tensors/commit/67bc3247c9f8524e3fe28b2db77f98cb3c2a451b))
* **linalg:** o(n^2) schur back-substitution for decimal eig; half/bf16 via double ([41bc2bf](https://github.com/ooples/AiDotNet.Tensors/commit/41bc2bf8636f8c7d90beb12ea01a7b20dab0baea))
* max and min propagate nan on the cpu and every gpu backend ([f651c0e](https://github.com/ooples/AiDotNet.Tensors/commit/f651c0e0d560002515965e0fa6e02dbc52a43ac4))
* **metal:** compile the float64 libraries in the shader catalog test ([19a66b7](https://github.com/ooples/AiDotNet.Tensors/commit/19a66b7dc1fe8b047c631bcac83e1a170fa1850a))
* **metal:** end the compute encoder when a launch throws before commit ([b5b74f2](https://github.com/ooples/AiDotNet.Tensors/commit/b5b74f2d82bb1de9add4ad10452b3d968696e8f3))
* **onnx,broadcast:** export the broadcast tile as Expand; keep the zero-add for an empty target ([d1a2ca5](https://github.com/ooples/AiDotNet.Tensors/commit/d1a2ca539fb4ee3a2bb827872ca8a3abc14c4378))
* **onnx,tests:** review findings on the broadcast-tile export and the probe and autocast tests; coverage on any device ([bf9a7cf](https://github.com/ooples/AiDotNet.Tensors/commit/bf9a7cf2fcc6789dd395e7bf6876140e6ac51d4f))
* **opencl:** compute 16-bit floats on the bits ([5025a3d](https://github.com/ooples/AiDotNet.Tensors/commit/5025a3dcdfe426d23b7f3a24b0dfc76d4896acb0))
* **opencl:** decline the fused Winograd forward where its local memory does not fit ([dc622c2](https://github.com/ooples/AiDotNet.Tensors/commit/dc622c23a5a0288c30455f8becf70705dd033936))
* **opencl:** keep subnormals in the bias add and leaky ReLU, fused and separate ([ef14377](https://github.com/ooples/AiDotNet.Tensors/commit/ef143771cef4ae7c70f64d3dc0b69f5b471c735e))
* **opencl:** never bind a kernel's output as its unused bias argument ([18cb2f7](https://github.com/ooples/AiDotNet.Tensors/commit/18cb2f7d3092838bf0bf7f3464b0324787e46c46))
* **opencl:** wrap int.MinValue floor-divided by -1 ([cf8d8ce](https://github.com/ooples/AiDotNet.Tensors/commit/cf8d8ce68a2fb598e5c13a9a47ff71a18e790ede))
* report or refuse the remaining silent fallbacks, free what was leaked ([2e927be](https://github.com/ooples/AiDotNet.Tensors/commit/2e927be455377595a966ddb7441116a18851a8b5))
* review findings on the combined GPU/GEMM fixes ([f3d9756](https://github.com/ooples/AiDotNet.Tensors/commit/f3d97561828b387272363cf4311adb44589dd0b1))
* review findings: co-own excludes broker plans, retire the conv ones vector, oneDNN resolver via the registry ([846cac9](https://github.com/ooples/AiDotNet.Tensors/commit/846cac98b93e0742b83afbc04a912328b9083d2a))
* second review round, the tests the first round lacked, and the remaining page-locked frees ([a075644](https://github.com/ooples/AiDotNet.Tensors/commit/a07564477654f9af83bc90d668cedfe3bfe57dc2))
* seventh review round ([64c9c82](https://github.com/ooples/AiDotNet.Tensors/commit/64c9c82991556c6b9c92addef1eecfa8abeb1487))
* sixth review round ([24969d5](https://github.com/ooples/AiDotNet.Tensors/commit/24969d596ac1fa175f04380b2d6a437de5cd94fb))
* speed-first strided scatter leaves untouched doubles exact ([6c2f41d](https://github.com/ooples/AiDotNet.Tensors/commit/6c2f41d4e9be4838756f3137ce16fec3729d3a93))
* **tests:** unique kernel locals for the arity audit; FFT WGSL fragment; RFFT closure off the first-order path ([d26e301](https://github.com/ooples/AiDotNet.Tensors/commit/d26e3018ebc57c777bf7c827125efec536424550))
* third review round, and direct tests for the second round's fixes ([230ddf1](https://github.com/ooples/AiDotNet.Tensors/commit/230ddf1a5e062e4c439a7db2d23efb635eaa4816))
* uploading a paged-out bf16 weight makes no resident host copy ([e0ce34a](https://github.com/ooples/AiDotNet.Tensors/commit/e0ce34a8f86989f85f17cf44580120717150a65d))
* **vulkan,tests:** the double-support probe is detection evidence; autocast conv cases need a 16-bit route ([6ccf8e4](https://github.com/ooples/AiDotNet.Tensors/commit/6ccf8e4cbb0dbffe5e60856885f96ebf33aa0a17))
* **vulkan:** attribute the float64 capability probe to detection ([fb677b7](https://github.com/ooples/AiDotNet.Tensors/commit/fb677b77a804aefd2889459c724e54a62000060e))
* **vulkan:** bound the device memory held by buffers released behind queued submissions ([fb4872c](https://github.com/ooples/AiDotNet.Tensors/commit/fb4872c3c548d8300f8fa35e7f0c1f8e26ae1b17))
* **vulkan:** bound the device memory held by buffers released behind queued submissions ([be8e771](https://github.com/ooples/AiDotNet.Tensors/commit/be8e771772e77a90cdfbf3161249bceb5d91c525))
* **vulkan:** elu backward takes the negative branch at zero, as every other engine does ([f36373b](https://github.com/ooples/AiDotNet.Tensors/commit/f36373bb56048c990e2b51e94901c48b4a50e8d4))
* **vulkan:** fill buffers past the 65535-workgroup minimum ([d660715](https://github.com/ooples/AiDotNet.Tensors/commit/d6607156e5165b45eda0f7f951956d09c4cd8d86))
* **vulkan:** honour CUDA_VISIBLE_DEVICES when picking the Vulkan device ([c8c9430](https://github.com/ooples/AiDotNet.Tensors/commit/c8c9430e8bd448a453d5dcce89c2178a6b8ee8da))
* **vulkan:** the spiral angle takes the sign of a zero cross product from its operands ([b18c413](https://github.com/ooples/AiDotNet.Tensors/commit/b18c413e8fb7ca299698221585c2957e0891b2af))
* **vulkan:** the spiral ring order pins the angle of a neighbour opposite the reference ([b07be45](https://github.com/ooples/AiDotNet.Tensors/commit/b07be45c91c1b42110ec9e7b5d43751df1e21247))
* **webgpu:** accurate log2/log10 and fused-GELU tanh, counted launches, handle-free aliasing; Dawn harness transfers ([0ba7812](https://github.com/ooples/AiDotNet.Tensors/commit/0ba78123bdf9fd1f690488c38e54d3e2045b9feb))
* **webgpu:** anomaly mode and oversized products take the cpu, async download checks the layout ([86b2848](https://github.com/ooples/AiDotNet.Tensors/commit/86b28487ca418164d4a20a27a9f986fcd39abbb8))
* **webgpu:** build UniquePositionAdd only where the WebGPU backend exists ([cbf2321](https://github.com/ooples/AiDotNet.Tensors/commit/cbf2321706540b335acd381b33c3a0c5b3c2e941))
* **webgpu:** dawn suites green on windows and mesa, with a lavapipe ci lane ([9d5b19c](https://github.com/ooples/AiDotNet.Tensors/commit/9d5b19c9c4e45970c8786e42992fc62aa5dbca9a))
* **webgpu:** exact double-float arithmetic and nan-safe zero tests under mesa ([5055bfb](https://github.com/ooples/AiDotNet.Tensors/commit/5055bfbf19a82eeb4f5688d8b723d251c0915a10))
* **webgpu:** keep the two-sum residual operands from reassociating ([f1da0e2](https://github.com/ooples/AiDotNet.Tensors/commit/f1da0e268f2669dc3dc8dd021e64efdfdc395bd0))
* **webgpu:** refuse unfoldable grids past the limit and apply the buffer cap policy ([fbc1615](https://github.com/ooples/AiDotNet.Tensors/commit/fbc1615a4765b60888c152da8fd0fa4f809196b5))
* **webgpu:** release the backend on every path that hands it to no engine ([89406ba](https://github.com/ooples/AiDotNet.Tensors/commit/89406bac955e3a68f49dc4e1871280d68aaed77a))


### Performance

* **compilation:** a gradient exchange moves the gradients through one contiguous device buffer ([6bd2517](https://github.com/ooples/AiDotNet.Tensors/commit/6bd2517ab97f36a678b3bbbc268b488046586b06))
* **compilation:** channels-last convolution stacks behind IChannelsLastBackend ([e2f9ea4](https://github.com/ooples/AiDotNet.Tensors/commit/e2f9ea4b934edb1048be9c0742dbf216719fc2c6))
* **compilation:** compiled-plan memory, in-plan mixed precision and the DDP exchange point ([9ba8db2](https://github.com/ooples/AiDotNet.Tensors/commit/9ba8db251b75bfc5ba537dc31913f343abbaff1f))
* **compilation:** fold a convolution's bias into the training BatchNorm that follows it ([a9ef6d5](https://github.com/ooples/AiDotNet.Tensors/commit/a9ef6d5b5dc996d418ee8e666e92446e1017a4d7))
* **compilation:** fuse training BatchNorm + LeakyReLU into one kernel each way ([d618942](https://github.com/ooples/AiDotNet.Tensors/commit/d6189428619f78fb10162c3fca9e2ebad4920cbb))
* **compilation:** run a convolution's bias, inference BatchNorm and activation as one pass ([30daf07](https://github.com/ooples/AiDotNet.Tensors/commit/30daf079e72999aae6378e0782ae53a23fa57500))
* **compilation:** the gradient exchange keeps its host staging across steps ([a4555c1](https://github.com/ooples/AiDotNet.Tensors/commit/a4555c1fe7293de42179242a767f20bb907b5bf3))
* **compiled-plan:** idle step memory is found only when an allocation between steps asks for it ([ef4938e](https://github.com/ooples/AiDotNet.Tensors/commit/ef4938e2abfd04d432e2895a822f2d0d2663e6ad))
* **compile:** fused conv/bias backward skips a frozen kernel's weight gradient ([f53d48e](https://github.com/ooples/AiDotNet.Tensors/commit/f53d48e65bb615c6c66f77adb9bf01da3ae13d15))
* **conv:** fuse cuDNN bias/activation into a channels-last epilogue; fuse ConvTranspose2D bias ([6f32a46](https://github.com/ooples/AiDotNet.Tensors/commit/6f32a467a8b23da45a7eaddb3dd767b18c181d01))
* **cpu:** chunked small-output weight gradient; stem input gradient as the per-image transposed conv ([7bcce16](https://github.com/ooples/AiDotNet.Tensors/commit/7bcce16fb9fc3762f757c629a1fd51f29c16d731))
* **cpu:** compiled plans of one group co-own their step-transient host arrays ([5b92d7b](https://github.com/ooples/AiDotNet.Tensors/commit/5b92d7b0c4fe06b75fcb71e4d9f551a4cf86d67b))
* **cpu:** per-image GEMM routes for short-K conv forward and transposed-conv forward ([cf1911a](https://github.com/ooples/AiDotNet.Tensors/commit/cf1911a2f68b07fe61010a546201441bd267c2bf))
* **cpu:** single-pass pack for a partial channel block ([435c4e2](https://github.com/ooples/AiDotNet.Tensors/commit/435c4e241c2500be1ce7e775ad57a4d11ba07dcd))
* **cpu:** vectorized, chunk-parallel compiled LeakyReLU backward; rent conv weight-gradient result ([62cf597](https://github.com/ooples/AiDotNet.Tensors/commit/62cf597cc98ab6c08cbcf2e72c184815977b9e57))
* **cuda:** a pooled buffer serves an allocation between steps before idle step memory ([55e2eeb](https://github.com/ooples/AiDotNet.Tensors/commit/55e2eebaa6e85915ef34b9b3667694d90274ab29))
* **cuda:** free an outgrown cuDNN workspace that no capture can name ([085f061](https://github.com/ooples/AiDotNet.Tensors/commit/085f061a68c6f070a65bbdc46892c08dd89153f0))
* **cuda:** measured route per autocast GEMM (cuBLASLt, classic GemmEx, fixed-order split-K) ([380d1a9](https://github.com/ooples/AiDotNet.Tensors/commit/380d1a951f5569c671cc0202fdf7eb26ee6393c0))
* **cuda:** separate unreachable garbage reclamation from reusable pool eviction in ReclaimUnderPressure ([93d36bb](https://github.com/ooples/AiDotNet.Tensors/commit/93d36bb362cded1f76910da9dc0c817ff300921a))
* **fused-ce:** one in-place [chunk, V] block per direction at the 672-row chunks ([a3d4084](https://github.com/ooples/AiDotNet.Tensors/commit/a3d4084b80591dfd8609728c25c87ce51e878d2c))
* **gemm:** size sgemmtiled packing buffers to the problem ([e7e3c2a](https://github.com/ooples/AiDotNet.Tensors/commit/e7e3c2a9c381ea102eefc9e57b7655f710d3aa6e))
* **gpu:** a convolution's backward narrows dY once for both of its passes ([e88601c](https://github.com/ooples/AiDotNet.Tensors/commit/e88601c497d8e41174490156819fb4f33060b7c5))
* **gpu:** a gradient exchange can move the step's gradients through one page-locked batch ([15e61d0](https://github.com/ooples/AiDotNet.Tensors/commit/15e61d051c59009346c118c52b8586eb6bbff30d))
* **gpu:** captured plans co-own step scratch; size the cuDNN workspace exactly ([9b328bc](https://github.com/ooples/AiDotNet.Tensors/commit/9b328bc93e3f0944044d232edc796a414faca354))
* **gpu:** co-own step scratch under mixed precision and channels-last fusion ([be93df4](https://github.com/ooples/AiDotNet.Tensors/commit/be93df49801d34194ba71dd933d905900a43bccb))
* **gpu:** DCGAN training step: channels-last fused convolutions, BF16 resident capture, shared step scratch ([2f011ee](https://github.com/ooples/AiDotNet.Tensors/commit/2f011eeb5ec9696450d660bbf616e02eb1e19b65))
* **gpu:** keep BF16 autocast steps resident; WeightNorm workspace for fused ConvTranspose ([3f81ba8](https://github.com/ooples/AiDotNet.Tensors/commit/3f81ba877071d3c71d3ce90c8896e602e2226d10))
* **gpu:** keep BF16 in-place, normalization, index and reduction ops resident; capture test ([ddf3dfc](https://github.com/ooples/AiDotNet.Tensors/commit/ddf3dfc1566544825f83ee185c468ad3fbc4e431))
* **gpu:** one growable transient scratch slot per local index ([34c5569](https://github.com/ooples/AiDotNet.Tensors/commit/34c5569556e30e34a03b770a99b3eef410f8600d))
* **gpu:** stage autocast convolutions channels-last; BF16 transposed convolutions ([e5c0246](https://github.com/ooples/AiDotNet.Tensors/commit/e5c02460862e7c801be8da2b643ffda01d3a119b))
* LogSumExp and TakeAlongDim backward stay on the device under a tape ([7bcbaa8](https://github.com/ooples/AiDotNet.Tensors/commit/7bcbaa8909381302cfbb4445574649397236988a))
* LogSumExp stays on the device under a tape ([5c6b612](https://github.com/ooples/AiDotNet.Tensors/commit/5c6b612167545d7319ead955f6e0b488d5eead7c))
* TakeAlongDim backward on the device ([1f286d9](https://github.com/ooples/AiDotNet.Tensors/commit/1f286d91eb69cf3e6d5ad8c3b00d2676fef1bc0e))
* the GLU family stays on the device under a tape; GQA keys get their gradient ([c5a95dd](https://github.com/ooples/AiDotNet.Tensors/commit/c5a95dd3e871b09584e71f28ab69e915886d8bd9))


### Refactoring

* **compilation:** no null-forgiving operators or silent catches in the plan memory work ([0298a81](https://github.com/ooples/AiDotNet.Tensors/commit/0298a81d469188c77391ed7562f778f964bc1e08))
* **compilation:** upload in-place state without a filtering type test ([e55825c](https://github.com/ooples/AiDotNet.Tensors/commit/e55825c38264543f411a9b21c5d3d2f4fec302d3))
* **compile:** remove the environment-gated FP16 heterogeneous training path ([6c4187c](https://github.com/ooples/AiDotNet.Tensors/commit/6c4187c1871f6325b7f2b13ace9702c1dcff9cb2))
* filter the closure fields with where before scanning them for index tensors ([56ef1f0](https://github.com/ooples/AiDotNet.Tensors/commit/56ef1f0adbf8905200a1330c31f59326b957ebbc))
* **gpu:** allocate the engine's half buffers as typed float16 ([f9eac6a](https://github.com/ooples/AiDotNet.Tensors/commit/f9eac6a49b1d9da3be336e362a5da7d8140d111a))
* **gpu:** filter the function kernels explicitly ([4beeaf4](https://github.com/ooples/AiDotNet.Tensors/commit/4beeaf4c9409b89b44a070ec012e2d359534cdf1))
* **gpu:** filter the kernel generator's loops explicitly, drop a redundant type test ([c3fab14](https://github.com/ooples/AiDotNet.Tensors/commit/c3fab1407be436b678b5649d1f2495b00392a3a8))
* **gpu:** key the engine's resident registry by element type ([50dc558](https://github.com/ooples/AiDotNet.Tensors/commit/50dc558b05c4969d093a7eb4acd1be8937b6f9c8))
* **webgpu:** each reduction pass owns its partials with a using ([353b304](https://github.com/ooples/AiDotNet.Tensors/commit/353b30453c664f344e4f6e2fa4fb9b875a450bd0))


### Build System

* **gpu:** record the two removed fp16 side interfaces for package validation ([f197010](https://github.com/ooples/AiDotNet.Tensors/commit/f197010ea3d145427c48cd6d11cff638a47bb78c))


### Documentation

* **vulkan:** describe the narrow types the backend now holds ([99e49dd](https://github.com/ooples/AiDotNet.Tensors/commit/99e49dd3fadd0393d9e11edcc0a649fcc20396c0))

## [0.152.1](https://github.com/ooples/AiDotNet.Tensors/compare/v0.152.0...v0.152.1) (2026-11-22)


### Bug Fixes

* address code scanning findings in kv cache append and cuda library discovery ([97be5d8](https://github.com/ooples/AiDotNet.Tensors/commit/97be5d80fd0a5c9d5dbb3640c0d084083790651b))
* **autotune:** restore the 2 s default online tuning budget ([313c13a](https://github.com/ooples/AiDotNet.Tensors/commit/313c13a47f625054740c414b5ca298de59541930))
* certify only the writes an optimized graph still performs ([f776fc7](https://github.com/ooples/AiDotNet.Tensors/commit/f776fc78f87caef0f910f02c5a652cfc0a207eaa))
* **compile:** advance every in-place target once at capture and recapture after a reset ([6764d67](https://github.com/ooples/AiDotNet.Tensors/commit/6764d6775b6380105b10d56b8090c40712ee6bff))
* **compile:** leave the closure untouched when TryRecordCompiledOp declines ([7ef48c4](https://github.com/ooples/AiDotNet.Tensors/commit/7ef48c4ccd32bb36f8bb5b0f1ac09e5c6c7f8e6c))
* **compile:** restore decode-op state on abandon; snapshot it only when the trace runs the op ([53bbf49](https://github.com/ooples/AiDotNet.Tensors/commit/53bbf49cacf4bcf17319d889208852efcc8e5691))
* **cuda-tuning:** drop held timing graphs when workspaces or nested decisions change ([01688ba](https://github.com/ooples/AiDotNet.Tensors/commit/01688babc9cdd02cd25e1c17ac798714e85be5ce))
* **cuda:** check buffer capacity and 64-bit indexing in the decode-step and copy kernels ([20b4284](https://github.com/ooples/AiDotNet.Tensors/commit/20b4284abfc37be0da11f17280f988a21b707bc3))
* **cuda:** guard conv scratch replacement, offload context liveness and cublas paths ([d7e9852](https://github.com/ooples/AiDotNet.Tensors/commit/d7e9852fe546aa564828f9520963b153e3ed52bd))
* **cuda:** refuse to free captured clip partials; correct the GDN fallback summary ([0f43618](https://github.com/ooples/AiDotNet.Tensors/commit/0f43618ee05fb2b8c14953579005dd2d2f3cef8c))
* drop a view's lent gpu buffer once its storage gives the buffer up ([fe4a135](https://github.com/ooples/AiDotNet.Tensors/commit/fe4a1354ea7f73f632f260fe94a1493884868dbb))
* **engines:** validate GatedDeltaNetStep operand shapes, not just lengths ([2cfa182](https://github.com/ooples/AiDotNet.Tensors/commit/2cfa182a734dc778637020bf00378b9d55298b0b))
* **gpu:** [#1784](https://github.com/ooples/AiDotNet.Tensors/issues/1784) review follow-ups, cross-backend tensor policy and webgpu transcendental accuracy ([a9241b8](https://github.com/ooples/AiDotNet.Tensors/commit/a9241b808102bbf61373d0b005bcd0cf3bc8019f))
* **gpu:** close cross-backend migration gaps from review ([1c03119](https://github.com/ooples/AiDotNet.Tensors/commit/1c031191d07d1b2012e94d3100d4faf359961db2))
* **gpu:** close the cross-backend review gaps in migration, policy and plan state ([a7a9fab](https://github.com/ooples/AiDotNet.Tensors/commit/a7a9fab64180566cf1e8859f605dd4119ec5f69f))
* **gpu:** compact the device result registry after a garbage collection ([f1fe44e](https://github.com/ooples/AiDotNet.Tensors/commit/f1fe44e9a32ef367c0c4bb69617b5483a2b85283))
* **gpu:** cross-backend review fixes: views, stale device copies, teardown, launch counts ([8f5b280](https://github.com/ooples/AiDotNet.Tensors/commit/8f5b2803aef3ee2c1984859336286b8eb1b9490c))
* **gpu:** cross-backend review: wrappers, streams, inputs, views, deferred reads, leases ([e0345ad](https://github.com/ooples/AiDotNet.Tensors/commit/e0345ad1a8021daf9d256a63e8d59e36c1ae1ea0))
* **gpu:** dispose a popped float64 pool buffer on every path that does not return it ([6c8cab6](https://github.com/ooples/AiDotNet.Tensors/commit/6c8cab6ec015924bf3eece8d9263b04746541f1d))
* **gpu:** fall back to cpu for double pooling under the type-preserving policy ([292ec21](https://github.com/ooples/AiDotNet.Tensors/commit/292ec21d6a6e5a8d55b8040c47abb05af94eab65))
* **gpu:** give buffer views and offload wrappers their parent's element type ([e5a7aba](https://github.com/ooples/AiDotNet.Tensors/commit/e5a7ababd2bdfbcae90b0905adcfe052af1eb87b))
* **gpu:** GPU detection's probe readback is not the first op's; cuBLAS prewarm peeks Current ([885034f](https://github.com/ooples/AiDotNet.Tensors/commit/885034f4e17738737e14676f25356863502db9cc))
* **gpu:** harden double residency eviction, invalidation and fallbacks ([d01af73](https://github.com/ooples/AiDotNet.Tensors/commit/d01af73704bd4632f8a1ff55527aefd4f7fff468))
* **gpu:** keep a failing trace listener from escaping the nvrtc and hip probes ([699fca8](https://github.com/ooples/AiDotNet.Tensors/commit/699fca8de32085f0b3ec41de0c57abb0a9d7b095))
* **gpu:** keep declared write order for a state next that views another state ([a45c5eb](https://github.com/ooples/AiDotNet.Tensors/commit/a45c5eb8ba29e60655eee49a7f4f4892e28c44a7))
* **gpu:** keep nvrtc and hip availability probes from caching discovery and logging failures ([b1517f8](https://github.com/ooples/AiDotNet.Tensors/commit/b1517f8af9453cdd7fad18c6dfbbc18082ad6ea6))
* **gpu:** never compute a double op through a float kernel under the type-preserving policy ([b23afbf](https://github.com/ooples/AiDotNet.Tensors/commit/b23afbf9a05100b5c7f2a767c1758798cd4cf6f0))
* **gpu:** never hand a kernel another backend's buffer; CrossBackendPolicy ([0ba2616](https://github.com/ooples/AiDotNet.Tensors/commit/0ba26164e72cd602c732de9377967030be86ce42))
* **gpu:** probe CUDA libraries name-first and rank toolkit versions numerically ([c684508](https://github.com/ooples/AiDotNet.Tensors/commit/c684508fbda2e3624ba143a5a2ef283feea8e60d))
* **gpu:** rehome cross-backend tensors resident; share one CUDA context ([5f6dcbe](https://github.com/ooples/AiDotNet.Tensors/commit/5f6dcbed134ff5f5729648d2229b77b0d68cc3bc))
* **gpu:** reject duplicate IndexCopy destinations on first use, without a per-replay set ([926eb31](https://github.com/ooples/AiDotNet.Tensors/commit/926eb313d9fe9a4a628cc1ec1ea9d8095bab5dc6))
* **gpu:** scope-aware rollback, wgsl typing and unexecuted-result tracking from review ([2f946b8](https://github.com/ooples/AiDotNet.Tensors/commit/2f946b86ebcb5661cfd35b863f884c90c929e83a))
* keep shadowed wgsl builtins and mark compiled graph writes executed ([31523ba](https://github.com/ooples/AiDotNet.Tensors/commit/31523bad56c81ffd3b8532306c706e7bb34a412c))
* keep trace state snapshots per target and accept a row gate lead ending in one ([5caad89](https://github.com/ooples/AiDotNet.Tensors/commit/5caad89124b37d1a4231a13652903aa4df8ae5bf))
* **kv-cache:** validate the append position on every backend and surface device faults ([d9a2355](https://github.com/ooples/AiDotNet.Tensors/commit/d9a23553634385e76b1e30f1aca85c6e65571254))
* make causal decomposed attention capturable in inference graphs ([e564454](https://github.com/ooples/AiDotNet.Tensors/commit/e564454a0db38196f874c4b5ef4d5cee17e35286))
* make causal decomposed attention capturable in inference graphs ([88de2fb](https://github.com/ooples/AiDotNet.Tensors/commit/88de2fb48d458bb215cd4becf2056a16674f512b))
* report an oversized kronecker product by name instead of an overflow ([7f35ab4](https://github.com/ooples/AiDotNet.Tensors/commit/7f35ab475fa47c6185ecf363a3ce403b009681b3))
* state the foreign-buffer preconditions as patterns instead of null-forgiving ([21e416e](https://github.com/ooples/AiDotNet.Tensors/commit/21e416e58b504b6c8812fe6862ed0fc469da959f))
* **tests,gpu:** suite failures on main: WebGPU fused-epilogue geometry, launch counts, WGSL fragment ([0197d62](https://github.com/ooples/AiDotNet.Tensors/commit/0197d623b8e4a68922dcf6bab650f3005c7daa1e))
* **topk:** rank NaN above every number on the CPU, as the GPU paths do; test NaN and -0 ([ae7db00](https://github.com/ooples/AiDotNet.Tensors/commit/ae7db0040ef8a1d89d33008a7e60279432fa2cbd))
* **webgpu:** accurate tanh, log and log2 in every f32 WGSL module ([ff466f0](https://github.com/ooples/AiDotNet.Tensors/commit/ff466f0cff9ec116b263c437b8f5d3070dd31786))
* **webgpu:** keep the builtin where a scalar f32 helper cannot stand in ([72ec856](https://github.com/ooples/AiDotNet.Tensors/commit/72ec856c5571ef65c30b257faf6c19853ee17505))
* **webgpu:** read wgsl types and early-evaluated contexts from code, not comments ([27febbe](https://github.com/ooples/AiDotNet.Tensors/commit/27febbea7512de096569539d5c11ffff3a955c9d))
* **webgpu:** saturate the tanh helper before exp can overflow and read enable directives as tokens ([9c950df](https://github.com/ooples/AiDotNet.Tensors/commit/9c950df00f211c230f16362664c97a81c4490694))


### Refactoring

* **gpu:** clear the stale op entry in a static helper and split the wgsl context check ([112d0fb](https://github.com/ooples/AiDotNet.Tensors/commit/112d0fb57b645f9a94f8187d4b2ce58941ad2990))
* **gpu:** dispose an unusable pooled float64 buffer in a using as it is popped ([1028f71](https://github.com/ooples/AiDotNet.Tensors/commit/1028f7108d9c7877a255feae00e8770ea4c524a0))
* **gpu:** judge a pooled float64 buffer before it leaves the pool ([a63da73](https://github.com/ooples/AiDotNet.Tensors/commit/a63da73dff2c28beac408d88c97132d3fab1b160))
* **webgpu:** select scalar-valued calls with explicit filters ([b02c20b](https://github.com/ooples/AiDotNet.Tensors/commit/b02c20b12700d6d721cffe1ba0054d8d0eb5ec3b))


### Documentation

* **engines:** flag CausalConv1dStep's newest-first kernel and cross-check it against DepthwiseConv1D ([6998cd2](https://github.com/ooples/AiDotNet.Tensors/commit/6998cd29068ef9213dcbb27eea87c5100bd3712c))
* **engines:** state GatedDeltaRule.Standard's bound as non-expansive for 0 &lt;= a, b &lt;= 1 ([e3eb7f4](https://github.com/ooples/AiDotNet.Tensors/commit/e3eb7f4af8e402871851d0a570480bf822921c77))
* record the four abstract IEngine members 0.152.0 added (PR [#1784](https://github.com/ooples/AiDotNet.Tensors/issues/1784)) ([092acfc](https://github.com/ooples/AiDotNet.Tensors/commit/092acfc0e13c397b47ae184cc141bc25ab412418))

## [0.152.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.151.2...v0.152.0) (2026-11-22)


### Features

* **compilation:** GPU TensorIndexSelect defers to the host recorder in inference traces too ([8eea4fa](https://github.com/ooples/AiDotNet.Tensors/commit/8eea4fa523220a135952769048d34b36f5115aab))
* **compilation:** inference plans record TensorIndexSelect over concrete int indices and TensorTopK ([d44e2fa](https://github.com/ooples/AiDotNet.Tensors/commit/d44e2fafd858dbbd2182c7e85d217d7aef2339e6))
* **compilation:** public CompiledModelCache.CompileStatefulInference ([42ace3b](https://github.com/ooples/AiDotNet.Tensors/commit/42ace3be5220e9e9c3c4a8cf76c179aba299afdd))
* **compilation:** stateful compiled inference plans (state leaves advanced in place, CUDA-graph replayable) ([5ce5253](https://github.com/ooples/AiDotNet.Tensors/commit/5ce525381711d0cea2f52eec1e06d21a8a670b71))
* **engines:** consolidate weight normalization, real FFTs and workspace ownership ([d1f3faa](https://github.com/ooples/AiDotNet.Tensors/commit/d1f3faae0e75c0c62b4b1172f043dfac47cb09c9))
* **engines:** IEngine.CausalConv1dStep, a causal depthwise convolution's decode step with the history shifted in place ([28a1fcd](https://github.com/ooples/AiDotNet.Tensors/commit/28a1fcdaba2aeb1e9358764cf0be7ee93977d774))
* **engines:** IEngine.GatedDeltaNetStep, the gated delta rule's decode step with the state updated in place ([6c12877](https://github.com/ooples/AiDotNet.Tensors/commit/6c128771b9f7716d0efd7ec00bd1a11dba768fb9))
* **engines:** IEngine.KvCacheAppend, a decode cache written at a device-side position ([8539185](https://github.com/ooples/AiDotNet.Tensors/commit/85391854fb2126d31641891e912b654492998aae))
* **gdn:** the Gated DeltaNet scan and decode step take the paper's rule (GatedDeltaRule.Standard) ([889fab6](https://github.com/ooples/AiDotNet.Tensors/commit/889fab6688d13cdd362a45a3fc7fde4bdb562490))
* **gpu:** consolidate compiled replay, sparse attention and GPU dispatch ([52ba751](https://github.com/ooples/AiDotNet.Tensors/commit/52ba7513a1639ad88b7beedeace727086a8f48bc))
* **gpu:** implement CSR graph attention across all GPU backends ([4d373fe](https://github.com/ooples/AiDotNet.Tensors/commit/4d373fe4c4ebb85abbe9b968ff46acab2e6a21e4))
* **inference:** captured plans replay in-place writes to their leaves exactly once ([6da780f](https://github.com/ooples/AiDotNet.Tensors/commit/6da780f7c537a9b10dce8434d670d0fc478b7520))


### Bug Fixes

* address [#1793](https://github.com/ooples/AiDotNet.Tensors/issues/1793) review on cuDNN workspace, WeightNorm backward precision and console trace ([426455d](https://github.com/ooples/AiDotNet.Tensors/commit/426455d4b6c86ac873f85891d8b4d2c304988557))
* **api:** declare the Vulkan and WebGPU fused CE position shaders (RS0016) ([dbb4607](https://github.com/ooples/AiDotNet.Tensors/commit/dbb46079949a0aa5fdfe56b8c67d11161de06986))
* **autodiff:** classify TensorWeightNorm, keep its first-order backward closure-free, GAT create-graph ([9d596ef](https://github.com/ooples/AiDotNet.Tensors/commit/9d596efd0f77ebfdee90e261b4a1f38bc28d0e4f))
* **autodiff:** clear post-[#1766](https://github.com/ooples/AiDotNet.Tensors/issues/1766) audit findings ([#1768](https://github.com/ooples/AiDotNet.Tensors/issues/1768)) ([011b677](https://github.com/ooples/AiDotNet.Tensors/commit/011b6776a223d4016b6bff4e0faff5ee699eaad2))
* **autodiff:** keep the index producer out of index select's gradient edges ([1cd7ab3](https://github.com/ooples/AiDotNet.Tensors/commit/1cd7ab3ac5ede2a9c81a454856d66ab547b79e7d))
* **autotune:** warm-replay reused timing graphs; default budget 15 s ([5b330c7](https://github.com/ooples/AiDotNet.Tensors/commit/5b330c72390cf9fa5617e6e1ac4077953fe3f73e))
* **ci:** keep a multi-line initialization failure in one log record ([ee45e47](https://github.com/ooples/AiDotNet.Tensors/commit/ee45e4725528aac396bccb4fc23a1909bafb8a2a))
* **compilation:** a captured inference plan downloads each replay's output from the buffer the graph writes ([d492f8f](https://github.com/ooples/AiDotNet.Tensors/commit/d492f8f0272d21ddeb01bc3137feda069861521d))
* **compilation:** a GPU engine's inference plan runs only the device-agnostic CPU passes ([5f4e2cc](https://github.com/ooples/AiDotNet.Tensors/commit/5f4e2ccd42f96e5c7b76d20cacef57764fcbed4d))
* **compilation:** a stateful plan's next-state producers stay observable through fusion ([27b4f65](https://github.com/ooples/AiDotNet.Tensors/commit/27b4f65c6a1f853635ad56f40905d6c99f353acd))
* **compilation:** a trace that realizes a state-writing decode op no longer advances its state ([b7499c2](https://github.com/ooples/AiDotNet.Tensors/commit/b7499c216250483370490ea726c6622dafac5c71))
* **compilation:** add a lower-rank broadcast operand's gradient from the live output gradient ([6bce717](https://github.com/ooples/AiDotNet.Tensors/commit/6bce71758edc6eb3d2a5e71cb2aa6c64e7845be8))
* **compilation:** an observable output may end a fused chain; update the index-copy test helper ([f8dc3cd](https://github.com/ooples/AiDotNet.Tensors/commit/f8dc3cd74d420694fe7f02d6501c31114495636e))
* **compilation:** count scratch lent to the shared workspace in the step scratch bytes ([0c447f9](https://github.com/ooples/AiDotNet.Tensors/commit/0c447f9dec7c5008cb25a70399fb9a76ba940045))
* **compilation:** GPU inference plans skip memory planning; captures draw step scratch from a plan pool; an abandoned capture unpins the inputs ([d027cf7](https://github.com/ooples/AiDotNet.Tensors/commit/d027cf79741a1cc2392fd1b2da230ed1a32ad7a8))
* **compilation:** inference plans record TensorTopK, and TensorOneHot over its indices ([337fadd](https://github.com/ooples/AiDotNet.Tensors/commit/337fadd5dfd830bb71df817d6952523f404b6ba9))
* **compilation:** keep compiled plans from changing eager matmul numerics ([7acfe7a](https://github.com/ooples/AiDotNet.Tensors/commit/7acfe7a0472a3b7b86084510b2fe35ba6191d7ab))
* **compilation:** keep the captured step graph when the device rewrites an input ([fc48548](https://github.com/ooples/AiDotNet.Tensors/commit/fc48548eee15a64ca7e8be1c739297fe25f97139))
* **compilation:** memory planning keeps a carried state on its own storage ([3c7629a](https://github.com/ooples/AiDotNet.Tensors/commit/3c7629a3b46ca75efcd3e806e6b87a4d8fdb9b78))
* **compilation:** pin every leaf a captured inference plan reads; GPU TopK records in inference traces ([1b3e100](https://github.com/ooples/AiDotNet.Tensors/commit/1b3e100c44131959e6f89984196e7c091b3ec274))
* **compilation:** preserve shared GPU state bindings during step eviction ([e0e00f9](https://github.com/ooples/AiDotNet.Tensors/commit/e0e00f905e3330a2617dac9adce119182e08c960))
* **compilation:** publish running-state writes after GPU graph replay ([1c0725e](https://github.com/ooples/AiDotNet.Tensors/commit/1c0725ed1b77216aafd4801e8ee098c05e6ef2ae))
* **compilation:** refuse a training trace through the forward-only step ops ([8f58b75](https://github.com/ooples/AiDotNet.Tensors/commit/8f58b75cf53abbad029e56905c229999fa5e5c41))
* **compilation:** restore traced running statistics into the bound device buffer ([2d96ad5](https://github.com/ooples/AiDotNet.Tensors/commit/2d96ad53870dd8ad99312e6bfb7ba7ade2c758d2))
* **compilation:** trace inference plans on a plan-owned input copy ([cf2ff73](https://github.com/ooples/AiDotNet.Tensors/commit/cf2ff735c11acb432342b74bd9e7e21c8fa5831b))
* **compiled:** a host write to any device-resident Tensor&lt;int&gt; reaches the compiled step and its replays ([3182283](https://github.com/ooples/AiDotNet.Tensors/commit/3182283305cc02f9a88d7875bc84f7a1b16ddb4f))
* **compiled:** a host write to device-resident fused CE labels reaches the compiled step and its replays ([639ce33](https://github.com/ooples/AiDotNet.Tensors/commit/639ce33a2c877935738eed56c936bf22251e7da1))
* **compiled:** an out-of-range device label throws at the finiteness check instead of being skipped ([8ef59cc](https://github.com/ooples/AiDotNet.Tensors/commit/8ef59cc6a480def5e03473204a1ce42451f32535))
* **compiled:** capture the fused linear cross-entropy with per-step host labels ([c6d0d66](https://github.com/ooples/AiDotNet.Tensors/commit/c6d0d662a72b2a7266a20ebd992cf3e26165370b))
* **compiled:** check a captured ids tensor's baked buffer before the version gate ([36bf7b7](https://github.com/ooples/AiDotNet.Tensors/commit/36bf7b7a48af138ace4c255a509027adb402936e))
* **compiled:** convert device-resident float embedding ids on the device (captured replays used last step's computed ids) ([e8bdac3](https://github.com/ooples/AiDotNet.Tensors/commit/e8bdac306242996c2662e66dbe35962190bc7206))
* **compiled:** create the fused CE labels' device copy before the capture ([4afa484](https://github.com/ooples/AiDotNet.Tensors/commit/4afa484ed928702355c7485662f918b0bd88d146))
* **compiled:** detect an out-of-range device label with a finite count, not NaN semantics ([c607329](https://github.com/ooples/AiDotNet.Tensors/commit/c60732995fc60ba6e4597e90964284cbac0576d8))
* **compiled:** host Tensor&lt;int&gt; ids get a persistent device copy on the resident step (index ops can be captured) ([bec5879](https://github.com/ooples/AiDotNet.Tensors/commit/bec58792fb53b5e35e1ac3cd9bdbed839306cb7c))
* **compiled:** normalize resident float embedding index values ([df27d20](https://github.com/ooples/AiDotNet.Tensors/commit/df27d20ffb43d45e5db652c79a2ba907e11274fd))
* **compiled:** per-plan label-error check, cleared every step, reported on dispose ([e616da7](https://github.com/ooples/AiDotNet.Tensors/commit/e616da7ec42b6b367f878fd8b3bf1775d94ff560))
* **compiled:** resident int refresh for whole-vector tensors only; keep the captured handle ([45995dc](https://github.com/ooples/AiDotNet.Tensors/commit/45995dc54c331edb0a09a8632de4d32e88ae8958))
* **compiled:** retire finalized index bindings and validate replayed embedding IDs ([8b8b77e](https://github.com/ooples/AiDotNet.Tensors/commit/8b8b77e53236a71f57e57e0b75f887f199000b27))
* **compiled:** review fixes for the captured fused cross-entropy ([e9d9849](https://github.com/ooples/AiDotNet.Tensors/commit/e9d984945a85722c2d11bf3b2b422cdfeac0d01a))
* **compiled:** scope index replay bindings and validate labels before updates ([749f0bd](https://github.com/ooples/AiDotNet.Tensors/commit/749f0bd64ac02d4dbca89e43aa1061f6912d72ff))
* **cuda:** keep the cuBLAS prewarm and context syncs out of open captures ([591cc2c](https://github.com/ooples/AiDotNet.Tensors/commit/591cc2c6060a1fa1e59a2c9c9e1895fcb635b48f))
* **cuda:** share GLA tile launch contract constants ([38f7540](https://github.com/ooples/AiDotNet.Tensors/commit/38f7540ee7864d9177761212735f5b694aa797e4))
* **curand:** probe the library, not a device generator; resolve it from the CUDA install ([0501fe9](https://github.com/ooples/AiDotNet.Tensors/commit/0501fe971a8b851fea79139d5e7369ff6e3eb945))
* **embedding:** bound resident gathers and atomic gradient writes ([098e0a1](https://github.com/ooples/AiDotNet.Tensors/commit/098e0a1279b9fb3264006e65f076ebd772474102))
* **engine:** resolve code scanning findings in the gated delta step and planning passes ([ff7fddb](https://github.com/ooples/AiDotNet.Tensors/commit/ff7fddbb41c1bab0c0b09054633a55ecd12928c1))
* **engines:** preserve replayed gradients and support odd-length device FFTs ([2c6452f](https://github.com/ooples/AiDotNet.Tensors/commit/2c6452fe4ac602c3cb06b6c6c26a18d095f30970))
* **engines:** publish AiDotNetEngine.Current through named helpers; CPU plan tests for the decode steps ([870a3c3](https://github.com/ooples/AiDotNet.Tensors/commit/870a3c36dc668142081abea952f51b8602359054))
* **forecasting:** clarify erfc replay and index-add ownership states ([0bd9ba4](https://github.com/ooples/AiDotNet.Tensors/commit/0bd9ba47c5f3b242fc0e55d46efc0c373792f099))
* **gpu:** address consolidated replay, sparse attention and workspace review ([3031c07](https://github.com/ooples/AiDotNet.Tensors/commit/3031c0754578661f46fba970826151eeb14074d6))
* **gpu:** allow repeated source indices in gather ([2240e59](https://github.com/ooples/AiDotNet.Tensors/commit/2240e59d330e1f9a0daff641472849428d0a18fc))
* **gpu:** bound arg-reduce traversal and strengthen edge parity ([97428e3](https://github.com/ooples/AiDotNet.Tensors/commit/97428e34db133c4c4f98458eb00f6b9478db37fb))
* **gpu:** committing a device-updated parameter allocates nothing once its entry exists ([a484a47](https://github.com/ooples/AiDotNet.Tensors/commit/a484a47b4af9b7d540b8be21e631e677e18f277c))
* **gpu:** empty and mismatched vector operands answer as the cpu engine does ([7e5d069](https://github.com/ooples/AiDotNet.Tensors/commit/7e5d069d92112177f06c8b0761dfb1d9fc7e51d0))
* **gpu:** fused CE target positions on every backend (HIP, OpenCL, Metal, Vulkan, WebGPU) ([eab0ac0](https://github.com/ooples/AiDotNet.Tensors/commit/eab0ac043b7d9f06fed312debc476afa96d736e6))
* **gpu:** join the toolkit directory and file name so the directory is never discarded ([c23813e](https://github.com/ooples/AiDotNet.Tensors/commit/c23813eac489af9061ea327507c173a71526264d))
* **gpu:** keep the host value when a resident mirror has no live reader ([94dedd9](https://github.com/ooples/AiDotNet.Tensors/commit/94dedd9a44d3e450c00e7f69ad782dd78f0a47aa))
* **gpu:** keep the merged assembly under the CS8103 user-string cap ([278aecc](https://github.com/ooples/AiDotNet.Tensors/commit/278aecc21de564e36c15eade93de6c3293c7ff7d))
* **gpu:** lease kv-cache and state lookups and guard device positions ([7f658d7](https://github.com/ooples/AiDotNet.Tensors/commit/7f658d7765b221c981e90c43355b0647aa753e45))
* **gpu:** pinned-lifetime tests re-register the offload allocator a registry reset removed ([a6c7a66](https://github.com/ooples/AiDotNet.Tensors/commit/a6c7a66cba734920f783fc7ca4363a4670743ce0))
* **gpu:** preserve grouped gradients and validate WebGPU output bindings ([a250616](https://github.com/ooples/AiDotNet.Tensors/commit/a25061625e2ecc7682f9a5c157aa2d0288bcc7f7))
* **gpu:** re-registration replaces an offload allocator whose context is gone ([4e1fce1](https://github.com/ooples/AiDotNet.Tensors/commit/4e1fce1e24795dd9444d5ec2edb330d03bb98635))
* **gpu:** resolve code scanning findings in the stateful plan and cuda teardown ([2995721](https://github.com/ooples/AiDotNet.Tensors/commit/2995721b21205e96c030ffa1e02b77fb6edde132))
* **gpu:** run a real fft magnitude of unsupported length on the host engine ([131dfcb](https://github.com/ooples/AiDotNet.Tensors/commit/131dfcbc817cfcd7507e9228869a9c464118f18a))
* **gpu:** run non-power-of-two real FFTs on the device Bluestein path without cuFFT ([05c5874](https://github.com/ooples/AiDotNet.Tensors/commit/05c58745c3bde91ed862a6d062d3f3656275c0ae))
* **gpu:** synchronize nvrtc and hip availability probes across threads ([7b1d3f8](https://github.com/ooples/AiDotNet.Tensors/commit/7b1d3f86a6f1202195f5eecff5aeca36197d015d))
* **gpu:** take the lease before the backend lookup in the decode step ops ([8f69648](https://github.com/ooples/AiDotNet.Tensors/commit/8f69648836f26b8c0b0504464506f1eb8690fca3))
* **gpu:** the toolkit directory probe never combines a rooted candidate ([dc771c0](https://github.com/ooples/AiDotNet.Tensors/commit/dc771c093d31d4677aad45a05611d739c0b77ae3))
* **gpu:** use ties-to-even rounding in CUDA and HIP tensor kernels ([9820645](https://github.com/ooples/AiDotNet.Tensors/commit/9820645c57d55067e72f721255759562a49d2e95))
* **gpu:** vector reductions honour the allocation cap policy and reject null operands ([864036f](https://github.com/ooples/AiDotNet.Tensors/commit/864036fca0ca1001c9fe02941f8d3d7c279fb6d4))
* **gpu:** vector Sum, Mean and DotProduct of non-float T use the CPU engine ([5c4cf3a](https://github.com/ooples/AiDotNet.Tensors/commit/5c4cf3a99bffc735987f0e8ecbad4416cd53fed7))
* **gpu:** vector Sum, Mean and DotProduct of non-float T use the CPU engine ([52cc123](https://github.com/ooples/AiDotNet.Tensors/commit/52cc123712629b002e0f63da23575ced9ad5d58e))
* **graph:** validate sparse attention and own device patterns per engine ([f1fecd0](https://github.com/ooples/AiDotNet.Tensors/commit/f1fecd06f62f038e36b3d81f0747ca0c5472e79e))
* **inference:** keep declared-order state writes when batched copies would overlap ([0535ff8](https://github.com/ooples/AiDotNet.Tensors/commit/0535ff8fb6c1a510b682ab38a21e9f961922da93))
* **inference:** preserve observable outputs and replay index dependencies ([079d5c4](https://github.com/ooples/AiDotNet.Tensors/commit/079d5c48a6aefee943411b8fb3b57d46284bc200))
* **inference:** preserve state across graph failures and host updates ([fb19523](https://github.com/ooples/AiDotNet.Tensors/commit/fb195237b9abd27ed089f3b126f30633a366714d))
* **inference:** reach the batched state copy through a backend capability ([21db8c4](https://github.com/ooples/AiDotNet.Tensors/commit/21db8c4dc84b5c430a4e4ecfa959e271e584fd77))
* **metal:** count compute dispatches in the launch probe ([3a95951](https://github.com/ooples/AiDotNet.Tensors/commit/3a95951bc21d9eaa20339c34db6df61f73337e36))
* **moe:** the fused SwiGLU MoE records in inference traces, so a decode plan re-routes every execution ([ef69534](https://github.com/ooples/AiDotNet.Tensors/commit/ef69534e49fa59825cfe6fcfe6c671cc0f7b5b4f))
* **opencl:** compute dgn_to_double's sticky bit with masks so POCL 6 / LLVM 18 compiles it ([43de363](https://github.com/ooples/AiDotNet.Tensors/commit/43de36337eb4b4f2089ceb1dfd592c1fe317d297))
* **opencl:** drop the quotes from the kernel comment that ended the C# string early ([b29c807](https://github.com/ooples/AiDotNet.Tensors/commit/b29c807fcee40e7ca039312690b66dea1c83c518))
* **opencl:** two more POCL 6 / LLVM 18 instcombine aborts (dgn_roundshift, top-k model ops) ([8572e70](https://github.com/ooples/AiDotNet.Tensors/commit/8572e706e41b88dc94d220e6348d07aee9994096))
* **optimization:** ForwardCSEPass never copies from storage memory planning already reused ([5530193](https://github.com/ooples/AiDotNet.Tensors/commit/553019358f4bd4165bfacb4a3b0d7af69cca3797))
* **optimization:** pointwise fusion stops at an intermediate other steps read ([f57cc48](https://github.com/ooples/AiDotNet.Tensors/commit/f57cc48c3ae18b59c9012e7ddc102624669b874b))
* **parity:** force cuda for aidotnet cuda runs and caption recordings with the measured gpu ([c6da1f8](https://github.com/ooples/AiDotNet.Tensors/commit/c6da1f872a37e0785a0b45c1565eb056d6472ed8))
* **parity:** refuse battery power in recordings, usb-powered laptops and the last process ([ec7ec85](https://github.com/ooples/AiDotNet.Tensors/commit/ec7ec856ebfc9e6d8780814aff4c4a5b8578e8dd))
* **tensors:** preserve gradient accuracy, buffer ownership and workspace bounds ([37b3651](https://github.com/ooples/AiDotNet.Tensors/commit/37b365190a89edd09d36e5f20b4bd0c18c11b8b7))
* **tests:** measure weight norm allocations on desktop CLR ([46d3098](https://github.com/ooples/AiDotNet.Tensors/commit/46d3098cbf9ea5421f4aeaf07bf18e90f4434980))
* **webgpu:** fused CE positions kernel built a constant NaN, which Tint rejects ([9be5170](https://github.com/ooples/AiDotNet.Tensors/commit/9be51702fdc64f9fb43ecb2f77e15ea5130859cf))


### Performance

* **#2561:** bound grouped convolution and fuse six-backend epilogues ([#1782](https://github.com/ooples/AiDotNet.Tensors/issues/1782)) ([fe44c79](https://github.com/ooples/AiDotNet.Tensors/commit/fe44c79eae58a12c2a24c98107d919627d2a336d))
* accelerate grouped convolution and reduce HiFi-GAN gradient storage ([9c7dd5f](https://github.com/ooples/AiDotNet.Tensors/commit/9c7dd5fb1e8e816af9ab9c3f6cfa66a39f610f76))
* **autotune:** spend the tuning budget on timing, not first launches or graph setup ([ea0cc3b](https://github.com/ooples/AiDotNet.Tensors/commit/ea0cc3b0c0673dd7db290fc634259a0cf60100bc))
* **compilation:** capture the global-norm gradient clip with the optimizer update ([187531b](https://github.com/ooples/AiDotNet.Tensors/commit/187531baf5176ea550d6695447645d21fd8c3e13))
* **compilation:** replay a device-copied batch without a host round trip ([d8f4db5](https://github.com/ooples/AiDotNet.Tensors/commit/d8f4db508a6f0d3797ad1915f3e369155e4e6d9f))
* **cpu:** pack waveform convolutions directly into GEMM microtiles ([b8afe80](https://github.com/ooples/AiDotNet.Tensors/commit/b8afe806b68b2b2c9eb3bc44c12fd71237412a3f))
* **cpu:** reuse packed convolution panels across waveform tiles ([428c648](https://github.com/ooples/AiDotNet.Tensors/commit/428c648ee4b86af431b1bfa9f77e0c32a07ae1f6))
* **cuda,hip:** block-per-row argmax/argmin for wide rows (49,152-wide row: ~3.5 ms -&gt; 36 us) ([eba5f84](https://github.com/ooples/AiDotNet.Tensors/commit/eba5f84066894a90c12091cfe2285f9c10180160))
* **cuda:** GLA scan backward on 1024-thread tiles, no atomics (v2 shape 14.5 ms -&gt; 2.15 ms) ([6f9d894](https://github.com/ooples/AiDotNet.Tensors/commit/6f9d894ab80fd46e8ec58424cd0ccfcd070ca83e))
* **forecasting:** reuse CPU replay buffers and accelerate GPU index-add ([3855357](https://github.com/ooples/AiDotNet.Tensors/commit/385535792dce400f9a18135db4fc23a2b89fa350))
* fuse inference convolution bias and residual output stores ([a9866a3](https://github.com/ooples/AiDotNet.Tensors/commit/a9866a3fc8ddb8dad4fe75600b8dc926fff3294c))
* **gpu:** broadcast multiply of a leading-prefix scale or a trailing tile runs one kernel, any rank ([0f9c3c8](https://github.com/ooples/AiDotNet.Tensors/commit/0f9c3c8d740dc0415febb9cd761fb74d4d57d4ed))
* **gpu:** commit device parameters and run first-order backwards without allocating ([2a1f1de](https://github.com/ooples/AiDotNet.Tensors/commit/2a1f1deeb979d1c427d8cd3466f7c2e20582f64b))
* **gpu:** one-launch TopK for short rows on every backend ([8dbc586](https://github.com/ooples/AiDotNet.Tensors/commit/8dbc586aaa9eea40f31d1b9273320a75fbbf87af))
* hand idle compiled workspace to the allocator cache for prediction ([cdf065a](https://github.com/ooples/AiDotNet.Tensors/commit/cdf065ab9eccf5491cfb0bd8d496e7c67c292c3c))
* **inference:** a captured lookup of device float ids gathers in one launch ([2747ee9](https://github.com/ooples/AiDotNet.Tensors/commit/2747ee9746d5ac3a5e2e09a55b95c4b383ee5f6e))
* **inference:** a replayed graph skips re-uploading an unchanged input ([67f227b](https://github.com/ooples/AiDotNet.Tensors/commit/67f227ba0df5db8295e32d4a6508776d1d8efad1))
* **inference:** a stateful plan writes all its states in one copy launch ([b68253d](https://github.com/ooples/AiDotNet.Tensors/commit/b68253de096e8cf12800fe42c171eec772f60348))
* **inference:** ops that write every element skip the zero fill in captured graphs ([de5eb7e](https://github.com/ooples/AiDotNet.Tensors/commit/de5eb7e50bcae9a266876a2c8be97f026f1ad3ce))
* pack compiled gradient banks by size and lifetime ([3175ecb](https://github.com/ooples/AiDotNet.Tensors/commit/3175ecb1d22930d298dd1c4f36dc871b2181346d))
* retire incompatible idle GPU cache cells on larger inference misses ([613ecff](https://github.com/ooples/AiDotNet.Tensors/commit/613ecffc6bd14220a95ee0aa45e16187e06f05bb))
* reuse compiled training storage on CPU and GPU for HiFi-GAN ([9db10d0](https://github.com/ooples/AiDotNet.Tensors/commit/9db10d047477b99599b334e9d3dca0c5dfb93e67))
* reuse plan-owned leaf uploads after input rebinding ([ad4f249](https://github.com/ooples/AiDotNet.Tensors/commit/ad4f2499b581dac1e0ff81c4112590deae5c3cd3))
* run the module-init gpu detection on the first read of the engine ([0600e6e](https://github.com/ooples/AiDotNet.Tensors/commit/0600e6e2cf62dea213d0873eeeffd126bf3ac304))


### Refactoring

* **blas:** release the packed b buffer without a redundant null check ([f29f2ba](https://github.com/ooples/AiDotNet.Tensors/commit/f29f2bae6b2352d06fa62372e62a155a59725a92))
* **compilation:** read typed tensors without an object round trip and lock a private gate ([dcf5dae](https://github.com/ooples/AiDotNet.Tensors/commit/dcf5dae7d15bb2b24b613e609c94b3ba64fe6523))
* **engine:** move the locked engine assignment out of the property setter ([dc020ae](https://github.com/ooples/AiDotNet.Tensors/commit/dc020ae20124e96a2865da19b168ff6f033796a4))
* **engine:** move the locked engine assignment out of the property setter ([03ebf74](https://github.com/ooples/AiDotNet.Tensors/commit/03ebf74e6b0919d9ead7ae00cef1fb9ba163e925))


### Documentation

* **compilation:** the GPU memory-planning skip states what was measured, not an unverified mechanism ([a6630c6](https://github.com/ooples/AiDotNet.Tensors/commit/a6630c662dffe21fafb264816f3b204aadfc7217))
* **compiled:** only a compiled Step() reaches the device-id label check ([20f1573](https://github.com/ooples/AiDotNet.Tensors/commit/20f15731622933b1d91b1bf1ae5f10f7e7ce44a1))

## [0.151.2](https://github.com/ooples/AiDotNet.Tensors/compare/v0.151.1...v0.151.2) (2026-11-17)


### Bug Fixes

* **compilation:** preserve gradient enrollment and training replay ([#1766](https://github.com/ooples/AiDotNet.Tensors/issues/1766)) ([dcfcf5a](https://github.com/ooples/AiDotNet.Tensors/commit/dcfcf5ad1fceddfddf7008c2590a30463c82e141))

## [0.151.1](https://github.com/ooples/AiDotNet.Tensors/compare/v0.151.0...v0.151.1) (2026-11-16)


### Bug Fixes

* **autodiff:** sign generic LayerNorm replay scratch ([#1762](https://github.com/ooples/AiDotNet.Tensors/issues/1762)) ([cf8f863](https://github.com/ooples/AiDotNet.Tensors/commit/cf8f8639dae2c5defa92c4275007d22dd9c86a4d))
* **compile:** no backward for ops traced under NoGradScope, plus the [#1744](https://github.com/ooples/AiDotNet.Tensors/issues/1744)/[#1760](https://github.com/ooples/AiDotNet.Tensors/issues/1760) review fixes ([#1764](https://github.com/ooples/AiDotNet.Tensors/issues/1764)) ([7fd8a02](https://github.com/ooples/AiDotNet.Tensors/commit/7fd8a02f75a6ca59fb652157df51a593fbac90ab))

## [0.151.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.150.0...v0.151.0) (2026-11-16)


### ⚠ BREAKING CHANGES

* **engines:** match tiny-cuda-nn's multiresolution hash encoding on every backend ([#1759](https://github.com/ooples/AiDotNet.Tensors/issues/1759))
* **engines:** add data loading, numeric recurrences and generic linalg ([#1720](https://github.com/ooples/AiDotNet.Tensors/issues/1720))
* **engines:** add portable programs, model ops and higher-order autodiff ([#1722](https://github.com/ooples/AiDotNet.Tensors/issues/1722))

### Features

* **compile:** compiled plans for closure objectives (probe, signature, guard) ([#1752](https://github.com/ooples/AiDotNet.Tensors/issues/1752)) ([8a21da6](https://github.com/ooples/AiDotNet.Tensors/commit/8a21da6dcbb1b7927809b5aa103071d395f9a328))
* **engines:** add data loading, numeric recurrences and generic linalg ([#1720](https://github.com/ooples/AiDotNet.Tensors/issues/1720)) ([3a84eac](https://github.com/ooples/AiDotNet.Tensors/commit/3a84eaceb10887b6a28748e1eecd2f9984906100))
* **engines:** add portable programs, model ops and higher-order autodiff ([#1722](https://github.com/ooples/AiDotNet.Tensors/issues/1722)) ([fcedf2a](https://github.com/ooples/AiDotNet.Tensors/commit/fcedf2a7f8e9531636e82300a9869fbf8fc76d77))
* **gpu,moe,routing,ce:** fused SwiGLU MoE, grouped GEMM, sparse routing, detached CE & FP16/BF16 autocast conv across all 6 backends ([#1718](https://github.com/ooples/AiDotNet.Tensors/issues/1718)) ([0e8db6a](https://github.com/ooples/AiDotNet.Tensors/commit/0e8db6a5f2f541234dc4ac8bd28173f68992f3c4))
* **gpu:** extend buffer pooling, cache release, and on-chip BatchedFFT across all 6 DirectGpu backends ([#1729](https://github.com/ooples/AiDotNet.Tensors/issues/1729)) ([cffdd53](https://github.com/ooples/AiDotNet.Tensors/commit/cffdd534d4c87264cdf5aff7648eb7751c5ab66a))


### Bug Fixes

* add both lanes in the double sum-of-squares horizontal sum ([#1733](https://github.com/ooples/AiDotNet.Tensors/issues/1733)) ([6181b58](https://github.com/ooples/AiDotNet.Tensors/commit/6181b588b9d629ff686ea47fa613fe67d13e6bc3))
* carry the digamma asymptotic series to machine precision in double ([#1732](https://github.com/ooples/AiDotNet.Tensors/issues/1732)) ([58d2094](https://github.com/ooples/AiDotNet.Tensors/commit/58d2094e5bfd159d38c57a32c091ecea67d02a48))
* correct the bessel i0 family and its gradients ([#1745](https://github.com/ooples/AiDotNet.Tensors/issues/1745)) ([bc892c6](https://github.com/ooples/AiDotNet.Tensors/commit/bc892c6e12b6cfe72fa1e1d93bdd3e88356e67ca))
* **cuda,opencl,ci:** standalone PTX runtimes share the primary context; drained OpenCL queues report complete; quiet-GPU wait and disk preflight for the 3090 runners ([#1736](https://github.com/ooples/AiDotNet.Tensors/issues/1736)) ([983e640](https://github.com/ooples/AiDotNet.Tensors/commit/983e6401255842841515ba2b42f5046fcf7ca033))
* **cuda:** make the sm_75 lane pass: fp16 gemm algorithm, cuda headers, honest skips ([#1741](https://github.com/ooples/AiDotNet.Tensors/issues/1741)) ([32fadb1](https://github.com/ooples/AiDotNet.Tensors/commit/32fadb1bf81006c22d58fea0f97c23b2992f94cf))
* **engines:** accumulate half and bfloat16 reductions and matmul in float32 ([#1735](https://github.com/ooples/AiDotNet.Tensors/issues/1735)) ([b5fe4f6](https://github.com/ooples/AiDotNet.Tensors/commit/b5fe4f6b71bfe290b1fb03d58401dbdd2c35eccb))
* **engines:** match tiny-cuda-nn's multiresolution hash encoding on every backend ([#1759](https://github.com/ooples/AiDotNet.Tensors/issues/1759)) ([b5a61d7](https://github.com/ooples/AiDotNet.Tensors/commit/b5a61d7e2c2c974d6ea02b894f1f420bbc134c64))
* **gpu:** fix softmax_rows barrier, state buffer liveness, and coverage ([#1756](https://github.com/ooples/AiDotNet.Tensors/issues/1756)) ([6876702](https://github.com/ooples/AiDotNet.Tensors/commit/6876702837cd405a5bc4745b1e76fc36b3e2b388))
* **gpu:** step-1 host parameter residency and [#1737](https://github.com/ooples/AiDotNet.Tensors/issues/1737) audit fixes ([#1755](https://github.com/ooples/AiDotNet.Tensors/issues/1755)) ([c41f6a6](https://github.com/ooples/AiDotNet.Tensors/commit/c41f6a64077a240518ec8ed93caf51fe688597d0))
* **hip:** make the MFMA GEMM compile and compute correctly on CDNA, add gfx950 ([#1748](https://github.com/ooples/AiDotNet.Tensors/issues/1748)) ([151f0bf](https://github.com/ooples/AiDotNet.Tensors/commit/151f0bf24ff34bb9a063dfc46949eff03bb6f729))
* **linalg:** accurate float/double svd; half-safe cpu engine fallbacks ([#1757](https://github.com/ooples/AiDotNet.Tensors/issues/1757)) ([719ce1b](https://github.com/ooples/AiDotNet.Tensors/commit/719ce1bdaad7931bf5645eeb46fb1cda48ae767b))
* **ptx:** emit a PTX ISA version that can name the target; report why CudaBackend is unavailable ([#1719](https://github.com/ooples/AiDotNet.Tensors/issues/1719)) ([906dd37](https://github.com/ooples/AiDotNet.Tensors/commit/906dd377f1e08f68cbaee1e5177ad8db19f914e1))
* unwrap the engine in gradient checkpointing so main builds under tens0020 ([#1731](https://github.com/ooples/AiDotNet.Tensors/issues/1731)) ([2c90aed](https://github.com/ooples/AiDotNet.Tensors/commit/2c90aed63b16626e628c0ee316eef147c43ee93e))


### Performance

* **gpu,autodiff:** eliminate GPU/CPU sync stalls, LOH churn, and checkpoint VRAM retention across 6 backends ([#1737](https://github.com/ooples/AiDotNet.Tensors/issues/1737)) ([515bf91](https://github.com/ooples/AiDotNet.Tensors/commit/515bf91b070da5b418a3d603f7c97a526053d665))
* **gpu:** eager TensorNegate allocates nothing on the host ([#1730](https://github.com/ooples/AiDotNet.Tensors/issues/1730)) ([bb8beec](https://github.com/ooples/AiDotNet.Tensors/commit/bb8beec823cfdf81499a4aa66b1e43a3b15eb67f))
* **gpu:** release intermediates at last use, CUDA step arena, compiled MoE/in-place/embedding training fixes, fused LR schedules ([#1742](https://github.com/ooples/AiDotNet.Tensors/issues/1742)) ([e76d5af](https://github.com/ooples/AiDotNet.Tensors/commit/e76d5afda770af2ef5179ec526094e0c914524cb))

## [0.150.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.149.0...v0.150.0) (2026-11-10)


### Features

* **compiled-plan:** choose which side of the gradient clip the l2 term goes on ([#1697](https://github.com/ooples/AiDotNet.Tensors/issues/1697)) ([14e6dcb](https://github.com/ooples/AiDotNet.Tensors/commit/14e6dcb614654e71e3b6cc29e8f1bc7aa1eed02b))
* element-wise gate and engine fixes from the AiDotNet parity profiles ([#1702](https://github.com/ooples/AiDotNet.Tensors/issues/1702)) ([5ffc6f8](https://github.com/ooples/AiDotNet.Tensors/commit/5ffc6f87f99c33e26a8d6ab4c3eea115fa9c4b4b))


### Bug Fixes

* **compilation:** realize lazy operands before a recorded op's trace-time run ([#1704](https://github.com/ooples/AiDotNet.Tensors/issues/1704)) ([42acf96](https://github.com/ooples/AiDotNet.Tensors/commit/42acf964ee14fe3d4fcca12cc53b121cfbb4faa1))
* **compile:** a recorded custom node's output owns its storage ([#1705](https://github.com/ooples/AiDotNet.Tensors/issues/1705)) ([#1707](https://github.com/ooples/AiDotNet.Tensors/issues/1707)) ([3b4ed5b](https://github.com/ooples/AiDotNet.Tensors/commit/3b4ed5be95e253ea568d32eb15cc66c8f3949558))
* **cpu:** any-rank axis ops, with a reflection-driven rank conformance gate ([#1700](https://github.com/ooples/AiDotNet.Tensors/issues/1700)) ([c140cd6](https://github.com/ooples/AiDotNet.Tensors/commit/c140cd6d4c7d5ebf3b3152a863ed10e8724cc413))
* **topk:** TopK/TensorTopK on any rank and axis; heap selection; GPU rank-N last axis ([#1696](https://github.com/ooples/AiDotNet.Tensors/issues/1696)) ([22f815a](https://github.com/ooples/AiDotNet.Tensors/commit/22f815aebd53f70e5642c5d71ad90d1b8e4c6187))


### Performance

* **gpu:** checkpointed GLA backward on all backends, broadcast without full expansion, fused causal depthwise conv ([#1712](https://github.com/ooples/AiDotNet.Tensors/issues/1712)) ([22caa5a](https://github.com/ooples/AiDotNet.Tensors/commit/22caa5aaeee4e7280eda911cc56389a87d23b956))
* **gpu:** one-launch on-chip batched FFT on CUDA/HIP/OpenCL/Vulkan; drop per-op host syncs in RFFT/IRFFT ([#1714](https://github.com/ooples/AiDotNet.Tensors/issues/1714)) ([dc1bb7e](https://github.com/ooples/AiDotNet.Tensors/commit/dc1bb7e557601fe905e36b9e22e2edfdf0873778))

## [0.149.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.148.1...v0.149.0) (2026-11-09)


### Features

* **engine:** report whether the current engine is the caller's choice ([#1695](https://github.com/ooples/AiDotNet.Tensors/issues/1695)) ([838ddce](https://github.com/ooples/AiDotNet.Tensors/commit/838ddce76c06501690fd457c8afdf1e141ecf5eb))


### Bug Fixes

* **ci:** print hung test host stacks and stop gc pauses tripping wgsl translation ([#1690](https://github.com/ooples/AiDotNet.Tensors/issues/1690)) ([c3fe3a1](https://github.com/ooples/AiDotNet.Tensors/commit/c3fe3a1fae1b6968390d97d2b80b3c0417fd92bd))
* **simd:** extend FastExpDouble256 Taylor expansion to degree 13 (about 1 ULP: measured max 1.001 ULP over x in [-700, 700]; the release note originally claimed &lt;0.5 ULP, which rounding in the polynomial does not reach) ([#1692](https://github.com/ooples/AiDotNet.Tensors/issues/1692)) ([3ff6e9c](https://github.com/ooples/AiDotNet.Tensors/commit/3ff6e9c51bf1b98da00ba3d6f970d67bf7228621))


### Performance

* **gpu:** collect dead step results inside an open tape past 256 MB ([#1693](https://github.com/ooples/AiDotNet.Tensors/issues/1693)) ([8e8d8a1](https://github.com/ooples/AiDotNet.Tensors/commit/8e8d8a1b98080413d2e8a0aeccab2921e927cb29))
* **gpu:** fused linear-CE index forward in row chunks over backend GEMMs (44% of an LM step) ([#1691](https://github.com/ooples/AiDotNet.Tensors/issues/1691)) ([9556d47](https://github.com/ooples/AiDotNet.Tensors/commit/9556d47d3f9973cf31566d5b7af859ee17a43573))

## [0.148.1](https://github.com/ooples/AiDotNet.Tensors/compare/v0.148.0...v0.148.1) (2026-11-07)


### Bug Fixes

* **autodiff:** give every differentiable op its gradient, and differentiate every GridSample mode ([#1684](https://github.com/ooples/AiDotNet.Tensors/issues/1684)) ([6526ce5](https://github.com/ooples/AiDotNet.Tensors/commit/6526ce5de1e0c3a813d4c8c617fa2b122e5073c0))


### Performance

* **cpu:** arena idle byte budget, vectorised row broadcast, ToArray without zero-fill ([#1687](https://github.com/ooples/AiDotNet.Tensors/issues/1687)) ([fe99c62](https://github.com/ooples/AiDotNet.Tensors/commit/fe99c62f10dfd1bb8cf4e3eff85e5971871fad7f))
* **gpu:** keep GatedDeltaNetScanForward and BPTT backward on-device across all 6 GPU backends ([#1685](https://github.com/ooples/AiDotNet.Tensors/issues/1685)) ([510bf01](https://github.com/ooples/AiDotNet.Tensors/commit/510bf015b063d6e1b953860b7c61b72c29d93b41))

## [0.148.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.147.0...v0.148.0) (2026-11-06)


### Features

* hip dlpack exchange, metal tuned profile, fp16 gemm vs pytorch, avx-512 and cudnn evidence ([#1679](https://github.com/ooples/AiDotNet.Tensors/issues/1679)) ([4ac59b5](https://github.com/ooples/AiDotNet.Tensors/commit/4ac59b5f3e3a7726e2fc67d5bdc155057cccec25))


### Bug Fixes

* **autodiff:** record AffineGrid and mode GridSample in compiled training and on the GPU tape ([#1681](https://github.com/ooples/AiDotNet.Tensors/issues/1681)) ([515d226](https://github.com/ooples/AiDotNet.Tensors/commit/515d22610057e5ae243defeaa5e69e4a51d9b850))
* avx2-off crash in shared-weight 1x1 conv and fp32 gemm tiers without avx2 ([#1676](https://github.com/ooples/AiDotNet.Tensors/issues/1676)) ([2167d2f](https://github.com/ooples/AiDotNet.Tensors/commit/2167d2fc94f14517d7fe100fae0af92dea48703d))
* **memory:** stop leaking a gradient set per step and let the restore pool yield under pressure ([#1682](https://github.com/ooples/AiDotNet.Tensors/issues/1682)) ([7e5922b](https://github.com/ooples/AiDotNet.Tensors/commit/7e5922be564757c2ee5ba04f50650947efc38fd4))


### Performance

* **gpu:** cache IRFFT gather/sign tables and adjoint weights per backend and shape ([#1677](https://github.com/ooples/AiDotNet.Tensors/issues/1677)) ([1df0543](https://github.com/ooples/AiDotNet.Tensors/commit/1df05437965e9f892619761b212c22ad3bfff3ea))

## [0.147.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.146.0...v0.147.0) (2026-11-05)


### Features

* **clip:** deterministic, overflow-free global-norm clip, bit-identical on CPU and every GPU backend ([#1672](https://github.com/ooples/AiDotNet.Tensors/issues/1672)) ([0fa810a](https://github.com/ooples/AiDotNet.Tensors/commit/0fa810a99a79e754ce1d329d058d2562747a4f21))
* **compiled-plan:** stream fused optimizer state through ICompiledTrainingPlanStreamedState ([#1675](https://github.com/ooples/AiDotNet.Tensors/issues/1675)) ([9a603fb](https://github.com/ooples/AiDotNet.Tensors/commit/9a603fb94a2ef4aeb38870a5ba4bbbbc9dfa5295))


### Bug Fixes

* **autodiff:** accept output-padded gradients in conv-transpose backward ([#1674](https://github.com/ooples/AiDotNet.Tensors/issues/1674)) ([2a2bf42](https://github.com/ooples/AiDotNet.Tensors/commit/2a2bf42a7486ffe4d05f61dbbca395d609d434c1))
* **gpu:** detach copy-on-write parameters before an in-place device optimizer step ([#1671](https://github.com/ooples/AiDotNet.Tensors/issues/1671)) ([52102c3](https://github.com/ooples/AiDotNet.Tensors/commit/52102c3fc14aade0fa6e0547e8b6db5486ebc7b8))

## [0.146.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.145.0...v0.146.0) (2026-11-03)


### Features

* **optimizers:** int8 and bf16 moments for AdamW, eager and fused ([#1670](https://github.com/ooples/AiDotNet.Tensors/issues/1670)) ([242ea7b](https://github.com/ooples/AiDotNet.Tensors/commit/242ea7bb3ac0937e4027b8edfb18bc65da05af87))


### Bug Fixes

* **gpu:** device scatter for IndexSelect backward; slice CUDA BatchedFFT past gridDim.y ([#1663](https://github.com/ooples/AiDotNet.Tensors/issues/1663)) ([b3d1664](https://github.com/ooples/AiDotNet.Tensors/commit/b3d1664098902a5a0a091c7e9ff884625c1c8e06)), closes [#1661](https://github.com/ooples/AiDotNet.Tensors/issues/1661) [#1662](https://github.com/ooples/AiDotNet.Tensors/issues/1662)
* **gpu:** one-hot from device int32 indices in one launch, no host read ([#1668](https://github.com/ooples/AiDotNet.Tensors/issues/1668)) ([b98e352](https://github.com/ooples/AiDotNet.Tensors/commit/b98e352bdef5e9cea3351403b5ae9c400d86de70)), closes [#1664](https://github.com/ooples/AiDotNet.Tensors/issues/1664)
* **gpu:** ParameterBuffer views resolved at their offset (wrong eager GPU gradients); device clip_grad_norm_ ([#1666](https://github.com/ooples/AiDotNet.Tensors/issues/1666)) ([acee008](https://github.com/ooples/AiDotNet.Tensors/commit/acee008c418569f0fb2c133776b4f3aea71cd3d0))
* **review:** make startup-only SIMD flags readonly ([#1660](https://github.com/ooples/AiDotNet.Tensors/issues/1660)) ([383c7dd](https://github.com/ooples/AiDotNet.Tensors/commit/383c7dd1e541c0c3f5fd2c18141106b9ffbd5bad))


### Performance

* **blas:** route AVX2 thin-M weight gradients to PackBoth where it measured faster ([#1658](https://github.com/ooples/AiDotNet.Tensors/issues/1658)) ([1f00a2c](https://github.com/ooples/AiDotNet.Tensors/commit/1f00a2c4e84d52923dc83c922f1f10826c3cf5bd))
* **cudnn:** forward default runs cuDNN's heuristic pick (1.5-4x on budget-exhausted 3x3 shapes) ([#1657](https://github.com/ooples/AiDotNet.Tensors/issues/1657)) ([0cd1a9c](https://github.com/ooples/AiDotNet.Tensors/commit/0cd1a9c8f7fc27a145c83f3f2669092579642198))
* **gemm:** deterministic engine matmul takes GotoGemm where it is thread-count invariant ([#653](https://github.com/ooples/AiDotNet.Tensors/issues/653)) ([#1659](https://github.com/ooples/AiDotNet.Tensors/issues/1659)) ([2e174af](https://github.com/ooples/AiDotNet.Tensors/commit/2e174afd0136e92b010fc5355770fb27c74bde3c))


### Documentation

* **ptx-campaigns:** conv 3x3 continuation ($42.96): winner 1.61x fused Winograd in graph; correct [#1650](https://github.com/ooples/AiDotNet.Tensors/issues/1650) baseline ([#1667](https://github.com/ooples/AiDotNet.Tensors/issues/1667)) ([96ad9f3](https://github.com/ooples/AiDotNet.Tensors/commit/96ad9f384227e4e5a1ba8126230c3539634ab299))

## [0.145.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.144.2...v0.145.0) (2026-11-03)


### ⚠ BREAKING CHANGES

* on net462, net471 and netstandard2.0 the System.Runtime.Intrinsics Vector128/256/512 polyfill types (SimdCompat) are internal, and the Invoke(Vector128/256/512) overloads of AcoshOperatorDouble, AsinhOperatorDouble and AtanhOperatorDouble are removed from those targets. Both shipped publicly in 0.144.1.

### Features

* **compile:** batch-dynamic inference plans + compiled-vs-eager miscompile fixes (T-3) ([#1623](https://github.com/ooples/AiDotNet.Tensors/issues/1623)) ([f36f13e](https://github.com/ooples/AiDotNet.Tensors/commit/f36f13ef84ff5e551e0cc461f78e42bdc631c88b))
* DLPack interop, ONNX exporter, public API tracking and package validation (T-4) (+1 combined) ([#1581](https://github.com/ooples/AiDotNet.Tensors/issues/1581)) ([94a333b](https://github.com/ooples/AiDotNet.Tensors/commit/94a333bb109b175b445a150b1b203956715b46ea))
* **gpu:** device memory stats and TensorScope.Detach, with license headers 7, 9, 11/18 (combined) ([#1590](https://github.com/ooples/AiDotNet.Tensors/issues/1590)) ([88abbb4](https://github.com/ooples/AiDotNet.Tensors/commit/88abbb43f5905c1c939f9adace82037610c84542))


### Bug Fixes

* **api:** declare TunedKernelDecisionReason.Heuristic in the public API ([#1637](https://github.com/ooples/AiDotNet.Tensors/issues/1637)) ([e79ec80](https://github.com/ooples/AiDotNet.Tensors/commit/e79ec8031d2b776de08d11c91aadc63e4c075f27))
* **autodiff:** record conv2d gradients under create-graph ([#1655](https://github.com/ooples/AiDotNet.Tensors/issues/1655)) ([986f21c](https://github.com/ooples/AiDotNet.Tensors/commit/986f21cd603b75490961b4d5e8f53eb7c3fe72c8))
* **cpu:** kernel-cache hits allocate nothing; weight packs sized to their depth ([#1633](https://github.com/ooples/AiDotNet.Tensors/issues/1633)) ([9a6d0b1](https://github.com/ooples/AiDotNet.Tensors/commit/9a6d0b1f744ad56f1f814b73356bf542e9a26dc5))
* **cuda,vulkan:** CUDA-lane crashes and the Vulkan LSTM TDR — cuLinkCreate option lifetime, cublasDestroy sync during capture, LSTM sequence on parallel kernels ([#1628](https://github.com/ooples/AiDotNet.Tensors/issues/1628)) ([8686f7f](https://github.com/ooples/AiDotNet.Tensors/commit/8686f7ffcdeeb0b642a2ccb94871679281d6c82f))
* **cuda:** hold the cuDNN handle from cudnnSetStream through its launch ([#1614](https://github.com/ooples/AiDotNet.Tensors/issues/1614)) ([3c3ac63](https://github.com/ooples/AiDotNet.Tensors/commit/3c3ac63df7c204d4974016cd30ae192abe237257))
* **engines:** empty slices are valid, so concatenate backward handles zero-width inputs ([#1635](https://github.com/ooples/AiDotNet.Tensors/issues/1635)) ([92c320b](https://github.com/ooples/AiDotNet.Tensors/commit/92c320b345c14755171da056257d0560e3678f0e))
* **gpu:** batched attention route is tapeless only, smaller score budget, fused-branch test ([#1603](https://github.com/ooples/AiDotNet.Tensors/issues/1603)) ([e8b22ea](https://github.com/ooples/AiDotNet.Tensors/commit/e8b22eacc28c632994eb490e9fb5ff5bd09d76d9))
* relu keeps nan on every cpu and gpu path; cuda leaky relu honours alpha ([#1646](https://github.com/ooples/AiDotNet.Tensors/issues/1646)) ([0ee348e](https://github.com/ooples/AiDotNet.Tensors/commit/0ee348e26b1b03396325c842c2a0bc146109ab5f))
* **tests:** CrowdTrainer header on the four compilation tests from [#1623](https://github.com/ooples/AiDotNet.Tensors/issues/1623) ([#1634](https://github.com/ooples/AiDotNet.Tensors/issues/1634)) ([2feb5a7](https://github.com/ooples/AiDotNet.Tensors/commit/2feb5a7938079aece84b2e5fc2bd28f0ea6abe32))
* v1 gaps on main - tuning drains only its own pool buffers; drop committed trace ([#1645](https://github.com/ooples/AiDotNet.Tensors/issues/1645)) ([f951440](https://github.com/ooples/AiDotNet.Tensors/commit/f9514408690d1594db9162c18f6ca0eb384616f0))


### Performance

* **conv:** blocked Winograd weight gradient, 512-bit blocked GEMM, stride-2 12-position forward ([#1573](https://github.com/ooples/AiDotNet.Tensors/issues/1573)) ([#1651](https://github.com/ooples/AiDotNet.Tensors/issues/1651)) ([aaa6979](https://github.com/ooples/AiDotNet.Tensors/commit/aaa6979cc4ef54f88da9aebaf21d359e45a006e9))
* **conv:** few-patch Winograd F(4x4) for wide small-plane layers; forward floor 32 off full-rate AVX-512 ([#1573](https://github.com/ooples/AiDotNet.Tensors/issues/1573)) ([#1652](https://github.com/ooples/AiDotNet.Tensors/issues/1652)) ([30e3e4a](https://github.com/ooples/AiDotNet.Tensors/commit/30e3e4a1bc4a6e5a4a98f02c805bbd9eee1105ae))
* **cpu-gemm:** packed, k-blocked B strips for the 512-bit direct GEMM tiles ([#1644](https://github.com/ooples/AiDotNet.Tensors/issues/1644)) ([0c410e1](https://github.com/ooples/AiDotNet.Tensors/commit/0c410e15d9d56bb7bd551103e293fcb3edc8ab8a))
* **cpu:** AVX-512 narrow GEMM, 12-position conv on small planes, packed-A copy ([#1573](https://github.com/ooples/AiDotNet.Tensors/issues/1573)) ([#1648](https://github.com/ooples/AiDotNet.Tensors/issues/1648)) ([c4ff1a9](https://github.com/ooples/AiDotNet.Tensors/commit/c4ff1a96d2e52ce0d277e4ffe895a38279d85a9f))
* **cpu:** bind pool workers across every processor group ([#653](https://github.com/ooples/AiDotNet.Tensors/issues/653)) ([#1647](https://github.com/ooples/AiDotNet.Tensors/issues/1647)) ([e6f5c02](https://github.com/ooples/AiDotNet.Tensors/commit/e6f5c028cf83dfebf672405368876b6ce4416a79))
* **cuda:** PTX campaigns recorded with spend; conv 3x3 winner 1.92x cuDNN eager ([#1649](https://github.com/ooples/AiDotNet.Tensors/issues/1649)) ([#1650](https://github.com/ooples/AiDotNet.Tensors/issues/1650)) ([e8a498e](https://github.com/ooples/AiDotNet.Tensors/commit/e8a498e7ef8e79594bc7299d025358628fdb184e))
* **gemm:** A·Bᵀ on the 512-bit direct tiles with the transpose fused into strip packing; linear dX takes it ([#1573](https://github.com/ooples/AiDotNet.Tensors/issues/1573)) ([#1654](https://github.com/ooples/AiDotNet.Tensors/issues/1654)) ([d0614f2](https://github.com/ooples/AiDotNet.Tensors/commit/d0614f286f201e0ba1a9f3fc354a5b501180567f))
* **gemm:** resolve DisableAutotune before the tile choice; AVX-512 thin-K A^T to PackBoth ([#1573](https://github.com/ooples/AiDotNet.Tensors/issues/1573)) ([#1653](https://github.com/ooples/AiDotNet.Tensors/issues/1653)) ([438d943](https://github.com/ooples/AiDotNet.Tensors/commit/438d94368d132217de4fe575d555a961fee67fbe))
* gpu and cpu inference ahead of pytorch on every parity model and batch size ([#1636](https://github.com/ooples/AiDotNet.Tensors/issues/1636)) ([5806593](https://github.com/ooples/AiDotNet.Tensors/commit/58065936269eaa7aa4fb0c882312f7864ae5ad1f))
* **tensors:** move rarely-set tensor fields into a side object allocated on first use ([#1641](https://github.com/ooples/AiDotNet.Tensors/issues/1641)) ([503b5fa](https://github.com/ooples/AiDotNet.Tensors/commit/503b5fa1501bf2cd6fc3a064371ccf0181143d69))
* v1 T-3 CPU training and GRU (CUDA 0.61x, CPU 0.38x torch) ([#1639](https://github.com/ooples/AiDotNet.Tensors/issues/1639)) ([172dea1](https://github.com/ooples/AiDotNet.Tensors/commit/172dea12ce2cc9b62bc51a59cb3eef6497181513))

## [0.144.2](https://github.com/ooples/AiDotNet.Tensors/compare/v0.144.1...v0.144.2) (2026-10-29)


### Bug Fixes

* v1 T-1 correctness (AVX2-off kernel parity, [#1570](https://github.com/ooples/AiDotNet.Tensors/issues/1570), CPU parity flakes, coverage baseline) ([#1578](https://github.com/ooples/AiDotNet.Tensors/issues/1578)) ([3680d3b](https://github.com/ooples/AiDotNet.Tensors/commit/3680d3b3d3a6a22db06042ced7981023a2760763))

## [0.144.1](https://github.com/ooples/AiDotNet.Tensors/compare/v0.144.0...v0.144.1) (2026-10-29)


### Bug Fixes

* **gpu:** [#1570](https://github.com/ooples/AiDotNet.Tensors/issues/1570) residency regressions; fused gru, conv-bn folding and tensor scope; gpu smoke lane ([#1577](https://github.com/ooples/AiDotNet.Tensors/issues/1577)) ([f17bd87](https://github.com/ooples/AiDotNet.Tensors/commit/f17bd87d6d17b54c07d75ee0a49f381c61de839c))


### Performance

* **gpu:** batched multi-head attention forward instead of a per-head loop ([#1601](https://github.com/ooples/AiDotNet.Tensors/issues/1601)) ([29bb94f](https://github.com/ooples/AiDotNet.Tensors/commit/29bb94f4635f24d64cf18286d66dc58294fbd1a7))

## [0.144.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.143.0...v0.144.0) (2026-10-28)


### Features

* **codegen:** declare the remaining gather/scatter, sparse, index and attention kernels ([#1440](https://github.com/ooples/AiDotNet.Tensors/issues/1440), [#1441](https://github.com/ooples/AiDotNet.Tensors/issues/1441)) ([#1552](https://github.com/ooples/AiDotNet.Tensors/issues/1552)) ([bbaedac](https://github.com/ooples/AiDotNet.Tensors/commit/bbaedaca110d41281055404386f39d84957b3215))
* **cuda:** attention backward and flash backward as resident-program stages ([#1440](https://github.com/ooples/AiDotNet.Tensors/issues/1440)) ([#1550](https://github.com/ooples/AiDotNet.Tensors/issues/1550)) ([d732ae7](https://github.com/ooples/AiDotNet.Tensors/commit/d732ae71218c5fc837120ce0d03a4fa66f9c416a))


### Bug Fixes

* **compiled-gpu:** keep the captured training step on the device for every model ([#1564](https://github.com/ooples/AiDotNet.Tensors/issues/1564)) ([98ecfcd](https://github.com/ooples/AiDotNet.Tensors/commit/98ecfcd1f0209378a369f54fe39e718e3ba7c47d))
* **compiled:** retiring a replayed graph lands only the step's results ([#1554](https://github.com/ooples/AiDotNet.Tensors/issues/1554)) ([b217eb1](https://github.com/ooples/AiDotNet.Tensors/commit/b217eb14e7d88b94c5e24ee063dd12993e38fb64))
* five defects (seeding, regex timeouts, culture parsing, swallowed teardown, onnx conversions) ([#1560](https://github.com/ooples/AiDotNet.Tensors/issues/1560)) ([3e576a1](https://github.com/ooples/AiDotNet.Tensors/commit/3e576a10d23231010b038c61b62f89d45b13adcd))
* **gpu:** fp16 im2col conv writes nchw for batch &gt; 1 ([#1562](https://github.com/ooples/AiDotNet.Tensors/issues/1562)) ([9197575](https://github.com/ooples/AiDotNet.Tensors/commit/919757528cb89da68fab25251271688d2b087b0b))
* **gpu:** grid-stride batch-derived grid axes past 65535, and keep caller-owned TensorCopy destinations across a tape ([#1567](https://github.com/ooples/AiDotNet.Tensors/issues/1567)) ([767372a](https://github.com/ooples/AiDotNet.Tensors/commit/767372a27ab51b19c7001751cb22451d6c6f7f5e))


### Performance

* **cuda:** fp16 accumulation on stock geforce tensor cores - gemm routes, flash attention, rate probe ([#1549](https://github.com/ooples/AiDotNet.Tensors/issues/1549)) ([#1565](https://github.com/ooples/AiDotNet.Tensors/issues/1565)) ([f20fb4d](https://github.com/ooples/AiDotNet.Tensors/commit/f20fb4d92c74e27c3f74bc155c3083dc4989b346))
* fuse recurrent state gradients and preserve GPU residency ([#1566](https://github.com/ooples/AiDotNet.Tensors/issues/1566)) ([927a99b](https://github.com/ooples/AiDotNet.Tensors/commit/927a99b6f74994cb55b8fd9e7d3907632fc945e1))
* **gpu:** scatter-add the reduce-max and upsample-3d backwards in one launch ([#1551](https://github.com/ooples/AiDotNet.Tensors/issues/1551)) ([6dba14a](https://github.com/ooples/AiDotNet.Tensors/commit/6dba14a0b4cdef012d888000e5bfc2292d54e275))
* **streaming:** recycle restore arrays and write back in place (AiDotNet[#2322](https://github.com/ooples/AiDotNet.Tensors/issues/2322)) ([#1548](https://github.com/ooples/AiDotNet.Tensors/issues/1548)) ([bfbca3c](https://github.com/ooples/AiDotNet.Tensors/commit/bfbca3cfe2e68fb7e2fa6d6c04437b19d6a82eb4))

## [0.143.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.142.0...v0.143.0) (2026-10-26)


### ⚠ BREAKING CHANGES

* AiDotNet.Tensors.Engines.DirectGpu.HIP.RocSparse is removed; use CpuSparseDeviceOps, which it always delegated to.

### Features

* close the remaining gaps in [#921](https://github.com/ooples/AiDotNet.Tensors/issues/921), [#515](https://github.com/ooples/AiDotNet.Tensors/issues/515), [#1441](https://github.com/ooples/AiDotNet.Tensors/issues/1441) and [#833](https://github.com/ooples/AiDotNet.Tensors/issues/833) ([#1543](https://github.com/ooples/AiDotNet.Tensors/issues/1543)) ([4aea57d](https://github.com/ooples/AiDotNet.Tensors/commit/4aea57d7f3eb7366ea8d3c46cd93127db4f1e5d8))
* **codegen:** declare the row-wise scatter mean, softmax and csr spmm ([#1441](https://github.com/ooples/AiDotNet.Tensors/issues/1441)) ([#1546](https://github.com/ooples/AiDotNet.Tensors/issues/1546)) ([cc26929](https://github.com/ooples/AiDotNet.Tensors/commit/cc269299fd5860572f11222009ba4b21e6a18f32))


### Bug Fixes

* Clone() of an arena tensor deep-copies instead of sharing copy-on-write ([#653](https://github.com/ooples/AiDotNet.Tensors/issues/653)) ([#1547](https://github.com/ooples/AiDotNet.Tensors/issues/1547)) ([7b0f824](https://github.com/ooples/AiDotNet.Tensors/commit/7b0f8244bf74bdadb103a3f4869ee04176caff9e))
* **codegen:** SASS scheduler correctness - yield on long stalls, LDS/STS uniform offsets, R2UR spacing ([#1544](https://github.com/ooples/AiDotNet.Tensors/issues/1544)) ([4328dd4](https://github.com/ooples/AiDotNet.Tensors/commit/4328dd43af1af356735a0938d0a02bdb6af641d1))
* **codegen:** the SASS scheduler schedules the Tensor-Core GEMM again ([#1540](https://github.com/ooples/AiDotNet.Tensors/issues/1540)) ([778a846](https://github.com/ooples/AiDotNet.Tensors/commit/778a84685e3c4a50d5a96e8fcb224cc5f8253447))
* **simd:** the cached-b gemms and the fused gelu epilogue throw without avx2 ([#1542](https://github.com/ooples/AiDotNet.Tensors/issues/1542)) ([69c026e](https://github.com/ooples/AiDotNet.Tensors/commit/69c026e4ffabd41f1cec91581c32c3126ef26e51))


### Performance

* **codegen:** SASS instruction reordering - the [#1442](https://github.com/ooples/AiDotNet.Tensors/issues/1442) gate met (TF32 Tensor-Core GEMM 1.13x ptxas) ([#1545](https://github.com/ooples/AiDotNet.Tensors/issues/1545)) ([ec84bfd](https://github.com/ooples/AiDotNet.Tensors/commit/ec84bfd13e86d50e334a0e94c3236b56f35cd231))
* **ptx:** FP32 GEMM register-pipelined fragments and a leaner cp.async loop ([#1538](https://github.com/ooples/AiDotNet.Tensors/issues/1538)) ([5f78dc9](https://github.com/ooples/AiDotNet.Tensors/commit/5f78dc9f3f74a008650da16943af968c7aafb95f))

## [0.142.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.141.0...v0.142.0) (2026-10-23)


### Features

* **codegen:** [#890](https://github.com/ooples/AiDotNet.Tensors/issues/890) phase 4d/4e - paged prefill, segment reductions, and the routes that call them ([#1526](https://github.com/ooples/AiDotNet.Tensors/issues/1526)) ([8fe96aa](https://github.com/ooples/AiDotNet.Tensors/commit/8fe96aacf009e312bbdaab0f5caccdde4ff65577))
* **codegen:** own SASS backend - bit model, cubin rewriter, scheduler ([#1442](https://github.com/ooples/AiDotNet.Tensors/issues/1442)) ([#1504](https://github.com/ooples/AiDotNet.Tensors/issues/1504)) ([5e22587](https://github.com/ooples/AiDotNet.Tensors/commit/5e225873d4b99c368044d0bed3eec4ef65fd3e4b))


### Bug Fixes

* **autotune:** re-measure the fastest candidate once when two lost pairs refuse it, instead of letting a slower clean candidate win ([#1492](https://github.com/ooples/AiDotNet.Tensors/issues/1492)) ([8463b32](https://github.com/ooples/AiDotNet.Tensors/commit/8463b324f7b432a99fe000a72263f742ad0c599b))
* **compilation:** add the divide backward into a shared operand's gradient ([#1533](https://github.com/ooples/AiDotNet.Tensors/issues/1533)) ([82825df](https://github.com/ooples/AiDotNet.Tensors/commit/82825df47d914b7e7684cafa2c094c0c7ab9e014))
* **cuda:** build an attention shape's program at its first sighting so identical calls agree bit for bit ([#1532](https://github.com/ooples/AiDotNet.Tensors/issues/1532)) ([c0864fd](https://github.com/ooples/AiDotNet.Tensors/commit/c0864fd09e01c2f1a038029afb6c96e6f75b4aa9)), closes [#1529](https://github.com/ooples/AiDotNet.Tensors/issues/1529)
* **gpu:** compiled-plan steps on a non-cuda gpu - the readback waits where it is read, and the fusion expectations come from the backend ([#1503](https://github.com/ooples/AiDotNet.Tensors/issues/1503)) ([307a659](https://github.com/ooples/AiDotNet.Tensors/commit/307a6590b11f23c50e07d314162347b50770488b))
* **gpu:** ConvTranspose2D gradients include the output-padding rows ([#1508](https://github.com/ooples/AiDotNet.Tensors/issues/1508)) ([b4c0d94](https://github.com/ooples/AiDotNet.Tensors/commit/b4c0d94cf7d8ec6d7057a58cd01af2427e4eefd6))
* **gpu:** IndexSelect yields zeros for an out-of-range index ([#1515](https://github.com/ooples/AiDotNet.Tensors/issues/1515)) ([82b6d81](https://github.com/ooples/AiDotNet.Tensors/commit/82b6d81e150dea3714e22e607fabccb7df2464d1))
* **opencl:** keep the readback event before the commit can flush, and check the flush ([#1518](https://github.com/ooples/AiDotNet.Tensors/issues/1518)) ([01df74b](https://github.com/ooples/AiDotNet.Tensors/commit/01df74b9552b3440df50aced1a2ed067e4cf6af0))
* **vulkan:** GridSample reflection, AdaDelta weight decay, tuning timestamps in the runs' command buffer ([#1525](https://github.com/ooples/AiDotNet.Tensors/issues/1525)) ([87643de](https://github.com/ooples/AiDotNet.Tensors/commit/87643de8ca7e829f707643d48e19caac4f20b6b5)), closes [#1524](https://github.com/ooples/AiDotNet.Tensors/issues/1524)


### Performance

* **codegen:** [#890](https://github.com/ooples/AiDotNet.Tensors/issues/890) phase 4a/4b - normalisation, pointwise and convolution as declared specs (draft) ([#1457](https://github.com/ooples/AiDotNet.Tensors/issues/1457)) ([da43b2c](https://github.com/ooples/AiDotNet.Tensors/commit/da43b2c612268cfc01e41009d496167a21ab538d))
* **codegen:** [#890](https://github.com/ooples/AiDotNet.Tensors/issues/890) phase 4a/4d/4e - groupnorm backward, paged decode, deterministic scatter ([#1516](https://github.com/ooples/AiDotNet.Tensors/issues/1516)) ([ea29a79](https://github.com/ooples/AiDotNet.Tensors/commit/ea29a7980aa8e68671e3bc52c235f03cc2356eba))
* **cpu:** GEMMs at 2-4 threads no longer run 5-19x slower than at one ([#1534](https://github.com/ooples/AiDotNet.Tensors/issues/1534)) ([22679b1](https://github.com/ooples/AiDotNet.Tensors/commit/22679b155ea4fe7e2925ac0202cb16f4ffa4cacd))
* **cuda:** attention falls back to the flash kernel; fix the [#890](https://github.com/ooples/AiDotNet.Tensors/issues/890) route tool on sm_86 ([#1530](https://github.com/ooples/AiDotNet.Tensors/issues/1530)) ([5231b9b](https://github.com/ooples/AiDotNet.Tensors/commit/5231b9b0f14ac2c5738e72e5d71aa321e19f6b1b))
* **cuda:** fused Winograd dW accumulates a 1 x 8k x 8c block per thread - 32 FMAs per two shared loads (step -53 us) ([#1506](https://github.com/ooples/AiDotNet.Tensors/issues/1506)) ([4e62cc1](https://github.com/ooples/AiDotNet.Tensors/commit/4e62cc16e212be92c77ac1adf9bad5d5d250cdcf))
* **cuda:** one input-gradient launch for ResNet's strided conv and its projection shortcut (step -73 us) ([#1494](https://github.com/ooples/AiDotNet.Tensors/issues/1494)) ([c50bafe](https://github.com/ooples/AiDotNet.Tensors/commit/c50bafeb84e7f89d9e52e27c6562531a6a5b189c))
* **cuda:** ResNet-18 BatchNorm backwards skip two activation passes (step 2.241 -&gt; 2.211 ms) ([#1490](https://github.com/ooples/AiDotNet.Tensors/issues/1490)) ([254b38a](https://github.com/ooples/AiDotNet.Tensors/commit/254b38aa3614ba4de331f943120250320e6cbc88))
* **cuda:** tiled, atomic-free FlashAttention backward (21-33x the hand-written kernel) ([#1531](https://github.com/ooples/AiDotNet.Tensors/issues/1531)) ([6c73e63](https://github.com/ooples/AiDotNet.Tensors/commit/6c73e63ac7bda7619dd58f0db62f581b0b018750))
* **gpu:** device-side finiteness guard and non-blocking readback on HIP, Metal, Vulkan and WebGPU ([#1523](https://github.com/ooples/AiDotNet.Tensors/issues/1523)) ([9aba09d](https://github.com/ooples/AiDotNet.Tensors/commit/9aba09d81861cb624aa4665db403df94f7eb59b8))
* **gpu:** fused Winograd F(2x2,3x3) forward on Vulkan, Metal and WebGPU ([#1432](https://github.com/ooples/AiDotNet.Tensors/issues/1432)) ([#1536](https://github.com/ooples/AiDotNet.Tensors/issues/1536)) ([e0a9a32](https://github.com/ooples/AiDotNet.Tensors/commit/e0a9a32eef15e6363deee011818688a200e45d2c))
* **gpu:** port the stride-2 input gradients to HIP, Metal, OpenCL, Vulkan and WebGPU ([#1517](https://github.com/ooples/AiDotNet.Tensors/issues/1517)) ([7abd0d1](https://github.com/ooples/AiDotNet.Tensors/commit/7abd0d1f250134ac59980a5811ce0d61d5bf7976)), closes [#1479](https://github.com/ooples/AiDotNet.Tensors/issues/1479) [#1480](https://github.com/ooples/AiDotNet.Tensors/issues/1480) [#1481](https://github.com/ooples/AiDotNet.Tensors/issues/1481) [#1482](https://github.com/ooples/AiDotNet.Tensors/issues/1482) [#1483](https://github.com/ooples/AiDotNet.Tensors/issues/1483)
* **gpu:** Winograd F(4x4,3x3) forward on Vulkan, Metal, WebGPU and HIP through a shared pipeline ([#1432](https://github.com/ooples/AiDotNet.Tensors/issues/1432)) ([#1535](https://github.com/ooples/AiDotNet.Tensors/issues/1535)) ([8a0c9a7](https://github.com/ooples/AiDotNet.Tensors/commit/8a0c9a7eeece039bcba9ebc777a01488541fba21))

## [0.141.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.140.0...v0.141.0) (2026-10-21)


### ⚠ BREAKING CHANGES

* **simd:** float exp and softmax results on Intel CPUs change by float32 rounding (they now match AMD's).

### Bug Fixes

* **gpu:** fall back to the cpu for non-float element types in the gpu into-ops ([#1478](https://github.com/ooples/AiDotNet.Tensors/issues/1478)) ([0374212](https://github.com/ooples/AiDotNet.Tensors/commit/0374212ce6e016386768ebb4ab1c3ea75ff29860))
* **opencl:** diagnose a refused buffer view, and state the dispose-recycle preconditions ([#1486](https://github.com/ooples/AiDotNet.Tensors/issues/1486)) ([61ffe09](https://github.com/ooples/AiDotNet.Tensors/commit/61ffe0921e4f106b3ad737b3932def6a3a161895))
* **simd:** one exp kernel on every CPU vendor ([#1458](https://github.com/ooples/AiDotNet.Tensors/issues/1458)) ([f06c41c](https://github.com/ooples/AiDotNet.Tensors/commit/f06c41c32a2e9a0b7d1588324d381fb896be058b))
* **winograd:** compile and gate the fused weight gradient apart from the transforms ([#1484](https://github.com/ooples/AiDotNet.Tensors/issues/1484)) ([3edf831](https://github.com/ooples/AiDotNet.Tensors/commit/3edf8318fb3416fdc821a541c4fd4b7f2e84c819))


### Performance

* **compiled-gpu:** fuse a linear head into the MSE loss - one launch forward, one backward (LSTM step 63.8 -&gt; 61.1 us) ([#1447](https://github.com/ooples/AiDotNet.Tensors/issues/1447)) ([1565669](https://github.com/ooples/AiDotNet.Tensors/commit/1565669054b7711445a7d7598de58e9ef3bd93d9))
* **cuda-conv:** 8x8-micro-tile implicit-GEMM forward for the stride-2 3x3 layers (20-23% faster per layer, ResNet-18 step -31 us) ([#1461](https://github.com/ooples/AiDotNet.Tensors/issues/1461)) ([eb28289](https://github.com/ooples/AiDotNet.Tensors/commit/eb282891ad4fb6819c7d9af0c4d14ba9ee90780b))
* **cuda-conv:** fused Winograd dW writes dW directly when unsplit (ResNet-18 512@4 53 -&gt; 41 us, step 2.361 -&gt; 2.313 ms) ([#1468](https://github.com/ooples/AiDotNet.Tensors/issues/1468)) ([cc2d70e](https://github.com/ooples/AiDotNet.Tensors/commit/cc2d70ef0b4ae983b89f7b30f0923e42f5c74626))
* **cuda-conv:** fused Winograd F(3x3,2x2) weight gradient - one launch plus a split finish ([#1465](https://github.com/ooples/AiDotNet.Tensors/issues/1465)) ([9e9347e](https://github.com/ooples/AiDotNet.Tensors/commit/9e9347ed4ed62fce20907defc4950d07491b19c8))
* **cuda-lstm:** backward GEMMs - bias from the first column tile, one-wave row splits with an in-kernel reduce (10.2 -&gt; 8.6 us) ([#1434](https://github.com/ooples/AiDotNet.Tensors/issues/1434)) ([8352dc2](https://github.com/ooples/AiDotNet.Tensors/commit/8352dc239569679ecff3c307c3975a4c18727e56))
* **cuda-optimizer:** ping-pong SGD update for small models - bounded grid, loads before stores, one-warp tail, undo folded in (LSTM 7.3 -&gt; 3.4 us per step) ([#1463](https://github.com/ooples/AiDotNet.Tensors/issues/1463)) ([78ddc30](https://github.com/ooples/AiDotNet.Tensors/commit/78ddc302e4a4f0350ef7f1272cb0e399358c49f6))
* **cuda-step-io:** copy only changed step inputs (LSTM -3.6, MLP -22, Transformer -3.3 us) ([#1456](https://github.com/ooples/AiDotNet.Tensors/issues/1456)) ([a3238ab](https://github.com/ooples/AiDotNet.Tensors/commit/a3238ab09439e20a756bf660a93ece77a31d1d5b))
* **cuda-step-io:** one system fence in the epilogue; the loss-gradient reseed rides in the prologue ([#1454](https://github.com/ooples/AiDotNet.Tensors/issues/1454)) ([ce606dc](https://github.com/ooples/AiDotNet.Tensors/commit/ce606dc0107d9bbd35e46c6669e6feb399f41b75))
* **cuda:** fused winograd conv loads a 4x6 window per two neighbouring tiles ([#1470](https://github.com/ooples/AiDotNet.Tensors/issues/1470)) ([8509a4e](https://github.com/ooples/AiDotNet.Tensors/commit/8509a4e6ae4c8371e46322ebbb7cab578c897f93))
* **cuda:** stride-2 3x3 input gradient - all four sub-pixel phases in one kernel (ResNet step -60 us) ([#1477](https://github.com/ooples/AiDotNet.Tensors/issues/1477)) ([71e82c9](https://github.com/ooples/AiDotNet.Tensors/commit/71e82c9ceb388cf245d328fe2706db1e4e60279a))
* **gpu:** overlap host and device on opencl and fuse convolution bias and relu on every backend ([#1435](https://github.com/ooples/AiDotNet.Tensors/issues/1435)) ([dfa37a6](https://github.com/ooples/AiDotNet.Tensors/commit/dfa37a6cefd8b5b265a3150f67e47eade6104435))
* **lstm:** take the backward recurrence's coefficients off the per-step chain ([#1444](https://github.com/ooples/AiDotNet.Tensors/issues/1444)) ([7106951](https://github.com/ooples/AiDotNet.Tensors/commit/7106951c983b4f21abe235ef0a29836791796f1c))
* PyTorch parity at its best ([#1055](https://github.com/ooples/AiDotNet.Tensors/issues/1055)), conv data path ([#1335](https://github.com/ooples/AiDotNet.Tensors/issues/1335)), CPU saturation ([#653](https://github.com/ooples/AiDotNet.Tensors/issues/653)), resident programs ([#890](https://github.com/ooples/AiDotNet.Tensors/issues/890)) ([#1430](https://github.com/ooples/AiDotNet.Tensors/issues/1430)) ([d3edb84](https://github.com/ooples/AiDotNet.Tensors/commit/d3edb84444709f3da44841260610ce312208927c))
* webgpu dispatches past 65535 workgroups fold onto a 2-d grid; conv passes are one fma chain ([#1466](https://github.com/ooples/AiDotNet.Tensors/issues/1466)) ([6a827a3](https://github.com/ooples/AiDotNet.Tensors/commit/6a827a342132f6e9b902098454248a36909504b2))

## [0.140.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.139.2...v0.140.0) (2026-10-19)


### ⚠ BREAKING CHANGES

* **blas:** AIDOTNET_USE_BLAS, AIDOTNET_BLAS_PROVIDER and the AIDOTNET_OPENBLAS_* switches no longer do anything. Removed public API: NativeLibraryDetector.HasOpenBlas/HasMkl/HasCpuBlas/OpenBlasLoadDiagnostic/ RequireOpenBlas/RequireCpuBlas, NativeLibraryStatus and PlatformCapabilities HasOpenBlas/HasMkl/HasCpuBlas, OpenBlasNotFoundException, CpuInferenceConfig, NativeInferencePool, PrefersManagedCache, BlasManaged.PreferManaged and BlasManaged.AutotuneRouting.

### Features

* five-issue batch - Metal CI lane ([#915](https://github.com/ooples/AiDotNet.Tensors/issues/915), [#921](https://github.com/ooples/AiDotNet.Tensors/issues/921)), GPU SpMM ([#515](https://github.com/ooples/AiDotNet.Tensors/issues/515)), CPU saturation ([#653](https://github.com/ooples/AiDotNet.Tensors/issues/653)), fused basic block ([#1353](https://github.com/ooples/AiDotNet.Tensors/issues/1353)) ([#1400](https://github.com/ooples/AiDotNet.Tensors/issues/1400)) ([b98a640](https://github.com/ooples/AiDotNet.Tensors/commit/b98a64002a3a00b296346bd562b066603b7ba121))


### Bug Fixes

* **cuda-gemm:** a tuned GEMM route comes before the narrow kernels' shape heuristics; LastGemmDispatchPath reports NarrowGemm ([#1426](https://github.com/ooples/AiDotNet.Tensors/issues/1426)) ([dcdb2c9](https://github.com/ooples/AiDotNet.Tensors/commit/dcdb2c94653a100955e2693eb3586e67bc18f176))
* **vulkan:** double parity kernels request SignedZeroInfNanPreserve (float controls) so infinities and NaNs survive; device-gated ([#1416](https://github.com/ooples/AiDotNet.Tensors/issues/1416)) ([a7a01d4](https://github.com/ooples/AiDotNet.Tensors/commit/a7a01d499a345cd4fb49a54a9596768f85a5fef3))


### Performance

* **blas:** remove native OpenBLAS and Intel MKL; parallel row-times-matrix GEMV ([#1401](https://github.com/ooples/AiDotNet.Tensors/issues/1401)) ([9ca36ad](https://github.com/ooples/AiDotNet.Tensors/commit/9ca36ad1ed054bebc69ca8a468796c895537846d))
* **compiled-gpu:** a captured training step costs the host one graph launch (step I/O + optimizer in the graph) ([#1405](https://github.com/ooples/AiDotNet.Tensors/issues/1405)) ([b82bcef](https://github.com/ooples/AiDotNet.Tensors/commit/b82bcef0092b247de84330d1f46c68acc8f8432a))
* **compiled-gpu:** one fused node for a composed MSE loss (ReduceMean(Square(Subtract))) ([#1407](https://github.com/ooples/AiDotNet.Tensors/issues/1407)) ([fbb734f](https://github.com/ooples/AiDotNet.Tensors/commit/fbb734f05590cc0faf1be0abd3aec26a370c6011))
* **cuda-conv:** phase input gradient indexes dY without divisions and takes 1x1 stride-2 kernels ([#1415](https://github.com/ooples/AiDotNet.Tensors/issues/1415)) ([50eaf11](https://github.com/ooples/AiDotNet.Tensors/commit/50eaf11ac8ca07c30da18ea6cf574c5f5e1bffab))
* **cuda-conv:** r8v weight-gradient kernels read dY's K tile as 16-byte loads (ResNet-18 3x3 layers 3-18% faster) ([#1428](https://github.com/ooples/AiDotNet.Tensors/issues/1428)) ([fe01a5a](https://github.com/ooples/AiDotNet.Tensors/commit/fe01a5aec5b35222ab554c590d8d220a4b20d06c))
* **cuda-gemm:** skinny_gemm on four warp groups with prefetch + 16/8-row tiles, and small-shape GEMMs (Transformer GEMMs 50.7 -&gt; 24.7 us/step, head 16 -&gt; 8 us) ([#1410](https://github.com/ooples/AiDotNet.Tensors/issues/1410)) ([0c9fc80](https://github.com/ooples/AiDotNet.Tensors/commit/0c9fc809da30582ec7254c9d0c143dbdd08d0b7b))
* **cuda-lstm:** a last-step LSTM node outputs h_T directly and differentiates it alone (ILstmLastStepTraining) ([#1418](https://github.com/ooples/AiDotNet.Tensors/issues/1418)) ([e06e790](https://github.com/ooples/AiDotNet.Tensors/commit/e06e790b013e7e3acb6a0075da2ab01b3e4ec0cb))
* **cuda-lstm:** forward recurrence with a unit's four gates on adjacent lanes (13.7 -&gt; 10.9 us) ([#1420](https://github.com/ooples/AiDotNet.Tensors/issues/1420)) ([4099af8](https://github.com/ooples/AiDotNet.Tensors/commit/4099af855fad1f7081874e2bf8a69295eb4440fc))
* **cuda-lstm:** register-resident recurrence + one fused backward-GEMM kernel; LSTM step 309 -&gt; 155 us ([#1403](https://github.com/ooples/AiDotNet.Tensors/issues/1403)) ([12b5393](https://github.com/ooples/AiDotNet.Tensors/commit/12b5393c0970cede78ca1b2ffc92c510a10fdc73))
* **cuda-optimizer:** ping-pong captured SGD - one fused pass updates into the other weight set and takes the finiteness reduction (ResNet-18 step -43.5 us) ([#1422](https://github.com/ooples/AiDotNet.Tensors/issues/1422)) ([481b039](https://github.com/ooples/AiDotNet.Tensors/commit/481b039f9c458b8ad5ee936e6f41763b32c3d34c))
* **cuda:** Transformer step 140 -&gt; ~99 us - single-launch bias/LayerNorm/weight gradients, in-place packed QKV attention ([#1413](https://github.com/ooples/AiDotNet.Tensors/issues/1413)) ([427c164](https://github.com/ooples/AiDotNet.Tensors/commit/427c164602eb21bcc7c21bf45f73edc57293d9de))
* **cuda:** tune candidates as captured-graph replays (device time without per-launch host cost) ([#1427](https://github.com/ooples/AiDotNet.Tensors/issues/1427)) ([507f95f](https://github.com/ooples/AiDotNet.Tensors/commit/507f95f15eaa2b8f77a537271b26e10d3c998673))

## [0.139.2](https://github.com/ooples/AiDotNet.Tensors/compare/v0.139.1...v0.139.2) (2026-10-17)


### Bug Fixes

* **compiled-gpu:** max-pool recompute backward keeps its winner indices on the device ([#1396](https://github.com/ooples/AiDotNet.Tensors/issues/1396)) ([06b8eac](https://github.com/ooples/AiDotNet.Tensors/commit/06b8eac8a2f0412212e9a82b82cc40fef9b73faa))


### Performance

* **cpu:** training-step fixes vs PyTorch - JIT edge rows, tiering, parked-pool stealing, deep-K weight gradients ([#1362](https://github.com/ooples/AiDotNet.Tensors/issues/1362)) ([3ac6a4d](https://github.com/ooples/AiDotNet.Tensors/commit/3ac6a4dc7ae4cd62b6c0fae1634c4c5c3f220e43))
* **streaming:** fast lossless re-encode and no per-materialize weight-sized garbage ([#1394](https://github.com/ooples/AiDotNet.Tensors/issues/1394)) ([4ec7e92](https://github.com/ooples/AiDotNet.Tensors/commit/4ec7e926a34e31f4ce696f1e627cb58fc061ab9c))

## [0.139.1](https://github.com/ooples/AiDotNet.Tensors/compare/v0.139.0...v0.139.1) (2026-10-17)


### Bug Fixes

* **autotune:** an inconclusive tuning measurement is retried and never persisted ([#1382](https://github.com/ooples/AiDotNet.Tensors/issues/1382)) ([5ca97e2](https://github.com/ooples/AiDotNet.Tensors/commit/5ca97e23ac99f99d717ac140e04a964d35b60e71))
* **autotune:** the tuned-slot gate forgives one stalled timing pair ([#1358](https://github.com/ooples/AiDotNet.Tensors/issues/1358)) ([011fdea](https://github.com/ooples/AiDotNet.Tensors/commit/011fdea64e6ed47f07a940f1042c4e4d7ab0ae1e))
* **cuda:** fan-out gemms run each slice on its leased stream ([#1364](https://github.com/ooples/AiDotNet.Tensors/issues/1364)) ([ac9def1](https://github.com/ooples/AiDotNet.Tensors/commit/ac9def1ccff54e0fa9a3ea0b3ef34d469053c98b)), closes [#1315](https://github.com/ooples/AiDotNet.Tensors/issues/1315)
* **cuda:** per-thread cuBLAS setup zero-fills its skinny-GEMM counters on the device ([#1348](https://github.com/ooples/AiDotNet.Tensors/issues/1348)) ([4ef9c46](https://github.com/ooples/AiDotNet.Tensors/commit/4ef9c46e028667bd2d9be2ea99301477154f26b8))
* **memory:** disposing a large result returns its array to the thread cache ([#1363](https://github.com/ooples/AiDotNet.Tensors/issues/1363)) ([7c3247b](https://github.com/ooples/AiDotNet.Tensors/commit/7c3247ba9baacc59608a4f26471e3273f279bbf9)), closes [#1331](https://github.com/ooples/AiDotNet.Tensors/issues/1331)


### Performance

* **compiled-gpu:** register-resident BatchNorm + fused BN-&gt;ReLU and BN-&gt;add-&gt;ReLU (ResNet-18 step -20%) ([#1376](https://github.com/ooples/AiDotNet.Tensors/issues/1376)) ([78e9e46](https://github.com/ooples/AiDotNet.Tensors/commit/78e9e46d33df5d663a7323430ce415c6e54bcfbf))
* **compiled-gpu:** remove the captured training step's dead memsets and first-write gradient copies ([#1361](https://github.com/ooples/AiDotNet.Tensors/issues/1361)) ([f3dd5c0](https://github.com/ooples/AiDotNet.Tensors/commit/f3dd5c021f922213ca04ec25fd038a969832768d))
* **cpu:** single-thread gemm/conv routes and a staged direct conv for large planes ([#1380](https://github.com/ooples/AiDotNet.Tensors/issues/1380)) ([e8fb163](https://github.com/ooples/AiDotNet.Tensors/commit/e8fb16388df6485d6ab4e4350ecfbda055a237fd))
* **cuda-conv:** 8x8-micro-tile weight-gradient kernel (r8) replaces the engine rectangular tiles ([#1392](https://github.com/ooples/AiDotNet.Tensors/issues/1392)) ([491e84c](https://github.com/ooples/AiDotNet.Tensors/commit/491e84ceb726e34e844d9712ab8cc431c723f7f9))
* **cuda-conv:** engine forward/input-gradient K order keeps each weight run in L1 ([#1387](https://github.com/ooples/AiDotNet.Tensors/issues/1387)) ([3bab9bf](https://github.com/ooples/AiDotNet.Tensors/commit/3bab9bfb9cf9cd53c5d1f89ebf7110a2cd30cb63))
* **cuda-conv:** faster engine input gradient: balanced phase split-K, coalesced epilogue, tap-major weights ([#1374](https://github.com/ooples/AiDotNet.Tensors/issues/1374)) ([6290563](https://github.com/ooples/AiDotNet.Tensors/commit/6290563895c2808d822d143797ba9913e87caddb))
* **cuda-conv:** fused single-kernel Winograd F(2x2,3x3) for the input gradient and forward ([#1377](https://github.com/ooples/AiDotNet.Tensors/issues/1377)) ([84811af](https://github.com/ooples/AiDotNet.Tensors/commit/84811af535bb145f48cb5eda764265c9e6efa575))
* **cuda-conv:** fused Winograd transforms its filters in-kernel and splits the reduction on deep layers ([#1390](https://github.com/ooples/AiDotNet.Tensors/issues/1390)) ([11485f7](https://github.com/ooples/AiDotNet.Tensors/commit/11485f74098784b4934b1d9418dd1932a0317b57))
* **cuda-conv:** opt-in Winograd F(4x4,3x3) input gradient and F(3x3,4x4) weight gradient ([#1378](https://github.com/ooples/AiDotNet.Tensors/issues/1378)) ([22e1b68](https://github.com/ooples/AiDotNet.Tensors/commit/22e1b68d1da698bd5e004ee8ed69b95793d9ed9c))
* **cuda-conv:** rectangular engine weight-gradient tiles as dW candidates ([#1385](https://github.com/ooples/AiDotNet.Tensors/issues/1385)) ([93675af](https://github.com/ooples/AiDotNet.Tensors/commit/93675af404c5805e37fe2f489b911dd0c4b07f6d))
* **cuda-conv:** split-K for the sub-pixel phase input gradient on small grids ([#1356](https://github.com/ooples/AiDotNet.Tensors/issues/1356)) ([c083f50](https://github.com/ooples/AiDotNet.Tensors/commit/c083f5059c14bf0834cef75016e8871aad2d04a5))
* **cuda:** multi-tensor sum of squares sums each float4 in float ([#1367](https://github.com/ooples/AiDotNet.Tensors/issues/1367)) ([4fdf82a](https://github.com/ooples/AiDotNet.Tensors/commit/4fdf82a261382987e14851e7218b8838787368df))
* **cuda:** Winograd conv candidates: F(2x2,3x3) forward + input gradient, F(3x3,2x2) weight gradient ([#1350](https://github.com/ooples/AiDotNet.Tensors/issues/1350)) ([835da67](https://github.com/ooples/AiDotNet.Tensors/commit/835da67faaa91c397344243d1816cc5ddec6a774))
* **gpu:** Winograd conv routes on HIP, OpenCL, Vulkan, WebGPU and Metal; CUDA 13 startup, OpenCL BatchedGemm and tuning-attribution fixes ([#1379](https://github.com/ooples/AiDotNet.Tensors/issues/1379)) ([c3a6d70](https://github.com/ooples/AiDotNet.Tensors/commit/c3a6d70e086be0645d3ba31440fff1da09575795))
* **opencl:** choose clblast's direct or indirect gemm by problem volume ([#1383](https://github.com/ooples/AiDotNet.Tensors/issues/1383)) ([fe74992](https://github.com/ooples/AiDotNet.Tensors/commit/fe74992275e07b1255ad503352331dcef5d3fcf6)), closes [#377](https://github.com/ooples/AiDotNet.Tensors/issues/377)
* **ptx:** direct-PTX FP32 CSR segmented max and min (experimental) ([#1393](https://github.com/ooples/AiDotNet.Tensors/issues/1393)) ([78c093d](https://github.com/ooples/AiDotNet.Tensors/commit/78c093d5d7c010c3d4df3975c741da8aeac381a6)), closes [#852](https://github.com/ooples/AiDotNet.Tensors/issues/852)

## [0.139.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.138.1...v0.139.0) (2026-10-16)


### ⚠ BREAKING CHANGES

* **gpu-sparse:** AiDotNet.Tensors.Engines.DirectGpu.CUDA.CuSparse is removed; use CpuSparseDeviceOps, which it always delegated to.

### Features

* **#1265:** cuBLAS optional — every CUDA GEMM falls back to generated kernels (FP32/FP16/BF16/FP64); native-library inventory ([#1316](https://github.com/ooples/AiDotNet.Tensors/issues/1316)) ([01d8a7f](https://github.com/ooples/AiDotNet.Tensors/commit/01d8a7fe5685e11002650923d52bcc384c7f7a48))


### Bug Fixes

* **compile:** a training-mode BatchNorm no longer blocks whole-step CUDA-graph capture ([#1327](https://github.com/ooples/AiDotNet.Tensors/issues/1327)) ([1eaf48a](https://github.com/ooples/AiDotNet.Tensors/commit/1eaf48a083a0b329d9121fc9c247603449c45577))
* **cuda-conv:** resolve duplicate TryConv2DBackwardKernelImplicitGemm (CS0111 on main) ([#1329](https://github.com/ooples/AiDotNet.Tensors/issues/1329)) ([9a58451](https://github.com/ooples/AiDotNet.Tensors/commit/9a5845190f4c5b369032f98230b65eadd6444a4e)), closes [#1328](https://github.com/ooples/AiDotNet.Tensors/issues/1328)
* **gpu:** clamp the decode-linear gelu tanh argument so vulkan returns no nan ([#1318](https://github.com/ooples/AiDotNet.Tensors/issues/1318)) ([6489db1](https://github.com/ooples/AiDotNet.Tensors/commit/6489db11cf44339bec75632984dc1d0af045e987))
* **gpu:** eager results own their device buffer, so dispose returns it to the pool ([#1324](https://github.com/ooples/AiDotNet.Tensors/issues/1324)) ([0726bf6](https://github.com/ooples/AiDotNet.Tensors/commit/0726bf6c8b1d9f56179f38970385232fa0c00f0c)), closes [#1302](https://github.com/ooples/AiDotNet.Tensors/issues/1302)
* **hip:** batched gemm launch bounds and deformable conv backward argument order ([#1342](https://github.com/ooples/AiDotNet.Tensors/issues/1342)) ([b46835d](https://github.com/ooples/AiDotNet.Tensors/commit/b46835da21792bc7ff37cbd16a4ec0e0b9636ba1))
* **hip:** gemm beta-zero nan, variance layout, fast-math nan, dropout and reciprocal gradients ([#1346](https://github.com/ooples/AiDotNet.Tensors/issues/1346)) ([f3a0b6c](https://github.com/ooples/AiDotNet.Tensors/commit/f3a0b6c4c312fc63e9fcf6b404e3e4ee05424d68))
* **metal:** make five shader libraries compile with Apple's compiler, and check all of them ([#1299](https://github.com/ooples/AiDotNet.Tensors/issues/1299)) ([490f969](https://github.com/ooples/AiDotNet.Tensors/commit/490f9694571dfd047cdc685810f867a8cdd773ce))


### Performance

* **#1265:** strict-FP32 cp.async SGEMM + tuned per-shape routing of CUDA Gemm where it beats cuBLAS ([#1310](https://github.com/ooples/AiDotNet.Tensors/issues/1310)) ([ccba3fd](https://github.com/ooples/AiDotNet.Tensors/commit/ccba3fd119a7bd40da0062b755f613fe098327f9))
* **#1321:** GEMM routing measures and follows separate eager and CUDA-graph verdicts ([#1322](https://github.com/ooples/AiDotNet.Tensors/issues/1322)) ([7a27b7c](https://github.com/ooples/AiDotNet.Tensors/commit/7a27b7cc4dd5bb7e6ee74329f4ff42258536075a))
* **#436:** fused CPU MLP/MHA/LSTM inference beat PyTorch's best settings by 2.4-2.9x (real PyTorch, paired rounds) ([#1332](https://github.com/ooples/AiDotNet.Tensors/issues/1332)) ([9ff5279](https://github.com/ooples/AiDotNet.Tensors/commit/9ff5279c418ab880618bb54ae6b1099e8a7a128d))
* **autodiff:** direct per-head attention, vectorized softmax for the fused mha node ([#1320](https://github.com/ooples/AiDotNet.Tensors/issues/1320)) ([8b47413](https://github.com/ooples/AiDotNet.Tensors/commit/8b4741356f3ee1691ea6c9ea7fb90b28129ee4bd))
* **autodiff:** fused float multi-head attention backward as one tape node ([#1307](https://github.com/ooples/AiDotNet.Tensors/issues/1307)) ([93679cf](https://github.com/ooples/AiDotNet.Tensors/commit/93679cf95072aa37e883a0da23dc824888285946))
* **compile:** plain SGD takes the one-launch, device-guarded multi-tensor path ([#1338](https://github.com/ooples/AiDotNet.Tensors/issues/1338)) ([1696304](https://github.com/ooples/AiDotNet.Tensors/commit/1696304b322822d5ee2fbb6ee851cce5311b0c54))
* **compile:** shared-operand matmul backward accumulates in place, column-bias broadcast without a materialized result ([#1303](https://github.com/ooples/AiDotNet.Tensors/issues/1303)) ([83782c4](https://github.com/ooples/AiDotNet.Tensors/commit/83782c451c274d1af0145d09b86f43043c8e9ab2))
* **conv:** parallel implicit-gemm conv; route k&gt;512 large-spatial convs to it ([#1305](https://github.com/ooples/AiDotNet.Tensors/issues/1305)) ([193e1ac](https://github.com/ooples/AiDotNet.Tensors/commit/193e1ac74979d9ac558fd9741e20c6a24459e44d))
* **cpu:** default the parallel width to physical cores, like pytorch ([#1333](https://github.com/ooples/AiDotNet.Tensors/issues/1333)) ([e431cf6](https://github.com/ooples/AiDotNet.Tensors/commit/e431cf63bc4937381bf9738c91dd100c760ba5ac))
* **cpu:** training-step gemm routing and lstm backward vs pytorch ([#1330](https://github.com/ooples/AiDotNet.Tensors/issues/1330)) ([d4e7352](https://github.com/ooples/AiDotNet.Tensors/commit/d4e7352c0ba71c0abfa1615f9e23a99a80e4f75d))
* **cuda-conv:** implicit-GEMM conv engine (forward + backward) with per-shape tuned routing ([#1336](https://github.com/ooples/AiDotNet.Tensors/issues/1336)) ([6c7df8f](https://github.com/ooples/AiDotNet.Tensors/commit/6c7df8f52c6d978a2a6ed85494b0e062feb7124d))
* **cuda-conv:** implicit-GEMM Conv2D backward (weight and input gradients) without im2col ([#1323](https://github.com/ooples/AiDotNet.Tensors/issues/1323)) ([614981a](https://github.com/ooples/AiDotNet.Tensors/commit/614981a8734b76046e081fac4ab86ed38f476dad))
* **cuda-conv:** tensor-core mainloops (3xTF32, opt-in TF32) as measured conv-engine candidates ([#1344](https://github.com/ooples/AiDotNet.Tensors/issues/1344)) ([bc58001](https://github.com/ooples/AiDotNet.Tensors/commit/bc5800154c09beaeff326efda063ffaebf22d4d7))
* **cuda:** LSTM and Transformer training - GEMM-form LSTM, fused attention node, sync-free compiled step ([#1274](https://github.com/ooples/AiDotNet.Tensors/issues/1274)) ([4e9f54a](https://github.com/ooples/AiDotNet.Tensors/commit/4e9f54a9311be0260b84a63831958fb7a501a2e3))
* **gemm:** split-k and tall-m routes for backward-pass gemms ([#1319](https://github.com/ooples/AiDotNet.Tensors/issues/1319)) ([c3cee65](https://github.com/ooples/AiDotNet.Tensors/commit/c3cee65dd01482fb8391481849b0022019498b68))
* **gpu:** BatchNorm converts its saved statistic in one launch each way and skips the dead running-stat fills ([#1340](https://github.com/ooples/AiDotNet.Tensors/issues/1340)) ([a6e092f](https://github.com/ooples/AiDotNet.Tensors/commit/a6e092f8afaa305f0e0c46df93676cf541e6d9f8))
* **hip:** run clblast's indirect gemm from 64, not from opencl's 448 ([#1334](https://github.com/ooples/AiDotNet.Tensors/issues/1334)) ([7214c57](https://github.com/ooples/AiDotNet.Tensors/commit/7214c5738db168dc8e270c50878b7c072821c49c)), closes [#1266](https://github.com/ooples/AiDotNet.Tensors/issues/1266)
* parallel cpu adam and hash-encoding capture in training traces ([#1341](https://github.com/ooples/AiDotNet.Tensors/issues/1341)) ([381fffc](https://github.com/ooples/AiDotNet.Tensors/commit/381fffc69a97bb43146d5267b1c0214719e31118))
* **parity:** fp32 PyTorch side, multi-tensor SGD, ResNet-18 CUDA baseline ([#1317](https://github.com/ooples/AiDotNet.Tensors/issues/1317)) ([a24c082](https://github.com/ooples/AiDotNet.Tensors/commit/a24c08283b7b8f609f08008923581a7a02034a4f))
* **ptx:** direct-PTX FP32 CSR SpMM (experimental) + [#852](https://github.com/ooples/AiDotNet.Tensors/issues/852) sparse coverage manifest ([#1312](https://github.com/ooples/AiDotNet.Tensors/issues/1312)) ([f2dc89c](https://github.com/ooples/AiDotNet.Tensors/commit/f2dc89ca36a0e90a2704fad9c16eb4d25adc96d5))


### Refactoring

* **gpu-sparse:** remove cuSPARSE from the library ([#1308](https://github.com/ooples/AiDotNet.Tensors/issues/1308)) ([beb11dc](https://github.com/ooples/AiDotNet.Tensors/commit/beb11dc91042ac8ac9b99e5f927387ebb161f840))

## [0.138.1](https://github.com/ooples/AiDotNet.Tensors/compare/v0.138.0...v0.138.1) (2026-10-14)


### Bug Fixes

* **#1293:** keep chunk ownership in the parallel executor (1M elementwise ops 4-11x faster than 0.138.0), steal only unstarted chunks ([#1294](https://github.com/ooples/AiDotNet.Tensors/issues/1294)) ([9cf0b43](https://github.com/ooples/AiDotNet.Tensors/commit/9cf0b43b72ecc79dd588c8c4d71c3a36b042be1b))
* **arena:** clear autograd state when the tensor ring reissues a tensor ([#1301](https://github.com/ooples/AiDotNet.Tensors/issues/1301)) ([8a3f409](https://github.com/ooples/AiDotNet.Tensors/commit/8a3f409f9aba9bb2c0a2f97ba4e8a03cb604bb04))
* **autodiff:** multi-head attention forward lost every gradient under an eager tape ([#1290](https://github.com/ooples/AiDotNet.Tensors/issues/1290)) ([e0cacb9](https://github.com/ooples/AiDotNet.Tensors/commit/e0cacb975d1843663d195ce487e7efd6f8afbf3a))
* **gpu-sparse:** fix cuSPARSE bindings, freeze the GPU SpMM bar, share one CUDA backend ([#1297](https://github.com/ooples/AiDotNet.Tensors/issues/1297)) ([486e0f9](https://github.com/ooples/AiDotNet.Tensors/commit/486e0f974f61251872bd7de10afc1f7448019beb))
* **hip:** make the managed sparse kernels run and route them ahead of rocsparse ([#1289](https://github.com/ooples/AiDotNet.Tensors/issues/1289)) ([e361613](https://github.com/ooples/AiDotNet.Tensors/commit/e36161367416697e6da574302aa7b9143f79540c))


### Performance

* **#1263:** evolve the generated PTX GEMM with AiDotNet.Evolution ([#1291](https://github.com/ooples/AiDotNet.Tensors/issues/1291)) ([08c0cf8](https://github.com/ooples/AiDotNet.Tensors/commit/08c0cf80088d4756791156c23149e225c3efc7d9))
* **#1264:** Tensor-Core PTX GEMM (TF32 and FP16), evolved; TF32 parity with cuBLAS at 4096^3 ([#1296](https://github.com/ooples/AiDotNet.Tensors/issues/1296)) ([70d1e72](https://github.com/ooples/AiDotNet.Tensors/commit/70d1e723419e5f26c3699ba27f7ed6c9c34a7cc2))
* **cuda-attention:** walk causal grids heaviest-first across every head ([#1300](https://github.com/ooples/AiDotNet.Tensors/issues/1300)) ([f88d978](https://github.com/ooples/AiDotNet.Tensors/commit/f88d978b646e6eebb1893f655e3f019889a0e2e8)), closes [#1184](https://github.com/ooples/AiDotNet.Tensors/issues/1184)
* **hip:** run clblast's indirect gemm on hip, 3-8x the hip kernels ([#1288](https://github.com/ooples/AiDotNet.Tensors/issues/1288)) ([24ddfaa](https://github.com/ooples/AiDotNet.Tensors/commit/24ddfaab4e28281d8a24ca1d8b28dff2d9fb156b))

## [0.138.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.137.1...v0.138.0) (2026-10-14)


### Features

* **#1262:** parameterized direct-PTX SGEMM generator with runtime shapes ([#1283](https://github.com/ooples/AiDotNet.Tensors/issues/1283)) ([ce7e55a](https://github.com/ooples/AiDotNet.Tensors/commit/ce7e55a6da683b5902a856a22e9608ea9b0e2b23))
* **curand:** generate MRG32k3a and Sobol32 uniforms on the CUDA device ([#1273](https://github.com/ooples/AiDotNet.Tensors/issues/1273)) ([cb44cb7](https://github.com/ooples/AiDotNet.Tensors/commit/cb44cb7151f3e8725245ec80005350af4fd7d814))


### Bug Fixes

* **#1285:** tuning rejections carry compilation, resource and benchmark statuses ([#1286](https://github.com/ooples/AiDotNet.Tensors/issues/1286)) ([510049b](https://github.com/ooples/AiDotNet.Tensors/commit/510049b004d9c8843896b75dd620c500cd88462d))
* **autodiff:** differentiate TensorGatherClassValues' backward under create-graph ([#1278](https://github.com/ooples/AiDotNet.Tensors/issues/1278)) ([b6aaae2](https://github.com/ooples/AiDotNet.Tensors/commit/b6aaae25d689cab7580c490925e6009a6964119e))
* **autodiff:** seed the graph backward with the loss's own shape ([#1287](https://github.com/ooples/AiDotNet.Tensors/issues/1287)) ([e1bc334](https://github.com/ooples/AiDotNet.Tensors/commit/e1bc33421db2ebffd6ebe75333677dc451ec6c81))
* **conv:** make the implicit-vs-full conv parity test compare the two routes ([#1279](https://github.com/ooples/AiDotNet.Tensors/issues/1279)) ([53f9740](https://github.com/ooples/AiDotNet.Tensors/commit/53f9740eabfa0db67b0e524cec135907b5db3957))
* page a dropped streaming weight back in before GetFlattenedData reads it ([#1277](https://github.com/ooples/AiDotNet.Tensors/issues/1277)) ([96096cd](https://github.com/ooples/AiDotNet.Tensors/commit/96096cdefdc68ae22f17536ecf926011cd416948))
* **spectral:** compute widebandlogbinpool bin edges in exact integers everywhere ([#1280](https://github.com/ooples/AiDotNet.Tensors/issues/1280)) ([c6b5025](https://github.com/ooples/AiDotNet.Tensors/commit/c6b5025985465d1bae6f1421643e625837be5e80))

## [0.137.1](https://github.com/ooples/AiDotNet.Tensors/compare/v0.137.0...v0.137.1) (2026-10-13)


### Bug Fixes

* **arena:** keep arena-backed tensors out of TensorPool ([#1275](https://github.com/ooples/AiDotNet.Tensors/issues/1275)) ([b0319db](https://github.com/ooples/AiDotNet.Tensors/commit/b0319db415c92c0498f3111132a379433672adda))

## [0.137.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.136.0...v0.137.0) (2026-10-13)


### Features

* **#1116:** Metal through the tuned-kernel registry, from a template shared with OpenCL ([#1255](https://github.com/ooples/AiDotNet.Tensors/issues/1255)) ([820ad93](https://github.com/ooples/AiDotNet.Tensors/commit/820ad938d2d3fd9d518a4018c087322747567109))
* **#1216:** direct FP16 convolution kernel on Vulkan ([#1217](https://github.com/ooples/AiDotNet.Tensors/issues/1217)) ([37b56e2](https://github.com/ooples/AiDotNet.Tensors/commit/37b56e2ebfb0e9794c8304b8645cb57ff6f2c2a2))
* **#974:** fused decode-linear routes on every GPU backend ([#1219](https://github.com/ooples/AiDotNet.Tensors/issues/1219)) ([c106175](https://github.com/ooples/AiDotNet.Tensors/commit/c10617534c22080a51bfdb85ab542dd16385713b))
* **rng:** MRG32k3a on every backend, bit-equal to cuRAND ([#1251](https://github.com/ooples/AiDotNet.Tensors/issues/1251)) ([1cfd24b](https://github.com/ooples/AiDotNet.Tensors/commit/1cfd24be96195a94eac907e1cebf0e177d2f27bd)), closes [#1250](https://github.com/ooples/AiDotNet.Tensors/issues/1250)
* **rng:** Sobol and seeded scrambled Sobol on every backend ([#1259](https://github.com/ooples/AiDotNet.Tensors/issues/1259)) ([7ed7721](https://github.com/ooples/AiDotNet.Tensors/commit/7ed772100b1d755337d3dc82821ae5d3d9766bb9)), closes [#1257](https://github.com/ooples/AiDotNet.Tensors/issues/1257) [#1258](https://github.com/ooples/AiDotNet.Tensors/issues/1258)
* target net462 and netstandard2.0 (Tensors + Onnx) ([#1243](https://github.com/ooples/AiDotNet.Tensors/issues/1243)) ([55bfa58](https://github.com/ooples/AiDotNet.Tensors/commit/55bfa5830f48fe2f6d82afb43a421f31758ff471))


### Bug Fixes

* **#1211:** reduce over the requested axis in MaxOverAxis / SumOverAxisDirect / MeanOverAxisDirect ([#1215](https://github.com/ooples/AiDotNet.Tensors/issues/1215)) ([d843df5](https://github.com/ooples/AiDotNet.Tensors/commit/d843df5c027a8643f110b89ec9bcecd518a37bf5))
* **#1221:** restore the thread's exact codec options and determinism override after compile/import ([#1226](https://github.com/ooples/AiDotNet.Tensors/issues/1226)) ([6724751](https://github.com/ooples/AiDotNet.Tensors/commit/67247513f310f230e82cffcdce94b18c5cbaefa5))
* **#1223:** let tools/ConvParallelProbe see the library's internals so gemm-bench builds ([#1224](https://github.com/ooples/AiDotNet.Tensors/issues/1224)) ([1bfab31](https://github.com/ooples/AiDotNet.Tensors/commit/1bfab318e23e26e2da1423642b3de1569df53493))
* **#1235:** version-guard the sparse-linear CSR snapshot; ONNX plans name their output (MatMul/Gemm imports returned zeros) ([#1240](https://github.com/ooples/AiDotNet.Tensors/issues/1240)) ([753ae14](https://github.com/ooples/AiDotNet.Tensors/commit/753ae1403172be3f244dcb61eda3eebff8119b1a))
* **build:** grant AiDotNet benchmark and serving assemblies internals access ([#1249](https://github.com/ooples/AiDotNet.Tensors/issues/1249)) ([1c27ccf](https://github.com/ooples/AiDotNet.Tensors/commit/1c27ccfb7f5e36140e6db614a496801386b33b09))
* compile the gpu kernels backends looked up but never loaded, and the bugs that hid ([#1260](https://github.com/ooples/AiDotNet.Tensors/issues/1260)) ([90ed603](https://github.com/ooples/AiDotNet.Tensors/commit/90ed6034e77adb03c455d1e4918fdcd3ef9156c8))
* **gpu:** PowerToDb and DbToPower take a power reference on every backend ([#1237](https://github.com/ooples/AiDotNet.Tensors/issues/1237)) ([b2c95e2](https://github.com/ooples/AiDotNet.Tensors/commit/b2c95e26bf1fc07aa371e759b1f8cb7744c79bb8)), closes [#1236](https://github.com/ooples/AiDotNet.Tensors/issues/1236)
* **parallel:** claim plain dispatch chunks so a starved worker cannot stall it ([#1267](https://github.com/ooples/AiDotNet.Tensors/issues/1267)) ([80df3fd](https://github.com/ooples/AiDotNet.Tensors/commit/80df3fdc06accb56ea631766af79a1826cb538f9))
* **webgpu:** run the real backend on Dawn in tests, and fix the JS glue so parameterized kernels dispatch ([#1245](https://github.com/ooples/AiDotNet.Tensors/issues/1245)) ([9a8ca15](https://github.com/ooples/AiDotNet.Tensors/commit/9a8ca1552c7bebb7224be3fb1078e40fabb957e0))
* **webgpu:** run the remaining WebGPU tests on the Dawn-hosted backend and fix what they found ([#1272](https://github.com/ooples/AiDotNet.Tensors/issues/1272)) ([a500ef9](https://github.com/ooples/AiDotNet.Tensors/commit/a500ef9ee1a7b1114489d5e580d9208f33aba12a)), closes [#1271](https://github.com/ooples/AiDotNet.Tensors/issues/1271)


### Performance

* **#1113:** hip serves softmax, layernorm, rmsnorm and conv2d through the tuned-kernel registry ([#1252](https://github.com/ooples/AiDotNet.Tensors/issues/1252)) ([54da857](https://github.com/ooples/AiDotNet.Tensors/commit/54da857c263f21357100aaa4a02ce6334c1c2537))
* **#1192:** coalesce Vulkan dispatches into one submission (64-op chains 33 -&gt; 17.5 µs per op) ([#1225](https://github.com/ooples/AiDotNet.Tensors/issues/1225)) ([fd49e4a](https://github.com/ooples/AiDotNet.Tensors/commit/fd49e4a5200f57371cdb7d80aa34cc8742bed88c))
* **#1218:** pack A/B, clear C and run the M/N edges in parallel in the machine-kernel GEMM (FP64 1000³ 16.2 -&gt; 10.2 ms) ([#1222](https://github.com/ooples/AiDotNet.Tensors/issues/1222)) ([5276a4d](https://github.com/ooples/AiDotNet.Tensors/commit/5276a4dfe0ec83deb8f8548b0d72ca9a48298b12))
* **#1241:** zero large rented buffers in parallel (Softmax_ZeroAlloc 363 -&gt; 198 µs) ([#1242](https://github.com/ooples/AiDotNet.Tensors/issues/1242)) ([a23db0a](https://github.com/ooples/AiDotNet.Tensors/commit/a23db0aa7788fb7ef3cb46da649c620fcf1ab473))
* **blas:** symm reads the stored triangle in place for a narrow panel ([#1253](https://github.com/ooples/AiDotNet.Tensors/issues/1253)) ([45536e4](https://github.com/ooples/AiDotNet.Tensors/commit/45536e477a08ee7875c3c5590f866aaa69cc88b7)), closes [#517](https://github.com/ooples/AiDotNet.Tensors/issues/517)
* **conv:** use the full-matrix im2col gemm for high-channel 3x3 convs at 8+ threads ([#1270](https://github.com/ooples/AiDotNet.Tensors/issues/1270)) ([14a91a1](https://github.com/ooples/AiDotNet.Tensors/commit/14a91a195f7cfce15dba1641bfc98239022c67e0))
* **registry:** tuned-kernel registry dispatch on webgpu ([#1268](https://github.com/ooples/AiDotNet.Tensors/issues/1268)) ([a74fe76](https://github.com/ooples/AiDotNet.Tensors/commit/a74fe76b45f330ac67f5e9497a1d4e7476f6ed94))


### Build System

* restore AiDotNet.Tensors.Onnx before its net462 build, since the solution does not include it ([55bfa58](https://github.com/ooples/AiDotNet.Tensors/commit/55bfa5830f48fe2f6d82afb43a421f31758ff471))

## [0.136.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.135.0...v0.136.0) (2026-10-12)


### Features

* **#1135:** bitwise / shift / gcd / lcm / floor_divide on int64 and narrow integer tensors on the GPU ([#1144](https://github.com/ooples/AiDotNet.Tensors/issues/1144)) ([69fa161](https://github.com/ooples/AiDotNet.Tensors/commit/69fa161366331f034ac302aaf74fa2a1303a0703))
* **#1165 phase 2:** measure GPU capability fallbacks; WebGPU LoRA takes any rank ([#1208](https://github.com/ooples/AiDotNet.Tensors/issues/1208)) ([3c72c43](https://github.com/ooples/AiDotNet.Tensors/commit/3c72c433117917f45e7aca0f622a916509f0783b))
* **#889:** weightless tape-aware fused SDPA (FlashAttention-2 fwd/bwd, mask/causal/softcap in-kernel) -- O(S) training memory ([#1139](https://github.com/ooples/AiDotNet.Tensors/issues/1139)) ([d503c9a](https://github.com/ooples/AiDotNet.Tensors/commit/d503c9a98412b4a0c1eff97b970b7e7289ee5428))
* **gpu:** enforce six-backend parity, implement the gaps, and five issues ([#1167](https://github.com/ooples/AiDotNet.Tensors/issues/1167)) ([1fbe3d2](https://github.com/ooples/AiDotNet.Tensors/commit/1fbe3d26e8ae2376ddc77ed04e5d1643ce363506))
* **opencl:** device kernels for the pytorch-parity ops; vulkan device features ([#1175](https://github.com/ooples/AiDotNet.Tensors/issues/1175)) ([7bdb578](https://github.com/ooples/AiDotNet.Tensors/commit/7bdb578ec824ac7d600423d243d2407c67bf20b1))
* **parity:** close all 387 PyTorch gaps; fix in-place autograd and Fill on views ([#1120](https://github.com/ooples/AiDotNet.Tensors/issues/1120)) ([4e7183a](https://github.com/ooples/AiDotNet.Tensors/commit/4e7183a7fc7096f99d8c184986942c74f1799ed6))
* tuned-kernel registry dispatch on opencl and vulkan (softmax, layernorm, conv2d) ([#1168](https://github.com/ooples/AiDotNet.Tensors/issues/1168)) ([398c478](https://github.com/ooples/AiDotNet.Tensors/commit/398c478a945c1537dfe936f714aa33a1361f14d9))
* **vulkan:** device kernels for the PyTorch-parity ops ([#1149](https://github.com/ooples/AiDotNet.Tensors/issues/1149)) ([a0d8389](https://github.com/ooples/AiDotNet.Tensors/commit/a0d8389e266bd1059f889aa7b21c2ce63e65d46f))
* **vulkan:** generic-precision, arbitrary-length FFT ([#1151](https://github.com/ooples/AiDotNet.Tensors/issues/1151)) ([de2bc5e](https://github.com/ooples/AiDotNet.Tensors/commit/de2bc5e43203225180789dae647ee10c1facab9f))


### Bug Fixes

* **#1131, #1134:** CPU sort in PyTorch's NaN/stable order; non-finite GridSample points sample nothing on CPU and CUDA ([#1143](https://github.com/ooples/AiDotNet.Tensors/issues/1143)) ([161a262](https://github.com/ooples/AiDotNet.Tensors/commit/161a2620222afc7579583b5c6209a6b247f068dd))
* **#1165 phase 1:** GPU dispatch falls back only for missing capabilities; faults propagate ([#1203](https://github.com/ooples/AiDotNet.Tensors/issues/1203)) ([7cd7f21](https://github.com/ooples/AiDotNet.Tensors/commit/7cd7f21b58879de6c8ef06cc920b0a2b8729a217))
* **#1172:** implement the int64-layout parity ops on Vulkan so main compiles ([#1173](https://github.com/ooples/AiDotNet.Tensors/issues/1173)) ([c40c295](https://github.com/ooples/AiDotNet.Tensors/commit/c40c2952bd980abd84b812bd82c02281741805c6))
* **#1196:** Vulkan LDL factorization picks the reference pivots (volatile buffers in the translated solvers) ([#1197](https://github.com/ooples/AiDotNet.Tensors/issues/1197)) ([9579ca3](https://github.com/ooples/AiDotNet.Tensors/commit/9579ca3cb929f3882aa393f622e4b5b883a25e7c))
* **#1199:** make main build again (stale parity baseline entry, concrete-backend test) ([#1200](https://github.com/ooples/AiDotNet.Tensors/issues/1200)) ([a38a385](https://github.com/ooples/AiDotNet.Tensors/commit/a38a3859e93ffaa00ef70483b1ac8df9f542ef9e))
* **alloc:** keep this alive in storage readers; release escaped arrays; net471 sin/cos reference ([#1187](https://github.com/ooples/AiDotNet.Tensors/issues/1187)) ([3764a7d](https://github.com/ooples/AiDotNet.Tensors/commit/3764a7d93c52f298a1d6dc71d5ccc49137726bf3))
* compute erf and erfc to double precision ([#1180](https://github.com/ooples/AiDotNet.Tensors/issues/1180)) ([bbb00b7](https://github.com/ooples/AiDotNet.Tensors/commit/bbb00b7a954f90cf686ecbac005c11d95cd93507))
* **conv:** stale transposed kernel in the stride-1 conv input gradient on Linux (0.135.0 regression) ([#1146](https://github.com/ooples/AiDotNet.Tensors/issues/1146)) ([8eb97ab](https://github.com/ooples/AiDotNet.Tensors/commit/8eb97abb6ea15dbbcfa67071b6a4b345f0688bc1))
* **cuda:** ReduceSumBackward / ReduceMeanBackward launch one thread per element ([#1195](https://github.com/ooples/AiDotNet.Tensors/issues/1195)) ([4225158](https://github.com/ooples/AiDotNet.Tensors/commit/4225158b42637632380664371b8f8b8839dc39fc))
* full reductions return rank 0 on cpu and gpu, eager and compiled ([#1145](https://github.com/ooples/AiDotNet.Tensors/issues/1145)) ([bea7504](https://github.com/ooples/AiDotNet.Tensors/commit/bea7504db2dca1fc47f3e77534b8838273c9dcc4))
* **gpu:** device-side complex split, 12 parity elementwise kernels, scale-aware GPU/CPU comparison ([#1112](https://github.com/ooples/AiDotNet.Tensors/issues/1112), part 1) ([#1121](https://github.com/ooples/AiDotNet.Tensors/issues/1121)) ([18a119b](https://github.com/ooples/AiDotNet.Tensors/commit/18a119b998353d830a8648cc2bc1acce32c6c127))
* **gpu:** unbreak main's parity build, fused attention and non-finite grid sample on every backend ([#1181](https://github.com/ooples/AiDotNet.Tensors/issues/1181)) ([680df86](https://github.com/ooples/AiDotNet.Tensors/commit/680df860b7cc92e5d104785e67baa7f072e16fd0))
* **vulkan:** binary gcd and 8-byte-aligned 64-bit push pairs on amd ([#1174](https://github.com/ooples/AiDotNet.Tensors/issues/1174)) ([21a52e4](https://github.com/ooples/AiDotNet.Tensors/commit/21a52e4d4eafe45a09b2506a5882ec3ea09c424e))
* **vulkan:** pass every push-constant word to the fused kernels ([#1147](https://github.com/ooples/AiDotNet.Tensors/issues/1147)) ([#1148](https://github.com/ooples/AiDotNet.Tensors/issues/1148)) ([91d10b6](https://github.com/ooples/AiDotNet.Tensors/commit/91d10b6a70f20578333a3cbbb7b40e47decdc592))


### Performance

* **#1141 phase 1:** tiled fused CUDA attention for SDPA and MultiHeadAttentionCore -- 11-35x faster, near PyTorch ([#1159](https://github.com/ooples/AiDotNet.Tensors/issues/1159)) ([2ce4420](https://github.com/ooples/AiDotNet.Tensors/commit/2ce442044065d0a1ea71221fd1e5c553743dee79))
* **#1141 phase 2:** fused CUDA attention backward with atomic dQ (deterministic two-pass under DeterministicReductions) ([#1182](https://github.com/ooples/AiDotNet.Tensors/issues/1182)) ([26f3121](https://github.com/ooples/AiDotNet.Tensors/commit/26f3121a254132935a36b22ec9d5f1ddaee6e927))
* **#1150:** split a GPU full ReduceSum across blocks -- 37.8 ms to 0.24 ms for 33.5M floats, on par with PyTorch ([#1153](https://github.com/ooples/AiDotNet.Tensors/issues/1153)) ([d2a86ae](https://github.com/ooples/AiDotNet.Tensors/commit/d2a86ae9526b0c10b34b182b458ad0bc8c46ab6c))
* **#1157:** SumAxis takes a work-group per long row and splits few long rows on OpenCL, HIP, Metal and WebGPU ([#1209](https://github.com/ooples/AiDotNet.Tensors/issues/1209)) ([3e9dc2c](https://github.com/ooples/AiDotNet.Tensors/commit/3e9dc2c3f4ee16ef0c75c87ae36f2c8686739483))
* **#1164:** Vulkan max pooling without indices is 9-19x faster (no per-call placeholder buffer) ([#1166](https://github.com/ooples/AiDotNet.Tensors/issues/1166)) ([11b2967](https://github.com/ooples/AiDotNet.Tensors/commit/11b2967e973de610ee14dba22e25698f3907e1bd))
* **#1169:** Vulkan device-buffer pool (allocation 0.7-6 ms -&gt; under 1 µs; ops that allocate their result 11-52x faster) ([#1170](https://github.com/ooples/AiDotNet.Tensors/issues/1170)) ([02b9bd2](https://github.com/ooples/AiDotNet.Tensors/commit/02b9bd28170dab5c02ba921472d3c41143df756f))
* **#1176:** Vulkan pipeline lookup without hashing the shader source (2-4 µs -&gt; ~50 ns per op) ([#1177](https://github.com/ooples/AiDotNet.Tensors/issues/1177)) ([37d1fbd](https://github.com/ooples/AiDotNet.Tensors/commit/37d1fbd1b779572e3714520700b282cce5ddb5ba))
* **#1183:** collect on a Vulkan pool miss when no tape is open (cold stretch ~2,000 -&gt; ~256 allocations) ([#1185](https://github.com/ooples/AiDotNet.Tensors/issues/1185)) ([cbc0054](https://github.com/ooples/AiDotNet.Tensors/commit/cbc0054310a6e5f3777142c726eb4508b36a64de))
* **#1188:** Vulkan 256-lane row variants (6-9x on 4,096-column rows), RTX 3080 profile, device-clock gate timing ([#1189](https://github.com/ooples/AiDotNet.Tensors/issues/1189)) ([0c85149](https://github.com/ooples/AiDotNet.Tensors/commit/0c851494769905101adbbdc0b2172e2f093337d6))
* **#1190:** a pending Vulkan dispatch keeps its own descriptor set (batches no longer flush when a kernel repeats) ([#1193](https://github.com/ooples/AiDotNet.Tensors/issues/1193)) ([bd929b5](https://github.com/ooples/AiDotNet.Tensors/commit/bd929b525e73e83c02471f80c171e0c7b1360c5d))
* **#1191:** Vulkan dispatches are submitted without waiting (small-op chains 107 -&gt; 35 µs per op) ([#1214](https://github.com/ooples/AiDotNet.Tensors/issues/1214)) ([659acc9](https://github.com/ooples/AiDotNet.Tensors/commit/659acc93dc05e9ed04057cec97cb16f59221e615))
* **#1198:** Vulkan full-tensor sum 1,338 ms -&gt; 0.5 ms for 33.5M floats (long rows split across workgroups) ([#1201](https://github.com/ooples/AiDotNet.Tensors/issues/1201)) ([394e072](https://github.com/ooples/AiDotNet.Tensors/commit/394e0727b9ee7831382f2c2ff9c17cf9e61a31fd))
* **#475:** MLP layers dispatch the parallel managed GEMM (AIsEval MLP 1.5 ms -&gt; 0.23 ms, torch 0.33 ms) ([#1210](https://github.com/ooples/AiDotNet.Tensors/issues/1210)) ([5f50e27](https://github.com/ooples/AiDotNet.Tensors/commit/5f50e2777796e6f77a6e9620c618aec0f8f44bd2)), closes [#475](https://github.com/ooples/AiDotNet.Tensors/issues/475)
* **#639:** float depthwise conv and direct pointwise gradients -- every MobileNetV3 conv phase beats PyTorch on CPU ([#1140](https://github.com/ooples/AiDotNet.Tensors/issues/1140)) ([7af33e1](https://github.com/ooples/AiDotNet.Tensors/commit/7af33e10d2e7898ff4557021038337bc5797226a))
* **#653 phase 2:** fan out broadcast, reductions, ELU, Concat and strided copies -- bit-identical at any DOP ([#1137](https://github.com/ooples/AiDotNet.Tensors/issues/1137)) ([78b68de](https://github.com/ooples/AiDotNet.Tensors/commit/78b68dec4c485989dea9a2289f829ff68d1aed26))
* **#653 phase 3:** fan out norm backward and CPU attention across cores, bit-identical at any thread count ([#1178](https://github.com/ooples/AiDotNet.Tensors/issues/1178)) ([36b49ff](https://github.com/ooples/AiDotNet.Tensors/commit/36b49ffb622216c48b174c2f038352ac0f374cb4))
* **alloc:** recycle undisposed large result buffers (tracked owners, escape flag, leases) ([#1118](https://github.com/ooples/AiDotNet.Tensors/issues/1118)) ([0eb9c9b](https://github.com/ooples/AiDotNet.Tensors/commit/0eb9c9b7244c5810136a1ee274b0ff66bff29506))
* **cpu:** run the float elementwise dispatchers on every platform ([#1130](https://github.com/ooples/AiDotNet.Tensors/issues/1130)) ([8f2cc99](https://github.com/ooples/AiDotNet.Tensors/commit/8f2cc999ed6f218a8ff9346f94f4b788e897ba9c))
* **cuda:** one-launch last-axis slices and no zero-fill on fully written outputs -- transformer step 0.7x PyTorch ([#1138](https://github.com/ooples/AiDotNet.Tensors/issues/1138)) ([2b3c718](https://github.com/ooples/AiDotNet.Tensors/commit/2b3c718f66fe2f87453babd62342f6fa3c6609f5))
* **vulkan:** fused two-vector Meshgrid2D for the admitted shapes ([#1152](https://github.com/ooples/AiDotNet.Tensors/issues/1152)) ([7399786](https://github.com/ooples/AiDotNet.Tensors/commit/73997865df856a623ffe78185efc288e160e6baf))

## [0.135.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.134.4...v0.135.0) (2026-10-10)


### Features

* **parity:** close the pytorch api gaps reported by fullparity_nogapremains ([#1111](https://github.com/ooples/AiDotNet.Tensors/issues/1111)) ([351d46c](https://github.com/ooples/AiDotNet.Tensors/commit/351d46c7b9fa6075fd2585991ec7598540cb5d70))


### Bug Fixes

* arena gradients, nchw grid sample, atanh; perf: short-m gemm routing, write-through into ops ([#1098](https://github.com/ooples/AiDotNet.Tensors/issues/1098)) ([ec52dad](https://github.com/ooples/AiDotNet.Tensors/commit/ec52dad2e25697720047921881280391da4dd444))
* **autodiff:** tape step context accepts sparse parameter-buffer leaves ([#1102](https://github.com/ooples/AiDotNet.Tensors/issues/1102)) ([f25afe9](https://github.com/ooples/AiDotNet.Tensors/commit/f25afe9a9d0ace6ca7697006bf848b97ab04403c))
* **codegen:** validate CPU kernel contracts and preserve IEEE signs ([#1110](https://github.com/ooples/AiDotNet.Tensors/issues/1110)) ([92670df](https://github.com/ooples/AiDotNet.Tensors/commit/92670dff606c12987ce8e3868ea8787b943f4347))
* **cpu:** float elementwise dispatch ran serially above the parallel grain ([#1119](https://github.com/ooples/AiDotNet.Tensors/issues/1119)) ([9f7bcb2](https://github.com/ooples/AiDotNet.Tensors/commit/9f7bcb2120e513e96d7dfe45ac09a9386c35f39c))
* hip broke vulkan loading; perf: fp16-weight gemm, vectorized tanh and mish; real avx-512 gate ([#1103](https://github.com/ooples/AiDotNet.Tensors/issues/1103)) ([44ca3f0](https://github.com/ooples/AiDotNet.Tensors/commit/44ca3f0c2b899c43f171e1131c904b28a9ff64bb))


### Performance

* **cpu:** all five cpu parity gates pass; zero-copy optimizer api (cpu + gpu); real zero-1 ([#1109](https://github.com/ooples/AiDotNet.Tensors/issues/1109)) ([356bf44](https://github.com/ooples/AiDotNet.Tensors/commit/356bf44f6617adc2eeb86b2d69c24ef83032fcee))
* **cpu:** parallel conv, bias-gradient and pool paths for CPU training ([#1093](https://github.com/ooples/AiDotNet.Tensors/issues/1093)) ([dc1ce96](https://github.com/ooples/AiDotNet.Tensors/commit/dc1ce96439b9c73904661de35d0369801a4b6b03))
* fair NumSharp 0.70 benchmarks + allocator GC, GPU-reduction, transpose and subtract fixes ([#1094](https://github.com/ooples/AiDotNet.Tensors/issues/1094)) ([#1101](https://github.com/ooples/AiDotNet.Tensors/issues/1101)) ([ed13d28](https://github.com/ooples/AiDotNet.Tensors/commit/ed13d286c8e9faed352df5f1a012495fb5b272a5))

## [0.134.4](https://github.com/ooples/AiDotNet.Tensors/compare/v0.134.3...v0.134.4) (2026-10-03)


### Bug Fixes

* **arena:** key the tensor ring by element type as well as count ([#1088](https://github.com/ooples/AiDotNet.Tensors/issues/1088)) ([b0d4b64](https://github.com/ooples/AiDotNet.Tensors/commit/b0d4b64911337d048a6ed107d9a101c435002fd3))

## [0.134.3](https://github.com/ooples/AiDotNet.Tensors/compare/v0.134.2...v0.134.3) (2026-10-02)


### Bug Fixes

* **engines:** engine tanh copies returned nan above x ~44 in float ([#1087](https://github.com/ooples/AiDotNet.Tensors/issues/1087)) ([4d783dd](https://github.com/ooples/AiDotNet.Tensors/commit/4d783dd7ca970685f55b33603875d841a148b25f))
* **helpers:** mathhelper.tanh returned nan above x ~44 in float ([#1085](https://github.com/ooples/AiDotNet.Tensors/issues/1085)) ([23a5c67](https://github.com/ooples/AiDotNet.Tensors/commit/23a5c674b1f8a5eedf60680036660d5354e2c5c5))

## [0.134.2](https://github.com/ooples/AiDotNet.Tensors/compare/v0.134.1...v0.134.2) (2026-10-02)


### Bug Fixes

* **compilation:** keep a double plan's optimizer hyperparameters in double ([#1083](https://github.com/ooples/AiDotNet.Tensors/issues/1083)) ([16d3ddc](https://github.com/ooples/AiDotNet.Tensors/commit/16d3ddcb010bd946244879d22c64ecb99d5ff5a8))
* **engines:** ResetToCpu keeps a cpu engine that is already current ([#1082](https://github.com/ooples/AiDotNet.Tensors/issues/1082)) ([2f2f8cb](https://github.com/ooples/AiDotNet.Tensors/commit/2f2f8cb5de1075d52635064fe27121819f98ee08))
* **precision:** exact large-offset norms on CPU; preserve double/int/long/decimal on the GPU by default ([#1081](https://github.com/ooples/AiDotNet.Tensors/issues/1081)) ([5da848e](https://github.com/ooples/AiDotNet.Tensors/commit/5da848e3b8d702d90f8b4380c788b325b565d64a))

## [0.134.1](https://github.com/ooples/AiDotNet.Tensors/compare/v0.134.0...v0.134.1) (2026-10-01)


### Bug Fixes

* **compilation:** weight paper amsgrad gradients by the caller's exact 1 - beta ([#1079](https://github.com/ooples/AiDotNet.Tensors/issues/1079)) ([b6b909f](https://github.com/ooples/AiDotNet.Tensors/commit/b6b909fbdf8a00ec29c9ca2bc2e8e5f6a3240eb1))


### Performance

* **gpu:** no gen-2 GC pauses in GPU training loops ([#1078](https://github.com/ooples/AiDotNet.Tensors/issues/1078)) ([a44706b](https://github.com/ooples/AiDotNet.Tensors/commit/a44706b9c04804473cf186d2c6b45010d8d779ad))

## [0.134.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.133.2...v0.134.0) (2026-10-01)


### Features

* add paper-faithful amsgrad mode to the fused training plan ([#1077](https://github.com/ooples/AiDotNet.Tensors/issues/1077)) ([033a0d7](https://github.com/ooples/AiDotNet.Tensors/commit/033a0d760f36a244d99af74c0ecb12c9719e8aed))
* **compilation:** add an external learning rate the caller drives between steps ([#1075](https://github.com/ooples/AiDotNet.Tensors/issues/1075)) ([654842b](https://github.com/ooples/AiDotNet.Tensors/commit/654842bc426f8b19a19292f99e82e024dd444230))


### Performance

* **gpu:** a caching device allocator keeps a warm, fixed-shape training loop's device memory flat, with the pool bounded at a quarter of device memory ([#1074](https://github.com/ooples/AiDotNet.Tensors/issues/1074)) ([a9c0482](https://github.com/ooples/AiDotNet.Tensors/commit/a9c04823e1e2f71861e21e3fc547496a0fdaadb5))

## [0.133.2](https://github.com/ooples/AiDotNet.Tensors/compare/v0.133.1...v0.133.2) (2026-09-30)


### Bug Fixes

* **compile:** BatchNorm's running-statistics update uses the current step's batch statistics ([#1068](https://github.com/ooples/AiDotNet.Tensors/issues/1068)) ([2fe8826](https://github.com/ooples/AiDotNet.Tensors/commit/2fe88262f1de3d1339df789a3bd9ce4de5efd016))
* **gpu:** disposing a view no longer unbinds the device copy other views share ([#1072](https://github.com/ooples/AiDotNet.Tensors/issues/1072)) ([e907b5a](https://github.com/ooples/AiDotNet.Tensors/commit/e907b5a49fa6ef882a434a23a83e635432076c3a))
* **gpu:** L2 on the resident step, view residency and tape release after device-owned storage ([#1070](https://github.com/ooples/AiDotNet.Tensors/issues/1070)) ([d624085](https://github.com/ooples/AiDotNet.Tensors/commit/d624085ba3eaa24acb6e634766ac50a00d1660a0))
* **opencl:** revive the pocl gpu-parity job and cover the device training paths ([#1073](https://github.com/ooples/AiDotNet.Tensors/issues/1073)) ([7b2363d](https://github.com/ooples/AiDotNet.Tensors/commit/7b2363d3a5d90de6d4eb7eba09f9c5614b404978))
* **simd:** one sigmoid on every CPU vendor, and saturate infinities in the Padé kernel ([#1069](https://github.com/ooples/AiDotNet.Tensors/issues/1069)) ([c1827e6](https://github.com/ooples/AiDotNet.Tensors/commit/c1827e61f739b82a3aeb7a2bae529e8191bd9372))


### Performance

* **gpu:** device-resident training steps on every GPU backend, plus Vulkan shader and op fixes ([#1067](https://github.com/ooples/AiDotNet.Tensors/issues/1067)) ([010b9e5](https://github.com/ooples/AiDotNet.Tensors/commit/010b9e5bd643cf50c8d511f897c753b7c1f0edaa))

## [0.133.1](https://github.com/ooples/AiDotNet.Tensors/compare/v0.133.0...v0.133.1) (2026-09-29)


### Bug Fixes

* **compile:** accumulate a lower-rank broadcast operand's gradient; stop discarding pre-packed weights ([#1062](https://github.com/ooples/AiDotNet.Tensors/issues/1062)) ([3101ee7](https://github.com/ooples/AiDotNet.Tensors/commit/3101ee71e26967b00291cf7e87d69e731d10ee1b))
* **engine:** make TensorCopy write into the destination's live storage ([#1054](https://github.com/ooples/AiDotNet.Tensors/issues/1054)) ([589122c](https://github.com/ooples/AiDotNet.Tensors/commit/589122cd39fe345170c0f49646409b7b1d9cf179))
* optimizer formulas agree on every backend; failed graph captures no longer corrupt training; more backwards on the device ([#1065](https://github.com/ooples/AiDotNet.Tensors/issues/1065)) ([582b470](https://github.com/ooples/AiDotNet.Tensors/commit/582b470fb6a083320950d2afc4c91bd6272079b6))
* **optimizers:** dynamic-codebook 8-bit Adam, min_8bit_size, and a real LAMB on every backend ([#1061](https://github.com/ooples/AiDotNet.Tensors/issues/1061)) ([295b135](https://github.com/ooples/AiDotNet.Tensors/commit/295b1355610ff7a40aa89cc1f258e161135ea55f))

## [0.133.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.132.1...v0.133.0) (2026-09-27)


### Features

* **compilation:** export and import a compiled plan's fused optimizer state ([#1051](https://github.com/ooples/AiDotNet.Tensors/issues/1051)) ([b922a2b](https://github.com/ooples/AiDotNet.Tensors/commit/b922a2bcad820fe8d424ccd6723870133f8ceafb))


### Bug Fixes

* **compilation:** record the saved state graph-mode backwards read ([#1047](https://github.com/ooples/AiDotNet.Tensors/issues/1047)) ([0cb0459](https://github.com/ooples/AiDotNet.Tensors/commit/0cb0459d7f8a579be91b5ee872b39054369a853f))


### Performance

* **gpu:** keep the GLA scan and matmul-transposed/fused-linear backwards on the device during training ([#1050](https://github.com/ooples/AiDotNet.Tensors/issues/1050)) ([29e61a1](https://github.com/ooples/AiDotNet.Tensors/commit/29e61a14dd060316362f1715d4fb37dde778af3e))
* **gpu:** keep the remaining tape-bailing ops on the device ([#1053](https://github.com/ooples/AiDotNet.Tensors/issues/1053)) ([0cfdd72](https://github.com/ooples/AiDotNet.Tensors/commit/0cfdd72df9766a30dfb142d7a7d9bf16881b6769))
* **gpu:** keep training on the device — tape-time residency, device backwards, pooled/uninitialized/staged CUDA memory ([#1052](https://github.com/ooples/AiDotNet.Tensors/issues/1052)) ([ba3c2fd](https://github.com/ooples/AiDotNet.Tensors/commit/ba3c2fd8043a8beeea75de990bed5054842c021c))
* **gpu:** single-launch tile / concat / last-axis slice and batched IRFFT (up to ~100,000x) ([#1048](https://github.com/ooples/AiDotNet.Tensors/issues/1048)) ([3be2431](https://github.com/ooples/AiDotNet.Tensors/commit/3be24314144ce39a0d5cc9f4b09bbd6ef78db98b))

## [0.132.1](https://github.com/ooples/AiDotNet.Tensors/compare/v0.132.0...v0.132.1) (2026-09-24)


### Bug Fixes

* **deps:** depend on the stable aidotnet.evolution so releases can pack ([#1043](https://github.com/ooples/AiDotNet.Tensors/issues/1043)) ([e07fd0b](https://github.com/ooples/AiDotNet.Tensors/commit/e07fd0b3a37b2cdcf6bbeeac9fa4ada3d26d3273))

## [0.132.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.131.0...v0.132.0) (2026-09-20)


### Features

* **autotune:** deliver US-25 artifact promotion and bounded retuning ([#1030](https://github.com/ooples/AiDotNet.Tensors/issues/1030)) ([94aeab8](https://github.com/ooples/AiDotNet.Tensors/commit/94aeab898c8d6d59e0a25d9aa50f269414483a79))


### Bug Fixes

* align grouped query attention causal mask with the kv-cache offset ([#1042](https://github.com/ooples/AiDotNet.Tensors/issues/1042)) ([b9f0f7a](https://github.com/ooples/AiDotNet.Tensors/commit/b9f0f7a613b314a2761d5d12b04293f4f3232adf))
* **attention:** make the query-offset contract total for rectangular noncausal attention ([#1035](https://github.com/ooples/AiDotNet.Tensors/issues/1035)) ([f1d6387](https://github.com/ooples/AiDotNet.Tensors/commit/f1d63878316ee287c254625ef964aff55230c76d))
* **blas:** ignore the timed autotune cache in deterministic mode ([#1034](https://github.com/ooples/AiDotNet.Tensors/issues/1034)) ([d976f6c](https://github.com/ooples/AiDotNet.Tensors/commit/d976f6cd93648c3dc3397a8ef3e10120bef45b4c))
* correct transposed octonion linear backward jacobian tables ([#1041](https://github.com/ooples/AiDotNet.Tensors/issues/1041)) ([026da97](https://github.com/ooples/AiDotNet.Tensors/commit/026da97b96e856d952438c7557ae0d13bf06d1b5))
* **gpu:** return a tensor to the CPU when its resident buffer is invalidated ([#1039](https://github.com/ooples/AiDotNet.Tensors/issues/1039)) ([c68b163](https://github.com/ooples/AiDotNet.Tensors/commit/c68b1633a7780acf948c6bccf98aae46636bc00c))
* make convtranspose3d bit-identical across thread budgets ([#1040](https://github.com/ooples/AiDotNet.Tensors/issues/1040)) ([8fca226](https://github.com/ooples/AiDotNet.Tensors/commit/8fca226e81601a5e32cb8be440b6e1f028f06c8a))

## [0.131.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.130.3...v0.131.0) (2026-09-16)


### Features

* add differentiable tensor asin, acos, atan and atan2 ([#1019](https://github.com/ooples/AiDotNet.Tensors/issues/1019)) ([6ab238a](https://github.com/ooples/AiDotNet.Tensors/commit/6ab238ac7f6d80a50be20b68efcbba70da37db94))
* **autotune:** persist quarantine and restore validated deployments ([#1024](https://github.com/ooples/AiDotNet.Tensors/issues/1024)) ([cbbc0c0](https://github.com/ooples/AiDotNet.Tensors/commit/cbbc0c03d40afd3d09b1ee194da856d15f04be95))
* integrate evolutionary kernel autotuning ([#1011](https://github.com/ooples/AiDotNet.Tensors/issues/1011)) ([67ceb6e](https://github.com/ooples/AiDotNet.Tensors/commit/67ceb6ed328a7764d7ee45864d32188fd659e363))
* make istft differentiable and add stftphase ([#1021](https://github.com/ooples/AiDotNet.Tensors/issues/1021)) ([a93cb60](https://github.com/ooples/AiDotNet.Tensors/commit/a93cb606f6947d59c45ff805cd042060ced66272))


### Bug Fixes

* **autodiff:** reduce the rank-3 weight gradient to its rank-2 leaf ([#1037](https://github.com/ooples/AiDotNet.Tensors/issues/1037)) ([87adda3](https://github.com/ooples/AiDotNet.Tensors/commit/87adda331fdc48ed2d08426709166e8d4cfe7e53))
* **broadcast:** follow NumPy's rule for zero-size extents across engine, autodiff and ONNX ([#1033](https://github.com/ooples/AiDotNet.Tensors/issues/1033)) ([16344d2](https://github.com/ooples/AiDotNet.Tensors/commit/16344d25a0eb32c2257e5d1056d0e5bfb346cb8c))
* **gpu:** preserve lazy activation ownership ([#1020](https://github.com/ooples/AiDotNet.Tensors/issues/1020)) ([cba12d7](https://github.com/ooples/AiDotNet.Tensors/commit/cba12d7e42ce01d43e3f0a0d4d5980cd68dafe77))
* preserve streaming transactions and GPU cache coherence ([#1029](https://github.com/ooples/AiDotNet.Tensors/issues/1029)) ([1c71f40](https://github.com/ooples/AiDotNet.Tensors/commit/1c71f4091acc39196b8d9604f555a265b110a30c))
* preserve warmup phases and version schedule checkpoints ([#1036](https://github.com/ooples/AiDotNet.Tensors/issues/1036)) ([a5c299a](https://github.com/ooples/AiDotNet.Tensors/commit/a5c299aee81ec8e93368d3f3b5b3c353e9c63b9a))

## [0.130.3](https://github.com/ooples/AiDotNet.Tensors/compare/v0.130.2...v0.130.3) (2026-09-08)


### Bug Fixes

* **tensors:** preserve COW storage across inference reads ([#1012](https://github.com/ooples/AiDotNet.Tensors/issues/1012)) ([d3a78d1](https://github.com/ooples/AiDotNet.Tensors/commit/d3a78d152539d1cced7a246c9bcba4ad16a557dc))

## [0.130.2](https://github.com/ooples/AiDotNet.Tensors/compare/v0.130.1...v0.130.2) (2026-09-04)


### Bug Fixes

* **compilation:** classify empty graph capture failures ([#1009](https://github.com/ooples/AiDotNet.Tensors/issues/1009)) ([04cdf64](https://github.com/ooples/AiDotNet.Tensors/commit/04cdf641f3a8cc168efc100001a2be1313dc833e))

## [0.130.1](https://github.com/ooples/AiDotNet.Tensors/compare/v0.130.0...v0.130.1) (2026-09-04)


### Bug Fixes

* **compilation:** preserve dynamic masks across graph replay ([fa2c2b7](https://github.com/ooples/AiDotNet.Tensors/commit/fa2c2b72cfc3e3ccbda85446f1d057c5972100e5))

## [0.130.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.129.8...v0.130.0) (2026-09-03)


### Features

* **ptx:** add experimental pointwise GLU blueprint ([#864](https://github.com/ooples/AiDotNet.Tensors/issues/864)) ([4341b37](https://github.com/ooples/AiDotNet.Tensors/commit/4341b371b2a08f205d39ec6d6203955917919143))
* **ptx:** fuse residual LayerNorm GELU ([#863](https://github.com/ooples/AiDotNet.Tensors/issues/863)) ([e7c88ca](https://github.com/ooples/AiDotNet.Tensors/commit/e7c88ca3368706872ba00b11da5ea004d8dc33a7))
* **ptx:** port dense GEMM, linear, LoRA, and vector kernels ([#859](https://github.com/ooples/AiDotNet.Tensors/issues/859)) ([352f2e0](https://github.com/ooples/AiDotNet.Tensors/commit/352f2e0bda99a8d5da07db214bed3b4715730a20))


### Bug Fixes

* complete graph capture and GPU backend parity ([#1006](https://github.com/ooples/AiDotNet.Tensors/issues/1006)) ([b7ab57b](https://github.com/ooples/AiDotNet.Tensors/commit/b7ab57b41fc8be2e04a10e7393315c1c80ebe2f5))
* create the diagnostics dump directory instead of failing silently ([#1001](https://github.com/ooples/AiDotNet.Tensors/issues/1001)) ([28b0cf8](https://github.com/ooples/AiDotNet.Tensors/commit/28b0cf82b29ef16c67cbf60cb773c873107b9279))
* **gpu:** avoid COW detach during cache maintenance ([#1002](https://github.com/ooples/AiDotNet.Tensors/issues/1002)) ([9e34d7d](https://github.com/ooples/AiDotNet.Tensors/commit/9e34d7d5d962c0b01a0b049fb904a35da92f770e))
* preserve compiled tensor lifetimes and CPU invariants ([#1005](https://github.com/ooples/AiDotNet.Tensors/issues/1005)) ([4141c01](https://github.com/ooples/AiDotNet.Tensors/commit/4141c01c281068ac5ee02d07bd5c378de5ce4b60))

## [0.129.8](https://github.com/ooples/AiDotNet.Tensors/compare/v0.129.7...v0.129.8) (2026-08-30)


### Bug Fixes

* **cpu:** preserve Conv2D parity without Windows regression ([#998](https://github.com/ooples/AiDotNet.Tensors/issues/998)) ([37260c3](https://github.com/ooples/AiDotNet.Tensors/commit/37260c33a1c5bcaa35d823dbc42396c3c3afecc8))
* register the crash diagnostics dump at assembly load, not first use ([#999](https://github.com/ooples/AiDotNet.Tensors/issues/999)) ([b21aab5](https://github.com/ooples/AiDotNet.Tensors/commit/b21aab57a341d9f877383844102ee6f14b6558b1))

## [0.129.7](https://github.com/ooples/AiDotNet.Tensors/compare/v0.129.6...v0.129.7) (2026-08-30)


### Bug Fixes

* gpu tape gaps, categorical residency, and non-power-of-two reductions ([#995](https://github.com/ooples/AiDotNet.Tensors/issues/995)) ([28a42f5](https://github.com/ooples/AiDotNet.Tensors/commit/28a42f5e901084dd5a48db0c17bf202c20e4dcf8))
* mask the b loads in the n-tail direct gemm kernels ([#991](https://github.com/ooples/AiDotNet.Tensors/issues/991)) ([44c1ac1](https://github.com/ooples/AiDotNet.Tensors/commit/44c1ac147f2865c7242d4ac66503f1d75273ae7e))
* mask the column tail in the 4-row block gemm kernels ([#993](https://github.com/ooples/AiDotNet.Tensors/issues/993)) ([4faaa5d](https://github.com/ooples/AiDotNet.Tensors/commit/4faaa5d9a0204647f552b46904d91bdaa358d9a4))
* upload the sequence file crash evidence actually writes ([#997](https://github.com/ooples/AiDotNet.Tensors/issues/997)) ([a378851](https://github.com/ooples/AiDotNet.Tensors/commit/a3788514152fcfd742860392f7793b1fb3eb9fe5))

## [0.129.6](https://github.com/ooples/AiDotNet.Tensors/compare/v0.129.5...v0.129.6) (2026-08-29)


### Bug Fixes

* reject gemm operands too small for their own strides ([#989](https://github.com/ooples/AiDotNet.Tensors/issues/989)) ([70bf7e2](https://github.com/ooples/AiDotNet.Tensors/commit/70bf7e2a3fac8717e617813a9b051dd0b54c1016))

## [0.129.5](https://github.com/ooples/AiDotNet.Tensors/compare/v0.129.4...v0.129.5) (2026-08-28)


### Bug Fixes

* cut lossless streaming-encode peak from ~3.7n to ~1.7n ([#986](https://github.com/ooples/AiDotNet.Tensors/issues/986)) ([5101076](https://github.com/ooples/AiDotNet.Tensors/commit/510107653c027c4dec00b3f33d6b121b5855596f))
* make conv2d and conv2dinto compute the same function for float ([#988](https://github.com/ooples/AiDotNet.Tensors/issues/988)) ([459539d](https://github.com/ooples/AiDotNet.Tensors/commit/459539da0adb60118fb0b49fb000db305b2cc014))

## [0.129.4](https://github.com/ooples/AiDotNet.Tensors/compare/v0.129.3...v0.129.4) (2026-08-26)


### Bug Fixes

* make the arena allocation counters impossible to misread ([#984](https://github.com/ooples/AiDotNet.Tensors/issues/984)) ([4f71f71](https://github.com/ooples/AiDotNet.Tensors/commit/4f71f71f2ba707283ceb5f200515876093717547))


### Performance

* stop CloneDeepCopy staging through a full intermediate array ([#983](https://github.com/ooples/AiDotNet.Tensors/issues/983)) ([96e88a3](https://github.com/ooples/AiDotNet.Tensors/commit/96e88a3ab8d2c8f5e65d9fbf7f47cae26f928b85))

## [0.129.3](https://github.com/ooples/AiDotNet.Tensors/compare/v0.129.2...v0.129.3) (2026-08-26)


### Performance

* **gpu:** stop uploading a tensor to the GPU just because it was registered ([#981](https://github.com/ooples/AiDotNet.Tensors/issues/981)) ([34a45d5](https://github.com/ooples/AiDotNet.Tensors/commit/34a45d5c578d198250eb184cc90d4bf37c1fbab0))

## [0.129.2](https://github.com/ooples/AiDotNet.Tensors/compare/v0.129.1...v0.129.2) (2026-08-22)


### Bug Fixes

* **perf:** bound the buffer caches in aggregate, not just per bucket ([#979](https://github.com/ooples/AiDotNet.Tensors/issues/979)) ([0ce3bb7](https://github.com/ooples/AiDotNet.Tensors/commit/0ce3bb7d53a052a699b7130f68cb7a555488f940))

## [0.129.1](https://github.com/ooples/AiDotNet.Tensors/compare/v0.129.0...v0.129.1) (2026-08-22)


### Bug Fixes

* **compilation:** capture custom fused training kernels ([#978](https://github.com/ooples/AiDotNet.Tensors/issues/978)) ([b761bdd](https://github.com/ooples/AiDotNet.Tensors/commit/b761bdd88734ed74a10dc232431b6f4e659599b0))
* **compilation:** record RoPE dependencies in fused training ([#975](https://github.com/ooples/AiDotNet.Tensors/issues/975)) ([f592c46](https://github.com/ooples/AiDotNet.Tensors/commit/f592c46a9b6ea84571d1f865ed2eac74b5bd7094))
* **directgpu:** break the infinite recursion in the TensorTanh CPU fallback ([#977](https://github.com/ooples/AiDotNet.Tensors/issues/977)) ([d40cfb5](https://github.com/ooples/AiDotNet.Tensors/commit/d40cfb5de5dd2f67e7fb8981fb435155a44a567a))

## [0.129.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.128.0...v0.129.0) (2026-08-21)


### Features

* **autodiff:** add fused abc slot-recurrence scan kernel ([#973](https://github.com/ooples/AiDotNet.Tensors/issues/973)) ([47b72b6](https://github.com/ooples/AiDotNet.Tensors/commit/47b72b6f393fa4dac5bdc62e2cf154926bf98d8f))
* **gpu:** add capability-aware autocast and type parity ([#971](https://github.com/ooples/AiDotNet.Tensors/issues/971)) ([f301610](https://github.com/ooples/AiDotNet.Tensors/commit/f301610b46232630bdfe66197cabe88c22621252))
* **ptx:** add mixed-precision decode golden slice ([#837](https://github.com/ooples/AiDotNet.Tensors/issues/837)) ([#860](https://github.com/ooples/AiDotNet.Tensors/issues/860)) ([d95d3f7](https://github.com/ooples/AiDotNet.Tensors/commit/d95d3f7794a6c5f00f1004172d153c26f7189525))
* **ptx:** exact-shape FP32 row-softmax blueprint ([#840](https://github.com/ooples/AiDotNet.Tensors/issues/840)) ([#868](https://github.com/ooples/AiDotNet.Tensors/issues/868)) ([2ce70c3](https://github.com/ooples/AiDotNet.Tensors/commit/2ce70c3ef9ad4c91bc31951a4d9b423e3a649001))


### Bug Fixes

* **autodiff:** record MaxPool2DWithIndices on the GPU fast path ([#968](https://github.com/ooples/AiDotNet.Tensors/issues/968)) ([8c0e179](https://github.com/ooples/AiDotNet.Tensors/commit/8c0e179f2f3e206c23ab1538ee419090554f10bb))
* **compilation:** skip a fused optimizer step whose gradients are not finite ([#972](https://github.com/ooples/AiDotNet.Tensors/issues/972)) ([bc66bc0](https://github.com/ooples/AiDotNet.Tensors/commit/bc66bc0c0c2713ced924e5d752d9541cb4573975))
* **rglru:** clip the sqrt derivative, not just skip it at zero ([#969](https://github.com/ooples/AiDotNet.Tensors/issues/969)) ([bbe86f7](https://github.com/ooples/AiDotNet.Tensors/commit/bbe86f7135af800d0e91d2b3ce58556102ff20e0))

## [0.128.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.127.0...v0.128.0) (2026-08-18)


### Features

* **fft:** generic-precision, arbitrary-length GPU FFT path ([#959](https://github.com/ooples/AiDotNet.Tensors/issues/959)) ([3647d32](https://github.com/ooples/AiDotNet.Tensors/commit/3647d3267b8875a95e574cb186a419ec0f827ba7))
* **fused:** dispatch Fletcher-Reeves conjugate gradient through the plan ([#957](https://github.com/ooples/AiDotNet.Tensors/issues/957)) ([fa81d5e](https://github.com/ooples/AiDotNet.Tensors/commit/fa81d5e19858588dc145b93c54d687d04bcfe5bb))
* **fused:** dispatch linearized ADMM through the plan ([#956](https://github.com/ooples/AiDotNet.Tensors/issues/956)) ([7456fdd](https://github.com/ooples/AiDotNet.Tensors/commit/7456fddaff70240ee81c5d21eaaab084c92adee3))
* **ptx:** bf16 tensor-core wmma GEMM on the naive and staged paths ([#967](https://github.com/ooples/AiDotNet.Tensors/issues/967)) ([83db840](https://github.com/ooples/AiDotNet.Tensors/commit/83db8400009a3a6f9d0d99e13fa7966044e39d7f))
* **ptx:** exact-shape FP32 embedding gather blueprint ([#844](https://github.com/ooples/AiDotNet.Tensors/issues/844)) ([#870](https://github.com/ooples/AiDotNet.Tensors/issues/870)) ([f22edb1](https://github.com/ooples/AiDotNet.Tensors/commit/f22edb1ab76ac805806e13197bc4fbaa67ac87c8))
* **ptx:** exact-shape FP32 row-sum reduction blueprint ([#843](https://github.com/ooples/AiDotNet.Tensors/issues/843)) ([#869](https://github.com/ooples/AiDotNet.Tensors/issues/869)) ([31497cc](https://github.com/ooples/AiDotNet.Tensors/commit/31497cceb2be429ba7fcf5eed049d54a11e8c10a))
* **ptx:** exact-shape FP32→FP16 cast blueprint ([#845](https://github.com/ooples/AiDotNet.Tensors/issues/845)) ([#871](https://github.com/ooples/AiDotNet.Tensors/issues/871)) ([c283a93](https://github.com/ooples/AiDotNet.Tensors/commit/c283a93ed90428af5a0b244560fa9883e60fdf63))
* **ptx:** per-sample FP32 MSE-loss blueprint ([#847](https://github.com/ooples/AiDotNet.Tensors/issues/847)) ([#872](https://github.com/ooples/AiDotNet.Tensors/issues/872)) ([e64cd40](https://github.com/ooples/AiDotNet.Tensors/commit/e64cd409c72ca411d2ed193c03b419d87bc20c6a))


### Bug Fixes

* **fused:** make the dense LARS kernel implement You et al. Algorithm 1 ([#955](https://github.com/ooples/AiDotNet.Tensors/issues/955)) ([a781bd4](https://github.com/ooples/AiDotNet.Tensors/commit/a781bd4d330cc34c0ce0fc1c4b7b9c7a3075ed31))
* **fused:** RAdam rectification precision and Nadam bias-correction placement ([#960](https://github.com/ooples/AiDotNet.Tensors/issues/960)) ([8cec7b3](https://github.com/ooples/AiDotNet.Tensors/commit/8cec7b33f8f618770a0de80706462c902264dd4f))
* **streaming:** reclaim backing files when the pool's process is killed ([#966](https://github.com/ooples/AiDotNet.Tensors/issues/966)) ([4c5e975](https://github.com/ooples/AiDotNet.Tensors/commit/4c5e975038f6686819e728ca7cb4a806a3c8ca12))

## [0.127.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.126.0...v0.127.0) (2026-08-16)


### Features

* add differentiable UPRNet warp operations ([#951](https://github.com/ooples/AiDotNet.Tensors/issues/951)) ([34c2e86](https://github.com/ooples/AiDotNet.Tensors/commit/34c2e86dda435ada1fd131641328527b617454cd))
* **fused:** reach the last two unfused optimizer families — ProximalL1 and L-BFGS ([#953](https://github.com/ooples/AiDotNet.Tensors/issues/953)) ([dbbb562](https://github.com/ooples/AiDotNet.Tensors/commit/dbbb56263eb2449d1237ccdaf9d7c99312a1bd6b))
* **fused:** surface the nesterov flag and FTRL's beta term ([#949](https://github.com/ooples/AiDotNet.Tensors/issues/949)) ([92a793d](https://github.com/ooples/AiDotNet.Tensors/commit/92a793d936158f31577c2c7435a3d792e40e2dd9))
* **ptx:** fused SGD-momentum optimizer blueprint ([#848](https://github.com/ooples/AiDotNet.Tensors/issues/848)) ([#874](https://github.com/ooples/AiDotNet.Tensors/issues/874)) ([88c5d58](https://github.com/ooples/AiDotNet.Tensors/commit/88c5d58cf179c532b3ccd3c39321dbed9fe3e98c))


### Bug Fixes

* **sdpa:** broadcast the attention mask in the generic workers ([#674](https://github.com/ooples/AiDotNet.Tensors/issues/674) review) ([#947](https://github.com/ooples/AiDotNet.Tensors/issues/947)) ([7a3f802](https://github.com/ooples/AiDotNet.Tensors/commit/7a3f8026b85bcc258d7512b146303e8fdd7e2ca3))


### Performance

* **compilation:** bound lazy graph reordering cost ([#948](https://github.com/ooples/AiDotNet.Tensors/issues/948)) ([217786a](https://github.com/ooples/AiDotNet.Tensors/commit/217786a92aa261157ebd8a01f2f5e1a52bda6d64))

## [0.126.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.125.0...v0.126.0) (2026-08-14)


### Features

* **ptx:** add convolution golden slice ([#877](https://github.com/ooples/AiDotNet.Tensors/issues/877)) ([3038daf](https://github.com/ooples/AiDotNet.Tensors/commit/3038daffbcd56ffb899fc3201ca21c589880d43c))
* **ptx:** global average pooling blueprint ([#842](https://github.com/ooples/AiDotNet.Tensors/issues/842)) ([#876](https://github.com/ooples/AiDotNet.Tensors/issues/876)) ([6806cb9](https://github.com/ooples/AiDotNet.Tensors/commit/6806cb9e40e4bee0553c5550356e520c2285515c))


### Bug Fixes

* **streaming:** make paged tensor access mutation-safe ([#945](https://github.com/ooples/AiDotNet.Tensors/issues/945)) ([4965fc0](https://github.com/ooples/AiDotNet.Tensors/commit/4965fc0973083e5aa5c338ced7731423c6ced082))

## [0.125.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.124.0...v0.125.0) (2026-08-13)


### Features

* **engine:** add fused complex diagonal SSM scan ([#940](https://github.com/ooples/AiDotNet.Tensors/issues/940)) ([f319b03](https://github.com/ooples/AiDotNet.Tensors/commit/f319b032d9d9cd25f7bf60166081846d1a20ce8c))
* **engine:** fuse Mesa and routed SSM scans ([#944](https://github.com/ooples/AiDotNet.Tensors/issues/944)) ([05b33d5](https://github.com/ooples/AiDotNet.Tensors/commit/05b33d593f35c7b585c7f5057faa425227319514))
* **ptx:** add spectral complex multiply golden slice ([#879](https://github.com/ooples/AiDotNet.Tensors/issues/879)) ([9ac21c5](https://github.com/ooples/AiDotNet.Tensors/commit/9ac21c5e1609f0e7904bfa7fe4ea1c20883fbc8a))

## [0.124.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.123.0...v0.124.0) (2026-08-13)


### Features

* **ptx:** add register-resident dense solver family ([#882](https://github.com/ooples/AiDotNet.Tensors/issues/882)) ([f130995](https://github.com/ooples/AiDotNet.Tensors/commit/f1309950346d6e5bb68271fa4a156c784b3f2b6b))
* **ptx:** complete RNG and stochastic kernel family ([#880](https://github.com/ooples/AiDotNet.Tensors/issues/880)) ([c1c9afc](https://github.com/ooples/AiDotNet.Tensors/commit/c1c9afc57e9e32f8eff31f9ec7e34f7a52bd1c69))


### Bug Fixes

* engine dispatch must honor tensor residency (CPU-resident double silently computed in float) ([#937](https://github.com/ooples/AiDotNet.Tensors/issues/937)) ([aab2846](https://github.com/ooples/AiDotNet.Tensors/commit/aab28469507bc9846786f4926599a12d56e1c227))


### Performance

* **autodiff:** materialize the in-place gradient copy lazily ([#938](https://github.com/ooples/AiDotNet.Tensors/issues/938)) ([8550c42](https://github.com/ooples/AiDotNet.Tensors/commit/8550c42dfafd715aa1d9b765bbd47eba666c9ab5))

## [0.123.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.122.0...v0.123.0) (2026-08-12)


### Features

* **ptx:** add experimental vision kernel family ([#883](https://github.com/ooples/AiDotNet.Tensors/issues/883)) ([a85a9c3](https://github.com/ooples/AiDotNet.Tensors/commit/a85a9c3a35e345c5709e8db9a84bc32bc6c82daf))


### Bug Fixes

* **autodiff:** harden gradient ownership and composite contracts ([#935](https://github.com/ooples/AiDotNet.Tensors/issues/935)) ([c3b64d8](https://github.com/ooples/AiDotNet.Tensors/commit/c3b64d8fa3ec03cc83e31558d4032bb107f37efc))

## [0.122.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.121.0...v0.122.0) (2026-08-07)


### Features

* **autotune:** add stable GPU first-run and community tuning ([#887](https://github.com/ooples/AiDotNet.Tensors/issues/887)) ([219f9c1](https://github.com/ooples/AiDotNet.Tensors/commit/219f9c1c701dbfa5cf0c0eb410a5597f64de179f))
* **ptx:** add recurrent state golden slice ([#878](https://github.com/ooples/AiDotNet.Tensors/issues/878)) ([f65d334](https://github.com/ooples/AiDotNet.Tensors/commit/f65d3342ff217575f4a58f41193b7ec89a802d85))
* **ptx:** port softmax, log-softmax, masking, log-sum-exp, and backward fusions ([#840](https://github.com/ooples/AiDotNet.Tensors/issues/840)) ([#884](https://github.com/ooples/AiDotNet.Tensors/issues/884)) ([b1a48f9](https://github.com/ooples/AiDotNet.Tensors/commit/b1a48f9eedc2597d814e1c47ab448e186db5b3cd))
* **ptx:** port specialized scientific, hypercomplex, and hyperbolic kernels ([#854](https://github.com/ooples/AiDotNet.Tensors/issues/854)) ([#885](https://github.com/ooples/AiDotNet.Tensors/issues/885)) ([a96b638](https://github.com/ooples/AiDotNet.Tensors/commit/a96b638dcec7197473e32af43513ff1c3be987d6))
* **tensor:** deferred-storage tensors — shape without allocation ([#930](https://github.com/ooples/AiDotNet.Tensors/issues/930)) ([5596d07](https://github.com/ooples/AiDotNet.Tensors/commit/5596d077b7c432dd169905d3c24466a114d169c3))


### Bug Fixes

* **tests:** remove duplicate MeasuredResourceBudget theories in DirectPtxWmmaTests ([#934](https://github.com/ooples/AiDotNet.Tensors/issues/934)) ([98d7c47](https://github.com/ooples/AiDotNet.Tensors/commit/98d7c470cc23811170487dff7d1b04754866a93d))


### Performance

* **blas:** BCL Vector&lt;T&gt; FP64 streaming microkernel — net471 GEMM was fully scalar ([#933](https://github.com/ooples/AiDotNet.Tensors/issues/933)) ([6db5074](https://github.com/ooples/AiDotNet.Tensors/commit/6db507429d01a002c7d630d5435b639be25ebf64))
* **codegen:** harden staging and close the championship loop ([#909](https://github.com/ooples/AiDotNet.Tensors/issues/909)) ([c6b71ad](https://github.com/ooples/AiDotNet.Tensors/commit/c6b71ad1fe1dd053ea1893f73da860c42f1e0798))

## [0.121.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.120.6...v0.121.0) (2026-08-01)


### Features

* **engine:** broadcast implicitly on the element-wise operators ([#919](https://github.com/ooples/AiDotNet.Tensors/issues/919)) ([fc390f0](https://github.com/ooples/AiDotNet.Tensors/commit/fc390f007f6897d34443a66e5f3b55204f436e49))


### Bug Fixes

* **autodiff:** reject full-span views from tensor pool ([#925](https://github.com/ooples/AiDotNet.Tensors/issues/925)) ([1b0006f](https://github.com/ooples/AiDotNet.Tensors/commit/1b0006f43af5c49df44ac9054b0f152f31b8716e))

## [0.120.6](https://github.com/ooples/AiDotNet.Tensors/compare/v0.120.5...v0.120.6) (2026-07-31)


### Bug Fixes

* **autodiff:** Spectrogram backward was not the adjoint; make MelSpectrogram differentiable; close the OpRegistry out-param hole ([#911](https://github.com/ooples/AiDotNet.Tensors/issues/911)) ([4207a19](https://github.com/ooples/AiDotNet.Tensors/commit/4207a199169eff4c6921d6cffd2c1558303d39b3))
* **compilation:** key compiled plans on data-derived constants, not just structure ([#913](https://github.com/ooples/AiDotNet.Tensors/issues/913)) ([09e44fb](https://github.com/ooples/AiDotNet.Tensors/commit/09e44fb50eff18e2f0ec2c6b6a3288c3604eb616))
* **tests:** double.IsFinite does not exist on net471 (completes [#898](https://github.com/ooples/AiDotNet.Tensors/issues/898)) ([#910](https://github.com/ooples/AiDotNet.Tensors/issues/910)) ([a31a350](https://github.com/ooples/AiDotNet.Tensors/commit/a31a3508834c7b07ef496a4f7391b5214f753f14))


### Performance

* **codegen:** strengthen oracle-driven kernel foundation ([#897](https://github.com/ooples/AiDotNet.Tensors/issues/897)) ([998ac91](https://github.com/ooples/AiDotNet.Tensors/commit/998ac918334c2400d1bfe39947b901da4d585493))

## [0.120.5](https://github.com/ooples/AiDotNet.Tensors/compare/v0.120.4...v0.120.5) (2026-07-30)


### Bug Fixes

* **autodiff:** pin savedState tensors against pool/arena reuse (Issue [#338](https://github.com/ooples/AiDotNet.Tensors/issues/338) completion) ([#906](https://github.com/ooples/AiDotNet.Tensors/issues/906)) ([bb56e31](https://github.com/ooples/AiDotNet.Tensors/commit/bb56e313ca09d2ffad3189d6bc810058e1b408fd))
* **gpu:** correct LSTM forward+backward GPU kernels + POCL cpu-opencl parity gate ([#907](https://github.com/ooples/AiDotNet.Tensors/issues/907)) ([e54f99f](https://github.com/ooples/AiDotNet.Tensors/commit/e54f99f9ccec7dbbf76fd0cdee83b4199c9c1e95))

## [0.120.4](https://github.com/ooples/AiDotNet.Tensors/compare/v0.120.3...v0.120.4) (2026-07-30)


### Bug Fixes

* **compilation:** replay stop-gradient forward values ([#903](https://github.com/ooples/AiDotNet.Tensors/issues/903)) ([b80e21a](https://github.com/ooples/AiDotNet.Tensors/commit/b80e21a685cf510b0bb7383dc862ce02052a4ad0))

## [0.120.3](https://github.com/ooples/AiDotNet.Tensors/compare/v0.120.2...v0.120.3) (2026-07-29)


### Bug Fixes

* **compilation:** preserve COW parameter views and FP64 training ([#901](https://github.com/ooples/AiDotNet.Tensors/issues/901)) ([da337a4](https://github.com/ooples/AiDotNet.Tensors/commit/da337a469278100b72f79817ec1e5046cdb53522))

## [0.120.2](https://github.com/ooples/AiDotNet.Tensors/compare/v0.120.1...v0.120.2) (2026-07-29)


### Bug Fixes

* **autodiff:** elementwise backwards threw on a rank-0 operand vs a rank-1 seed gradient ([#898](https://github.com/ooples/AiDotNet.Tensors/issues/898)) ([ff15edc](https://github.com/ooples/AiDotNet.Tensors/commit/ff15edcc0eca4e44cf101a262421d0a33e03b67d))
* preserve COW isolation in compiled training ([#899](https://github.com/ooples/AiDotNet.Tensors/issues/899)) ([f74c0eb](https://github.com/ooples/AiDotNet.Tensors/commit/f74c0eb91bbf1c507815962a81b662c07ffa2d36))

## [0.120.1](https://github.com/ooples/AiDotNet.Tensors/compare/v0.120.0...v0.120.1) (2026-07-28)


### Bug Fixes

* **rwkv7:** implement the paper-faithful RWKV-7 generalized delta rule ([#895](https://github.com/ooples/AiDotNet.Tensors/issues/895)) ([ecad0b7](https://github.com/ooples/AiDotNet.Tensors/commit/ecad0b7870d401596bd2dfd293ae40ed8859ce34))

## [0.120.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.119.0...v0.120.0) (2026-07-28)


### Features

* add direct PTX fused-kernel blueprint ([#832](https://github.com/ooples/AiDotNet.Tensors/issues/832)) ([2832eb2](https://github.com/ooples/AiDotNet.Tensors/commit/2832eb21d7dd749a139e245a95ba0285d83490c4))
* **ptx:** compile and dispatch direct-PTX on net471 (all TFMs) ([#881](https://github.com/ooples/AiDotNet.Tensors/issues/881)) ([cc14666](https://github.com/ooples/AiDotNet.Tensors/commit/cc1466621664b53107033d8ce59742307aa55393))
* **ptx:** direct-PTX kernel foundation - 5 emitter gaps closed + convolution family ([#886](https://github.com/ooples/AiDotNet.Tensors/issues/886)) ([3b21f7d](https://github.com/ooples/AiDotNet.Tensors/commit/3b21f7df74546a9d9c8525004ae451a89463906e))
* **ptx:** fuse decode QKV, RoPE, and KV-cache update ([#858](https://github.com/ooples/AiDotNet.Tensors/issues/858)) ([51dace6](https://github.com/ooples/AiDotNet.Tensors/commit/51dace68e3e7d309ec2115555aa1a547648c08d1))
* **ptx:** implement fused attention kernel family ([#857](https://github.com/ooples/AiDotNet.Tensors/issues/857)) ([8fb5e91](https://github.com/ooples/AiDotNet.Tensors/commit/8fb5e91ea0ae2b8841e749b74f44cc26be258144))


### Bug Fixes

* **autodiff:** gather backward must scatter-ADD, not reuse the gather-shaped helper ([#894](https://github.com/ooples/AiDotNet.Tensors/issues/894)) ([b541350](https://github.com/ooples/AiDotNet.Tensors/commit/b54135043a75b6d2d9426d5c3c4c62fd2bf91f4a))
* **autodiff:** write fused-linear grads to live buffers ([#888](https://github.com/ooples/AiDotNet.Tensors/issues/888)) ([4ac1684](https://github.com/ooples/AiDotNet.Tensors/commit/4ac1684e7acb9064849997383b06eb50974e5b3b))
* **gpu:** gate sparsemax on where-select availability ([#866](https://github.com/ooples/AiDotNet.Tensors/issues/866)) ([9b1a88a](https://github.com/ooples/AiDotNet.Tensors/commit/9b1a88a11c451f81a494ead92a5ba0607542222a))


### Performance

* **ptx:** staged + double-buffered tensor-core GEMM - 0.05x to 0.57x at 4096^3 ([#893](https://github.com/ooples/AiDotNet.Tensors/issues/893)) ([6d41166](https://github.com/ooples/AiDotNet.Tensors/commit/6d41166df0176b4ee885822df5c3d89c4bfca709))
* **training:** element-parallel fused optimizer + reusable per-step arena ([#892](https://github.com/ooples/AiDotNet.Tensors/issues/892)) ([4d3c3ec](https://github.com/ooples/AiDotNet.Tensors/commit/4d3c3ecf83cbfadf41ab79557f9ae12c3c721e77))

## [0.119.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.118.0...v0.119.0) (2026-07-21)


### Features

* **simd:** block-Q8_0 GEMM matching ggml_vec_dot_q8_0_q8_0 ([#830](https://github.com/ooples/AiDotNet.Tensors/issues/830)) ([3a4d27e](https://github.com/ooples/AiDotNet.Tensors/commit/3a4d27eaadb2382f71277b3487fe64fde9f1ed33))


### Bug Fixes

* **cuda:** power_scalar returned NaN for negative bases under --use_fast_math ([#827](https://github.com/ooples/AiDotNet.Tensors/issues/827)) ([4ae2377](https://github.com/ooples/AiDotNet.Tensors/commit/4ae237736a7196c717efbe486cbf2b1d9653a19a))
* **gpu:** add tape-gradient parity and correct backward kernels ([#829](https://github.com/ooples/AiDotNet.Tensors/issues/829)) ([f1b6dff](https://github.com/ooples/AiDotNet.Tensors/commit/f1b6dff4df7f5aea7b675bb56201f82366d57ece))


### Performance

* **fft:** blittable buffer-reusing NativeComplexFFT — kill per-call Complex&lt;T&gt; allocation ([#855](https://github.com/ooples/AiDotNet.Tensors/issues/855)) ([b61cd73](https://github.com/ooples/AiDotNet.Tensors/commit/b61cd730c01e7596c522e7fdc630d789c268c9f7))
* **fft:** wire NativeComplexFFT butterfly to SIMD radix-2 (test-gated) ([#831](https://github.com/ooples/AiDotNet.Tensors/issues/831)) ([7938eb2](https://github.com/ooples/AiDotNet.Tensors/commit/7938eb2f5ddb24914e7fe0cc348dc6e2c360ec46))

## [0.118.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.117.2...v0.118.0) (2026-07-20)


### Features

* document and release the native ANN kernel stack (0.118.0) ([#825](https://github.com/ooples/AiDotNet.Tensors/issues/825)) ([fb5a70e](https://github.com/ooples/AiDotNet.Tensors/commit/fb5a70ea1457af41367388a5fffc13a13b2de184))

## [0.117.2](https://github.com/ooples/AiDotNet.Tensors/compare/v0.117.1...v0.117.2) (2026-07-20)


### Bug Fixes

* **gpu:** centralize the persistent-weight version gate into the shared cache lookup ([#821](https://github.com/ooples/AiDotNet.Tensors/issues/821) follow-up) ([#822](https://github.com/ooples/AiDotNet.Tensors/issues/822)) ([87aebd0](https://github.com/ooples/AiDotNet.Tensors/commit/87aebd067219a231d7852d16431f9b99d12188ef))
* **gpu:** version-gate the persistent weight buffer so in-place weight loads under a GPU engine are not served stale ([#821](https://github.com/ooples/AiDotNet.Tensors/issues/821)) ([4c44737](https://github.com/ooples/AiDotNet.Tensors/commit/4c447370c0a08e4777de2d5f68fe2868efe92f20))

## [0.117.1](https://github.com/ooples/AiDotNet.Tensors/compare/v0.117.0...v0.117.1) (2026-07-19)


### Bug Fixes

* GPU [#226](https://github.com/ooples/AiDotNet.Tensors/issues/226) concurrency crash and a chain of pre-existing DirectGpu bugs ([#818](https://github.com/ooples/AiDotNet.Tensors/issues/818)) ([4af7ba8](https://github.com/ooples/AiDotNet.Tensors/commit/4af7ba83ed755be90ad9b8017713a417366dc1e0))

## [0.117.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.116.0...v0.117.0) (2026-07-19)


### Features

* **attention:** optional logit soft-cap on ScaledDotProductAttention (CPU + all 6 GPU backends) ([#816](https://github.com/ooples/AiDotNet.Tensors/issues/816)) ([ddccf57](https://github.com/ooples/AiDotNet.Tensors/commit/ddccf57207c9000ad51399fbedbd4d6f81e27bc9))
* batch-vectorized float FFT + CUDA/HIP FFT correctness fixes ([#811](https://github.com/ooples/AiDotNet.Tensors/issues/811)) ([#812](https://github.com/ooples/AiDotNet.Tensors/issues/812)) ([77320ef](https://github.com/ooples/AiDotNet.Tensors/commit/77320ef07cf4ccf5f51115651cb2ec8c7345f2b1))


### Bug Fixes

* **cpu:** bound Tanh/Sigmoid SIMD backward by logical length (fixes OOB write / AccessViolation) ([#815](https://github.com/ooples/AiDotNet.Tensors/issues/815)) ([8daadc1](https://github.com/ooples/AiDotNet.Tensors/commit/8daadc1bd808bc48579cdf8416d69b346a970aba))

## [0.116.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.115.1...v0.116.0) (2026-07-18)


### Features

* **gpu:** expose paged-attention kernels via IPagedAttentionBackend capability interface ([#806](https://github.com/ooples/AiDotNet.Tensors/issues/806)) ([2ccba00](https://github.com/ooples/AiDotNet.Tensors/commit/2ccba0029f5e3e7ccad7740832e018cdbc4ee683))
* **licensing:** P1 mirror — scope binding + jti/CRL revocation on signed entitlements ([#808](https://github.com/ooples/AiDotNet.Tensors/issues/808)) ([1db25fc](https://github.com/ooples/AiDotNet.Tensors/commit/1db25fcdfc0c2c503c672c1c3373f54470a755f5))


### Refactoring

* **gpu:** put paged-attention on IDirectGpuBackend; drop the capability interface ([#809](https://github.com/ooples/AiDotNet.Tensors/issues/809)) ([c369ab4](https://github.com/ooples/AiDotNet.Tensors/commit/c369ab4cc692500b9fff42b8212352d1dd5bca2a))

## [0.115.1](https://github.com/ooples/AiDotNet.Tensors/compare/v0.115.0...v0.115.1) (2026-07-17)


### Bug Fixes

* **ci:** repair line-ending autofix workflow (revives release automation) ([#804](https://github.com/ooples/AiDotNet.Tensors/issues/804)) ([e35fc35](https://github.com/ooples/AiDotNet.Tensors/commit/e35fc355adf4eacfa025a0c65f853f003eef95be))

## [0.115.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.114.5...v0.115.0) (2026-07-16)


### Features

* **gpu:** cross-backend CPU/GPU op-parity + full residency ([#775](https://github.com/ooples/AiDotNet.Tensors/issues/775)) ([#790](https://github.com/ooples/AiDotNet.Tensors/issues/790)) ([106bbca](https://github.com/ooples/AiDotNet.Tensors/commit/106bbca0ff1213c503b5b83520a8b18c959088c4))

## [0.114.5](https://github.com/ooples/AiDotNet.Tensors/compare/v0.114.4...v0.114.5) (2026-07-15)


### Bug Fixes

* **autodiff:** manual backward tape nodes now propagate input gradients through the graph path ([#786](https://github.com/ooples/AiDotNet.Tensors/issues/786)) ([db5029d](https://github.com/ooples/AiDotNet.Tensors/commit/db5029dbe8b0e631ec7e939f195e70d08cf31329))
* **gpu:** LayerNorm forward returns TRUE variance, not inverse-std, so compiled-replay backward gradients are correct ([#784](https://github.com/ooples/AiDotNet.Tensors/issues/784)) ([e735690](https://github.com/ooples/AiDotNet.Tensors/commit/e7356905c2d8f1c6dfb1352ac4007c1716351784))

## [0.114.4](https://github.com/ooples/AiDotNet.Tensors/compare/v0.114.3...v0.114.4) (2026-07-15)


### Bug Fixes

* **autodiff:** manual backward tape nodes now propagate input gradients through the graph path ([#786](https://github.com/ooples/AiDotNet.Tensors/issues/786)) ([3547028](https://github.com/ooples/AiDotNet.Tensors/commit/3547028282295b735cb484451459be19bab7a32b))

## [0.114.3](https://github.com/ooples/AiDotNet.Tensors/compare/v0.114.2...v0.114.3) (2026-07-13)


### Performance

* **autotune:** eliminate per-GEMM disk touches on the compiled-replay hot path ([#782](https://github.com/ooples/AiDotNet.Tensors/issues/782)) ([4fedc3b](https://github.com/ooples/AiDotNet.Tensors/commit/4fedc3b9e78accd596849459ff50dd272663fd37))

## [0.114.2](https://github.com/ooples/AiDotNet.Tensors/compare/v0.114.1...v0.114.2) (2026-07-13)


### Bug Fixes

* **cpu:** support groups&lt;deformGroups in DeformableConv2DGrouped (key deform group on input channel) ([#780](https://github.com/ooples/AiDotNet.Tensors/issues/780)) ([2b5a080](https://github.com/ooples/AiDotNet.Tensors/commit/2b5a0809389eed4df45c105954dfb5a99abbde36))

## [0.114.1](https://github.com/ooples/AiDotNet.Tensors/compare/v0.114.0...v0.114.1) (2026-07-13)


### Bug Fixes

* **cpu:** race-free GridSample-3D backward scatter (thread-local accumulators) ([#774](https://github.com/ooples/AiDotNet.Tensors/issues/774)) ([7773fc8](https://github.com/ooples/AiDotNet.Tensors/commit/7773fc8ac2048d0920589375d9e349a3e9dbfb54))
* **fft:** guard IRFFT against degenerate length-1 transform (nFft=0 OOR) ([#779](https://github.com/ooples/AiDotNet.Tensors/issues/779)) ([f72615b](https://github.com/ooples/AiDotNet.Tensors/commit/f72615b7bf131f6124a6fccad36755f4b320268e))

## [0.114.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.113.0...v0.114.0) (2026-07-11)


### Features

* **engine:** DepthwiseConv1D primitive (CPU + GPU) for time-channel separable convs ([#772](https://github.com/ooples/AiDotNet.Tensors/issues/772)) ([f72b182](https://github.com/ooples/AiDotNet.Tensors/commit/f72b1827d3d145eb1c3fed12e737ff73eacb67ff))


### Performance

* **parallel:** adaptive spare-core-gated warm-spin for the persistent worker pool ([#762](https://github.com/ooples/AiDotNet.Tensors/issues/762)) ([e5b2a0a](https://github.com/ooples/AiDotNet.Tensors/commit/e5b2a0afece8bcbbefefcbcb23158d3dd87007e6))

## [0.113.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.112.0...v0.113.0) (2026-07-10)


### Features

* **training:** compiled backward + createGraph, DP-SGD, multi-slot persistent inputs ([#763](https://github.com/ooples/AiDotNet.Tensors/issues/763)) ([b7cf72e](https://github.com/ooples/AiDotNet.Tensors/commit/b7cf72e1838455f9d57b46e2e0d2e4f0c953e3d1))


### Performance

* **compile:** SIMD-vectorize global-L2 gradient clip in fused training ([#766](https://github.com/ooples/AiDotNet.Tensors/issues/766)) ([ea98267](https://github.com/ooples/AiDotNet.Tensors/commit/ea98267f85d5f762f07313563064627c952f3b5e))
* fused-training optimizer + arena suite (clip SIMD, moment pooling, AdamW fusion, bias-corr hoist, multi-tensor foreach) ([#770](https://github.com/ooples/AiDotNet.Tensors/issues/770)) ([7c8455d](https://github.com/ooples/AiDotNet.Tensors/commit/7c8455db631c4acc9abd9a6f4420b1e0251f400d))

## [0.112.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.111.2...v0.112.0) (2026-07-10)


### Features

* **sparse:** tape-auto-tracking sparse engine ops + fully GPU-resident sparse autograd backward ([#758](https://github.com/ooples/AiDotNet.Tensors/issues/758)) ([4a865b7](https://github.com/ooples/AiDotNet.Tensors/commit/4a865b70a1426f2ccdcebfff22bbf6902669e8fb))


### Bug Fixes

* **compile:** fill entire output in compiled ReduceMax axis reduction ([#765](https://github.com/ooples/AiDotNet.Tensors/issues/765)) ([d04f8ca](https://github.com/ooples/AiDotNet.Tensors/commit/d04f8cafcd8bda78fa2c6d73ac1cfae87103ff41))
* **compile:** zero only accumulating multi-consumer grad buffers (N-BEATS resident blow-up) ([#759](https://github.com/ooples/AiDotNet.Tensors/issues/759)) ([d94b919](https://github.com/ooples/AiDotNet.Tensors/commit/d94b91924e51d707dbdbfa29f98d4608df4ffdde))
* **gpu:** resident params no longer flat-line FP32 fused-Adam training ([#764](https://github.com/ooples/AiDotNet.Tensors/issues/764)) ([c8da627](https://github.com/ooples/AiDotNet.Tensors/commit/c8da6270bb3241b7cbf1202a7760adf419132a7d))

## [0.111.2](https://github.com/ooples/AiDotNet.Tensors/compare/v0.111.1...v0.111.2) (2026-07-09)


### Bug Fixes

* **cpu:** zero the unwritten tail in the axis-0 concat fast path ([#755](https://github.com/ooples/AiDotNet.Tensors/issues/755)) ([369b36a](https://github.com/ooples/AiDotNet.Tensors/commit/369b36a1a7f025fbb7cbfa094698d80906bd84a4))

## [0.111.1](https://github.com/ooples/AiDotNet.Tensors/compare/v0.111.0...v0.111.1) (2026-07-09)


### Bug Fixes

* **autodiff:** make 2D AffineGrid tape-aware + vectorize (fixes STN untrainable) ([#751](https://github.com/ooples/AiDotNet.Tensors/issues/751)) ([890ed53](https://github.com/ooples/AiDotNet.Tensors/commit/890ed53656bc409da8930380085feeb59390adcf))
* **compiled-training:** keep plan activation buffers alive across replay — unfreezes fused training at large vocab ([#754](https://github.com/ooples/AiDotNet.Tensors/issues/754)) ([687a4c6](https://github.com/ooples/AiDotNet.Tensors/commit/687a4c6ad2a031c9ad5f5caa91c0df009e0abdd9))


### Performance

* **arena:** recycle transformer-layer scratch between layers for memory-bounded inference (AiDotNet[#1824](https://github.com/ooples/AiDotNet.Tensors/issues/1824)) ([#753](https://github.com/ooples/AiDotNet.Tensors/issues/753)) ([2572bd8](https://github.com/ooples/AiDotNet.Tensors/commit/2572bd80e4b9014ffb9b3ee5ad2d9ad8f79fe44b))

## [0.111.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.110.1...v0.111.0) (2026-07-07)


### Features

* **simd:** dual-path Adam kernel — bit-exact default + AIDOTNET_FAST_MATH FMA fast path ([#738](https://github.com/ooples/AiDotNet.Tensors/issues/738)) ([d670ce3](https://github.com/ooples/AiDotNet.Tensors/commit/d670ce31750d09311125761c9d3db8ed2dcc7bdf))


### Bug Fixes

* **autodiff:** make 2D AffineGrid differentiable w.r.t. theta ([#747](https://github.com/ooples/AiDotNet.Tensors/issues/747)) ([72e4116](https://github.com/ooples/AiDotNet.Tensors/commit/72e4116efcf33c11310f54fe0401f3597477db06))
* **bench:** gate gemm-bench on the median per-round paired ON/OFF ratio ([#741](https://github.com/ooples/AiDotNet.Tensors/issues/741)) ([39194bb](https://github.com/ooples/AiDotNet.Tensors/commit/39194bbf96e44f40c5d13969b05a565f732405b1))
* **gpu:** fused-optimizer weight-coherence — refresh resident device buffer after in-place update ([#739](https://github.com/ooples/AiDotNet.Tensors/issues/739)) ([8e61dc4](https://github.com/ooples/AiDotNet.Tensors/commit/8e61dc48df46d7981044fe98b22dc1b6a453218c))
* **gpu:** hardware-independent determinism for scatter-add grad + cuBLAS under SetDeterministicMode (Closes [#742](https://github.com/ooples/AiDotNet.Tensors/issues/742)) ([#744](https://github.com/ooples/AiDotNet.Tensors/issues/744)) ([b815872](https://github.com/ooples/AiDotNet.Tensors/commit/b8158726d08bc72b85bef25fabffef811aef6938))


### Performance

* **gemm:** beta=0 write-first path — skip redundant C-zeroing on pure C:=A·B ([#743](https://github.com/ooples/AiDotNet.Tensors/issues/743)) ([2aac704](https://github.com/ooples/AiDotNet.Tensors/commit/2aac704a5d9a55ff2407903a95ad74bae164e18b))
* kill training spin-storm — lower resident-pool MRES spin (2047 -&gt; 32) ([#736](https://github.com/ooples/AiDotNet.Tensors/issues/736)) ([8ede01f](https://github.com/ooples/AiDotNet.Tensors/commit/8ede01f1c9802bbafbc6eed2fd7c6db2e94a00da))
* zero-alloc training via TensorArena reuse + shape-safe ring (DRAFT) ([#734](https://github.com/ooples/AiDotNet.Tensors/issues/734)) ([3f7178a](https://github.com/ooples/AiDotNet.Tensors/commit/3f7178af342e4ed33871501ae160fd51479e858d))

## [0.103.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.102.16...v0.103.0) (2026-07-07)


### Features

* **cuda:** wire BatchNorm to cuDNN + fix cuDNN context binding (AiDotNet[#1159](https://github.com/ooples/AiDotNet.Tensors/issues/1159)) ([#726](https://github.com/ooples/AiDotNet.Tensors/issues/726)) ([5ff173c](https://github.com/ooples/AiDotNet.Tensors/commit/5ff173c746319fcc7e00b1c2843cdad0c2683856))
* **dcnv3:** fused grouped/depthwise deformable convolution kernel [WIP] ([#691](https://github.com/ooples/AiDotNet.Tensors/issues/691)) ([c16345f](https://github.com/ooples/AiDotNet.Tensors/commit/c16345f007f7eb8fb8b95b45a87f04849b823199))
* **dcnv3:** single-launch fused grouped deformable backward (GPU) + CPU follow-up ([#693](https://github.com/ooples/AiDotNet.Tensors/issues/693)) ([4006015](https://github.com/ooples/AiDotNet.Tensors/commit/4006015f4cac96785b7a6b46c4c4941cde920415))
* **fused:** bfloat16 moment storage for fused Adam/AdamW (CPU) ([#713](https://github.com/ooples/AiDotNet.Tensors/issues/713)) ([02655de](https://github.com/ooples/AiDotNet.Tensors/commit/02655dea06ffba03c143e761f88e28e32a8e8f6b))
* **gemm:** opt-in bf16-B GEMM + fused epilogue on the fast GotoGemm path ([#706](https://github.com/ooples/AiDotNet.Tensors/issues/706)) ([5fa4cff](https://github.com/ooples/AiDotNet.Tensors/commit/5fa4cfffafd870b89734584646f995f3fb851cbb))
* **simd:** shared bit-identical simd adam/amsgrad step kernel ([#1757](https://github.com/ooples/AiDotNet.Tensors/issues/1757)) ([#721](https://github.com/ooples/AiDotNet.Tensors/issues/721)) ([628abd5](https://github.com/ooples/AiDotNet.Tensors/commit/628abd516a4029917861df88c92f94b4098f2df9))
* **streaming:** RAM-aware Auto store — step bf16 -&gt; int8 -&gt; int4 to fit resident cap in inference ([#720](https://github.com/ooples/AiDotNet.Tensors/issues/720)) ([c8681be](https://github.com/ooples/AiDotNet.Tensors/commit/c8681be3e8c303f19570c9ab6d2c96cdbda4d01d))


### Bug Fixes

* **#638:** CUDA-graph capture training now LEARNS (device barrier + deterministic scratch pool + LN resident-weight gate) ([#705](https://github.com/ooples/AiDotNet.Tensors/issues/705)) ([25132d1](https://github.com/ooples/AiDotNet.Tensors/commit/25132d1681130f40ef06f31b2eb97c61e3f7fe6d))
* **autodiff:** route backward row-parallelism through the cooperative pool (fixes single-core training) ([#729](https://github.com/ooples/AiDotNet.Tensors/issues/729)) ([c2b85f2](https://github.com/ooples/AiDotNet.Tensors/commit/c2b85f2110ccd523db01675956084601017d990c))
* **blas:** default to managed BlasManaged; make native CPU BLAS opt-in ([#686](https://github.com/ooples/AiDotNet.Tensors/issues/686)) ([5ca9f14](https://github.com/ooples/AiDotNet.Tensors/commit/5ca9f14476cfbff89352f872a396819d7c2eb06c))
* **blas:** real fix for the Linux FP32 GEMM crash — add a SysV-ABI tile kernel (GotoGemm now cross-platform) ([#699](https://github.com/ooples/AiDotNet.Tensors/issues/699)) ([b7d4582](https://github.com/ooples/AiDotNet.Tensors/commit/b7d4582c12a41fb2f7f288176e0aac658ffcf402))
* **build:** cold builds fail — split LicenseValidator Restore/Build into distinct-property MSBuild calls ([#724](https://github.com/ooples/AiDotNet.Tensors/issues/724)) ([ec9b889](https://github.com/ooples/AiDotNet.Tensors/commit/ec9b8892d6f42fb8146d6abd7a08e140dd0f8303))
* **ci:** harden historical shard regressions ([#718](https://github.com/ooples/AiDotNet.Tensors/issues/718)) ([6ac5bc6](https://github.com/ooples/AiDotNet.Tensors/commit/6ac5bc6628ec370d6f8c9a13eeeaa2c867421577))
* **compilation:** fused optimizer NREs on params with no materialized weights (kicks lazy/multi-modal models off the fused path) ([#712](https://github.com/ooples/AiDotNet.Tensors/issues/712)) ([b54086e](https://github.com/ooples/AiDotNet.Tensors/commit/b54086e3190aaa7fc56c5e3c04929f8fa6d0d73f))
* **compilation:** stop topo-sort dropping in-place op chains (BatchNorm running-stat staleness) ([#704](https://github.com/ooples/AiDotNet.Tensors/issues/704)) ([0f764be](https://github.com/ooples/AiDotNet.Tensors/commit/0f764bec1d5741f31a7dd8fbeef6de6eb6baed99))
* **compile:** refresh norm statistics in the fused training forward (GroupNorm + BatchNorm) ([#722](https://github.com/ooples/AiDotNet.Tensors/issues/722)) ([7f2c94d](https://github.com/ooples/AiDotNet.Tensors/commit/7f2c94da78329d8cc6bf6e2316f26ce3690a3f87))
* **cpu:** invalidate per-array derived-weight caches on invalidatepersistenttensor ([#708](https://github.com/ooples/AiDotNet.Tensors/issues/708)) ([d03ea35](https://github.com/ooples/AiDotNet.Tensors/commit/d03ea35f36dd8ab31ad1cf18b6c62667e505667f))
* **gpu:** apply softmax row-wise in fused-linear epilogue (was crash/garbage) ([#711](https://github.com/ooples/AiDotNet.Tensors/issues/711)) ([e7fbc3e](https://github.com/ooples/AiDotNet.Tensors/commit/e7fbc3e3d995d43a821b23c2e3c439e5ec6e9b9d))
* **gpu:** clamp box w/h in IoU-family loss kernels so degenerate boxes match CPU ([#709](https://github.com/ooples/AiDotNet.Tensors/issues/709)) ([deaf1ea](https://github.com/ooples/AiDotNet.Tensors/commit/deaf1eae23ebfe011fb3bb93253b6762649085d4))
* **gpu:** CUDA Dropout dropped 100% of activations (mask never generated) + stale seed ([#732](https://github.com/ooples/AiDotNet.Tensors/issues/732)) ([41d923f](https://github.com/ooples/AiDotNet.Tensors/commit/41d923f7c11313f5f398ed4642bfcc4aaad6a243))
* **gpu:** DetectVendor must prefer CUDA over OpenCL device-0 vendor on mixed boxes ([#733](https://github.com/ooples/AiDotNet.Tensors/issues/733)) ([e8a4fb4](https://github.com/ooples/AiDotNet.Tensors/commit/e8a4fb4b3160fb50bfe5a3e9a3cc7d1ef2e71d15))
* **gpu:** LayerNorm/RMSNorm normalize over gamma dims for rank &gt; 2 (fixes GPU transformer forward) ([#730](https://github.com/ooples/AiDotNet.Tensors/issues/730)) ([82464c2](https://github.com/ooples/AiDotNet.Tensors/commit/82464c2f8da9be6b50c3f940e5d19322178dbf45))
* **streaming:** defer ReleaseToPool drop on shared storage (refcount &gt; 1) ([#692](https://github.com/ooples/AiDotNet.Tensors/issues/692)) ([0cf29fd](https://github.com/ooples/AiDotNet.Tensors/commit/0cf29fd18830aefd52cbe07e64f5c8d0f10c82bf))
* **streaming:** quiesce in-flight prefetch workers before pool teardown ([#687](https://github.com/ooples/AiDotNet.Tensors/issues/687)) ([f833a92](https://github.com/ooples/AiDotNet.Tensors/commit/f833a92af0f9e8ee164f3586853a7779eaac20a1))
* **streaming:** reclaim WeightRegistry entries for GC'd owners ([#714](https://github.com/ooples/AiDotNet.Tensors/issues/714)) ([#716](https://github.com/ooples/AiDotNet.Tensors/issues/716)) ([be97676](https://github.com/ooples/AiDotNet.Tensors/commit/be97676be3916d1a4e59a832b9d312bdf46f4477))


### Performance

* **#1650/#638:** diffusion UNet forward CUDA-graph capture ENGAGES — whole forward resident, ~3.2x (replay bit-exact) ([#671](https://github.com/ooples/AiDotNet.Tensors/issues/671)) ([1eec88e](https://github.com/ooples/AiDotNet.Tensors/commit/1eec88e6ae30124d6e98c1b38a684cbf3366f03b))
* **#1672:** *Into resident-scratch engine kernels (SDPA/AdaLN/FusedLinear) for diffusion inference ([#696](https://github.com/ooples/AiDotNet.Tensors/issues/696)) ([3652dc7](https://github.com/ooples/AiDotNet.Tensors/commit/3652dc7e427251d34442ac7e4315d209900cd58f))
* **#1715:** bound streaming-pool resident set on the GetParameters/register path (foundation-scale OOM) ([#707](https://github.com/ooples/AiDotNet.Tensors/issues/707)) ([170517e](https://github.com/ooples/AiDotNet.Tensors/commit/170517ef49ae10eec762f095c30ce5ee1df9f809))
* **#653:** route StreamingStrategy GEMM dispatch onto low-latency PersistentParallelExecutor ([#688](https://github.com/ooples/AiDotNet.Tensors/issues/688)) ([d9e7462](https://github.com/ooples/AiDotNet.Tensors/commit/d9e7462e32a75bb06bff70d6b53737d9d1a85ba2))
* **attn:** scatter-lock GC sweep — GQA backward 6.4x less GC + 3D-op locks ([#698](https://github.com/ooples/AiDotNet.Tensors/issues/698)) ([f9596de](https://github.com/ooples/AiDotNet.Tensors/commit/f9596dee4b86f63ca7d11cc51ada0f53d771c287))
* **blas:** managed FP32 GEMM — GotoBLAS rewrite + CCX-2D-NUMA + short-M campaign (DiT ~2x; sq4096 &gt; throttled MKL) ([#695](https://github.com/ooples/AiDotNet.Tensors/issues/695)) ([5f19acb](https://github.com/ooples/AiDotNet.Tensors/commit/5f19acb843653604bec84c4d4d368a3c878c810a))
* cache per-op env-var reads + skip zero-fill on small RentUninitialized ([#731](https://github.com/ooples/AiDotNet.Tensors/issues/731)) ([61ab50f](https://github.com/ooples/AiDotNet.Tensors/commit/61ab50fae556adc9c47fbbc239d5e69b5667d823))
* **conv:** batch=1 conv-backward-kernel writes gradient straight into dest ([#689](https://github.com/ooples/AiDotNet.Tensors/issues/689)) ([a1d61e0](https://github.com/ooples/AiDotNet.Tensors/commit/a1d61e016e6c68f336bcf9edd635e60c18b6b214))
* **conv:** bounded large-array pool for conv-backward stacks — ~8x less GC traffic ([#1691](https://github.com/ooples/AiDotNet.Tensors/issues/1691) follow-up) ([#694](https://github.com/ooples/AiDotNet.Tensors/issues/694)) ([aeaa5ea](https://github.com/ooples/AiDotNet.Tensors/commit/aeaa5ea360fa7802660bda21861ce8e79b896896))
* **conv:** route large-spatial 3x3 convs to implicit-GEMM (24x forward, 8x CNN step) ([#690](https://github.com/ooples/AiDotNet.Tensors/issues/690)) ([7a8a727](https://github.com/ooples/AiDotNet.Tensors/commit/7a8a727e3b66766fcf8c5b0e9c90029c9b3f6011))
* **cpu:** parallelize 3 linear-attn scans + fused fp16-weight GEMM ([#723](https://github.com/ooples/AiDotNet.Tensors/issues/723)) ([c840501](https://github.com/ooples/AiDotNet.Tensors/commit/c84050122773ac7d35eea46735c71effa6cd4415))
* **dcnv3:** make grouped deformable-conv backward ~2.9× faster (GC + dispatch + lock contention) ([#697](https://github.com/ooples/AiDotNet.Tensors/issues/697)) ([9d60add](https://github.com/ooples/AiDotNet.Tensors/commit/9d60addd4cd8fe939838d6ec59eaacf94b91fc89))
* **gemm:** exceed OpenBLAS — JIT small-GEMM (1.5–6×) + scientific multi-thread routing/pinning wins on diffusion shapes ([#700](https://github.com/ooples/AiDotNet.Tensors/issues/700)) ([ce14327](https://github.com/ooples/AiDotNet.Tensors/commit/ce14327c77f06ea39190b0f55586714ed3fd5b21))
* **geometry:** kill per-element indexer/numOps dispatch in sampling+distance kernels (GridSample ~50x +3) ([#702](https://github.com/ooples/AiDotNet.Tensors/issues/702)) ([1224c7d](https://github.com/ooples/AiDotNet.Tensors/commit/1224c7d16cf5a28bc8d3620ce852a3df9e66ae54))
* **ssm:** parallelize fused SSM/linear-attention scan kernels (Mamba, GLA, RWKV-7, GatedDeltaNet, xLSTM) ([#703](https://github.com/ooples/AiDotNet.Tensors/issues/703)) ([26fb3f4](https://github.com/ooples/AiDotNet.Tensors/commit/26fb3f47f4e7e0d65a7ca79a0a7d527373fc0095))
