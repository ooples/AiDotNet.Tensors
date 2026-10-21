# Changelog

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
