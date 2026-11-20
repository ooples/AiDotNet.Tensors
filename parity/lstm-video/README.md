# Reproduce the 17-second AiDotNet vs PyTorch video

These are the two training programs and the screen recorder used for the published clip. Each model learns which of ten channels contains a pulse in a noisy signal. Both start from the same weights and see the same batches. [Watch the original video](evidence/original.mp4) · [C# training loop](original/Program.cs) · [Python PyTorch training loop](original/torch_train.py) · [Screen recorder](original/record_pair.py)

| Recorded on one GTX 1660 Ti | AiDotNet | Python PyTorch |
| --- | ---: | ---: |
| Elapsed time for 4,096 updates, including accuracy checks | 2.5996422 s | 15.0012375 s |
| Correct classifications on 320 held-out signals | 302 | 301 |
| Correct classifications before training | 37 | 37 |

The numbers are the last line in each trainer's log. Each log includes the elapsed time, loss and held-out score at every 64th update. [AiDotNet checkpoints](evidence/ai/live.jsonl) · [PyTorch checkpoints](evidence/torch/live.jsonl) · [Original run metadata](evidence/result.json)

## Run it yourself

Use Git, the .NET 10 SDK, Python 3.13 and an NVIDIA GPU with a compatible driver and CUDA development libraries. On Windows use the x64 Developer PowerShell for Visual Studio with the C++ tools. From the root of this Tensors checkout, activate a Python virtual environment and install the versions used for the clip:

```shell
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install numpy==2.5.4 psutil==7.2.2 safetensors==0.8.0
```

On Windows also install `python -m pip install triton-windows==3.9.0.post30`. Then run:

```shell
python parity/lstm-video/run.py
```

The first invocation fetches the pinned source, builds AiDotNet, generates the original data and checks its SHA-256. It runs PyTorch and AiDotNet one at a time. The output shows your times, your held-out scores, and the location of both raw logs. The build and initial PyTorch compilation can take several minutes. Use `--work PATH` to choose where to put the downloads and results.

To record your own comparison, install the optional capture tools and add `--record`:

```shell
python -m pip install playwright==1.58.0 imageio-ffmpeg==0.8.1
python -m playwright install chromium
python parity/lstm-video/run.py --record
```

This records two live AiDotNet.Dashboard panels with Playwright. Each panel reads checkpoints written by its trainer. The recorder aligns the two captures at the start of timing, plays both at normal speed and holds the first panel on its final frame after its run ends. The output includes a new MP4. The original, separate screen captures are available too. [AiDotNet screen capture](evidence/ai/raw.webm) · [PyTorch screen capture](evidence/torch/raw.webm) · [Dashboard panel code](original/panel.js)

## What is being compared?

- A single-layer LSTM has 32 inputs per time step, 32 time steps, 64 hidden units and a linear head with 10 outputs. Both implementations have 25,482 trainable parameters. PyTorch's standard `nn.LSTM` has two bias vectors. The model freezes the second at zero to match the C# model.
- Each batch has 64 sequences. Training cycles through 256 batches. Five separate batches supply the 320 held-out signals. The exact starting weights ship here and the data generator uses seed 20261119. [Generator](original/fixture.py) · [Starting weights](lstm.weights.safetensors)
- Both sides take eight actual warm-up updates before the clock starts and then run 4,096 more. Every 64 updates both evaluate all 320 held-out signals and synchronize the GPU before recording elapsed time. This includes validation in both times.
- Python uses CUDA PyTorch 2.14.1+cu126, `torch.compile(model)` with its default settings, cross-entropy, backpropagation, gradient clipping at 1.0 and fused AdamW. AiDotNet uses its fused CUDA LSTM training path. Its training program checks that the fused path handled all 4,104 updates. The original recording does not contain a profiler trace of which PyTorch operations were compiled. These times describe this particular pair of implementations on one laptop.

## Exact source and evidence

The six files under [original/](original) are byte-for-byte copies from the recording directory. The launcher makes a fresh run directory, changes the project reference to the restored local source and preloads the pinned Python model from that checkout. The original trainers, input generator, dashboard layout and recorder then execute as written. [Input, source, assembly and recording hashes](evidence/sha256.json)

The recorded AiDotNet training assembly identified commit `0f2902c916`, the dashboard identified `71a1dc700a`, and the Tensors assembly identified `75b03c253`. Only a test file changed between the two AiDotNet commits. The launcher builds the identical product sources at `71a1dc700a` and the exact Tensors revision, using Git bundles for the historical integration commits. Builds made in another directory may have different binary hashes. [Pinned source revisions](sources/revisions.json)

For the separate two-launch CPU/CUDA comparison across 11 model families, see the [model-family benchmark](../model-families/README.md).
