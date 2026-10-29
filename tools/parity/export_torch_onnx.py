"""Export the #1574 interop models from PyTorch to ONNX, with a seeded input and torch's own output.

    python tools/parity/export_torch_onnx.py --out DIR

Writes, per model, DIR/<name>.onnx, DIR/<name>.input.bin and DIR/<name>.torch.bin (little-endian float32) and
DIR/<name>.json (input name and shape, output shape). OnnxTorchExportTests (AiDotNet.Tensors.Onnx.Tests) reads
them when AIDOTNET_TORCH_ONNX_DIR names DIR: it imports each model into Tensors, exports it back to ONNX and checks
Tensors, the re-export in ONNX Runtime, and torch against each other.

Models:
  resnet18     torchvision's resnet18 architecture (defined here so only torch is needed), eval mode, 1x3x64x64.
  bert_encoder a 2-layer BERT-style encoder: nn.TransformerEncoder, post-LayerNorm, GELU, 4 heads, d=64, seq 16.
"""
import argparse
import json
import os

import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.downsample = None
        if stride != 1 or cin != cout:
            self.downsample = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + (x if self.downsample is None else self.downsample(x)))


class ResNet18(nn.Module):
    def __init__(self, classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        layers, cin = [], 64
        for cout, stride in ((64, 1), (128, 2), (256, 2), (512, 2)):
            layers.append(nn.Sequential(BasicBlock(cin, cout, stride), BasicBlock(cout, cout, 1)))
            cin = cout
        self.layer1, self.layer2, self.layer3, self.layer4 = layers
        self.fc = nn.Linear(512, classes)

    def forward(self, x):
        x = self.maxpool(torch.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(nn.functional.adaptive_avg_pool2d(x, 1), 1))


def randomize_bn(model, gen):
    # Fresh BN stats are the identity; non-trivial running stats exercise the fold the way a trained model does.
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.copy_(torch.randn(m.num_features, generator=gen) * 0.1)
            m.running_var.copy_(torch.rand(m.num_features, generator=gen) * 0.5 + 0.75)
            m.weight.data.copy_(torch.rand(m.num_features, generator=gen) * 0.4 + 0.8)
            m.bias.data.copy_(torch.randn(m.num_features, generator=gen) * 0.1)


def export(name, model, x, out_dir, opset):
    model.eval()
    with torch.no_grad():
        y = model(x)
        torch.onnx.export(model, (x,), os.path.join(out_dir, name + ".onnx"), input_names=["input"],
                          output_names=["output"], opset_version=opset, do_constant_folding=True, dynamo=False)
    x.numpy().astype("<f4").tofile(os.path.join(out_dir, name + ".input.bin"))
    y.numpy().astype("<f4").tofile(os.path.join(out_dir, name + ".torch.bin"))
    with open(os.path.join(out_dir, name + ".json"), "w") as f:
        json.dump({"input": "input", "inputShape": list(x.shape), "outputShape": list(y.shape),
                   "torch": torch.__version__, "opset": opset}, f)
    print(f"{name}: input {list(x.shape)} -> output {list(y.shape)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1)

    resnet = ResNet18()
    with torch.no_grad():
        randomize_bn(resnet, gen)
    export("resnet18", resnet, torch.randn(1, 3, 64, 64, generator=gen), args.out, 17)

    # The fused fast path of nn.MultiheadAttention is not exportable; the composed ops are what BERT exports to.
    torch.backends.mha.set_fastpath_enabled(False)
    layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256, dropout=0.0,
                                       activation="gelu", batch_first=True, norm_first=False)
    encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
    export("bert_encoder", encoder, torch.randn(1, 16, 64, generator=gen), args.out, 17)


if __name__ == "__main__":
    main()
