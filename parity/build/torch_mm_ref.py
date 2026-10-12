# GUARD_OK: PyTorch API-parity reference values for torch.nn.functional.grouped_mm / scaled_mm (fp8 GEMM operator
# semantics). Not HRE model-weight compression: no model, no weights, only operator outputs for fixed random inputs.
import torch, torch.nn.functional as F, json

S = F.ScalingType
g = torch.Generator().manual_seed(3)
R = lambda *s: torch.randn(*s, generator=g)
out = {}


def put(name, **t):
    d = {k: (v.flatten().tolist() if torch.is_tensor(v) else v) for k, v in t.items()}
    d.update({k + "_shape": list(v.shape) for k, v in t.items() if torch.is_tensor(v)})
    out[name] = d


# grouped_mm, the four layouts (CPU needs 16-byte aligned strides: dims multiples of 4)
a = R(12, 8); b = R(2, 8, 4); offs = torch.tensor([4, 12], dtype=torch.int32)   # jagged rows: 4 and 8
put("g_2d3d", a=a, b=b, offs=offs, out=F.grouped_mm(a, b, offs=offs))
a = R(2, 4, 8); b = R(2, 8, 4)
put("g_3d3d", a=a, b=b, out=F.grouped_mm(a, b))
a = R(4, 12); b = R(12, 8); offs = torch.tensor([4, 12], dtype=torch.int32)    # jagged K
put("g_2d2d", a=a, b=b, offs=offs, out=F.grouped_mm(a, b, offs=offs))
a = R(2, 4, 8); b = R(8, 12); offs = torch.tensor([4, 12], dtype=torch.int32)   # jagged N
put("g_3d2d", a=a, b=b, offs=offs, out=F.grouped_mm(a, b, offs=offs))
# (grouped_mm bias: torch 2.12 raises "Bias not supported yet"; the library adds it per group, tested by equivalence)

# scaled_mm: float8 operands, handed to the library as their decoded values plus the decoding scales
f8 = lambda x: x.to(torch.float8_e4m3fn)
a8 = f8(R(16, 32)); b8 = f8(R(24, 32))
sa = torch.tensor(1.75); sb = torch.tensor(0.5)
put("s_tensorwise", a=a8.float(), b=b8.float().t().contiguous(), sa=sa, sb=sb,
    out=F.scaled_mm(a8, b8.t(), sa, S.TensorWise, sb, S.TensorWise, output_dtype=torch.float32))
sa = torch.rand(16, 1, generator=g) + 0.5; sb = torch.rand(1, 24, generator=g) + 0.5; bias = R(24).bfloat16()
put("s_rowwise", a=a8.float(), b=b8.float().t().contiguous(), sa=sa, sb=sb, bias=bias.float(),
    out=F.scaled_mm(a8, b8.t(), sa, S.RowWise, sb, S.RowWise, bias=bias, output_dtype=torch.float32))

json.dump(out, open("torch_mm_ref.json", "w"))
print({k: v["out_shape"] for k, v in out.items()})
