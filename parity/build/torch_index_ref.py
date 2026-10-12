import torch, json
x = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
L = lambda *v: torch.tensor(v, dtype=torch.long)
cases = {
  "single_dim0":      [L(1, 0, 1)],
  "adjacent_12":      [None, L(0, 2), L(3, 1)],
  "broadcast_12":     [None, L(0, 2).reshape(2, 1), L(3, 1, 0)],
  "separated_0_2":    [L(1, 0), None, L(2, 3)],
  "negative":         [L(-1), None, L(-2, 0)],
  "trailing_none":    [None, None, L(1, 3)],
  "all_four":         [L(0, 1), L(2, 0), L(3, 1), L(4, 0)],
}
out = {}
for k, idx in cases.items():
    r = torch.ops.aten.index(x, idx)
    out[k] = {"idx": [None if t is None else {"v": t.flatten().tolist(), "s": list(t.shape)} for t in idx],
              "out": r.flatten().tolist(), "shape": list(r.shape)}
m = torch.tensor([[True, False, True], [False, True, False]])
r = torch.ops.aten.index(x, [m])
out["bool_mask_01"] = {"mask": m.flatten().tolist(), "mask_shape": list(m.shape), "out": r.flatten().tolist(), "shape": list(r.shape)}
json.dump(out, open("torch_index_ref.json", "w"))
print({k: v["shape"] for k, v in out.items()})
