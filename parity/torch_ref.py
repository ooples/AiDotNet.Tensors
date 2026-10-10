import torch, json
torch.manual_seed(0)
def traj(opt_ctor, shape, steps=3, seed=1, **kw):
    g = torch.Generator().manual_seed(seed)
    p = torch.randn(*shape, generator=g, dtype=torch.float32)
    p0 = p.clone()
    p = torch.nn.Parameter(p)
    opt = opt_ctor([p], **kw)
    grads = []
    for s in range(steps):
        gr = torch.randn(*shape, generator=g, dtype=torch.float32)
        grads.append(gr.flatten().tolist())
        p.grad = gr.clone()
        opt.step()
    return {"shape": list(shape), "p0": p0.flatten().tolist(), "grads": grads, "p": p.detach().flatten().tolist()}
out = {
  "adafactor_matrix": traj(torch.optim.Adafactor, (4, 6), lr=1e-2),
  "adafactor_vector": traj(torch.optim.Adafactor, (7,), lr=1e-2),
  "adafactor_wd": traj(torch.optim.Adafactor, (3, 5), lr=5e-2, weight_decay=0.1),
  "muon_tall": traj(torch.optim.Muon, (6, 4), lr=2e-2),
  "muon_wide_moonshot": traj(torch.optim.Muon, (3, 5), lr=2e-2, adjust_lr_fn="match_rms_adamw", nesterov=False),
}
json.dump(out, open("torch_ref.json", "w"))
print({k: v["p"][:3] for k, v in out.items()})

import torch.optim._muon as M
_orig = M._zeropower_via_newtonschulz
def ns_f32(grad, ns_coefficients, ns_steps, eps):
    a, b, c = ns_coefficients
    X = grad.float()
    if grad.size(0) > grad.size(1): X = X.T
    X = X / X.norm().clamp(min=eps)
    for _ in range(ns_steps):
        A = X @ X.T
        B = torch.addmm(A, A, A, beta=b, alpha=c)
        X = torch.addmm(X, B, X, beta=a)
    if grad.size(0) > grad.size(1): X = X.T
    return X
M._zeropower_via_newtonschulz = ns_f32
out["muon_tall_f32"] = traj(torch.optim.Muon, (6, 4), lr=2e-2)
out["muon_wide_moonshot_f32"] = traj(torch.optim.Muon, (3, 5), lr=2e-2, adjust_lr_fn="match_rms_adamw", nesterov=False)
M._zeropower_via_newtonschulz = _orig
json.dump(out, open("torch_ref.json", "w"))
print("f32", out["muon_tall_f32"]["p"][:3])
