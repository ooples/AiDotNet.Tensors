from pathlib import Path
import json, sys, time
import torch
import torch.nn.functional as F
import numpy as np
from safetensors.torch import load_file
sys.path.insert(0, str(Path.home()/'source/repos/AiDotNet-reddit-rc1/benchmarks/AiDotNet.PyTorchParity/pytorch'))
from benchmark import LSTMClassifier

run, fixture, weights = map(Path, sys.argv[1:])
torch.set_num_threads(8)
assert torch.cuda.is_available()
base = LSTMClassifier().cuda()
base.load_state_dict(load_file(str(weights)))
parameters = sum(p.numel() for p in base.parameters() if p.requires_grad)
assert parameters == 25482
model = torch.compile(base)
opt = torch.optim.AdamW(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.01,fused=True)
x = torch.from_numpy(np.fromfile(fixture/'x.f32', dtype='<f4').reshape(261,64,32,32)).cuda()
y = torch.from_numpy(np.fromfile(fixture/'y.u8', dtype='u1').reshape(261,64)).long().cuda()
def train(k):
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x[k]),y[k])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    opt.step()
    return loss.detach()
@torch.no_grad()
def evaluate():
    return sum((model(x[i]).argmax(dim=-1) == y[i]).sum() for i in range(256,261)).item()
initial = evaluate()
for i in range(8):
    loss = train(i)
    if i==0: first_loss=loss.item()
correct = evaluate()
torch.cuda.synchronize()
with (run/'live.jsonl').open('w',buffering=1) as out:
    def report(step,seconds,acc,loss):
        print(json.dumps(dict(step=step,seconds=seconds,correct=acc,loss=loss)),file=out,flush=True)
    report(0,0,correct,loss.item())
    (run/'ready').write_text(json.dumps(dict(parameters=parameters,initial=initial,firstLoss=first_loss,device=torch.cuda.get_device_name(),version=torch.__version__)))
    while not (run/'go').exists(): time.sleep(.001)
    start=time.perf_counter()
    for step in range(1,4097):
        loss=train((step-1+8)%256)
        if step%64==0:
            current=loss.item()
            correct=evaluate()
            torch.cuda.synchronize()
            report(step,time.perf_counter()-start,correct,current)
(run/'done').write_text(json.dumps(dict(correct=correct)))
