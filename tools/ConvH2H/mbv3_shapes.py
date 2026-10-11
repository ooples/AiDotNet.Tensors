# MobileNetV3-Large conv shapes at a 32x32 input, from torchvision's inverted_residual_setting (no torchvision needed).
import json
def make_divisible(v, d=8):
    n = max(d, int(v + d / 2) // d * d)
    return n + d if n < 0.9 * v else n
cfg = [  # in, k, exp, out, se, stride
    (16,3,16,16,0,1),(16,3,64,24,0,2),(24,3,72,24,0,1),(24,5,72,40,1,2),(40,5,120,40,1,1),(40,5,120,40,1,1),
    (40,3,240,80,0,2),(80,3,200,80,0,1),(80,3,184,80,0,1),(80,3,184,80,0,1),(80,3,480,112,1,1),(112,3,672,112,1,1),
    (112,5,672,160,1,2),(160,5,960,160,1,1),(160,5,960,160,1,1)]
convs = []
def conv(C, H, O, K, S, G): convs.append(dict(N=1, C=C, H=H, W=H, O=O, K=K, S=S, P=(K - 1) // 2, G=G))
H = 32
conv(3, H, 16, 3, 2, 1); H = 16
for cin, k, exp, cout, se, s in cfg:
    if exp != cin: conv(cin, H, exp, 1, 1, 1)
    conv(exp, H, exp, k, s, exp); H = (H + 2 * ((k - 1) // 2) - k) // s + 1
    if se:
        sq = make_divisible(exp // 4)
        conv(exp, 1, sq, 1, 1, 1); conv(sq, 1, exp, 1, 1, 1)
    conv(exp, H, cout, 1, 1, 1)
conv(160, H, 960, 1, 1, 1)
uniq = {}
for c in convs:
    key = tuple(c.values()); uniq[key] = uniq.get(key, 0) + 1
out = [dict(zip(['N','C','H','W','O','K','S','P','G'], key), count=n) for key, n in uniq.items()]
json.dump(out, open('mbv3_convs.json', 'w'), indent=1)
print(len(convs), 'convs,', len(out), 'unique')
