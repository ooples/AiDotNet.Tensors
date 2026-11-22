from pathlib import Path
import numpy as np
folder = Path(__file__).resolve().parent / 'fixture'
folder.mkdir(exist_ok=True)
rng = np.random.default_rng(20261119)
with (folder/'x.f32').open('wb') as xs, (folder/'y.u8').open('wb') as ys:
    for k in range(261):
        x = (rng.random((64,32,32),dtype=np.float32) - .5) * 1.8
        y = rng.integers(0,10,size=64,dtype=np.uint8) if k<256 else np.arange((k-256)*64,(k-255)*64).astype(np.int32).__mod__(10).astype(np.uint8)
        starts = rng.integers(2,21,size=64)
        for i,(channel,start) in enumerate(zip(y,starts)):
            x[i,start:start+8,channel] += .85
        x.tofile(xs)
        y.tofile(ys)
