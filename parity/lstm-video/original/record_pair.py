from pathlib import Path
import json, os, socket, subprocess, sys, time
from playwright.sync_api import sync_playwright

root=Path(__file__).resolve().parent
take=Path((root/'current.txt').read_text().strip())
ff=next((root.parent/'.media-tools/imageio_ffmpeg/binaries').glob('ffmpeg-*.exe'))
fixture=root/'fixture'
weights=root.parents[2]/'reference/lstm.weights.safetensors'
dll=root/'bin/Release/net10.0/AiDotNet.PyTorchParity.dll'
env=dict(os.environ,AIDOTNET_CPU_THREADS='8',AIDOTNET_BLAS_THREADS='8',AIDOTNET_FUSED_DIAG='1')
def wait_file(path,seconds,workers):
    end=time.monotonic()+seconds
    while not path.exists():
        for p in workers:
            if p.poll() is not None and p.returncode!=0: raise RuntimeError(f'{p.args[0]} exited {p.returncode}: {path.parent}')
        if time.monotonic()>end: raise TimeoutError(path)
        time.sleep(.03)
def sample(d):
    lines=(d/'live.jsonl').read_text().splitlines()
    for line in reversed(lines[-2:]):
        try: return json.loads(line)
        except json.JSONDecodeError: pass
def ffcall(*args):
    subprocess.run([str(ff),'-loglevel','error','-y',*map(str,args)],check=True)
def track_meta(raw, d):
    # A green 8px sync marker is painted when the browser releases the already-warm trainer.
    data=subprocess.check_output([str(ff),'-loglevel','error','-i',str(raw),'-vf','fps=25,crop=8:8:0:0,scale=1:1','-f','rawvideo','-pix_fmt','rgb24','-'])
    first=next(i//3 for i in range(0,len(data),3) if data[i+1]>175 and data[i]<100 and data[i+2]<100)
    rows=[json.loads(s) for s in (d/'live.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==list(range(0,4097,64))
    assert all(0<=r['correct']<=320 and r['loss']>=0 for r in rows)
    return dict(raw=str(raw),trim=first/25, seconds=rows[-1]['seconds'], correct=rows[-1]['correct'], initial=json.loads((d/'ready').read_text()), checkpoints=len(rows))
def record(side,pw):
    d=take/side
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    workers=[]
    with (d/'display.log').open('w') as dl, (d/'aidotnet-training.log').open('w') as al:
        name='AiDotNet' if side=='ai' else 'PyTorch'
        display=subprocess.Popen(['dotnet',str(dll),'display',str(d),str(port),name],env=env,stdout=dl,stderr=dl)
        workers.append(display)
        if side=='ai':
            train=subprocess.Popen(['dotnet',str(dll),'train',str(d),str(fixture),str(weights)],env=env,stdout=al,stderr=al)
            workers.append(train)
        browser=None
        try:
            wait_file(d/'ready',150,workers);wait_file(d/'url',20,workers)
            browser=pw.chromium.launch()
            size=dict(width=668,height=506)
            ctx=browser.new_context(viewport=size,record_video_dir=d/'raw',record_video_size=size)
            page=ctx.new_page()
            page.expose_function('getTrainingSample',lambda:sample(d))
            page.expose_function('beginRun',lambda:(d/'go').touch())
            page.goto((d/'url').read_text(),wait_until='load')
            page.wait_for_function("typeof Chart==='function'")
            page.evaluate((root/'panel.js').read_text(encoding='utf-8'),dict(name=name,accent='#58cfc7' if side=='ai' else '#ffab74',steps=4096))
            page.wait_for_function("document.querySelector('canvas') && document.querySelector('#accuracy').textContent.includes('%')")
            page.wait_for_timeout(500)
            page.evaluate('window.armTraining()')
            page.wait_for_function("document.querySelector('#phase').textContent==='Finished'",timeout=130000)
            page.wait_for_function("document.getElementById('data-points').textContent==='65'")
            wait_file(d/'done',12,workers)
            page.wait_for_timeout(900)
            page.screenshot(path=str(d/'final.png'))
            v=page.video;ctx.close();raw=Path(v.path());browser.close();browser=None
            (d/'captured').touch()
            for p in workers: assert p.wait(15)==0
            meta=track_meta(raw,d)
            (d/'capture.json').write_text(json.dumps(meta,indent=2))
            print(name,json.dumps(meta),flush=True)
            return meta
        finally:
            if browser:browser.close()
            for p in workers:
                if p.poll() is None:p.terminate();p.wait(10)

def render(pw, meta):
    a,b=meta['ai'],meta['torch']
    assert a['initial']['initial']==b['initial']['initial'] and abs(a['initial']['firstLoss']-b['initial']['firstLoss'])<.002,(a,b)
    assert min(a['correct'],b['correct'])>160
    browser=pw.chromium.launch()
    page=browser.new_page(viewport=dict(width=1440,height=810))
    page.set_content("""<style>*{box-sizing:border-box}body{margin:0;background:#0b1320;color:#f3f6fc;font-family:'Segoe UI',sans-serif}.hero{position:absolute;left:40px;top:24px}.hero b{font-size:16px;letter-spacing:.08em;color:#89bcbf}.hero h1{font-size:43px;font-weight:650;letter-spacing:-1.4px;margin:4px 0}.hero p,.foot p{font-size:20px;color:#c6d3e2;margin:4px 0}.tile{position:absolute;top:162px;height:512px;width:674px;background:#111b2a;border:1px solid #354354;border-radius:8px}.a{left:40px}.b{left:726px}.foot{position:absolute;top:694px;left:40px}.foot strong{color:#f3f6fc;font-size:23px;font-weight:600}.foot p{font-size:18px}</style>
    <div class=hero><b>LIVE LSTM TRAINING  /  GTX 1660 Ti</b><h1>Which channel carried the pulse?</h1><p>Two frameworks train the same 25,482-parameter classifier on noisy signals. Each batch contains 64 sequences.</p></div>
    <div class='tile a'></div><div class='tile b'></div><div class=foot><strong>Same weights, batches and held-out signals</strong><p>Separate runs on one GPU, aligned at the first timed update and played at 1×. Both begin after 8 warm-up updates.</p><p>4,096 timed updates  ·  AdamW  ·  Accuracy checked on the same 320 signals every 64 updates</p></div>""")
    frame=take/'frame.png';page.screenshot(path=str(frame))
    # Static caption layer keeps both codec bylines and axis labels legible on the captured charts.
    page.set_content("""<style>*{box-sizing:border-box}html,body{margin:0;background:transparent;font-family:'Segoe UI',sans-serif}.by,.axis{position:absolute;background:#111b2a;color:#b0bdd0}.by{top:233px;width:598px;height:27px;font-size:16px;line-height:24px}.axis{top:615px;width:618px;height:44px;padding:0 18px}.ticks{position:relative;height:17px}.ticks span{position:absolute;transform:translateX(-50%);font-size:12px}.label{text-align:center;color:#b0bdd0;font-size:13px;margin-top:2px}</style>""")
    page.evaluate("""() => {
      for(const [offset,label] of [[0,'CUDA  /  fused LSTM training  /  AdamW'],[686,'CUDA  /  torch.compile  /  fused AdamW']]){
        const by=document.createElement('div');by.className='by';by.style.left=(68+offset)+'px';by.textContent=label;
        const axis=document.createElement('div');axis.className='axis';axis.style.left=(84+offset)+'px';
        const ticks=document.createElement('div');ticks.className='ticks';
        [0,1024,2048,3072,4096].forEach((n,i)=>{const t=document.createElement('span');t.style.left=(i*25)+'%';t.textContent=n.toLocaleString('en-US');ticks.append(t)});
        const axisLabel=document.createElement('div');axisLabel.className='label';axisLabel.textContent='Timed training updates';
        axis.append(ticks,axisLabel);document.body.append(by,axis);
      }
    }""")
    overlay=take/'chart-labels.png';page.screenshot(path=str(overlay),omit_background=True);browser.close()
    duration=max(a['seconds'],b['seconds'])+2.0
    # Freeze a completed panel while the other continues, without changing either track's rate.
    filters=[]
    for i,x in enumerate((a,b),1):
        filters.append(f"[{i}:v]fps=25,trim=start={x['trim']:.3f},setpts=PTS-STARTPTS,crop=668:498:0:8,tpad=stop_mode=clone:stop_duration={duration:.2f}[p{i}]")
    filters+=['[0:v][p1]overlay=43:173[tmp]','[tmp][p2]overlay=729:173[live]','[live][3:v]overlay=0:0[out]']
    dest=root.parent.parent/'cuda-lstm-side-by-side-review.mp4'
    stage=take/'composite.mp4'
    ffcall('-loop','1','-framerate','25','-i',frame,'-i',a['raw'],'-i',b['raw'],'-i',overlay,'-filter_complex',';'.join(filters),'-map','[out]','-t',f'{duration:.3f}','-an','-c:v','libx264','-crf','17','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',stage)
    ffcall('-i',stage,'-f','null','-')
    stage.replace(dest)
    for pct in (.25,.9):
        ffcall('-ss',str(duration*pct),'-i',dest,'-frames:v','1','-update','1',take/f'check-{pct}.png')
    result=dict(tracks=meta,videoSeconds=duration,video=str(dest))
    (root/'result.json').write_text(json.dumps(result,indent=2))
with sync_playwright() as pw:
    meta={}
    for side in ('torch','ai'):
        path=take/side/'capture.json'
        meta[side]=json.loads(path.read_text()) if path.exists() else record(side,pw)
    render(pw,meta)
