({ name, accent, steps }) => {
    const torch = name === 'PyTorch';
    const bg = '#111b2a';
    const css = document.createElement('style');
    css.textContent = `
    *{box-sizing:border-box} html, body{background:${bg}!important; width:100vw; height:100vh; overflow:hidden; margin:0!important; color:#f1f5fa; font-family:"Segoe UI",sans-serif}
    header, #summary {display:none!important} main {padding:0!important;margin:0!important;max-width:none!important}
    #film {padding:22px 26px 0}
    #byline {font-size:16px;color:#b0bdd0; margin:0}
    #title-row{display:flex;align-items:center;justify-content:space-between}
    #title-row h2{font-size:39px;letter-spacing:-1px;color:#f1f5fa;margin:0;line-height:1.25}
    #phase{font-size:16px;color:${accent};font-weight:600}
    .numbers{display:grid;grid-template-columns:1.1fr 1fr 1fr;margin:24px 0 12px}
    .numbers small{color:#b0bdd0;font-size:15px;display:block}
    .numbers b{color:#fff;font-weight:600;font-size:34px;line-height:1.3;font-variant-numeric:tabular-nums}
    #track{height:4px;background:#314052;border-radius:4px} #bar{height:100%;background:${accent};width:0}
    #charts-container{display:block!important;margin:16px 26px 0}
    .chart-card{border:none!important;background:none!important;box-shadow:none!important;padding:0!important;margin:0!important}
    .chart-card h3{font-size:17px!important;margin:0 0 8px!important;color:#e6ebf2!important}
    .chart-container{height:225px!important;}
    #marker{position:fixed;top:0;left:0;width:8px;height:8px;background:${bg}}
    `;
    document.head.append(css);
    const film=document.createElement('section');
    film.id='film';
    film.innerHTML='<div id="title-row"><h2></h2><span id="phase">Ready</span></div><p id="byline"></p><div class="numbers"><div><small>Timed updates</small><b id="count">0</b></div><div><small>Elapsed</small><b id="elapsed">0.00 s</b></div><div><small>Held-out</small><b id="accuracy">—</b></div></div><div id="track"><div id="bar"></div></div><i id="marker"></i>';
    document.querySelector('main').prepend(film);
    film.querySelector('h2').textContent=name;
    document.getElementById('byline').textContent=torch?'CUDA  ·  torch.compile  ·  fused AdamW':'CUDA  ·  fused LSTM training  ·  AdamW';
    const labels=Array.from({length:65},(_,i)=>i*64);
    Chart.register({id:'cameraScale',beforeUpdate(chart){
      const ds=chart.data.datasets[0]; if(!ds) return;
      ds.borderColor=accent; ds.backgroundColor=accent+'1F';ds.borderWidth=3;ds.pointRadius=0;ds.tension=0;
      chart.data.labels=labels;
      chart.options.scales.y.min=0; chart.options.scales.y.max=100;
      chart.options.scales.y.ticks.stepSize=25;
      chart.options.scales.x.ticks={color:'#aab8ca',autoSkip:false,maxRotation:0,callback:(_,i)=>i%16===0?labels[i].toLocaleString('en-US'):null};
      chart.options.scales.x.title={display:true,text:'Timed training updates',color:'#aab8ca',font:{size:13}};
    }});
    let reading=false, timer=null, ended=false, latest=null;
    window.armTraining=()=>{
      window.startedWall=performance.now(); document.getElementById('marker').style.background='#00ff00';
      document.getElementById('phase').textContent='Training'; timer=setInterval(()=>{
        if(!ended)document.getElementById('elapsed').textContent=((performance.now()-window.startedWall)/1000).toFixed(2)+' s';
      },35);
      return window.beginRun();
    };
    async function poll(){
      if(reading || ended)return; reading=true;
      try{
        const v=await window.getTrainingSample(); if(!v)return;
        latest=v;
        document.getElementById('count').textContent=v.step.toLocaleString('en-US');
        document.getElementById('accuracy').textContent=(v.correct*100/320).toFixed(1)+'%';
        document.getElementById('bar').style.width=(100*v.step/steps)+'%';
        if(v.step===steps){
          ended=true;clearInterval(timer);
          document.getElementById('elapsed').textContent=v.seconds.toFixed(2)+' s';
          document.getElementById('phase').textContent='Finished';
        }
      }finally{reading=false;}
    }
    poll(); setInterval(poll,45);
}
