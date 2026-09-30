"""Playable blinded comparisons with explicit accompaniment provenance.

Use the same original accompaniment within a case when export is supported;
otherwise use the same deterministic external-chord guide for every variant.
Corrected bar-grid scores always use the guide: their timeline is not raw MIDI.
Missing generations remain unavailable variants. No human scores are invented.
"""
import ctypes as ct
import ctypes.util
from collections import defaultdict
from pathlib import Path
import json
import math
import random
import wave

import mido
import numpy as np

from tri.data.chords import load_chord_sidecar
from tri.data.midi import write_grid_midi
from tri.data.prepare import discover_primary_midis
from tri.evaluation.batch import load_evaluation_windows
from tri.evaluation.cohorts import file_identity
from tri.evaluation.listening import export_context_window, ListeningSourceError, _context_signature
from tri.runtime import atomic_json, read_jsonl


def render_wav(midi_path,wav_path,soundfont='/usr/share/sounds/sf2/TimGM6mb.sf2'):
    """Offline FluidSynth file renderer, using the installed C runtime only."""
    library=ctypes.util.find_library('fluidsynth')
    if not library or not Path(soundfont).is_file():
        raise RuntimeError('installed FluidSynth library and SoundFont are required')
    lib=ct.CDLL(library);ptr=ct.c_void_p
    def api(name,args,result=ct.c_int):
        fn=getattr(lib,name);fn.argtypes=args;fn.restype=result;return fn
    settings_new=api('new_fluid_settings',[],ptr)
    setstr=api('fluid_settings_setstr',[ptr,ct.c_char_p,ct.c_char_p])
    setnum=api('fluid_settings_setnum',[ptr,ct.c_char_p,ct.c_double])
    setint=api('fluid_settings_setint',[ptr,ct.c_char_p,ct.c_int])
    synth_new=api('new_fluid_synth',[ptr],ptr)
    sfload=api('fluid_synth_sfload',[ptr,ct.c_char_p,ct.c_int])
    player_new=api('new_fluid_player',[ptr],ptr)
    add=api('fluid_player_add',[ptr,ct.c_char_p]);play=api('fluid_player_play',[ptr])
    status=api('fluid_player_get_status',[ptr])
    renderer_new=api('new_fluid_file_renderer',[ptr],ptr)
    block=api('fluid_file_renderer_process_block',[ptr])
    deleters={name:api('delete_fluid_'+name,[ptr],None) for name in ('file_renderer','player','synth','settings')}
    dest=Path(wav_path).resolve();dest.parent.mkdir(parents=True,exist_ok=True)
    temporary=dest.with_suffix('.tmp.wav');settings=settings_new();synth=player=renderer=None
    try:
        for name,value in [('player.timing-source','sample'),('audio.file.name',str(temporary)),('audio.file.type','wav'),('audio.file.format','s16')]:
            if setstr(settings,name.encode(),value.encode())<0:
                raise RuntimeError(f'FluidSynth setting failed: {name}')
        setnum(settings,b'synth.sample-rate',22050.);setnum(settings,b'synth.gain',.2)
        setint(settings,b'synth.reverb.active',0);setint(settings,b'synth.chorus.active',0)
        synth=synth_new(settings)
        if not synth or sfload(synth,str(Path(soundfont).resolve()).encode(),1)<0:
            raise RuntimeError('could not create offline synth or load SoundFont')
        player=player_new(synth)
        if not player or add(player,str(Path(midi_path).resolve()).encode())<0:
            raise RuntimeError('could not load MIDI player')
        renderer=renderer_new(synth)
        if not renderer or play(player)<0:
            raise RuntimeError('could not start offline renderer')
        limit=int(math.ceil((mido.MidiFile(midi_path).length+15)*22050/64))+100
        for _ in range(limit):
            if status(player)!=1:
                break
            if block(renderer)<0:
                raise RuntimeError('offline rendering failed')
        else:
            raise RuntimeError('offline renderer exceeded MIDI-duration bound')
    finally:
        for name,value in [('file_renderer',renderer),('player',player),('synth',synth),('settings',settings)]:
            if value:
                deleters[name](value)
    with wave.open(str(temporary),'rb') as f:
        frames,rate=f.getnframes(),f.getframerate()
        samples=np.frombuffer(f.readframes(frames),dtype='<i2')
        if not frames or not np.any(samples):
            raise RuntimeError('rendered audio is empty or silent')
        clipped=float(np.mean(np.abs(samples.astype(np.int32))>=32767))
    temporary.replace(dest)
    return {'seconds':frames/rate,'sample_rate':rate,'clipped_sample_fraction':clipped}


def export_chord_guide(tokens,features,path,*,initial_pitch=None,tempo_bpm=100):
    write_grid_midi(tokens,path,initial_pitch=initial_pitch,tempo=int(round(60_000_000/tempo_bpm)))
    midi=mido.MidiFile(path);track=mido.MidiTrack([mido.MetaMessage('track_name',name='HARMONIC_GUIDE')])
    track.append(mido.Message('program_change',channel=1,program=0))
    active=();last_tick=0;events=[]
    for i in range(len(tokens)+1):
        row=features[i] if i<len(tokens) else None
        notes=tuple(48+int(pc) for pc in np.flatnonzero(row[12:24])) if row is not None and row[37] and not row[36] else ()
        if notes!=active or i%4==0 or i==len(tokens):
            tick=i*(midi.ticks_per_beat//4)
            events.extend((tick,mido.Message('note_off',note=p,velocity=0,channel=1)) for p in active)
            events.extend((tick,mido.Message('note_on',note=p,velocity=42,channel=1)) for p in notes)
            active=notes
    for tick,message in events:
        track.append(message.copy(time=tick-last_tick));last_tick=tick
    track.append(mido.MetaMessage('end_of_track',time=len(tokens)*midi.ticks_per_beat//4-last_tick))
    midi.tracks.append(track);midi.save(path)


def _page(items,title):
    data=json.dumps(items,ensure_ascii=False).replace('<','\\u003c')
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TRI 盲听比较</title><style>body{font:16px/1.6 system-ui;margin:32px auto;max-width:1100px;padding:0 20px;color:#172d32;background:#f3f7f6}h1{font-size:28px}section{background:white;border:1px solid #dbe5e1;border-radius:12px;padding:20px;margin:20px 0}.variants{display:grid;grid-template-columns:repeat(auto-fit,minmax(225px,1fr));gap:16px}.variant{background:#f5f8fa;border-radius:8px;padding:14px}audio{width:100%}label{display:block;margin:10px 0}select,input{font:inherit;padding:5px;border:1px solid #abbcb6;border-radius:5px}button{font:inherit;background:#156955;color:white;border:0;border-radius:6px;padding:12px 20px;cursor:pointer}.muted{color:#52696b}.sticky{position:sticky;top:0;background:#f3f7f6;padding:12px 0;border-bottom:1px solid #ccd8d4}h2{font-size:19px}</style>
<h1>音乐补全 · 盲听比较</h1><p>''' + title + '''</p><p>同一组的固定旋律和伴奏相同。请比较旋律自然度、和声适配及衔接，1 分最低、5 分最高；无法评价可留空。先听再评分，不查看私有答案。已知模板的节奏固定，不要求它表现出随机变化。</p><p class="muted">部分组使用统一和弦伴奏，组内不会混用原伴奏。没有生成结果的版本明确标记，不自动计为任何分数。评分仅保存在当前浏览器；下载文件后交给负责人。</p>
<div class="sticky"><label>听者代号 <input id="listener" placeholder="自选匿名代号"></label><button id="download">下载评分 CSV</button> <span id="progress"></span></div><main id="items"></main>
<script>const items=''' + data + ''';
const key='tri_blind_'+location.pathname;let saved={};try{saved=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
const fields=[['naturalness','旋律自然度'],['harmony','和声适配'],['continuity','前后衔接'],['relationship','片段关系清晰度']];
function save(){try{localStorage.setItem(key,JSON.stringify(saved))}catch(e){}document.getElementById('progress').textContent='已填写 '+Object.values(saved).filter(x=>x!=='').length+' 项';}
const listener=document.getElementById('listener');listener.value=saved.listener||'';listener.oninput=()=>{saved.listener=listener.value;save()};
for(const item of items){const sec=document.createElement('section');const h=document.createElement('h2');h.textContent=item.item_id+' · '+(item.accompaniment==='original'?'原始伴奏':'统一和弦伴奏');sec.append(h);const grid=document.createElement('div');grid.className='variants';sec.append(grid);
for(const v of item.variants){const card=document.createElement('div');card.className='variant';const strong=document.createElement('strong');strong.textContent='版本 '+v.label;card.append(strong);grid.append(card);
if(v.available){const audio=document.createElement('audio');audio.controls=true;audio.preload='none';audio.src=v.audio;card.append(audio);for(const [field,label] of fields){const l=document.createElement('label');l.textContent=label+' ';const s=document.createElement('select');const id=item.item_id+'|'+v.label+'|'+field;for(const val of ['',1,2,3,4,5]){const o=document.createElement('option');o.value=val;o.textContent=val===''?'未评分 / 无法评价':val;s.append(o)}s.value=saved[id]||'';s.onchange=()=>{saved[id]=s.value;save()};l.append(s);card.append(l)}}else{const p=document.createElement('p');p.textContent='该版本未产生可用样本';card.append(p)}}
const l=document.createElement('label');l.textContent='整体更偏好： ';const s=document.createElement('select');const id=item.item_id+'|preference';for(const val of ['',...item.variants.filter(v=>v.available).map(v=>v.label),'并列','无法判断']){const o=document.createElement('option');o.value=val;o.textContent=val||'请选择';s.append(o)}s.value=saved[id]||'';s.onchange=()=>{saved[id]=s.value;save()};l.append(s);sec.append(l);document.getElementById('items').append(sec)}
document.getElementById('download').onclick=()=>{const csv=[['listener','item_id','variant','available','accompaniment',...fields.map(x=>x[0]),'preference']];for(const item of items)for(const v of item.variants)csv.push([saved.listener||'',item.item_id,v.label,v.available,item.accompaniment,...fields.map(([f])=>saved[item.item_id+'|'+v.label+'|'+f]||''),saved[item.item_id+'|preference']||'']);const encoded=csv.map(r=>r.map(x=>'"'+String(x).replaceAll('"','""')+'"').join(',')).join('\\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['\\ufeff'+encoded],{type:'text/csv;charset=utf-8'}));a.download='tri_blind_ratings.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),500)};save();</script></html>'''


def build_blind_audio(cohort_path,results_path,data_root,output_dir,*,source_mode='original',seed=20260915,render=True):
    if source_mode not in ('original','aligned'):
        raise ValueError('unknown listening source mode')
    output=Path(output_dir).resolve();public=output/'public';public.mkdir(parents=True,exist_ok=True)
    identity={'cohort':file_identity(cohort_path),'results':file_identity(results_path),'source_mode':source_mode,
              'seed':seed,'render':render,'soundfont':file_identity('/usr/share/sounds/sf2/TimGM6mb.sf2') if render else None}
    if (output/'report.json').exists():
        report=json.loads((output/'report.json').read_text())
        if report['identity']!=identity:
            raise ValueError('listening inputs changed; do not overwrite an existing blind assignment')
        if report['status']=='completed':
            if not (public/'index.html').is_file() or not (output/'private_key.json').is_file():
                raise ValueError('completed listening bundle is missing its page or private key')
            for item in json.loads((public/'items.json').read_text())['items']:
                for v in item['variants']:
                    if v['available'] and (not (public/v['midi']).is_file() or render and not (public/v['audio']).is_file()):
                        raise ValueError('completed listening bundle is missing media')
            return report
    plan=json.loads(Path(cohort_path).read_text());dataset=plan['options']['dataset']['path'];chords=plan['options']['chords']['path']
    windows={w.source_index:w for w in load_evaluation_windows(dataset,limit=2**31-1,split='validation')}
    sidecar=load_chord_sidecar(dataset,chords)
    rows=[r for r in read_jsonl(results_path) if r['replicate']==0]  # Fixed before inspecting outputs.
    expected_methods={'pooled_template','one_shot_joint','tri_direct','smc_4'}
    by_case=defaultdict(list)
    for r in rows:
        by_case[r['cohort_case_id']].append(r)
    sources={p.stem:p for p in discover_primary_midis(data_root)} if source_mode=='original' else {}
    with np.load(dataset,allow_pickle=False) as f:
        tempos=f['source_tempos'].copy() if 'source_tempos' in f.files else None
    rng=random.Random(seed);cases=list(plan['requests']);rng.shuffle(cases)
    items=[];private=[];fallbacks=[];audio_stats=[]
    for ordinal,case in enumerate(cases):
        w=windows[case['source_index']];features=sidecar['chord_features'][w.source_index]
        mode='original' if source_mode=='original' else 'chord_guide';tempo=100 if tempos is None else float(tempos[w.source_index])
        if mode=='original':
            probe=output/'context_probe.mid'
            try:
                export_context_window(sources[w.work_id],w.tokens,w.start_cell,probe,initial_pitch=w.initial_pitch)
            except ListeningSourceError as error:
                mode='chord_guide';fallbacks.append({'case_id':case['case_id'],'reason':error.reason})
            finally:
                probe.unlink(missing_ok=True)
            if mode=='chord_guide':
                from tri.data.chords import read_tempo_map
                tm=read_tempo_map(sources[w.work_id]);seconds=tm.cell_seconds(w.start_cell,len(w.tokens),4)
                tempo=60*len(w.tokens)/4/(seconds[-1]-seconds[0])
        candidates=sorted(by_case[case['case_id']],key=lambda r:r['method']);rng.shuffle(candidates)
        if {r['method'] for r in candidates}!=expected_methods or len(candidates)!=len(expected_methods):
            raise ValueError('missing or duplicate fixed-replicate listening candidates')
        item={'item_id':f'item_{ordinal:04d}','accompaniment':mode,'variants':[]};key=[];signature=None
        for index,row in enumerate(candidates):
            label=chr(65+index);name=f"{item['item_id']}_{label}";midi_path=public/f'{name}.mid'
            variant={'label':label,'available':bool(row['valid'])}
            key.append({'label':label,'method':row['method'],'status':row['status']})
            if row['valid']:
                if mode=='original':
                    export_context_window(sources[w.work_id],row['raw_tokens'],w.start_cell,midi_path,initial_pitch=w.initial_pitch)
                else:
                    export_chord_guide(row['raw_tokens'],features,midi_path,initial_pitch=w.initial_pitch,tempo_bpm=tempo)
                sig=_context_signature(midi_path)
                if signature is not None and sig!=signature:
                    raise RuntimeError('blind variants have different accompaniment')
                signature=sig;variant['midi']=midi_path.name
                if render:
                    wav_path=public/f'{name}.wav';audio_stats.append(render_wav(midi_path,wav_path));variant['audio']=wav_path.name
                else:
                    variant['audio']=''
            item['variants'].append(variant)
        items.append(item);private.append({'item_id':item['item_id'],'case_id':case['case_id'],'variants':key})
        atomic_json(output/'status.json',{'status':'running','completed':len(items),'planned':len(cases)})
    atomic_json(public/'items.json',{'items':items})
    atomic_json(output/'private_key.json',{'items':private})
    (public/'index.html').write_text(_page(items,'固定第 1 次采样；版本顺序已随机化。缺失版本与伴奏来源均保留。'))
    report={'status':'completed','identity':identity,'groups':len(items),
            'available_variants':sum(v['available'] for i in items for v in i['variants']),
            'unavailable_variants':sum(not v['available'] for i in items for v in i['variants']),
            'original_accompaniment_groups':sum(i['accompaniment']=='original' for i in items),
            'chord_guide_groups':sum(i['accompaniment']=='chord_guide' for i in items),'fallbacks':fallbacks,
            'audio_files':len(audio_stats),'max_clipped_sample_fraction':max((r['clipped_sample_fraction'] for r in audio_stats),default=0),
            'outputs':{'page':str(public/'index.html'),'public':str(public),'private_key':str(output/'private_key.json')},
            'human_ratings':'pending; browser form is empty until a human listens',
            'limitations':['Fixed replicate zero; no success-based resampling or substitution.',
                           'Source-export failures switch the whole case to a declared chord guide; analyze accompaniment strata separately.',
                           'Bar data uses aligned symbolic timing and constant annotated tempo, not original MIDI accompaniment.']}
    atomic_json(output/'report.json',report);atomic_json(output/'status.json',{'status':'completed','completed':len(items),'planned':len(cases)})
    return report
