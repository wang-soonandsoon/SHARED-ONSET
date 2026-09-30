"""Offline human-listening handoff and honest import of listener-supplied CSV.

Preparing audio is not a human result. No model is trained or asked to rate
music here. Original revision public/private assignments are read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import filecmp
import io
import json
from pathlib import Path
import shutil
from statistics import mean
import wave
import zipfile

from tri.runtime import atomic_json, atomic_jsonl, read_jsonl

STAGES = ('short', 'bar8', 'bar16')
KINDS = ('unknown', 'partial', 'known', 'harmony')
METHODS = ('pooled_template', 'one_shot_joint', 'tri_direct', 'smc_4')
METRICS = ('naturalness', 'harmony', 'continuity', 'relationship')
FIELDS = ('schema', 'listener', 'stage', 'item_id', 'variant', 'available',
          'accompaniment', *METRICS, 'preference')
SCHEMA = 'human_listening_v1'


def _identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size, 'modified_ns': stat.st_mtime_ns}


def _write_same(path, value):
    """Do not silently replace a frozen mapping when rerunning preparation."""
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'existing listening mapping differs: {path}')
    else:
        atomic_json(path, value)


def _copy_same(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not filecmp.cmp(source, target, shallow=False):
            raise ValueError(f'existing copied audio differs: {target}')
    else:
        shutil.copy2(source, target)


def _regions(midi_path, spec):
    """Approximate player positions from the already exported MIDI timeline."""
    import mido
    midi = mido.MidiFile(midi_path)
    changes = [(0, 500000)]
    tick = 0
    for message in mido.merge_tracks(midi.tracks):
        tick += message.time
        if message.type == 'set_tempo':
            if changes[-1][0] == tick:
                changes[-1] = (tick, message.tempo)
            else:
                changes.append((tick, message.tempo))

    def seconds(cell):
        target, elapsed = cell * midi.ticks_per_beat / 4, 0.
        for i, (start, tempo) in enumerate(changes):
            if start >= target:
                break
            end = changes[i + 1][0] if i + 1 < len(changes) else target
            elapsed += mido.tick2second(min(end, target) - start, midi.ticks_per_beat, tempo)
        return round(elapsed, 3)

    pairs = sorted(spec['equal_onsets'])
    return [{'name': f'片段 {side + 1}', 'start_seconds': seconds(pairs[0][side]),
             'end_seconds': seconds(pairs[-1][side] + 1),
             'given': all(str(pair[side]) in spec['observed'] for pair in pairs)} for side in (0, 1)]


def prepare_listening(revision_root='runs/revision', output='runs/decision_study/listening', *, archive=True):
    revision_root, output = Path(revision_root).resolve(), Path(output).resolve()
    public = output / 'public'
    public.mkdir(parents=True, exist_ok=True)
    (output / 'private/imports').mkdir(parents=True, exist_ok=True)
    items, private_items, sources = [], [], []
    pilot_rank = 0
    for stage in STAGES:
        source = revision_root / stage
        listening = source / 'listening'
        cohort_path = source / 'cohort.json'
        cohort = json.loads(cohort_path.read_text())
        if cohort['status'] != 'completed' or cohort['options']['split'] != 'validation':
            raise ValueError('human handoff requires completed frozen validation cohorts')
        source_public = json.loads((listening / 'public/items.json').read_text())['items']
        source_private = json.loads((listening / 'private_key.json').read_text())['items']
        report = json.loads((listening / 'report.json').read_text())
        if report['status'] != 'completed':
            raise ValueError('source audio preparation is not complete')
        by_id = {item['item_id']: item for item in source_private}
        cases = {row['case_id']: row for row in cohort['requests']}
        if len(by_id) != len(source_private) or len(by_id) != len(source_public):
            raise ValueError('duplicate or incomplete frozen blind mapping')
        if {item['item_id'] for item in source_public} != set(by_id):
            raise ValueError('public/private item identities disagree')
        selected = [next(row['case_id'] for row in cohort['requests'] if row['kind'] == kind) for kind in KINDS]
        ranks = {case_id: pilot_rank + i for i, case_id in enumerate(selected)}
        pilot_rank += len(selected)
        sources.append({'stage': stage, 'cohort': _identity(cohort_path),
                        'public': _identity(listening / 'public/items.json'),
                        'private_key': _identity(listening / 'private_key.json'),
                        'source_page': str(listening / 'public/index.html')})
        for old in source_public:
            mapping = by_id[old['item_id']]
            case = cases[mapping['case_id']]
            item_id = f'{stage}_{old["item_id"]}'
            methods = {variant['label']: variant for variant in mapping['variants']}
            if set(method['method'] for method in methods.values()) != set(METHODS) or len(methods) != 4:
                raise ValueError('expected four frozen generation strategies')
            if set(methods) != {v['label'] for v in old['variants']}:
                raise ValueError('public/private variant labels disagree')
            item = {'item_id': item_id, 'stage': stage, 'source_item_id': old['item_id'],
                    'accompaniment': old['accompaniment'], 'pilot': case['case_id'] in ranks,
                    'pilot_rank': ranks.get(case['case_id']), 'variants': [], 'regions': []}
            for variant in old['variants']:
                label = variant['label']
                available = bool(variant['available'])
                if available != (methods[label]['status'] == 'exported'):
                    raise ValueError('frozen available/status mismatch')
                copied = {'label': label, 'available': available}
                if available:
                    wav = listening / 'public' / variant['audio']
                    midi = listening / 'public' / variant['midi']
                    if not wav.is_file() or not midi.is_file():
                        raise ValueError('a frozen available variant is missing its media')
                    relative = f'audio/{item_id}_{label}.wav'
                    _copy_same(wav, public / relative)
                    with wave.open(str(wav), 'rb') as audio:
                        duration = audio.getnframes() / audio.getframerate()
                        if duration <= 0:
                            raise ValueError('empty source audio')
                    copied.update(audio=relative, seconds=duration)
                    if not item['regions']:
                        item['regions'] = _regions(midi, case['spec'])
                item['variants'].append(copied)
            items.append(item)
            private_items.append({'item_id': item_id, 'stage': stage, 'source_item_id': old['item_id'],
                                  'case_id': case['case_id'], 'work_id': case['work_id'],
                                  'kind': case['kind'], 'pilot': item['pilot'],
                                  'accompaniment': item['accompaniment'], 'variants': mapping['variants']})
    private = {'schema': SCHEMA, 'sources': sources, 'items': private_items,
               'selection': 'First frozen cohort request per kind per stage; replicate zero; preserve original A-D labels.'}
    _write_same(output / 'private/manifest.json', private)
    _write_same(public / 'items.json', {'schema': SCHEMA, 'items': items})
    (public / 'index.html').write_text(_page(items), encoding='utf-8')
    pilot = [item for item in items if item['pilot']]
    duration = sum(v.get('seconds', 0.) for item in pilot for v in item['variants'])
    inventory = {'materials_status': 'completed',
                 'groups': len(items), 'pilot_groups': len(pilot),
                 'available_audio': sum(v['available'] for item in items for v in item['variants']),
                 'unavailable_slots': sum(not v['available'] for item in items for v in item['variants']),
                 'pilot_available_audio': sum(v['available'] for item in pilot for v in item['variants']),
                 'pilot_unavailable_slots': sum(not v['available'] for item in pilot for v in item['variants']),
                 'pilot_unique_works': len({i['work_id'] for i in private_items if i['pilot']}),
                 'full_unique_works': len({i['work_id'] for i in private_items}),
                 'pilot_audio_once_minutes': round(duration / 60, 2),
                 'accompaniment_groups': dict(Counter(i['accompaniment'] for i in items)),
                 'pilot_accompaniment_groups': dict(Counter(i['accompaniment'] for i in pilot)),
                 'page': str(public / 'index.html'), 'public_directory': str(public),
                 'private_manifest': str(output / 'private/manifest.json'),
                 'rating_summary': str(output / 'ratings_summary.json'),
                 'scope': 'Compare generation strategies, not equivalent exact inference backends.',
                 'human_results': 'See ratings_summary.json; material completion does not imply human evaluation.'}
    if archive:
        archive_path = output / 'listener_package.zip'
        with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
            for path in sorted(public.rglob('*')):
                if path.is_file():
                    bundle.write(path, path.relative_to(public))
        inventory['listener_package'] = str(archive_path)
    atomic_json(output / 'materials.json', inventory)
    summarize_ratings(output)
    return inventory


def _page(items):
    data = json.dumps(items, ensure_ascii=False).replace('<', '\\u003c')
    return r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>旋律补全 · 匿名听评</title>
<style>body{font:16px/1.6 system-ui;margin:24px auto;max-width:1180px;padding:0 18px;color:#163433;background:#f4f7f5}
h1{font-size:29px}section{background:white;border:1px solid #d7e2df;border-radius:12px;padding:20px;margin:20px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px}.card{background:#f4f7f7;padding:14px;border-radius:8px}
audio{width:100%}label{display:block;margin:8px 0}select,input,button{font:inherit;border:1px solid #adc1ba;border-radius:6px;padding:6px}
button{background:#176752;color:white;cursor:pointer}.jump{font-size:12px;margin-right:4px;padding:5px}.muted{color:#526963;font-size:14px}
.controls{position:sticky;top:0;background:#f4f7f5;padding:10px 0;border-bottom:1px solid #bcd0c8;z-index:2}.note{background:#e9f1ed;padding:12px}
</style><h1>旋律补全 · 匿名听评</h1>
<p>同一组有四个版本，固定旋律和伴奏一致。版本 A–D 的含义每组不同，请只根据听感评价。</p>
<p>先完整听完一组的可用版本，再比较自然度、和声与衔接。1=很差，2=较差，3=一般，4=较好，5=很好；无法判断请留空。
“片段关系清晰度”指两个标示片段的节奏联系是否自然清楚，不要求音高相同。已给定的片段也标出供对照。</p>
<p class="muted">可以分次完成。音量保持一致，可反复切换版本。没有音频的版本保留为空缺，不自动算低分。
评分仅保存在本机浏览器，导出后手动交给负责人；本页不会上传数据。不需要姓名或联系方式。</p>
<div class="controls"><label>匿名听者代号 <input id="listener" maxlength="80" placeholder="例如 L01（每人固定一个）"></label>
<label>听评范围 <select id="scope"><option value="pilot">先导 12 组</option><option value="all">完整 72 组</option></select></label>
<button id="download">导出已填写评分 CSV</button> <span id="progress"></span><p id="notice" class="muted"></p></div>
<main id="items"></main><script>
const items=__ITEMS__;
const fields=[['naturalness','旋律自然度'],['harmony','和声适配'],['continuity','前后衔接'],['relationship','片段关系清晰度']];
const storage='human_listening_v1_'+location.pathname;let saved={};try{saved=JSON.parse(localStorage.getItem(storage)||'{}')}catch(e){}
const listener=document.getElementById('listener');listener.value=saved.listener||'';
function save(){try{localStorage.setItem(storage,JSON.stringify(saved))}catch(e){document.getElementById('notice').textContent='浏览器无法自动保存，请及时导出。'}
const n=items.filter(item=>hasResponse(item)).length;document.getElementById('progress').textContent='已有填写内容：'+n+' 组';}
function hasResponse(item){return !!saved[item.item_id+'|preference']||item.variants.some(v=>fields.some(([f])=>!!saved[item.item_id+'|'+v.label+'|'+f]));}
listener.oninput=()=>{saved.listener=listener.value.trim();save()};
document.getElementById('scope').value=saved.scope||'pilot';
document.getElementById('scope').onchange=e=>{saved.scope=e.target.value;render();save()};
function choose(id,options){const s=document.createElement('select');for(const [value,title]of options){const o=document.createElement('option');o.value=value;o.textContent=title;s.append(o)}
s.value=saved[id]||'';s.onchange=()=>{saved[id]=s.value;save()};return s;}
function render(){const container=document.getElementById('items');container.replaceChildren();
const shown=items.filter(i=>(saved.scope||'pilot')==='all'||i.pilot);if((saved.scope||'pilot')==='pilot')shown.sort((a,b)=>a.pilot_rank-b.pilot_rank);
for(const item of shown){const sec=document.createElement('section');const h=document.createElement('h2');h.textContent=item.item_id+' · '+({short:'短片段',bar8:'8 小节窗口',bar16:'16 小节窗口'}[item.stage]);sec.append(h);
const info=document.createElement('p');info.className='muted';info.textContent=(item.accompaniment==='original'?'原始伴奏':'统一和弦伴奏')+'；'+item.regions.map(r=>r.name+' 约 '+r.start_seconds.toFixed(1)+'–'+r.end_seconds.toFixed(1)+' 秒'+(r.given?'（已给定）':'')).join('；');sec.append(info);
const grid=document.createElement('div');grid.className='grid';sec.append(grid);
for(const v of item.variants){const card=document.createElement('div');card.className='card';const title=document.createElement('strong');title.textContent='版本 '+v.label;card.append(title);grid.append(card);
if(!v.available){const missing=document.createElement('p');missing.textContent='此版本未产生可用音频，保留空缺。';card.append(missing);continue;}
const audio=document.createElement('audio');audio.controls=true;audio.preload='none';audio.src=v.audio;audio.onplay=()=>document.querySelectorAll('audio').forEach(other=>{if(other!==audio)other.pause()});card.append(audio);
for(const region of item.regions){const b=document.createElement('button');b.className='jump';b.textContent='定位'+region.name;b.onclick=()=>{audio.currentTime=Math.max(0,region.start_seconds-.5);audio.play().catch(()=>{})};card.append(b);}
for(const[field,label]of fields){const l=document.createElement('label');l.textContent=label+' ';l.append(choose(item.item_id+'|'+v.label+'|'+field,[['','未评分 / 无法判断'],...[1,2,3,4,5].map(n=>[String(n),String(n)])]));card.append(l);}}
const preference=document.createElement('label');preference.textContent='整体更偏好： ';preference.append(choose(item.item_id+'|preference',[['','未选择'],...item.variants.filter(v=>v.available).map(v=>[v.label,v.label]),['tie','并列'],['unsure','无法判断']]));sec.append(preference);container.append(sec);}}
document.getElementById('download').onclick=()=>{const name=(saved.listener||'').trim();if(!name){document.getElementById('notice').textContent='请先填写一个匿名听者代号。';return}
const chosen=items.filter(hasResponse);if(!chosen.length){document.getElementById('notice').textContent='还没有填写评分。请实际聆听后再填写；空表不算完成。';return}
const csv=[['schema','listener','stage','item_id','variant','available','accompaniment',...fields.map(x=>x[0]),'preference']];
for(const item of chosen)for(const v of item.variants)csv.push(['human_listening_v1',name,item.stage,item.item_id,v.label,v.available,item.accompaniment,...fields.map(([f])=>saved[item.item_id+'|'+v.label+'|'+f]||''),saved[item.item_id+'|preference']||'']);
const encoded=csv.map(r=>r.map(x=>'"'+String(x).replaceAll('"','""')+'"').join(',')).join('\n');
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['\ufeff'+encoded],{type:'text/csv;charset=utf-8'}));a.download='human_listening_'+name.replace(/[^a-zA-Z0-9_-]/g,'_')+'.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),500);
document.getElementById('notice').textContent='已导出 '+chosen.length+' 组。请将 CSV 手动交给负责人；页面不会自动发送。';};
render();save();
</script></html>'''.replace('__ITEMS__', data)


def _normalize_csv(path, public_items, *, legacy_stage=None):
    text = Path(path).read_text(encoding='utf-8-sig')
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        return [], 0
    required = {'listener', 'item_id', 'variant', 'available', 'accompaniment', *METRICS, 'preference'}
    if not required <= set(reader.fieldnames):
        raise ValueError(f'rating CSV missing columns: {sorted(required - set(reader.fieldnames))}')
    output, blank_rows = {}, 0
    preferences = defaultdict(set)
    for index, raw in enumerate(reader, start=2):
        if None in raw:
            raise ValueError(f'extra or malformed CSV fields on row {index}')
        raw = {key: (value or '').strip() for key, value in raw.items()}
        blank = not any(raw.get(field) for field in (*METRICS, 'preference'))
        if blank:
            blank_rows += 1
            # Anonymous empty legacy forms are harmless; named blank rows in a
            # new export can clear an existing value and still require checks.
            if not raw['listener']:
                continue
        listener = raw['listener']
        if not listener or len(listener) > 80 or any(ord(c) < 32 for c in listener):
            raise ValueError(f'row {index}: a nonempty anonymous listener ID is required')
        stage = raw.get('stage') or legacy_stage
        item_id = raw['item_id']
        if item_id.startswith('item_'):
            if stage not in STAGES:
                raise ValueError('legacy item IDs require an explicit --legacy-stage')
            item_id = f'{stage}_{item_id}'
        if item_id not in public_items:
            raise ValueError(f'unknown blind item: {item_id}')
        item = public_items[item_id]
        if stage != item['stage']:
            raise ValueError('rating stage does not match frozen blind item')
        if raw.get('schema') not in (None, '', SCHEMA):
            raise ValueError('unsupported rating schema')
        variants = {v['label']: v for v in item['variants']}
        label = raw['variant']
        if label not in variants:
            raise ValueError('unknown variant label')
        if raw['available'].lower() not in ('true', 'false'):
            raise ValueError('available must be true or false')
        available = raw['available'].lower() == 'true'
        if available != variants[label]['available'] or raw['accompaniment'] != item['accompaniment']:
            raise ValueError('ratings changed frozen availability/accompaniment')
        scores = {}
        for metric in METRICS:
            value = raw[metric]
            if value and value not in ('1', '2', '3', '4', '5'):
                raise ValueError(f'{metric} must be an integer 1–5 or empty')
            scores[metric] = int(value) if value else None
        if not available and any(value is not None for value in scores.values()):
            raise ValueError('unavailable audio must not receive a score')
        preference = {'并列': 'tie', '无法判断': 'unsure'}.get(raw['preference'], raw['preference'])
        if preference not in ('', 'tie', 'unsure') and not (preference in variants and variants[preference]['available']):
            raise ValueError('preference refers to an unavailable or unknown variant')
        row = {'listener': listener, 'stage': stage, 'item_id': item_id, 'variant': label,
               'available': available, 'accompaniment': item['accompaniment'], **scores, 'preference': preference}
        key = listener, item_id, label
        if key in output and output[key] != row:
            raise ValueError('conflicting duplicate listener/item/variant rows in one CSV')
        output[key] = row
        if preference:
            preferences[(listener, item_id)].add(preference)
    if any(len(values) > 1 for values in preferences.values()):
        raise ValueError('conflicting preferences within one listener/item')
    return list(output.values()), blank_rows


def _has_response(row):
    return bool(row['preference']) or any(row[metric] is not None for metric in METRICS)


def _validate_preferences(rows):
    preferences = defaultdict(set)
    for row in rows:
        if row['preference']:
            preferences[row['listener'], row['item_id']].add(row['preference'])
    if any(len(values) > 1 for values in preferences.values()):
        raise ValueError('conflicting saved item preferences; import a complete corrected group')


def import_ratings(csv_paths, output='runs/decision_study/listening', *, legacy_stage=None):
    """Keep raw imports; newer supplied exports replace the same row identity."""
    output = Path(output).resolve()
    public = json.loads((output / 'public/items.json').read_text())
    by_id = {item['item_id']: item for item in public['items']}
    previous = read_jsonl(output / 'private/ratings.jsonl')
    rows = {(row['listener'], row['item_id'], row['variant']): row for row in previous}
    log_path = output / 'private/imports.json'
    imports = json.loads(log_path.read_text()) if log_path.exists() else []
    # Validate every file before changing saved ratings or importing raw files.
    parsed = [(Path(path).resolve(), *_normalize_csv(path, by_id, legacy_stage=legacy_stage)) for path in csv_paths]
    pending = []
    for path, values, blank_rows in parsed:
        identical = next((record for record in imports if Path(record['saved_path']).is_file()
                          and filecmp.cmp(path, record['saved_path'], shallow=False)
                          and record.get('legacy_stage') == legacy_stage), None)
        if identical:
            continue
        destination = output / f'private/imports/{len(imports) + 1:04d}.csv'
        updated = accepted = 0
        for row in values:
            key = row['listener'], row['item_id'], row['variant']
            if not _has_response(row) and key not in rows:
                continue
            updated += key in rows and rows[key] != row
            rows[key] = row
            accepted += 1
        imports.append({'source_filename': path.name, 'saved_path': str(destination),
                        'source': 'listener-supplied CSV; identity/listening not independently verified',
                        'legacy_stage': legacy_stage, 'accepted_rows': accepted,
                        'blank_source_rows': blank_rows, 'updated_existing_rows': updated})
        pending.append((path, destination))
    # Validate the final combined snapshot before saving raw files or ratings.
    # A partially corrected group must not leave its older votes inconsistent.
    _validate_preferences(rows.values())
    for path, destination in pending:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    atomic_json(log_path, imports)
    atomic_jsonl(output / 'private/ratings.jsonl', sorted(rows.values(), key=lambda row: (row['listener'], row['item_id'], row['variant'])))
    return summarize_ratings(output)


def summarize_ratings(output='runs/decision_study/listening'):
    output = Path(output).resolve()
    manifest = json.loads((output / 'private/manifest.json').read_text())
    items = {item['item_id']: item for item in manifest['items']}
    rows = read_jsonl(output / 'private/ratings.jsonl')
    annotated = []
    for row in rows:
        item = items[row['item_id']]
        methods = {variant['label']: variant['method'] for variant in item['variants']}
        annotated.append({**row, 'method': methods[row['variant']], 'work_id': item['work_id'],
                          'kind': item['kind'], 'pilot': item['pilot']})
    groups = defaultdict(list)
    for row in annotated:
        for metric in METRICS:
            if row[metric] is not None:
                groups[(row['stage'], row['accompaniment'], row['method'], metric)].append(row)
    ratings = []
    for (stage, accompaniment, method, metric), group in sorted(groups.items()):
        listener_values = defaultdict(list)
        for row in group:
            listener_values[row['listener']].append(row[metric])
        ratings.append({'stage': stage, 'accompaniment': accompaniment, 'method': method, 'metric': metric,
                        'rated_variants': len(group), 'listeners': len(listener_values),
                        'works': len({r['work_id'] for r in group}), 'items': len({r['item_id'] for r in group}),
                        'listener_macro_mean': mean(mean(values) for values in listener_values.values()),
                        'raw_rating_mean': mean(r[metric] for r in group)})
    lookup = {(row['listener'], row['item_id'], row['method']): row for row in annotated}
    differences = defaultdict(list)
    for row in annotated:
        if row['method'] == 'one_shot_joint':
            continue
        reference = lookup.get((row['listener'], row['item_id'], 'one_shot_joint'))
        if reference is None:
            continue
        for metric in METRICS:
            if row[metric] is not None and reference[metric] is not None:
                differences[(row['stage'], row['accompaniment'], row['method'], metric)].append(
                    {'listener': row['listener'], 'work_id': row['work_id'], 'item_id': row['item_id'],
                     'difference': row[metric] - reference[metric]})
    paired = []
    for (stage, accompaniment, method, metric), group in sorted(differences.items()):
        per_listener_work = defaultdict(list)
        for row in group:
            per_listener_work[(row['listener'], row['work_id'])].append(row['difference'])
        per_listener = defaultdict(list)
        for (listener, _), values in per_listener_work.items():
            per_listener[listener].append(mean(values))
        paired.append({'stage': stage, 'accompaniment': accompaniment, 'candidate': method,
                       'reference': 'one_shot_joint', 'metric': metric, 'matched_listener_items': len(group),
                       'listeners': len(per_listener), 'works': len({r['work_id'] for r in group}),
                       'listener_work_macro_difference': mean(mean(values) for values in per_listener.values()),
                       'matched_differences': group})
    preferences = {}
    for row in annotated:
        if row['preference']:
            key = row['listener'], row['item_id']
            value = row['preference']
            # Imports are snapshots; repeated per-variant preference is one vote.
            if key in preferences and preferences[key]['label'] != value:
                raise ValueError('saved imports contain inconsistent item preferences; reimport a complete corrected group')
            item = items[row['item_id']]
            methods = {v['label']: v['method'] for v in item['variants']}
            preferences[key] = {'listener': row['listener'], 'item_id': row['item_id'], 'stage': row['stage'],
                                'accompaniment': row['accompaniment'], 'work_id': row['work_id'],
                                'label': value, 'preferred_method': methods.get(value), 'outcome': value if value in ('tie', 'unsure') else 'preference'}
    usable_scores = sum(row[metric] is not None for row in annotated for metric in METRICS)
    usable_preferences = sum(row['outcome'] != 'unsure' for row in preferences.values())
    status = 'human_ratings_received' if usable_scores or usable_preferences else 'pending_human'
    summary = {'status': status, 'materials_status': 'completed', 'rating_rows': len(rows),
               'listeners_with_responses': len({row['listener'] for row in annotated if _has_response(row)}),
               'usable_numeric_scores': usable_scores, 'usable_preferences': usable_preferences,
               'rated_items': len({row['item_id'] for row in annotated if any(row[m] is not None for m in METRICS)}),
               'rated_works': len({row['work_id'] for row in annotated if any(row[m] is not None for m in METRICS)}),
               'ratings_by_stage_accompaniment_method': ratings,
               'paired_differences_vs_one_shot': paired, 'item_preferences': list(preferences.values()),
               'limitations': ['No human results are inferred from audio preparation or empty CSVs.',
                               'Scores are user-supplied; software validation does not verify listener identity or actual listening.',
                               'One fixed replicate per request; repeated variants/conditions/works are not independent listeners.',
                               'Analyze stages and original/chord-guide accompaniment separately; no significance or confidence claims.',
                               'These are generation-strategy comparisons, not quality comparisons of equal-target exact solvers.',
                               'Received ratings do not automatically establish a completed or adequately powered human study.']}
    atomic_json(output / 'ratings_summary.json', summary)
    atomic_json(output / 'status.json', {'materials_status': 'completed', 'human_evaluation_status': status,
                                       'listeners_with_responses': summary['listeners_with_responses'],
                                       'usable_numeric_scores': usable_scores, 'usable_preferences': usable_preferences})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'import', 'summary'))
    parser.add_argument('--output', default='runs/decision_study/listening')
    parser.add_argument('--revision-root', default='runs/revision')
    parser.add_argument('--csv', nargs='*', default=[])
    parser.add_argument('--legacy-stage', choices=STAGES)
    parser.add_argument('--no-archive', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        report = prepare_listening(args.revision_root, args.output, archive=not args.no_archive)
    elif args.command == 'import':
        report = import_ratings(args.csv, args.output, legacy_stage=args.legacy_stage)
    else:
        report = summarize_ratings(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
