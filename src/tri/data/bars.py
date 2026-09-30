"""Bar-grid adapter for the locally available manually aligned POP909 release.

Use hierarchical-structure-analysis melody.txt/finalized_chord.txt together,
never combine their shifted timelines with original MIDI timestamps. Sixteenth
durations and a zero bar origin are explicitly documented by that release.
Restrict to constant 4/4 confirmed by curated POP909-CL metadata. The original
work split is retained; this revision materializes train/validation only.
"""
from collections import Counter
from pathlib import Path
import json
import re

import mido
import numpy as np

from tri.data.chords import encode_chord_features, load_chord_sidecar
from tri.data.prepare import make_work_splits, discover_primary_midis
from tri.request_builder import sounding_pitches
from tri.runtime import atomic_json


def read_aligned_score(directory):
    directory=Path(directory)
    tokens=[]
    for number,line in enumerate((directory/'melody.txt').read_text().splitlines(),1):
        if not line.strip():
            continue
        parts=line.split()
        if len(parts)!=2:
            raise ValueError(f'melody line {number}: expected pitch and duration')
        pitch,duration=map(int,parts)
        if not 0<=pitch<=127 or duration<=0:
            raise ValueError(f'melody line {number}: invalid pitch/duration')
        tokens.extend([0]*duration if pitch==0 else [pitch+2]+[1]*(duration-1))
    labels=[]
    for number,line in enumerate((directory/'finalized_chord.txt').read_text().splitlines(),1):
        if not line.strip():
            continue
        match=re.fullmatch(r'\s*(\S+)\s+\[([^\]]*)\]\s+(-?\d+)\s+([\d.]+)\s*',line)
        no_chord=re.fullmatch(r'\s*N\s+\[\]\s+([\d.]+)\s*',line)
        if match:
            label=match[1];duration=float(match[4])*4
            tones=[int(v.strip()) for v in match[2].split(',') if v.strip()]
            if any(not 0<=v<=11 for v in tones):
                raise ValueError('invalid annotated chord tone')
        elif no_chord:
            label='N';duration=float(no_chord[1])*4
        else:
            raise ValueError(f'chord line {number}: ambiguous format')
        if not np.isfinite(duration) or duration<=0 or duration!=int(duration):
            raise ValueError(f'chord line {number}: non-grid duration')
        labels.extend([label]*int(duration))
    tempo=float((directory/'tempo.txt').read_text().strip())
    if not 20<=tempo<=300 or not tokens or not labels:
        raise ValueError('empty score or unsupported tempo')
    # Different trailing coverage is reported, never padded or time-warped.
    coverage=min(len(tokens),len(labels))
    return np.asarray(tokens[:coverage],dtype=np.int64),np.asarray(labels[:coverage]),tempo,{
        'melody_cells':len(tokens),'chord_cells':len(labels),'common_cells':coverage,
        'trailing_cells_outside_common_coverage':abs(len(tokens)-len(labels))}


def require_curated_four_four(path):
    midi=mido.MidiFile(path)
    signatures={(m.numerator,m.denominator) for t in midi.tracks for m in t if m.type=='time_signature'}
    if signatures!={(4,4)}:
        raise ValueError(f'curated meter is not explicit constant 4/4: {sorted(signatures)}')


def prepare_bars(data_root, output_dir, *, bars=8, max_windows_per_work=12, seed=20260912):
    if bars not in (8,16) or max_windows_per_work<1:
        raise ValueError('bar revision supports 8 or 16 bars and a positive window cap')
    root=Path(data_root).resolve();output=Path(output_dir).resolve()
    if root/'raw' in output.parents or output==root/'raw':
        raise ValueError('cannot write original data')
    configuration={'bars':bars,'max_windows_per_work':max_windows_per_work,'seed':seed,'source':'structure_manually_aligned_v1','splits':['train','validation']}
    report_path=output/'report.json'
    if report_path.exists():
        report=json.loads(report_path.read_text())
        if report['configuration']!=configuration:
            raise ValueError('bar dataset configuration changed; use a new output directory')
        load_chord_sidecar(output/'windows.npz',output/'chords/chords.npz')
        return report
    works=discover_primary_midis(root)
    splits=make_work_splits([p.stem for p in works],seed=seed)
    length=bars*16; rows=[];labels=[];ids=[];starts=[];initials=[];names=[];tempos=[]
    excluded=[];accepted=[]
    for source in works:
        work=source.stem
        if splits[work]=='test':
            continue  # Existing test works are not read by this revised dataset.
        try:
            require_curated_four_four(root/'raw/pop909_cl/POP909_processed'/f'{work}.mid')
            tokens,chords,tempo,coverage=read_aligned_score(root/'raw/pop909_structure'/work)
            sounding=sounding_pitches(tokens,None)
            candidates=[start for start in range(0,len(tokens)-length+1,length)
                        if np.count_nonzero(tokens[start:start+length]>=2)>=bars]
            rng=np.random.default_rng(seed+int(work))
            chosen=sorted(rng.choice(candidates,size=min(len(candidates),max_windows_per_work),replace=False).tolist()) if candidates else []
            if not chosen:
                raise ValueError('no complete active bar window in common annotation coverage')
            for start in chosen:
                rows.append(tokens[start:start+length]);labels.append(chords[start:start+length])
                ids.append(work);starts.append(start);names.append(splits[work]);tempos.append(tempo)
                pitch=None if start==0 else sounding[start-1]
                initials.append(-1 if pitch is None else pitch)
            accepted.append({'work_id':work,'windows':len(chosen),'split':splits[work],**coverage})
        except (ValueError,OSError,EOFError) as error:
            excluded.append({'work_id':work,'reason':str(error)})
    if not rows or not {'train','validation'}<=set(names):
        raise ValueError('bar data lacks training or validation windows')
    output.mkdir(parents=True,exist_ok=True);(output/'chords').mkdir(exist_ok=True)
    identity={'work_ids':np.asarray(ids,dtype='U3'),'start_cells':np.asarray(starts,dtype=np.int64),'splits':np.asarray(names,dtype='U10')}
    np.savez_compressed(output/'windows.npz',tokens=np.stack(rows),**identity,
                        initial_pitches=np.asarray(initials,dtype=np.int64),time_signatures=np.tile([4,4],(len(rows),1)),
                        source_tempos=np.asarray(tempos),bar_indices=np.asarray(starts)//16)
    chord_labels=np.stack(labels);known=np.ones(chord_labels.shape,dtype=bool)
    features,feature_known=encode_chord_features(chord_labels,known)
    seconds=(np.asarray(starts)[:,None]+np.arange(length+1)[None,:])*(60/np.asarray(tempos)[:,None]/4)
    np.savez_compressed(output/'chords/chords.npz',**identity,chord_labels=chord_labels,chord_known=known,
                        chord_features=features,chord_feature_known=feature_known,cell_seconds=seconds,
                        row_status=np.full(len(rows),'aligned',dtype='U12'))
    load_chord_sidecar(output/'windows.npz',output/'chords/chords.npz')
    report={'status':'completed','configuration':configuration,'windows':len(rows),'length':length,
            'split_counts':dict(Counter(names)),'accepted_works':len(accepted),'excluded_works':len(excluded),
            'accepted':accepted,'excluded':excluded,'unknown_feature_cells':int((~feature_known).sum()),
            'alignment':{'bar_origin_cell':0,'cells_per_bar':16,'meter':'explicit 4/4 in curated metadata',
                         'source_readme':'https://github.com/Dsqvival/hierarchical-structure-analysis/blob/master/README.txt',
                         'timeline':'manual aligned melody.txt and finalized_chord.txt, not raw MIDI time'},
            'limitations':['Release-provided manual alignment; not independently reannotated by this project.',
                           'First annotated tempo is constant; original expressive timing/accompaniment is not transferred.',
                           'Whole work remains in its original split. No test work is materialized or evaluated.']}
    atomic_json(output/'splits.json',{'seed':seed,'unit':'primary_work','work_splits':splits})
    atomic_json(report_path,report)
    return report
