import mido
import numpy as np
import pytest

from tri.data import bars


def score(path):
    path.mkdir(parents=True,exist_ok=True)
    (path/'melody.txt').write_text('60 4\n60 4\n0 8\n64 120\n')
    (path/'finalized_chord.txt').write_text('C:maj [0, 4, 7] 0 16\nN [] 16\n')
    (path/'tempo.txt').write_text('120\n')


def test_release_durations_reattacks_rests_and_trailing_coverage(tmp_path):
    score(tmp_path)
    tokens,labels,tempo,info=bars.read_aligned_score(tmp_path)
    assert len(tokens)==len(labels)==128 and tempo==120
    assert tokens[:16].tolist()==[62,1,1,1,62,1,1,1]+[0]*8
    assert labels[63]=='C:maj' and labels[64]=='N'
    assert info['trailing_cells_outside_common_coverage']==8


@pytest.mark.parametrize('text',['C:maj [0, 4, 7] 0 1 2\n','C:maj [0, 4, 7] 0 .3\n','N [] 0\n'])
def test_ambiguous_chord_columns_and_non_grid_durations_rejected(tmp_path,text):
    score(tmp_path);(tmp_path/'finalized_chord.txt').write_text(text)
    with pytest.raises(ValueError):
        bars.read_aligned_score(tmp_path)


@pytest.mark.parametrize('meters',[[],[(3,4)],[(4,4),(3,4)]])
def test_missing_or_changing_meter_not_assumed_four_four(tmp_path,meters):
    midi=mido.MidiFile();midi.tracks.append(mido.MidiTrack([
        mido.MetaMessage('time_signature',numerator=n,denominator=d) for n,d in meters]))
    path=tmp_path/'meter.mid';midi.save(path)
    with pytest.raises(ValueError,match='explicit constant'):
        bars.require_curated_four_four(path)


def test_adapter_keeps_work_split_and_never_opens_test_annotations(tmp_path,monkeypatch):
    works=[tmp_path/f'{i:03d}.mid' for i in range(1,4)]
    monkeypatch.setattr(bars,'discover_primary_midis',lambda _:works)
    monkeypatch.setattr(bars,'make_work_splits',lambda *a,**k:{'001':'train','002':'validation','003':'test'})
    inspected=[]
    monkeypatch.setattr(bars,'require_curated_four_four',lambda p:inspected.append(p.stem))
    for work in ('001','002'):
        path=tmp_path/'raw/pop909_structure'/work;score(path)
        (path/'melody.txt').write_text('60 4\n'*32)
    report=bars.prepare_bars(tmp_path,tmp_path/'derived',bars=8)
    assert inspected==['001','002'] and report['split_counts']=={'train':1,'validation':1}
    with np.load(tmp_path/'derived/windows.npz') as f:
        assert f['tokens'].shape==(2,128) and not np.any(f['start_cells']%16)
    assert bars.prepare_bars(tmp_path,tmp_path/'derived',bars=8)==report
