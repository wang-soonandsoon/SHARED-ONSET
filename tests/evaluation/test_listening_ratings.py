import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
import wave
import zipfile

import mido
import pytest

from tri.evaluation.listening_ratings import (
    FIELDS, KINDS, METHODS, METRICS, SCHEMA, STAGES,
    import_ratings, prepare_listening, summarize_ratings,
)


@pytest.fixture
def handoff(tmp_path):
    revision, output = tmp_path / 'revision', tmp_path / 'handoff'
    for stage in STAGES:
        source = revision / stage
        public = source / 'listening/public'
        public.mkdir(parents=True)
        requests, items, private = [], [], []
        for index, kind in enumerate(KINDS):
            item_id, case_id = f'item_{index:04d}', f'case-{stage}-{kind}'
            requests.append({'case_id': case_id, 'work_id': 'work-one', 'kind': kind,
                             'spec': {'equal_onsets': [[0, 4], [1, 5]], 'observed': {'0': 60, '1': 1} if kind == 'known' else {}}})
            variants, keys = [], []
            for label, method in zip('ABCD', METHODS):
                available = not (stage == 'bar8' and kind == 'partial' and label == 'D')
                variant = {'label': label, 'available': available}
                keys.append({'label': label, 'method': method, 'status': 'exported' if available else 'failed'})
                if available:
                    wav, midi = f'{item_id}_{label}.wav', f'{item_id}_{label}.mid'
                    with wave.open(str(public / wav), 'wb') as audio:
                        audio.setnchannels(1)
                        audio.setsampwidth(2)
                        audio.setframerate(8000)
                        audio.writeframes(b'\x00\x01' * 800)
                    music = mido.MidiFile(ticks_per_beat=480)
                    music.tracks.append(mido.MidiTrack([
                        mido.MetaMessage('set_tempo', tempo=500000, time=0),
                        mido.Message('note_on', note=60, time=0),
                        mido.MetaMessage('set_tempo', tempo=1000000, time=480),
                        mido.Message('note_off', note=60, time=480),
                    ]))
                    music.save(public / midi)
                    variant.update(audio=wav, midi=midi)
                variants.append(variant)
            items.append({'item_id': item_id, 'accompaniment': 'chord_guide', 'variants': variants})
            private.append({'item_id': item_id, 'case_id': case_id, 'variants': keys})
        (source / 'cohort.json').write_text(json.dumps({'status': 'completed', 'options': {'split': 'validation'}, 'requests': requests}))
        (source / 'listening/report.json').write_text(json.dumps({'status': 'completed'}))
        (source / 'listening/private_key.json').write_text(json.dumps({'items': private}))
        (public / 'items.json').write_text(json.dumps({'items': items}))
        (public / 'index.html').write_text('original frozen page')
    prepare_listening(revision, output)
    return revision, output


def row(item='short_item_0000', variant='A', **changes):
    value = dict.fromkeys(FIELDS, '')
    value.update(schema=SCHEMA, listener='L01', stage=item.split('_item_')[0], item_id=item,
                 variant=variant, available='true', accompaniment='chord_guide', naturalness='4')
    value.update(changes)
    return value


def write_csv(path, rows, fields=FIELDS):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_preparation_freezes_audio_labels_and_excludes_private_data(handoff):
    revision, output = handoff
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in revision.rglob('*') if p.is_file()}
    inventory = prepare_listening(revision, output)
    assert inventory['groups'] == inventory['pilot_groups'] == 12
    assert inventory['available_audio'] == inventory['pilot_available_audio'] == 47
    assert inventory['unavailable_slots'] == 1
    assert inventory['pilot_unique_works'] == 1
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before} == before
    items = json.loads((output / 'public/items.json').read_text())['items']
    assert items[0]['regions'] == [
        {'name': '片段 1', 'start_seconds': 0., 'end_seconds': .25, 'given': False},
        {'name': '片段 2', 'start_seconds': .5, 'end_seconds': 1., 'given': False}]
    assert items[3]['regions'][0]['given'] is False  # harmony remains hidden
    assert items[2]['regions'][0]['given'] is True
    failed = items[5]['variants'][3]
    assert failed == {'label': 'D', 'available': False}
    assert (output / 'public/audio/short_item_0000_A.wav').read_bytes() == (revision / 'short/listening/public/item_0000_A.wav').read_bytes()
    page = (output / 'public/index.html').read_text()
    for private in (*METHODS, 'work-one', 'case-short', 'private_key'):
        assert private not in page
    with zipfile.ZipFile(output / 'listener_package.zip') as bundle:
        assert set(bundle.namelist()) == {'index.html', 'items.json', *[v['audio'] for i in items for v in i['variants'] if v['available']]}
        assert not any('private' in name for name in bundle.namelist())
    status = summarize_ratings(output)
    assert status['status'] == 'pending_human'
    assert status['rating_rows'] == status['listeners_with_responses'] == 0


def test_blank_csv_cannot_be_a_human_result(handoff, tmp_path):
    _, output = handoff
    path = write_csv(tmp_path / 'empty.csv', [row(naturalness='', listener='')])
    summary = import_ratings([path], output)
    assert summary['status'] == 'pending_human'
    assert summary['usable_numeric_scores'] == summary['usable_preferences'] == summary['rating_rows'] == 0
    path2 = write_csv(tmp_path / 'named-empty.csv', [row(naturalness='')])
    assert import_ratings([path2], output)['listeners_with_responses'] == 0


def test_import_frozen_mapping_matched_scores_and_single_preference(handoff, tmp_path):
    _, output = handoff
    values = [row(variant=label, naturalness=str(score), preference='C') for label, score in zip('ABCD', (2, 3, 5, 4))]
    path = write_csv(tmp_path / 'real.csv', values)
    summary = import_ratings([path], output)
    assert summary['status'] == 'human_ratings_received'
    assert summary['usable_numeric_scores'] == 4
    assert summary['usable_preferences'] == 1
    assert summary['listeners_with_responses'] == summary['rated_works'] == summary['rated_items'] == 1
    assert summary['item_preferences'][0]['preferred_method'] == 'tri_direct'
    pairs = {p['candidate']: p for p in summary['paired_differences_vs_one_shot']}
    assert pairs['tri_direct']['listener_work_macro_difference'] == 2
    assert pairs['pooled_template']['listener_work_macro_difference'] == -1
    assert pairs['smc_4']['listener_work_macro_difference'] == 1
    assert import_ratings([path], output) == summary
    assert len(json.loads((output / 'private/imports.json').read_text())) == 1


def test_paired_macro_counts_listener_work_not_rows_as_independent(handoff, tmp_path):
    _, output = handoff
    values = [row(variant='B', naturalness='3'), row(variant='C', naturalness='5'),
              row(item='short_item_0001', variant='B', naturalness='3'),
              row(item='short_item_0001', variant='C', naturalness='3'),
              row(variant='B', naturalness='3', listener='L02'), row(variant='C', naturalness='2', listener='L02')]
    summary = import_ratings([write_csv(tmp_path / 'ratings.csv', values)], output)
    paired, = summary['paired_differences_vs_one_shot']
    assert paired['matched_listener_items'] == 3
    assert paired['listeners'] == 2 and paired['works'] == 1
    assert paired['listener_work_macro_difference'] == 0  # (mean(2,0) + -1) / 2


@pytest.mark.parametrize('changes', [
    {'naturalness': '6'}, {'naturalness': 'nan'}, {'naturalness': '2.0'},
    {'variant': 'X'}, {'available': 'false'}, {'accompaniment': 'original'},
    {'listener': ''}, {'stage': 'bar8'}, {'item_id': 'bad'}, {'preference': 'Z'},
    {'schema': 'made-up'}, {'available': '1'},
    {'item_id': 'bar8_item_0001', 'stage': 'bar8', 'variant': 'D', 'available': 'false'},
    {'item_id': 'bar8_item_0001', 'stage': 'bar8', 'preference': 'D'},
])
def test_invalid_import_does_not_mutate_saved_ratings(handoff, tmp_path, changes):
    _, output = handoff
    good = write_csv(tmp_path / 'good.csv', [row()])
    import_ratings([good], output)
    before = {p: p.read_bytes() for p in output.rglob('*') if p.is_file()}
    bad = write_csv(tmp_path / 'bad.csv', [row(**changes)])
    with pytest.raises(ValueError):
        import_ratings([bad], output)
    assert {p: p.read_bytes() for p in output.rglob('*') if p.is_file()} == before


def test_conflicting_preferences_are_transactional_and_full_update_clears(handoff, tmp_path):
    _, output = handoff
    initial = write_csv(tmp_path / 'initial.csv', [row(variant=v, preference='A') for v in 'ABCD'])
    import_ratings([initial], output)
    before = (output / 'private/ratings.jsonl').read_bytes()
    partial = write_csv(tmp_path / 'partial.csv', [row(preference='B')])
    with pytest.raises(ValueError, match='conflicting saved'):
        import_ratings([partial], output)
    assert (output / 'private/ratings.jsonl').read_bytes() == before
    assert len(list((output / 'private/imports').glob('*.csv'))) == 1
    corrected = write_csv(tmp_path / 'corrected.csv', [row(variant=v, naturalness='', preference='B') for v in 'ABCD'])
    corrected_summary = import_ratings([corrected], output)
    assert corrected_summary['usable_numeric_scores'] == 0
    assert corrected_summary['item_preferences'][0]['preferred_method'] == 'one_shot_joint'
    cleared = write_csv(tmp_path / 'cleared.csv', [row(variant=v, naturalness='') for v in 'ABCD'])
    clear_summary = import_ratings([cleared], output)
    assert clear_summary['status'] == 'pending_human'
    assert clear_summary['listeners_with_responses'] == 0


def test_conflicting_duplicates_and_within_file_preferences_fail(handoff, tmp_path):
    _, output = handoff
    for values in ([row(), row(naturalness='1')], [row(preference='A'), row(variant='B', preference='B')]):
        path = write_csv(tmp_path / 'bad.csv', values)
        with pytest.raises(ValueError, match='conflicting'):
            import_ratings([path], output)
    assert not (output / 'private/ratings.jsonl').exists()


def test_legacy_stage_is_explicit_and_chinese_vote_normalizes(handoff, tmp_path):
    _, output = handoff
    fields = [name for name in FIELDS if name not in ('schema', 'stage')]
    value = {key: value for key, value in row(item_id='item_0000', preference='并列').items() if key in fields}
    path = write_csv(tmp_path / 'legacy.csv', [value], fields)
    with pytest.raises(ValueError, match='legacy-stage'):
        import_ratings([path], output)
    summary = import_ratings([path], output, legacy_stage='short')
    assert summary['item_preferences'][0]['outcome'] == 'tie'
    assert summary['item_preferences'][0]['item_id'] == 'short_item_0000'


def test_unsure_without_scores_is_not_usable_human_evidence(handoff, tmp_path):
    _, output = handoff
    summary = import_ratings([write_csv(tmp_path / 'unsure.csv', [row(naturalness='', preference='unsure')])], output)
    assert summary['status'] == 'pending_human'
    assert summary['usable_preferences'] == 0
    assert summary['listeners_with_responses'] == 1


def test_mismatched_frozen_audio_refuses_overwrite(handoff):
    revision, output = handoff
    target = output / 'public/audio/short_item_0000_A.wav'
    target.write_bytes(b'changed')
    with pytest.raises(ValueError, match='copied audio differs'):
        prepare_listening(revision, output, archive=False)
    assert target.read_bytes() == b'changed'


def test_offline_form_export_logic_in_node_stub(handoff, tmp_path):
    """Test script/CSV behavior in a tiny DOM stub, not actual browser playback."""
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is optional for standalone page script test')
    _, output = handoff
    script = (output / 'public/index.html').read_text().split('<script>', 1)[1].split('</script>', 1)[0]
    harness = r'''
const assert = require('node:assert/strict');
(async()=>{
class Element {
  constructor(tag){this.tag=tag;this.children=[];this.value='';this.textContent=''}
  append(...children){this.children.push(...children)}
  replaceChildren(...children){this.children=children}
  click(){} pause(){} play(){return Promise.resolve()}
}
const nodes = Object.fromEntries(['listener','scope','progress','items','notice','download'].map(id=>[id,new Element(id)]));
const document={getElementById:id=>nodes[id],createElement:tag=>new Element(tag),querySelectorAll:()=>[]};
const localStorage={getItem:()=>null,setItem:()=>{}};
const location={pathname:'/test-only'};
let exportedBlob;
const URL={createObjectURL:blob=>{exportedBlob=blob;return 'test-only'},revokeObjectURL:()=>{}};
const setTimeout=callback=>callback();
__SCRIPT__
assert.equal(nodes.items.children.length,12);
nodes.download.onclick();assert.match(nodes.notice.textContent,/匿名听者/);assert.equal(exportedBlob,undefined);
saved.listener='TEST_ONLY';nodes.download.onclick();assert.match(nodes.notice.textContent,/还没有填写/);assert.equal(exportedBlob,undefined);
saved['short_item_0000|A|naturalness']='5';saved['short_item_0000|preference']='tie';
nodes.download.onclick();assert.ok(exportedBlob);process.stdout.write(await exportedBlob.text());
})().catch(error=>{console.error(error);process.exit(1)});
'''.replace('__SCRIPT__', script)
    path = tmp_path / 'form-test.cjs'
    path.write_text(harness)
    result = subprocess.run([node, str(path)], check=True, capture_output=True, text=True, timeout=10)
    rows = list(csv.DictReader(io.StringIO(result.stdout.lstrip('\ufeff'))))
    assert len(rows) == 4
    assert rows[0]['schema'] == SCHEMA
    assert rows[0]['listener'] == 'TEST_ONLY'
    assert rows[0]['naturalness'] == '5'
    assert all(row['preference'] == 'tie' for row in rows)
    assert rows[1]['naturalness'] == ''
    assert not (output / 'private/ratings.jsonl').exists()
