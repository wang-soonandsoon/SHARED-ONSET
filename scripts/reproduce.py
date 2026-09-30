"""Run the paper's fixed-input experiments, or inspect the recorded results."""
import argparse
import json
import os
from pathlib import Path
import zipfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    'synthetic': ('shared_rhythm_extension.yaml', 'synthetic', 'synthetic'),
    'real': ('shared_rhythm_extension.yaml', 'real', 'real_sampling'),
    'real_proposal': ('shared_rhythm_proposals.yaml', 'real_proposal', 'real_proposal'),
    'long': ('shared_rhythm_proposals.yaml', 'long', 'long_sampling'),
}


def extract(archive, destination):
    """Extract bundled assets once, without overwriting an edited input."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError('Archive member escapes its destination')
            content = bundle.read(info)
            if target.exists():
                if target.read_bytes() != content:
                    raise ValueError(f'Existing file differs: {target}; use a new --out')
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)


def configuration(stage, inputs, cpu):
    filename, key, _ = STAGES[stage]
    config = yaml.safe_load((ROOT / 'configs' / filename).read_text())
    config['common']['cpu'] = cpu
    family = 'real' if stage == 'real_proposal' else stage
    config['stages'][key]['manifest'] = str((inputs / family / 'cases.json').resolve())
    return config, key


def retain_model_metadata(out):
    """The supplied q retains its original, separately measured provider cost."""
    with zipfile.ZipFile(ROOT / 'assets/recorded_runs.zip') as bundle:
        for family in ('real_inputs', 'long_inputs'):
            target = out / family / 'cases.json'
            content = bundle.read(f'{family}/cases.json')
            if target.exists() and target.read_bytes() != content:
                raise ValueError(f'Existing model provenance differs: {target}')
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['extract', 'smoke', *STAGES, 'recorded', 'report'])
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/reproduction')
    parser.add_argument('--cpu', type=int, default=min(os.sched_getaffinity(0)))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    if args.cpu not in os.sched_getaffinity(0):
        parser.error('--cpu must be available in this process CPU affinity')
    if args.stage == 'recorded':
        extract(ROOT / 'assets/recorded_runs.zip', out / 'recorded')
        from tri.evaluation.shared_rhythm_extension_delivery import build_delivery
        build_delivery(out / 'recorded', out / 'recorded_report.md', plots=False)
        print(out / 'recorded_report.md')
        return
    if args.stage == 'report':
        from tri.evaluation.shared_rhythm_extension_delivery import build_delivery
        build_delivery(out, out / 'report.md', plots=False)
        print(out / 'report.md')
        return
    for family in ('synthetic', 'real', 'long'):
        extract(ROOT / f'assets/{family}_inputs.zip', out / 'inputs' / family)
    retain_model_metadata(out)
    if args.stage == 'extract':
        print(out / 'inputs')
        return
    stage = 'real' if args.stage == 'smoke' else args.stage
    config, key = configuration(stage, out / 'inputs', args.cpu)
    target = out / STAGES[stage][2]
    if args.stage == 'smoke':
        # One unchanged learned R=3 target; this is a functional check, not a paper table.
        settings = config['stages'][key]
        original = Path(settings['manifest'])
        manifest = json.loads(original.read_text())
        entry = next(e for e in manifest['cases'] if e['metadata']['R'] == 3)
        entry['path'] = str(original.parent / entry['path'])
        manifest.update(cases=[entry], count=1)
        smoke_manifest = out / 'smoke_cases.json'
        content = json.dumps(manifest, indent=2) + '\n'
        if not smoke_manifest.exists() or smoke_manifest.read_text() != content:
            smoke_manifest.write_text(content)
        settings.update(manifest=str(smoke_manifest), expected_cases=1,
                        backends=['product_multi', 'onset_rejection_boundary',
                                  'onset_rejection_visible_early'], samples_per_worker=2)
        target = out / 'smoke'
    from tri.evaluation.shared_rhythm_extension import run
    result = run(config, key, target, resume=args.resume, plan_only=args.plan_only)
    if args.stage == 'smoke' and not args.plan_only and result['status_counts'] != {'completed': 3}:
        raise RuntimeError('The smoke check did not complete all three backends; inspect its retained records')
    print(target)


if __name__ == '__main__':
    main()
