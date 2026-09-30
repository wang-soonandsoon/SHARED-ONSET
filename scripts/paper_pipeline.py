"""Prepare aligned data, train, select visible-context requests, and freeze q."""
import argparse
import json
from pathlib import Path

import yaml

from tri.paths import shared_data_root

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare-data', 'train', 'requests', 'freeze'])
    parser.add_argument('--data-root', type=Path, default=shared_data_root())
    parser.add_argument('--dataset-dir', type=Path)
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/fresh')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'assets/bar16_inference.pt')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cohort', choices=['real', 'long'], default='real')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / 'configs/paper_training.yaml').read_text())
    dataset_dir = args.dataset_dir or args.data_root / 'processed/paper2_tri/aligned_bars16'
    dataset, chords = dataset_dir / 'windows.npz', dataset_dir / 'chords/chords.npz'
    output = args.out.resolve()
    if args.command == 'prepare-data':
        from tri.data.bars import prepare_bars
        result = prepare_bars(args.data_root, dataset_dir, **config['data'])
        print(json.dumps({k: result[k] for k in ('status', 'windows', 'length', 'split_counts')}))
    elif args.command == 'train':
        import torch
        from tri.models.grid import GridConfig
        from tri.models.training import fit
        torch.set_num_threads(1)
        result = fit(dataset, chords, output / 'train', config=GridConfig(**config['model']),
                     device=args.device, resume=args.resume, **config['training'])
        print(json.dumps(result, indent=2))
    elif args.command == 'requests':
        from tri.evaluation.shared_rhythm_real_cases import prepare_real_requests, _identity
        from tri.evaluation.shared_rhythm_scale_cases import prepare_long_requests
        provenance = {key: _identity(path) for key, path in
                      [('dataset', dataset), ('chords', chords), ('checkpoint', args.checkpoint)]}
        provenance.update(checkpoint_seed=config['training']['seed'],
                          source='explicit_public_release_inputs')
        options = yaml.safe_load((ROOT / f'configs/shared_rhythm_{args.cohort}.yaml').read_text())['requests']
        prepare = prepare_real_requests if args.cohort == 'real' else prepare_long_requests
        result = prepare(output / f'{args.cohort}_inputs', options=options, provenance=provenance)
        print(json.dumps({k: result[k] for k in ('status', 'selected_count', 'selected_work_count')}))
    else:
        from tri.evaluation.shared_rhythm_real_cases import freeze_real_probabilities
        from tri.evaluation.shared_rhythm_scale_cases import prepare_long_scale_cases
        result = freeze_real_probabilities(output / f'{args.cohort}_inputs', device=args.device)
        if args.cohort == 'long':
            result = prepare_long_scale_cases(output / 'long_inputs/cases.json', output / 'long_scale_inputs')
        print(json.dumps({k: result[k] for k in ('status', 'count') if k in result}))


if __name__ == '__main__':
    main()
