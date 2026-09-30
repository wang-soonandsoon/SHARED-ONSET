"""Fetch the three pinned upstream corpora used by the aligned-data adapter."""
import argparse
from pathlib import Path
import subprocess

from tri.paths import shared_data_root

SOURCES = [
    ('POP909-Dataset', 'https://github.com/music-x-lab/POP909-Dataset.git',
     'd83e6edba6872a704f5d3b8b32f5cb540088dae6', 'POP909', 'pop909'),
    ('hierarchical-structure-analysis', 'https://github.com/Dsqvival/hierarchical-structure-analysis.git',
     '3a8d0d096e8dd2f62e38e48500bf0b230eeca585', 'POP909', 'pop909_structure'),
    ('POP909-CL-Dataset', 'https://github.com/AndyWeasley2004/POP909-CL-Dataset.git',
     'be9094392903c471a930519e1c0bacf8b6be5d62', '.', 'pop909_cl'),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=shared_data_root())
    args = parser.parse_args()
    root = args.data_root.resolve()
    (root / 'upstream').mkdir(parents=True, exist_ok=True)
    (root / 'raw').mkdir(exist_ok=True)
    for name, url, commit, subdir, alias in SOURCES:
        checkout = root / 'upstream' / name
        if not checkout.exists():
            subprocess.run(['git', 'init', str(checkout)], check=True)
            subprocess.run(['git', '-C', str(checkout), 'remote', 'add', 'origin', url], check=True)
            subprocess.run(['git', '-C', str(checkout), 'fetch', '--depth=1', 'origin', commit], check=True)
            subprocess.run(['git', '-C', str(checkout), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
        actual = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != commit:
            raise ValueError(f'{checkout} is not at the paper version; use a fresh data root')
        source, target = (checkout / subdir).resolve(), root / 'raw' / alias
        if not source.is_dir():
            raise FileNotFoundError(source)
        if target.exists() or target.is_symlink():
            if target.resolve() != source:
                raise ValueError(f'Refusing to replace existing data: {target}')
        else:
            target.symlink_to(source, target_is_directory=True)
        print(f'{alias}: {source}')


if __name__ == '__main__':
    main()
