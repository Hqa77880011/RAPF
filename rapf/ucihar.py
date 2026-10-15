import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', default='data/ucihar.npz')
    parser.add_argument('--validation-subjects', nargs='+', type=int, default=[1, 3, 5])
    args = parser.parse_args()
    root = Path(args.root)
    values, labels, subjects, splits = [], [], [], []
    for role in ['train', 'test']:
        streams = []
        for modality in ['body_acc', 'body_gyro', 'total_acc']:
            axes = [np.loadtxt(root / role / 'Inertial Signals' / f'{modality}_{axis}_{role}.txt') for axis in ['x', 'y', 'z']]
            streams.append(np.stack(axes, 1))
        x = np.stack(streams, 1).astype(np.float32)
        y = np.loadtxt(root / role / f'y_{role}.txt', dtype=int) - 1
        group = np.loadtxt(root / role / f'subject_{role}.txt', dtype=int)
        split = np.full(len(y), role, dtype='<U5')
        if role == 'train':
            split[np.isin(group, args.validation_subjects)] = 'val'
        values.append(x)
        labels.append(y)
        subjects.append(group)
        splits.append(split)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, x=np.concatenate(values), y=np.concatenate(labels),
                        group=np.concatenate(subjects), split=np.concatenate(splits))


if __name__ == '__main__':
    main()
