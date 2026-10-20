from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['MPLCONFIGDIR'] = str(PACKAGE / 'results' / 'cache' / 'matplotlib')


def project_root(value=None):
    root = Path(value).resolve() if value else PACKAGE.parent
    if not (root / 'src' / 'models.py').is_file():
        raise FileNotFoundError('Place pri_prune_baseline in drpe_formal_experiments, or use --project-root PATH.')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('No rows to write.')
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def source_config(root, seed):
    from src.config import ExperimentConfig
    path = root / f'outputs/cifar10/resnet18/drpe/seed{seed}/resolved_config.json'
    if not path.is_file():
        path = root / f'configs/cifar10/resnet18/drpe/drpe_r030_p75_eta200_seed{seed}_fixed200.json'
    values = read_json(path)
    cfg = ExperimentConfig(**values)
    cfg.data_root = str(root / 'data')
    cfg.dataset_download = False
    # Only common training settings are reused. PriPrune has its own defense,
    # and is not augmented with DRPE's clipping, Gaussian noise or projection.
    cfg.method = 'fedavg'
    cfg.privacy_enabled = False
    cfg.clip_norm = None
    cfg.noise_std = 0.0
    cfg.noise_eta = 0.0
    cfg.calibration_path = None
    cfg.validate()
    return cfg, path


def settings():
    options = read_json(PACKAGE / 'config.json')
    if not 0 < options['base_keep_ratio'] <= 1 or not 0 < options['initial_nonsharing_probability'] < 1:
        raise ValueError('Invalid pruning ratio or initial alpha.')
    if options['lambda_accuracy'] <= 0 or options['lambda_privacy'] < 0 or options['lambda_nonsharing'] < 0:
        raise ValueError('Invalid loss coefficients.')
    if options['mask_learning_rate'] <= 0 or options['gumbel_temperature'] <= 0:
        raise ValueError('Invalid mask learning rate or temperature.')
    if options['prune_scope'] not in {'matrix_weights','all'}:
        raise ValueError('Unknown pruning scope.')
    return options


def output_dir(seed, smoke=False):
    return PACKAGE / 'results' / ('smoke' if smoke else 'formal') / f'seed{seed}'


def partition_digest(indices):
    return hashlib.sha256(json.dumps(indices, separators=(',', ':')).encode()).hexdigest()


def validate_protocol(root):
    p = read_json(root / 'privacy_main_benchmark/protocol.json')
    expected = dict(dataset='cifar10', model='resnet18', num_classes=10,
                    checkpoint_round=20, transmission_round=21, observed_client_id=0,
                    iterations=2000, optimizer='adam', matching_loss='cosine',
                    attack_lr=0.1, label_lr=0.05, tv_weight=1e-4,
                    label_entropy_weight=1e-3)
    for key, value in expected.items():
        if p.get(key) != value:
            raise ValueError(f'Existing attack protocol differs: {key}={p.get(key)!r}; expected {value!r}.')
    if p['model_seeds'] != [42, 43, 44] or p['main']['attack_base_seeds'] != [3401, 3402]:
        raise ValueError('Seed protocol changed. Review the baseline before running.')
    targets = read_csv(root / 'privacy_main_benchmark/target_samples.csv')
    if len(targets) != 10 or sorted(int(x['class_id']) for x in targets) != list(range(10)):
        raise ValueError('Attack targets must be the existing ten class-balanced examples.')
    return p, targets
