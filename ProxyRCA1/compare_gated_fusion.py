#!/usr/bin/env python3
import json
import os
import multiprocessing

import torch

from solvers.carca_gated import CarcaGatedSolver

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, 'runs')
DATA_ROOT = os.path.join(ROOT, 'data')

config = {
    'envs': {
        'RUN_ROOT': RUNS,
        'DATA_ROOT': DATA_ROOT,
        'RAW_ROOT': os.path.join(ROOT, 'raw'),
        'CPU_COUNT': max(8, multiprocessing.cpu_count() // 4),
        'GPU_COUNT': torch.cuda.device_count(),
    },
    'solver': 'CarcaGatedSolver',
    'dataset': 'beauty1_id',
    'dataloader': {
        'sequence_len': 35,
        'train_num_negatives': 100,
        'valid_num_negatives': 100,
        'random_cut_prob': 1.0,
        'replace_user_prob': 0.0,
        'replace_item_prob': 0.02,
        'random_seed': 12345,
    },
    'model': {
        'hidden_dim': 256,
        'temporal_dim': 32,
        'num_proxy_item': 128,
        'num_known_item': 0,
        'num_layers': 1,
        'num_heads': 4,
        'dropout_prob': 0.1,
        'temperature': 0.1,
        'random_seed': 12345,
        'feature_mode': 'fusion',
    },
    'train': {
        'epoch': 150,
        'every': 10,
        'patience': 80,
        'batch_size': 128,
        'optimizer': {
            'algorithm': 'adamw',
            'lr': 1e-4,
            'beta1': 0.9,
            'beta2': 0.999,
            'weight_decay': 0.1,
            'amsgrad': False,
        },
    },
    'metric': {
        'ks_valid': [10],
        'ks_test': [1, 5, 10, 20, 50, 100],
        'pivot': 'NDCG@10',
    },
    'memo': 'gated fusion comparison run',
    'name': 'gated_ID/proxyrca',
    'run_dir': os.path.join(RUNS, 'gated_ID', 'proxyrca'),
}

os.makedirs(config['run_dir'], exist_ok=True)
with open(os.path.join(config['run_dir'], 'config.json'), 'w') as fp:
    json.dump({
        'solver': 'CarcaGatedSolver',
        'model': {
            'num_known_item': 0,
            'feature_mode': 'fusion',
            'hidden_dim': 256,
            'num_layers': 1,
            'num_heads': 4,
            'dropout_prob': 0.1,
            'random_seed': 12345,
            'id_emb_dim': 64,
        },
        'train': {
            'epoch': 150,
            'every': 10,
            'patience': 80,
            'batch_size': 128,
            'optimizer': {
                'algorithm': 'adamw',
                'lr': 1e-4,
                'beta1': 0.9,
                'beta2': 0.999,
                'weight_decay': 0.1,
                'amsgrad': False,
            },
        },
    }, fp, indent=4)

solver = CarcaGatedSolver(config)
solver.solve()
