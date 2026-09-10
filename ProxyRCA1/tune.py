import os
import json
import signal
import argparse
import optuna

from entry import ROOT, RUNS, default_config, update_dict_diff
from solvers import *  # noqa: F401,F403


class TrialTimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TrialTimeoutError("Trial exceeded time limit")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('base_run', type=str, help="e.g. beauty1_id/proxyrca, beauty1_image/proxyrca")
    parser.add_argument('--n_trials', type=int, default=20)
    parser.add_argument('--trial_timeout', type=int, default=3600,
                         help="max seconds per trial before it's cancelled (default: 1 hour)")
    return parser.parse_args()


def build_config(trial, base_run_name, tune_run_name):
    config = dict(default_config)

    partial_names = base_run_name.split('/')
    for i in range(1, len(partial_names) + 1):
        p = os.path.join(ROOT, RUNS, '/'.join(partial_names[:i]), 'config.json')
        if os.path.isfile(p):
            with open(p) as fp:
                update_dict_diff(config, json.load(fp))

    # optimizer
    config['train']['optimizer']['lr'] = trial.suggest_float('lr', 1e-6, 1e-3, log=True)
    config['train']['optimizer']['weight_decay'] = trial.suggest_categorical(
        'weight_decay', [0.0, 1e-5, 1e-4, 1e-3]
    )

    # model architecture — hidden_dim and num_heads must be paired together
    # so the search space stays fixed across trials (Optuna requires this)
    head_options = [
        (128, 1), (128, 2), (128, 4),
        (256, 1), (256, 2), (256, 4),
        (512, 1), (512, 2), (512, 4),
        (90, 1), (90, 2), (90, 3), (90, 5),
        (450, 1), (450, 2), (450, 3), (450, 5),
    ]
    idx = trial.suggest_categorical('hidden_dim_num_heads_idx', list(range(len(head_options))))
    hidden_dim, num_heads = head_options[idx]
    config['model']['hidden_dim'] = hidden_dim
    config['model']['num_heads'] = num_heads

    config['model']['num_layers'] = trial.suggest_categorical('num_layers', [1, 2, 3, 4, 5])
    config['model']['dropout_prob'] = trial.suggest_float('dropout_prob', 0.0, 0.5)
    config['model']['num_proxy_item'] = trial.suggest_categorical('num_proxy_item', [64, 128, 256])

    trial_name = f"{tune_run_name}/trial_{trial.number}"
    config['name'] = trial_name
    config['run_dir'] = os.path.join(ROOT, RUNS, trial_name)

    return config


if __name__ == '__main__':
    args = parse_args()
    base_run_name = args.base_run
    tune_run_name = f"{base_run_name}_tuning"

    def objective(trial):
        config = build_config(trial, base_run_name, tune_run_name)
        solver_class = globals()[config['solver']]
        solver = solver_class(config)

        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(args.trial_timeout)
        try:
            solver.solve()
        except TrialTimeoutError:
            print(f"Trial {trial.number} timed out after {args.trial_timeout}s — cancelling and moving on.")
            raise optuna.TrialPruned()
        finally:
            signal.alarm(0)

        return solver.best_score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.n_trials, catch=(Exception,))

    print("Best params:", study.best_params)
    print("Best valid score:", study.best_value)

    out_dir = os.path.join(ROOT, RUNS, tune_run_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'best_params.json'), 'w') as fp:
        json.dump({'base_run': base_run_name, 'best_params': study.best_params, 'best_value': study.best_value}, fp, indent=4)