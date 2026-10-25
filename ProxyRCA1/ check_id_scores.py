import os
import json
import argparse

import torch

from entry import ROOT, RUNS, default_config, update_dict_diff
from solvers import *  # noqa: F401,F403


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, help="an existing run, e.g. beauty1_id/proxyrca")
    parser.add_argument('--num_users', type=int, default=15)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    run_root = os.path.join(ROOT, RUNS)
    run_dir = os.path.join(run_root, args.name)
    if not os.path.isdir(run_dir):
        raise Exception(f"run dir not found: {run_dir}")

    config = dict(default_config)
    partial_names = args.name.split('/')
    for i in range(1, len(partial_names) + 1):
        p = os.path.join(run_root, '/'.join(partial_names[:i]), 'config.json')
        if os.path.isfile(p):
            with open(p) as fp:
                update_dict_diff(config, json.load(fp))

    config['name'] = args.name
    config['run_dir'] = run_dir

    solver_class = globals()[config['solver']]
    solver = solver_class(config)
    solver.load_model('test')  # loads the best saved checkpoint
    solver.set_model_mode('eval')

    print(f"num_items in this dataset: {solver.test_dataset.num_items}")
    print(f"padding_token: {solver.test_dataset.padding_token}")
    print()

    checked = 0
    ranks = []
    with torch.no_grad():
        for batch in solver.test_dataloader:
            if checked >= args.num_users:
                break

            logits = solver.calculate_forward(batch)  # (1, num_candidates)
            logits = logits[0]

            true_score = logits[0].item()  # index 0 is always the true/answer item
            sorted_idx = torch.argsort(logits, descending=True)
            rank = (sorted_idx == 0).nonzero(as_tuple=True)[0].item() + 1  # 1-based rank

            num_candidates = logits.shape[0]
            neg_scores = logits[1:]

            print(
                f"user {checked}: true_score={true_score:.4f}  "
                f"rank={rank}/{num_candidates}  "
                f"neg_min={neg_scores.min().item():.4f}  "
                f"neg_max={neg_scores.max().item():.4f}  "
                f"neg_mean={neg_scores.mean().item():.4f}"
            )
            ranks.append(rank)
            checked += 1

    print()
    print(f"checked {checked} users")
    print(f"mean rank: {sum(ranks) / len(ranks):.1f}")
    print(f"median rank: {sorted(ranks)[len(ranks) // 2]}")
    num_candidates_seen = logits.shape[0]
    print(f"expected mean rank under pure random guessing: {num_candidates_seen / 2:.1f}")
