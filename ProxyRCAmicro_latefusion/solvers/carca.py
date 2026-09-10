import os
import pickle
from shutil import copyfile

import numpy as np
from torch import topk as torch_topk

from models import Carca
from .base import BaseLWPContrastiveSolver

__all__ = ('CarcaSolver',)


def _load_feature_matrix(data_dir, feature_mode: str):
    with open(os.path.join(data_dir, 'iid2iindex.pkl'), 'rb') as fp:
        iid2iindex = pickle.load(fp)
    iindex2iid = {iindex: iid for iid, iindex in iid2iindex.items()}

    if feature_mode == 'fusion':
        image_features = _load_feature_matrix(data_dir, 'image')
        text_features = _load_feature_matrix(data_dir, 'text')
        return np.concatenate([image_features, text_features], axis=1)

    feature_files = {
        'image': 'iid2imagefeature.pkl',
        'text': 'iid2textfeature.pkl',
    }

    feature_file = feature_files.get(feature_mode)
    if feature_file is None:
        raise ValueError(f'Unsupported feature mode: {feature_mode}')

    with open(os.path.join(data_dir, feature_file), 'rb') as fp:
        iid2feature = pickle.load(fp)

    feature_rows = []
    for iindex in range(1, len(iid2iindex) + 1):
        iid = iindex2iid[iindex]
        feature = iid2feature.get(iid, iid2feature.get(str(iid)))
        if feature is None:
            raise ValueError(f'Missing feature for iid={iid} in mode={feature_mode}')
        arr = np.asarray(feature, dtype=np.float32).reshape(-1)
        feature_rows.append(arr)

    feature_dim = feature_rows[0].shape[0]
    if any(arr.shape[0] != feature_dim for arr in feature_rows):
        raise ValueError(f'Feature dimension mismatch in mode={feature_mode}')

    zeros = np.zeros(feature_dim, dtype=np.float32)
    feature_rows = [zeros] + feature_rows
    return np.stack(feature_rows, axis=0)


class CarcaSolver(BaseLWPContrastiveSolver):

    def init_model(self) -> None:
        C = self.config
        CM = C['model']

        with open(os.path.join(C['envs']['DATA_ROOT'], C['dataset'], 'iid2iindex.pkl'), 'rb') as fp:
            self.iid2iindex = pickle.load(fp)
            self.num_items = len(self.iid2iindex)

        data_dir = os.path.join(C['envs']['DATA_ROOT'], C['dataset'])
        feature_mode = CM.get('feature_mode', 'fusion')

        if feature_mode == 'id':
            ifeatures = None
            ifeature_dim = 0
        else:
            ifeatures = _load_feature_matrix(data_dir, feature_mode)
            ifeature_dim = ifeatures.shape[1]

        if type(CM['num_known_item']) is float:
            num_known_item = int(self.num_items * CM['num_known_item'])
        else:
            num_known_item = CM['num_known_item']

        self.model = Carca(
            num_items=self.num_items,
            ifeatures=ifeatures,
            ifeature_dim=ifeature_dim,
            icontext_dim=self.train_dataset.icontext_dim,  # type: ignore
            hidden_dim=CM['hidden_dim'],
            num_known_item=num_known_item,
            num_layers=CM['num_layers'],
            num_heads=CM['num_heads'],
            dropout_prob=CM['dropout_prob'],
            random_seed=CM['random_seed'],
            feature_mode=feature_mode,
            id_emb_dim=CM.get('id_emb_dim', 64),
        ).to(self.device)

    def calculate_forward(self, batch):
        profile_tokens = batch['profile_tokens'].to(self.device)
        profile_icontexts = batch['profile_icontexts'].to(self.device)
        extract_tokens = batch['extract_tokens'].to(self.device)
        extract_icontexts = batch['extract_icontexts'].to(self.device)

        logits = self.model(
            profile_tokens,
            profile_icontexts,
            extract_tokens,
            extract_icontexts,
        )
        return logits

    def calculate_loss(self, batch):
        label = batch['label'].to(self.device)
        logits = self.calculate_forward(batch)
        loss = self.ce_losser(logits, label)
        return loss

    def calculate_rankers(self, batch):
        logits = self.calculate_forward(batch)
        _, rankers = torch_topk(logits, self.max_top_k, dim=1)
        return rankers

    def backup(self):
        copyfile('models/encoders/advanced.py', f'{self.data_dir}/encoder.py')
        copyfile('models/carca.py', f'{self.data_dir}/model.py')
        copyfile('solvers/carca.py', f'{self.data_dir}/solver.py')