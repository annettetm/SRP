import os
import pickle

from models.carca_gated import CarcaGated
from .carca import CarcaSolver, _load_feature_matrix


class CarcaGatedSolver(CarcaSolver):
    """Separate solver using gated fusion without changing the original implementation."""

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
            image_features = None
            text_features = None

        elif feature_mode == 'three_way':
            ifeatures = None
            ifeature_dim = 0
            image_features = _load_feature_matrix(data_dir, 'image')
            text_features = _load_feature_matrix(data_dir, 'text')

        else:
            ifeatures = _load_feature_matrix(data_dir, feature_mode)
            ifeature_dim = ifeatures.shape[1]
            image_features = None
            text_features = None

        if type(CM['num_known_item']) is float:
            num_known_item = int(self.num_items * CM['num_known_item'])
        else:
            num_known_item = CM['num_known_item']

        self.model = CarcaGated(
            num_items=self.num_items,
            ifeatures=ifeatures,
            ifeature_dim=ifeature_dim,
            image_features=image_features,
            text_features=text_features,
            icontext_dim=self.train_dataset.icontext_dim,
            hidden_dim=CM['hidden_dim'],
            num_known_item=num_known_item,
            num_layers=CM['num_layers'],
            num_heads=CM['num_heads'],
            dropout_prob=CM['dropout_prob'],
            random_seed=CM['random_seed'],
            feature_mode=feature_mode,
            id_emb_dim=CM.get('id_emb_dim', 64),
        ).to(self.device)
