import torch
import torch.nn as nn

from .carca import Carca
from .encoders import AdvancedItemEncoder


class CarcaGated(Carca):

    def __init__(self, *args, image_features=None, text_features=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.image_encoder = None
        self.text_encoder = None
        self.three_way_gate_proj = None

        if self.feature_mode == 'three_way': #id, image and text separate
            self.image_encoder = AdvancedItemEncoder(
                num_items=self.num_items,
                ifeatures=image_features,
                ifeature_dim=image_features.shape[1],
                icontext_dim=self.icontext_dim,
                hidden_dim=self.hidden_dim,
                num_known_item=self.num_known_item,
                random_seed=self.random_seed,
            )

            self.text_encoder = AdvancedItemEncoder(
                num_items=self.num_items,
                ifeatures=text_features,
                ifeature_dim=text_features.shape[1],
                icontext_dim=self.icontext_dim,
                hidden_dim=self.hidden_dim,
                num_known_item=self.num_known_item,
                random_seed=self.random_seed,
            )

            self.three_way_gate_proj = nn.Linear(self.hidden_dim * 3, 3)


        elif self.use_id and self.use_content: #two-way
            self.gate_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)


    def _encode_items(self, tokens, icontexts):
        parts = []

        if self.feature_mode == 'three_way':
            image_vec = self.image_encoder(tokens, icontexts)
            text_vec = self.text_encoder(tokens, icontexts)

            id_tokens = tokens.clamp(min=0, max=self.num_items)
            id_vec = self.id_proj(self.item_id_emb(id_tokens))

            vectors = torch.stack([image_vec, text_vec, id_vec], dim=-2)
            gate_input = torch.cat([image_vec, text_vec, id_vec], dim=-1)

            weights = torch.softmax(
                self.three_way_gate_proj(gate_input),
                dim=-1,
            ).unsqueeze(-1)

            out = (weights * vectors).sum(dim=-2)
            return self.item_layernorm(out)

        if self.use_content:
            content = self.item_encoder(tokens, icontexts)
            parts.append(content)

        if self.use_id:
            id_tokens = tokens.clamp(min=0, max=self.num_items)
            id_vec = self.item_id_emb(id_tokens)
            id_vec = self.id_proj(id_vec)
            parts.append(id_vec)

        

        if len(parts) == 1:
            out = parts[0]
        elif len(parts) == 2:
            content_vec, id_vec = parts
            gate = torch.sigmoid(self.gate_proj(torch.cat([content_vec, id_vec], dim=-1)))
            out = gate * content_vec + (1.0 - gate) * id_vec


        else:
            raise RuntimeError(f'Invalid number of parts for feature_mode={self.feature_mode}')

        return self.item_layernorm(out)
