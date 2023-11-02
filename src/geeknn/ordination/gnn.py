from __future__ import annotations

import numpy as np
from sknnr.transformers import CCATransformer

from ._base import Transformed


class GNN(Transformed):
    SPP_TRANSFORM_FUNC = {
        "SQRT": lambda x: np.sqrt(x),
        "LOG": lambda x: np.log(x),
        "NONE": lambda x: x,
    }

    def __init__(self, k=1, spp_transform="SQRT", num_cca_axes=8, max_duplicates=None):
        self.spp_transform = spp_transform
        self.num_cca_axes = num_cca_axes
        super().__init__(k=k, max_duplicates=max_duplicates)

    def _get_transformer(self):
        # return CCATransformer(self.n_components)
        return CCATransformer(n_components=self.num_cca_axes)

    def _get_modeling_X_y(self, *, client_fc, X_columns, y_columns):
        X = client_fc.properties_to_array(X_columns)
        y = client_fc.properties_to_array(y_columns)
        y = self.SPP_TRANSFORM_FUNC[self.spp_transform](y)
        return X, y

    def _get_X_means(self) -> list[float]:
        return self.transformer_.env_center_.tolist()

    def _get_X_stds(self) -> list[float]:
        return np.ones(len(self._get_X_means())).tolist()

    def _get_projector(self) -> list[list[float]]:
        return self.transformer_.projector_.tolist()
