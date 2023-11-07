from __future__ import annotations

import numpy as np
from sknnr.transformers import CCATransformer

from ._base import Transformed


class GNN(Transformed):
    Y_TRANSFORM_FUNC = {
        "SQRT": lambda x: np.sqrt(x),
        "LOG": lambda x: np.log(x),
        "NONE": lambda x: x,
    }

    def __init__(self, *, y_transform: str = "SQRT", n_components: int = 8, **kwargs):
        self.y_transform = y_transform
        self.n_components = n_components
        super().__init__(**kwargs)

    def _get_transformer(self):
        # return CCATransformer(self.n_components)
        return CCATransformer(n_components=self.n_components)

    def _get_modeling_X_y(self, *, client_fc, X_columns, y_columns):
        X = client_fc.properties_to_array(X_columns)
        y = client_fc.properties_to_array(y_columns)
        y = self.Y_TRANSFORM_FUNC[self.y_transform](y)
        return X, y

    def _get_X_means(self) -> list[float]:
        return self.transformer_.env_center_.tolist()

    def _get_X_stds(self) -> list[float]:
        return np.ones(len(self._get_X_means())).tolist()

    def _get_projector(self) -> list[list[float]]:
        return self.transformer_.projector_.tolist()
