from __future__ import annotations

from sknnr.transformers import CCorATransformer

from ._base import Transformed


class MSN(Transformed):
    def _get_transformer(self):
        # return CCorATransformer(self.n_components)
        return CCorATransformer()

    def _get_modeling_X_y(self, *, client_fc, X_columns, y_columns):
        X = client_fc.properties_to_array(X_columns)
        y = client_fc.properties_to_array(y_columns)
        return X, y

    def _get_X_means(self) -> list[float]:
        return self.transformer_.scaler_.mean_.tolist()

    def _get_X_stds(self) -> list[float]:
        return self.transformer_.scaler_.scale_.tolist()

    def _get_projector(self) -> list[list[float]]:
        return self.transformer_.projector_.tolist()
