from __future__ import annotations

from sknnr.transformers import MahalanobisTransformer

from ._base import TransformedKNNClassifier


class MahalanobisKNNClassifier(TransformedKNNClassifier):
    def _get_transformer(self):
        return MahalanobisTransformer()

    def _get_X_means(self) -> list[float]:
        return self.transformer_.scaler_.mean_.tolist()

    def _get_X_stds(self) -> list[float]:
        return self.transformer_.scaler_.scale_.tolist()

    def _get_projector(self) -> list[list[float]]:
        return self.transformer_.transform_.tolist()
