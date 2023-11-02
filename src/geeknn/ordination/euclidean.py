from __future__ import annotations

import numpy as np
from sknnr.transformers import StandardScalerWithDOF

from ._base import Transformed


class Euclidean(Transformed):
    def _get_transformer(self):
        return StandardScalerWithDOF(ddof=1)

    def _get_X_means(self) -> list[float]:
        return self.transformer_.mean_.tolist()

    def _get_X_stds(self) -> list[float]:
        return self.transformer_.scale_.tolist()

    def _get_projector(self) -> list[list[float]]:
        return np.diag(np.ones(len(self._get_X_means()))).tolist()
