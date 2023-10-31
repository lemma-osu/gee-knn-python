from __future__ import annotations

from itertools import chain
from typing import Any

import ee
from joblib import Parallel, delayed
from pydantic import BaseModel


class Geometry(BaseModel):
    """Client-side proxy for ee.Geometry object"""

    type: str
    coordinates: list[Any]


class Feature(BaseModel):
    """Client-side proxy for ee.Feature object"""

    type: str
    geometry: Geometry
    id: str
    properties: dict[str, Any]

    def get_property(self, property: str) -> Any:
        return self.properties.get(property)

    def get_properties(self, properties: list[str]) -> list[Any]:
        return [self.get_property(x) for x in properties]


class FeatureCollection(BaseModel):
    """Client-side proxy for ee.FeatureCollection object"""

    type: str
    columns: dict[str, Any]
    version: int
    id: str
    properties: dict[str, Any]
    features: list[Feature]

    def aggregate_array(self, property: str) -> list[Any]:
        return [f.get_property(property) for f in self.features]

    def properties_to_array(self, properties: list[str]) -> list[list[Any]]:
        return [f.get_properties(properties) for f in self.features]

    @classmethod
    def from_ee_feature_collection(
        cls, fc: ee.FeatureCollection, num_threads: int = -1, chunk_size: int = 5000
    ):
        """Create a client-side FeatureCollection from an ee.FeatureCollection
        using multiple threads and chunking to avoid memory issues.
        """
        size = fc.size().getInfo()
        info = fc.limit(0).getInfo()
        chunks = [fc.toList(chunk_size, i) for i in range(0, size, chunk_size)]
        with Parallel(n_jobs=num_threads, backend="threading") as p:
            chunk_data = p(delayed(chunk.getInfo)() for chunk in chunks)
        info.update({"features": list(chain.from_iterable(chunk_data))})
        return cls(**info)


class GeeKnnClassifier:
    def __init__(self, k=1, max_duplicates=None):
        self.k = k
        self.max_duplicates = max_duplicates if max_duplicates is not None else 5

    def get_ids(self, fc: FeatureCollection, id_field: str) -> list[Any]:
        return list(map(int, fc.aggregate_array(id_field)))

    @property
    def k_nearest(self):
        return self.k + self.max_duplicates
