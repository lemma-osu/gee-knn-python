from __future__ import annotations

from abc import ABC, abstractmethod
from functools import singledispatchmethod
from itertools import chain
from typing import Any, Optional

import ee
import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from pydantic import BaseModel
from sklearn.base import TransformerMixin

from .utils import crosswalk_to_ids, filter_neighbors, get_k_neighbors, scores_to_fc


class Geometry(BaseModel):
    """Client-side proxy for ee.Geometry object."""

    type: str
    coordinates: list[Any]


class Feature(BaseModel):
    """Client-side proxy for ee.Feature object."""

    type: str
    geometry: Optional[Geometry]  # noqa: UP007
    id: str
    properties: dict[str, Any]

    def get_property(self, property: str) -> Any:
        return self.properties.get(property)

    def get_properties(self, properties: list[str]) -> list[Any]:
        return [self.get_property(x) for x in properties]


class FeatureCollection(BaseModel):
    """Client-side proxy for ee.FeatureCollection object."""

    type: str
    columns: dict[str, Any]
    version: Optional[int]  # noqa: UP007
    id: Optional[str]  # noqa: UP007
    properties: Optional[dict[str, Any]]  # noqa: UP007
    features: list[Feature]

    def aggregate_array(self, property: str) -> NDArray:
        """Return a single property as a 1-D numpy array."""
        return np.array([f.get_property(property) for f in self.features])

    def properties_to_array(self, properties: list[str]) -> NDArray:
        """Return specified properties as a 2-D numpy array."""
        return np.array([f.get_properties(properties) for f in self.features])

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


class Raw:
    def __init__(self, k: int = 1, max_duplicates: int = 5):
        self.k = k
        self.max_duplicates = max_duplicates
        self.clf = ee.Classifier.minimumDistance(
            metric="euclidean", kNearest=self.k_nearest
        )

    @property
    def k_nearest(self):
        """Return the number of requested nearest neighbors plus max_duplicates
        in order to account for non-independent samples.
        """
        return self.k + self.max_duplicates

    def _train_classifier(
        self,
        *,
        fc: ee.FeatureCollection,
        class_property: ee.String,
        input_properties: ee.List,
    ) -> None:
        """Train the classifier with the given feature collection."""
        self.clf = self.clf.train(
            features=fc,
            classProperty=class_property,
            inputProperties=input_properties,
        )

    def _get_ordered_X_image(self, X_image: ee.Image) -> ee.Image:
        """Return the X_image ensuring that band names match X_columns."""
        return X_image.select(self.X_columns)

    def train(
        self, *, fc: ee.FeatureCollection, id_field: str, X_columns: list[str], **_
    ):
        """Train the estimator, storing the needed server-side objects used
        in prediction.
        """
        self.id_field = ee.String(id_field)
        self.X_columns = ee.List(X_columns)
        self._train_classifier(
            fc=fc, class_property=self.id_field, input_properties=self.X_columns
        )
        return self

    @singledispatchmethod
    def predict(self, arg, **kwargs):
        """
        Predict nearest neighbors based on the type of the first argument.

        Parameters:
        - arg: The first argument to predict, which can be either an ee.Image or
          ee.FeatureCollection
        - **kwargs: Additional keyword arguments to pass to the predict method

        Dispatched Types:
        - ee.Image: Predict the nearest neighbors for the given covariate image.
          - Additional keyword arguments:
            - mode: The mode to use for the classifier.  Either "CLASSIFICATION"
              or "REGRESSION".

        - ee.FeatureCollection: Predict the nearest neighbors for the given covariate
          feature collection.
          - Additional keyword arguments:
            - colocation_obj: A Colocation object used to filter out neighbors that
              are not independent.

        Returns:
        - ee.Image: An image with k bands, where each band is the ID of the kth nearest
          neighbor.

        or

        - ee.FeatureCollection: A feature collection with k properties, where each
          property is the ID of the kth nearest neighbor.
        """
        raise NotImplementedError

    @predict.register
    def _(self, X_image: ee.Image, mode: str = "CLASSIFICATION"):
        X_image = self._get_ordered_X_image(X_image)
        return self._predict_image(X_image, mode=mode)

    @predict.register
    def _(self, fc: ee.FeatureCollection, colocation_obj=None):
        ids = fc.aggregate_array(self.id_field)
        return self._predict_fc(fc, ids, colocation_obj=colocation_obj)

    def _predict_image(self, X_image: ee.Image, mode: str = "CLASSIFICATION"):
        """Predict the nearest neighbors for the given covariate image."""
        clf = self.clf.setOutputMode(ee.String(mode))

        def get_neighbor_band_name(i):
            return ee.String("NN").cat(ee.Number(i).int().format())

        band_names = ee.List.sequence(1, self.k).map(get_neighbor_band_name)

        return (
            X_image.classify(classifier=clf)
            .toArray()
            .arraySlice(0, 0, self.k)
            .arrayFlatten([band_names])
        )

    def _predict_fc(self, fc: ee.FeatureCollection, ids: NDArray, colocation_obj=None):
        """Predict the nearest neighbors for the given covariate feature collection."""
        neighbor_fc = fc.classify(classifier=self.clf, outputName="neighbors")
        neighbor_fc = crosswalk_to_ids(neighbor_fc, ids, self.id_field)
        if colocation_obj is not None:
            neighbor_fc = filter_neighbors(neighbor_fc, colocation_obj, self.id_field)
        return get_k_neighbors(neighbor_fc, self.k)


class Transformed(Raw, ABC):
    """Base class for estimators that require transforming the input data."""

    @abstractmethod
    def _get_transformer(self) -> TransformerMixin:
        """Return the transformer to use for fitting."""
        ...

    @abstractmethod
    def _get_X_means(self) -> list[float]:
        """Return the means of the X columns."""
        ...

    @abstractmethod
    def _get_X_stds(self) -> list[float]:
        """Return the sample standard deviations of the X columns."""
        ...

    @abstractmethod
    def _get_projector(self) -> list[list[float]]:
        """Return the projection matrix."""
        ...

    def _get_modeling_X_y(
        self, *, client_fc: FeatureCollection, X_columns: list[str], **_
    ) -> tuple[NDArray, NDArray]:
        """Return X and y arrays for modeling.  The base class implementation
        only returns the array associated with the X columns.  Other
        subclasses can override to return both X and y arrays.
        """
        X = client_fc.properties_to_array(X_columns)
        return X, None

    def train(
        self, *, fc: ee.FeatureCollection, id_field: str, X_columns: list[str], **kwargs
    ):
        """Train the estimator and store the needed server-side objects used
        in prediction.

        Notes
        -----
        Training the estimator requires the following steps:

        1. Convert the server-side ee.FeatureCollection to a client-side
           FeatureCollection

        2. Get the input arrays for modeling from the feature collection

        3. Get the transformer for this estimator and tranform the X data

        4. Create server-side objects of needed data for prediction, including
           setting the minimumDistance classifier
        """
        client_fc = FeatureCollection.from_ee_feature_collection(fc)

        X, y = self._get_modeling_X_y(
            client_fc=client_fc, X_columns=X_columns, **kwargs
        )

        self.transformer_ = self._get_transformer().fit(X, y)
        X_transformed = self.transformer_.transform(X)

        self.id_field = ee.String(id_field)
        self.X_columns = ee.List(X_columns)
        self.X_means = ee.Array([self._get_X_means()])
        self.X_stds = ee.Array([self._get_X_stds()])
        self.projector = ee.Array(self._get_projector())
        self.fc = scores_to_fc(
            client_fc.aggregate_array(id_field).tolist(),
            X_transformed.tolist(),
            id_field,
        )

        super()._train_classifier(
            fc=self.fc.get("fc"),
            class_property=id_field,
            input_properties=self.fc.get("axis_names"),
        )

        return self

    @singledispatchmethod
    def predict(self, args, **kwargs):
        raise NotImplementedError

    @predict.register
    def _(self, X_image: ee.Image, mode: str = "CLASSIFICATION"):
        X_image = self._get_ordered_X_image(X_image)
        transformed_image = self.transform(X_image)
        return super()._predict_image(transformed_image, mode=mode)

    @predict.register
    def _(self, fc: ee.FeatureCollection, colocation_obj=None):
        ids = fc.aggregate_array(self.id_field)
        transformed_fc = self.transform(fc)
        return self._predict_fc(transformed_fc, ids, colocation_obj=colocation_obj)

    @singledispatchmethod
    def transform(self, arg):
        """
        Transform covariate values into the ordination space based on the type

        Parameters:
        - arg: The argument to transform, which can be either an ee.Image or
          ee.FeatureCollection

        Dispatched Types:
        - ee.Image: Transform the raw covariate image into the ordination space

        - ee.FeatureCollection: Transform each feature in the collection into the
          ordination space.

        Returns:
        - ee.Image: The transformed covariate image.

        or

        - ee.FeatureCollection: A feature collection with transformed covariate
          properties.
        """

        raise NotImplementedError

    @transform.register
    def _(self, X_image: ee.Image) -> ee.Image:
        """Transform a covariate image into the ordination space."""
        X_array_image = X_image.toArray().toArray(1).arrayTranspose(0, 1)
        return (
            X_array_image.subtract(ee.Image(self.X_means))
            .divide(ee.Image(self.X_stds))
            .matrixMultiply(self.projector)
            .arrayProject([1])
            .arrayFlatten([self.fc.get("axis_names")])
        )

    @transform.register
    def _(self, fc: ee.FeatureCollection) -> ee.FeatureCollection:
        """Transform a covariate feature collection into the ordination space."""
        # Select out the X_columns and convert to array
        reducer = ee.Reducer.toList().repeat(self.X_columns.size())
        lookup = fc.reduceColumns(reducer, self.X_columns)
        X_array = ee.Array(lookup.get("list")).transpose()

        # Transform the covariates
        X_transformed_array = (
            X_array.subtract(self.X_means.repeat(0, fc.size()))
            .divide(self.X_stds.repeat(0, fc.size()))
            .matrixMultiply(self.projector)
            .toList()
        )

        # Convert back to feature collection in order to use with
        # FeatureCollection.classify()
        axis_names = self.fc.get("axis_names")

        def row_to_properties(row):
            return ee.Feature(None, ee.Dictionary.fromLists(axis_names, row))

        return ee.FeatureCollection(X_transformed_array.map(row_to_properties))
