from __future__ import annotations

import ee
from sknnr.transformers import StandardScalerWithDOF

from . import utils
from ._base import FeatureCollection, GeeKnnClassifier


class Euclidean(GeeKnnClassifier):
    def train(self, *, fc, id_field, env_columns, **kwargs):
        fc = ee.FeatureCollection(fc)
        self.id_field = ee.String(id_field)
        self.env_columns = ee.List(env_columns)

        # Create the initial arrays from the feature collection
        ids = utils.fc_to_array(fc, ee.List([self.id_field])).project([0]).toList()
        env_arr = utils.fc_to_array(fc, self.env_columns)

        # Get means and SDs for each environmental variable
        self.env_means = utils.column_means(env_arr)
        self.env_sds = utils.column_sds(env_arr)

        # Create a normalized environmental matrix based on column statistics
        env_normalized = utils.normalize_arr(env_arr).toList()

        # Convert normalized env scores to a feature collection for classification
        self.fc = utils.scores_to_fc(ids, env_normalized, self.id_field)

        # Train the classifier within the transformed space
        self.clf = ee.Classifier.minimumDistance(
            metric="euclidean",
            kNearest=self.k_nearest,
        ).train(
            features=self.fc.get("fc"),
            classProperty=self.id_field,
            inputProperties=self.fc.get("axis_names"),
        )
        return self

    def train_client(
        self,
        *,
        fc: ee.FeatureCollection,
        id_field: str,
        env_columns: list[str],
        **kwargs,
    ):
        client_fc = FeatureCollection.from_ee_feature_collection(fc)
        X = client_fc.properties_to_array(env_columns)
        transformer = StandardScalerWithDOF(ddof=1).fit(X)

        self.id_field = ee.String(id_field)
        self.env_columns = ee.List(env_columns)
        self.env_means = ee.Array([transformer.mean_.tolist()])
        self.env_sds = ee.Array([transformer.scale_.tolist()])
        self.fc = utils.scores_to_fc(
            self.get_ids(client_fc, id_field),
            ee.Array(transformer.transform(X).tolist()).toList(),
            id_field,
        )

        # Train the classifier within the transformed space
        self.clf = ee.Classifier.minimumDistance(
            metric="euclidean",
            kNearest=self.k_nearest,
        ).train(
            features=self.fc.get("fc"),
            classProperty=self.id_field,
            inputProperties=self.fc.get("axis_names"),
        )
        return self

    def predict(self, env_image, mode="CLASSIFICATION"):
        env_image = ee.Image(env_image)
        self.clf = self.clf.setOutputMode(ee.String(mode))

        # Ensure the env_image band names match the env_columns
        env_image = env_image.select(self.env_columns)

        # Convert this to an array image and normalize the variables
        arr_im = env_image.toArray().toArray(1).arrayTranspose(0, 1)
        normalized = arr_im.subtract(ee.Image(self.env_means)).divide(
            ee.Image(self.env_sds)
        )

        # Flatten the array
        transformed_image = normalized.arrayProject([1]).arrayFlatten(
            [self.fc.get("axis_names")]
        )

        # Return the nearest neighbors in this space
        def get_name(i):
            return ee.String("NN").cat(ee.Number(i).int().format())

        names = ee.List.sequence(1, self.k).map(get_name)

        return (
            transformed_image.classify(classifier=self.clf)
            .toArray()
            .arraySlice(0, 0, self.k)
            .arrayFlatten([names])
        )

    def predict_fc(self, fc, colocation_obj=None):
        fc = ee.FeatureCollection(fc)

        # Get the IDs from the fc to be linked against neighbors below
        ids = fc.aggregate_array(self.id_field)

        # Select out the env_columns and convert to array
        reducer = ee.Reducer.toList().repeat(self.env_columns.size())
        lookup = fc.reduceColumns(reducer, self.env_columns)
        env_arr = ee.Array(lookup.get("list")).transpose()

        # Normalize the variables
        transformed_arr = (
            env_arr.subtract(self.env_means.repeat(0, fc.size()))
            .divide(self.env_sds.repeat(0, fc.size()))
            .toList()
        )

        # Convert back to feature collection in order to use
        # FeatureCollection.classify()
        ax_names = self.fc.get("axis_names")

        def row_to_properties(row):
            return ee.Feature(None, ee.Dictionary.fromLists(ax_names, row))

        transformed_fc = ee.FeatureCollection(transformed_arr.map(row_to_properties))

        # Retrieve the nearest neighbors in this space
        neighbor_fc = transformed_fc.classify(
            classifier=self.clf, outputName="neighbors"
        )

        # Zip with IDs
        zipped = neighbor_fc.toList(neighbor_fc.size()).zip(ids)

        def zip_with_id(t):
            t = ee.List(t)
            f = ee.Feature(t.get(0))
            id_ = ee.Number(t.get(1))
            return (
                ee.Feature(None)
                .set("neighbors", ee.Array(f.get("neighbors")).toList())
                .set(self.id_field, id_)
            )

        neighbor_fc = ee.FeatureCollection(zipped.map(zip_with_id))

        # Filter plots if colocation_obj is present
        if colocation_obj is not None:
            neighbor_fc = utils.filter_neighbors(
                neighbor_fc, colocation_obj, self.id_field
            )

        # Return the neighbors as a list of lists (plots x k)
        return utils.return_k_neighbors(neighbor_fc, self.k)
