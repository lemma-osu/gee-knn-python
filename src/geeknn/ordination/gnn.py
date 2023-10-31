from __future__ import annotations

import ee
import numpy as np
from sknnr.transformers import CCATransformer

from . import utils
from ._base import FeatureCollection, GeeKnnClassifier
from .cca import cca


class GNN(GeeKnnClassifier):
    SPP_TRANSFORM_FUNC = {
        "SQRT": lambda x: np.sqrt(x),
        "LOG": lambda x: np.log(x),
        "NONE": lambda x: x,
    }

    def __init__(self, k=1, spp_transform="SQRT", num_cca_axes=8, max_duplicates=None):
        self.spp_transform = spp_transform
        self.num_cca_axes = num_cca_axes
        super().__init__(k=k, max_duplicates=max_duplicates)

    def train(
        self,
        *,
        fc: ee.FeatureCollection,
        id_field: str,
        spp_columns: list[str],
        env_columns: list[str],
        **kwargs,
    ):
        fc = ee.FeatureCollection(fc)
        self.id_field = ee.String(id_field)
        self.env_columns = ee.List(env_columns)

        # Create the initial arrays from the feature collection
        ids = utils.fc_to_array(fc, [id_field]).project([0]).toList()
        spp_arr = utils.fc_to_array(fc, spp_columns)
        env_arr = utils.fc_to_array(fc, env_columns)

        # Remove spp columns that are zero
        col_sums = spp_arr.reduce(ee.Reducer.sum(), [0])
        mask = col_sums.neq(0)
        spp_arr = spp_arr.mask(mask)

        # Get means and SDs for each environmental variable
        self.env_means = utils.column_means(env_arr)
        self.env_sds = utils.column_sds(env_arr)

        # Create a normalized environmental matrix based on column statistics
        env_normalized = utils.normalize_arr(env_arr)

        # Remove env columns that are zero - no variation in environment
        col_sums = env_normalized.reduce(ee.Reducer.sum(), [0])
        mask = col_sums.neq(0)
        env_normalized = env_normalized.mask(mask)
        self.env_means = self.env_means.mask(mask)
        self.env_sds = self.env_sds.mask(mask)

        # Remove names from self.env_columns if removed from
        # env_normalized
        def remove_name(item):
            item = ee.List(item)
            in_matrix = ee.Number(item.get(0))
            name = ee.String(item.get(1))
            return ee.Algorithms.If(in_matrix.eq(1), name, "remove")

        self.env_columns = (
            mask.neq(0)
            .project([1])
            .toList()
            .zip(self.env_columns)
            .map(remove_name)
            .filter(ee.Filter.neq("item", "remove"))
        )

        # Optionally, apply a transform to the species matrix
        spp_transformed = None
        if self.spp_transform == "SQRT":
            spp_transformed = spp_arr.pow(0.5)
        elif self.spp_transform == "LOG":
            spp_transformed = spp_arr.log()
        else:
            spp_transformed = spp_arr

        # Run CCA
        cca_obj = cca(spp_transformed, env_normalized)

        # Adjust CCA data to account for number of axes
        num_axes = cca_obj.get("num_cca_axes")
        self.num_cca_axes = ee.Number(
            ee.Algorithms.If(
                ee.Number(self.num_cca_axes).lte(num_axes),
                self.num_cca_axes,
                num_axes,
            )
        )
        self.cca_obj = ee.Dictionary(
            {
                "coeff": ee.Array(cca_obj.get("coeff")),
                "centers": ee.Array(cca_obj.get("centers")),
                "eig_matrix": (
                    ee.Array(cca_obj.get("eig_matrix"))
                    .slice(0, 0, self.num_cca_axes)
                    .slice(1, 0, self.num_cca_axes)
                ),
                "plot_scores": (
                    ee.Array(cca_obj.get("plot_scores"))
                    .slice(1, 0, self.num_cca_axes)
                    .toList()
                ),
            }
        )

        # Convert CCA plot scores to a feature collection for classification
        self.fc = utils.scores_to_fc(
            ids, self.cca_obj.get("plot_scores"), self.id_field
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

    def train_client(
        self,
        *,
        fc: ee.FeatureCollection,
        id_field: str,
        spp_columns: list[str],
        env_columns: list[str],
        **kwargs,
    ):
        client_fc = FeatureCollection.from_ee_feature_collection(fc)
        X = client_fc.properties_to_array(env_columns)
        y = client_fc.properties_to_array(spp_columns)
        y = self.SPP_TRANSFORM_FUNC[self.spp_transform](y)
        transformer = CCATransformer(n_components=self.num_cca_axes).fit(X, y=y)

        self.id_field = ee.String(id_field)
        self.env_columns = ee.List(env_columns)
        self.env_centers = ee.Array([transformer.env_center_.tolist()])
        self.projector = ee.Array(transformer.projector_.tolist())
        self.fc = utils.scores_to_fc(
            self.get_ids(client_fc, id_field),
            ee.Array(transformer.transform(X).tolist()).toList(),
            id_field,
        )
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
        # normalized = arr_im.subtract(ee.Image(self.env_means)).divide(
        #     ee.Image(self.env_sds)
        # )

        # # Center the variables based on species weights, multiply by
        # # the CCA coefficients and weight based on the eigenvalues matrix
        # transformed_image = (
        #     normalized.subtract(ee.Image(ee.Array(self.cca_obj.get("centers"))))
        #     .matrixMultiply(ee.Array(self.cca_obj.get("coeff")))
        #     .arraySlice(1, 0, self.num_cca_axes)
        #     .matrixMultiply(ee.Array(self.cca_obj.get("eig_matrix")))
        #     .arrayProject([1])
        #     .arrayFlatten([self.fc.get("axis_names")])
        # )

        transformed_image = (
            arr_im.subtract(ee.Image(self.env_centers))
            .matrixMultiply(self.projector)
            .arrayProject([1])
            .arrayFlatten([self.fc.get("axis_names")])
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

        # # Normalize the variables and project the plots
        # transformed_arr = (
        #     env_arr.subtract(self.env_means.repeat(0, fc.size()))
        #     .divide(self.env_sds.repeat(0, fc.size()))
        #     .subtract(ee.Array(self.cca_obj.get("centers")).repeat(0, fc.size()))
        #     .matrixMultiply(ee.Array(self.cca_obj.get("coeff")))
        #     .slice(1, 0, self.num_cca_axes)
        #     .matrixMultiply(ee.Array(self.cca_obj.get("eig_matrix")))
        #     .toList()
        # )

        transformed_arr = (
            env_arr.subtract(self.env_centers.repeat(0, fc.size()))
            .matrixMultiply(self.projector)
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
        id_field = self.id_field

        def zip_with_id(t):
            t = ee.List(t)
            f = ee.Feature(t.get(0))
            id_ = ee.Number(t.get(1))
            return (
                ee.Feature(None)
                .set("neighbors", ee.Array(f.get("neighbors")).toList())
                .set(id_field, id_)
            )

        neighbor_fc = ee.FeatureCollection(zipped.map(zip_with_id))

        # Filter plots if colocation_obj is present
        if colocation_obj is not None:
            neighbor_fc = utils.filter_neighbors(
                neighbor_fc, colocation_obj, self.id_field
            )

        # Return the neighbors as a list of lists (plots x k)
        return utils.return_k_neighbors(neighbor_fc, self.k)
