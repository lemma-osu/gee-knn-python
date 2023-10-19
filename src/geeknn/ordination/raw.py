import ee

from . import utils


class Raw:
    def __init__(self, k=1, max_duplicates=None):
        self.k = k
        self.max_duplicates = max_duplicates if max_duplicates is not None else 5

    @property
    def k_nearest(self):
        return self.k + self.max_duplicates

    def train(self, fc, id_field, env_columns, **kwargs):
        fc = ee.FeatureCollection(fc)
        self.id_field = ee.String(id_field)
        self.env_columns = ee.List(env_columns)

        self.clf = ee.Classifier.minimumDistance(
            metric="euclidean",
            kNearest=self.k_nearest,
        ).train(
            features=fc,
            classProperty=self.id_field,
            inputProperties=self.env_columns,
        )
        return self

    def predict(self, env_image, mode="CLASSIFICATION"):
        env_image = ee.Image(env_image)
        self.clf = self.clf.setOutputMode(ee.String(mode))

        # Ensure the env_image band names match the env_columns
        env_image = env_image.select(self.env_columns)

        # Return the nearest neighbors in this space\
        def get_name(i):
            return ee.String("NN").cat(ee.Number(i).int().format())

        names = ee.List.sequence(1, self.k).map(get_name)

        return (
            env_image.classify(classifier=self.clf)
            .toArray()
            .arraySlice(0, 0, self.k)
            .arrayFlatten([names])
        )

    def predict_fc(self, fc, colocation_obj=None):
        fc = ee.FeatureCollection(fc)

        # Get the IDs from the fc to be linked against neighbors below
        ids = fc.aggregate_array(self.id_field)

        # Retrieve the nearest neighbors in this space
        neighbor_fc = fc.classify(classifier=self.clf, outputName="neighbors")

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
