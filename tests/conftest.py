import ee
import pytest

from geeknn.utils import Colocation

ee.Initialize()

DATA_DIR = "users/gregorma/gee-knn/attribute_tables"

X_COLUMNS = [
    "ANNPRE",
    "ANNTMP",
    "AUGMAXT",
    "CONTPRE",
    "CVPRE",
    "DECMINT",
    "DIFTMP",
    "SMRTMP",
    "SMRTP",
    "LAT",
    "LON",
    "ASPTR",
    "DEM",
    "PRR",
    "SLPPCT",
    "TPI450",
]


@pytest.fixture()
def training_data():
    """Return the training data for the test model that consists of a feature
    collection with the ID field, species columns, and environmental columns
    identified.
    """
    id_field = "FCID"
    y = ee.FeatureCollection(f"{DATA_DIR}/test_species").sort(id_field)
    X = ee.FeatureCollection(f"{DATA_DIR}/test_environment").sort(id_field)

    # Join the properties of the two feature collections
    # See https://code.earthengine.google.com/87211cf5cc335585a992acefcd7ec9a4
    fltr = ee.Filter.equals(leftField=id_field, rightField=id_field)
    join = ee.Join.saveFirst(matchKey=id_field).apply(y, X, fltr)

    def join_feature(f):
        f1 = ee.Feature(f)
        f2 = ee.Feature(f.get(id_field)).toDictionary()
        return f1.set(f2)

    fc = join.map(join_feature)

    # Set the columns to extract.  For X_columns, they need to match the
    # ordering of the bands in the stacked X image
    unwanted = ee.List([id_field, "system:index"])
    y_columns = y.first().propertyNames().removeAll(unwanted).sort().getInfo()
    return {
        "fc": fc,
        "id_field": id_field,
        "y_columns": y_columns,
        "X_columns": X_COLUMNS,
    }


@pytest.fixture()
def observed_ids(training_data):
    return training_data["fc"].limit(10).aggregate_array("FCID").getInfo()


@pytest.fixture()
def colocation_obj():
    """Return the crosswalk between FCID and LOC_ID for the test data"""
    colocation_fc = ee.FeatureCollection(f"{DATA_DIR}/fcid_x_locid_eco")
    return Colocation(fc=colocation_fc, location_field="LOC_ID", plot_field="FCID")


@pytest.fixture()
def X_image():
    """Return the stacked environmental image for the test data"""
    return ee.Image("users/gregorma/gee-knn/test-input/all_600").rename(X_COLUMNS)
