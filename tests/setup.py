import ee

ee.Initialize()

DATA_DIR = "users/gregorma/gee-knn/attribute_tables"

ENV_COLUMNS = ee.List(
    [
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
)


def get_training_data():
    """Return the training data for the test model that consists of a feature
    collection with the ID field, species columns, and environmental columns
    identified."""
    id_field = "FCID"
    spp = ee.FeatureCollection(f"{DATA_DIR}/test_species").sort(id_field)
    env = ee.FeatureCollection(f"{DATA_DIR}/test_environment").sort(id_field)

    # Join the properties of the two feature collections
    # See https://code.earthengine.google.com/87211cf5cc335585a992acefcd7ec9a4
    fltr = ee.Filter.equals(leftField=id_field, rightField=id_field)
    join = ee.Join.saveFirst(matchKey=id_field).apply(spp, env, fltr)

    def join_feature(f):
        f1 = ee.Feature(f)
        f2 = ee.Feature(f.get(id_field)).toDictionary()
        return f1.set(f2)

    fc = join.map(join_feature)

    # Set the columns to extract.  For env_columns, they need to match the
    # ordering of the bands in the stacked environmental image
    unwanted = ee.List([id_field, "system:index"])
    spp_columns = spp.first().propertyNames().removeAll(unwanted).sort()
    return {
        "fc": fc,
        "id_field": id_field,
        "spp_columns": spp_columns,
        "env_columns": ENV_COLUMNS,
    }


def get_colocation_fc():
    """Return the crosswalk between FCID and LOC_ID for the test data"""
    return ee.FeatureCollection(f"{DATA_DIR}/fcid_x_locid_eco")


def get_covariate_image():
    """Return the stacked environmental image for the test data"""
    return ee.Image("users/gregorma/gee-knn/test-input/all_600").rename(ENV_COLUMNS)
