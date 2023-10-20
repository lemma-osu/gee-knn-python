import ee

ee.Initialize()

# Bring in sample species (spp) and environmental (env) feature classes
# and set the field that holds the plot IDs.  For the environmental matrix,
# this could be done at runtime.
spp = ee.FeatureCollection("users/gregorma/gee-knn/attribute_tables/test_species").sort(
    "FCID"
)
env = ee.FeatureCollection(
    "users/gregorma/gee-knn/attribute_tables/test_environment"
).sort("FCID")
id_field = "FCID"

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
env_columns = ee.List(
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

# Feature collection which stores colocation information
colocation_fc = ee.FeatureCollection(
    "users/gregorma/gee-knn/attribute_tables/fcid_x_locid_eco"
)

env_img = ee.Image("users/gregorma/gee-knn/test-input/all_600").rename(env_columns)
