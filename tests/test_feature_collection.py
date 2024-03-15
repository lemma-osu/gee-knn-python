import ee

from geeknn._base import FeatureCollection

ee.Initialize()


def test_feature_collection_missing_attributes():
    fc_server = ee.FeatureCollection(
        [
            ee.Feature(ee.Geometry.Point([-123, 44]), {"foo": 1}),
            ee.Feature(ee.Geometry.Point([-122, 44]), {"foo": 2}),
            ee.Feature(ee.Geometry.Point([-121, 44]), {"foo": 3}),
        ]
    )
    fc_client = FeatureCollection.from_ee_feature_collection(fc_server)
    assert len(fc_client.features) == 3
    assert fc_client.version is None
    assert fc_client.id is None
    assert fc_client.properties is None
