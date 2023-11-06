import ee
import pytest

from geeknn.ordination import GNN, MSN, Euclidean, Mahalanobis, Raw

ESTIMATOR_PARAMETERS = {
    "raw": (Raw, {}, 360000),
    "euc": (Euclidean, {}, 359998),
    "mah": (Mahalanobis, {}, 359710),
    "msn": (MSN, {}, 359828),
    "gnn": (GNN, {"spp_transform": "SQRT", "num_cca_axes": 16}, 352272),
}


def get_check_image(prefix: str) -> ee.Image:
    return ee.Image(f"users/gregorma/gee-knn/test-check/{prefix}_neighbors_600")


def run_method(kls, options, training_data, X_image, check_image):
    """Run predict on the given estimator, difference it against a
    reference images, and return the frequency of zero differences"""
    model = kls(**options).train(**training_data)
    nn_image = model.predict(X_image=X_image, mode="CLASSIFICATION").retile(32)
    diff_nn_image = nn_image.subtract(check_image).abs()
    frequency = diff_nn_image.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        scale=30,
        maxPixels=1e9,
        tileScale=8,
    )
    return (
        ee.List(diff_nn_image.bandNames())
        .map(lambda b: (ee.Number(ee.Dictionary(frequency.get(b)).get("0.0"))))
        .getInfo()
    )


@pytest.mark.parametrize(
    "estimator_parameter",
    ESTIMATOR_PARAMETERS.items(),
    ids=ESTIMATOR_PARAMETERS.keys(),
)
@pytest.mark.parametrize("k", [5])
def test_image_match(estimator_parameter, k, training_data, X_image):
    """Test that predicted kNN images match expected images for the given
    number of expected_matches"""
    method, (est, options, expected_matches) = estimator_parameter
    options["k"] = k
    check_image = get_check_image(method)
    matches = run_method(est, options, training_data, X_image, check_image)
    assert all(match >= expected_matches for match in matches)
