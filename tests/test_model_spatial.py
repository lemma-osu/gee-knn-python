import pytest
import ee

from geeknn.test_data import setup as sample_data
from geeknn.ordination import Raw, Euclidean, Mahalanobis, MSN, GNN


@pytest.fixture
def k():
    return 5


@pytest.fixture
def training_data():
    return {
        "fc": sample_data.fc,
        "id_field": sample_data.id_field,
        "spp_columns": sample_data.spp_columns,
        "env_columns": sample_data.env_columns,
    }


@pytest.fixture
def raw_check_img():
    return ee.Image("users/gregorma/gee-knn/test-check/raw_neighbors_600")


@pytest.fixture
def euc_check_img():
    return ee.Image("users/gregorma/gee-knn/test-check/euc_neighbors_600")


@pytest.fixture
def mah_check_img():
    return ee.Image("users/gregorma/gee-knn/test-check/mah_neighbors_600")


@pytest.fixture
def msn_check_img():
    return ee.Image("users/gregorma/gee-knn/test-check/msn_neighbors_600")


@pytest.fixture
def gnn_check_img():
    return ee.Image("users/gregorma/gee-knn/test-check/gnn_neighbors_600")


def run_method(kls, options, training_data, check_img):
    # Create the model
    model = kls(**options)

    # Train the model
    model = model.train(**training_data)

    # Spatially predict the model
    nn = model.predict(
        env_image=sample_data.env_img, mode="CLASSIFICATION"
    ).retile(32)

    # Get the per-pixel difference
    diff_nn = nn.subtract(check_img).abs()

    # Get a frequency histogram of difference values
    fh = diff_nn.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        scale=30,
        maxPixels=1e9,
        tileScale=8,
    )
    return (
        ee.List(diff_nn.bandNames())
        .map(lambda b: (ee.Number(ee.Dictionary(fh.get(b)).get("0.0"))))
        .getInfo()
    )


def test_raw(k, training_data, raw_check_img):
    # RAW - exactly matching for k=5
    matches = run_method(Raw, {"k": k}, training_data, raw_check_img)
    assert all(match >= 360000 for match in matches)


def test_euc(k, training_data, euc_check_img):
    # EUC - exactly matching for k=2, NN3, NN4 and NN5 don't match
    # Bands 3, 4, 5 have at most two pixels (of 360,000) different
    matches = run_method(Euclidean, {"k": k}, training_data, euc_check_img)
    assert all(match >= 359998 for match in matches)


def test_mah(k, training_data, mah_check_img):
    # MAH - not currently matching for any band
    # All bands have at least 359710 / 360000 pixels (99.92%) with no difference
    matches = run_method(Mahalanobis, {"k": k}, training_data, mah_check_img)
    assert all(match >= 359710 for match in matches)


def test_msn(k, training_data, msn_check_img):
    # MSN - not currently matching for any band
    # All bands have at least 359828 / 360000 pixels (99.95%) with no difference
    matches = run_method(MSN, {"k": k}, training_data, msn_check_img)
    assert all(match >= 359828 for match in matches)


def test_gnn(k, training_data, gnn_check_img):
    # GNN - not currently matching for any band
    # All bands have at least 352272 / 360000 pixels (97.85%) with no difference
    matches = run_method(
        GNN,
        {"k": k, "spp_transform": "SQRT", "num_cca_axes": 16},
        training_data,
        gnn_check_img,
    )
    assert all(match >= 352272 for match in matches)
