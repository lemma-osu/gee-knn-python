import pytest

from geeknn.ordination import GNN, MSN, Euclidean, Mahalanobis, Raw
from geeknn.ordination.utils import Colocation
from geeknn.test_data import setup as sample_data


@pytest.fixture()
def k():
    return 5


@pytest.fixture()
def training_data():
    return {
        "fc": sample_data.fc,
        "id_field": sample_data.id_field,
        "spp_columns": sample_data.spp_columns,
        "env_columns": sample_data.env_columns,
    }


@pytest.fixture()
def colocation_obj():
    return Colocation(
        fc=sample_data.colocation_fc, location_field="LOC_ID", plot_field="FCID"
    )


def run_method(kls, options, training_data, colocation_obj=None):
    # Create the model
    model = kls(**options)

    # Train the model
    model = model.train(**training_data)

    # Classify the first 10 records of the test feature collection
    return model.predict_fc(
        fc=sample_data.fc.limit(10),
        colocation_obj=colocation_obj,
    )


def get_obs_ids(fc, id_field="FCID", limit=10):
    return fc.limit(limit).aggregate_array(id_field).getInfo()


def test_raw_dependent(k, training_data):
    neighbors = run_method(Raw, {"k": k}, training_data)
    obs = get_obs_ids(training_data["fc"])
    prd = [x[0] for x in neighbors.getInfo()]
    assert all(x == y for x, y in zip(obs, prd))


def test_euc_dependent(k, training_data):
    neighbors = run_method(Euclidean, {"k": k}, training_data)
    obs = get_obs_ids(training_data["fc"])
    prd = [x[0] for x in neighbors.getInfo()]
    assert all(x == y for x, y in zip(obs, prd))


def test_mah_dependent(k, training_data):
    neighbors = run_method(Mahalanobis, {"k": k}, training_data)
    obs = get_obs_ids(training_data["fc"])
    prd = [x[0] for x in neighbors.getInfo()]
    assert all(x == y for x, y in zip(obs, prd))


def test_msn_dependent(k, training_data):
    neighbors = run_method(MSN, {"k": k}, training_data)
    obs = get_obs_ids(training_data["fc"])
    prd = [x[0] for x in neighbors.getInfo()]
    assert all(x == y for x, y in zip(obs, prd))


def test_gnn_dependent(k, training_data):
    neighbors = run_method(
        GNN,
        {"k": k, "spp_transform": "SQRT", "num_cca_axes": 16},
        training_data,
    )
    obs = get_obs_ids(training_data["fc"])
    prd = [x[0] for x in neighbors.getInfo()]
    assert all(x == y for x, y in zip(obs, prd))


def test_raw_independent(k, training_data, colocation_obj):
    neighbors = run_method(Raw, {"k": k}, training_data, colocation_obj=colocation_obj)
    obs = get_obs_ids(training_data["fc"])
    prd = [x[0] for x in neighbors.getInfo()]
    assert all(x != y for x, y in zip(obs, prd))


def test_euc_independent(k, training_data, colocation_obj):
    neighbors = run_method(
        Euclidean, {"k": k}, training_data, colocation_obj=colocation_obj
    )
    obs = get_obs_ids(training_data["fc"])
    prd = [x[0] for x in neighbors.getInfo()]
    assert all(x != y for x, y in zip(obs, prd))


def test_mah_independent(k, training_data, colocation_obj):
    neighbors = run_method(
        Mahalanobis, {"k": k}, training_data, colocation_obj=colocation_obj
    )
    obs = get_obs_ids(training_data["fc"])
    prd = [x[0] for x in neighbors.getInfo()]
    assert all(x != y for x, y in zip(obs, prd))


def test_msn_independent(k, training_data, colocation_obj):
    neighbors = run_method(MSN, {"k": k}, training_data, colocation_obj=colocation_obj)
    obs = get_obs_ids(training_data["fc"])
    prd = [x[0] for x in neighbors.getInfo()]
    assert all(x != y for x, y in zip(obs, prd))


def test_gnn_independent(k, training_data, colocation_obj):
    neighbors = run_method(
        GNN,
        {"k": k, "spp_transform": "SQRT", "num_cca_axes": 16},
        training_data,
        colocation_obj=colocation_obj,
    )
    obs = get_obs_ids(training_data["fc"])
    prd = [x[0] for x in neighbors.getInfo()]
    assert all(x != y for x, y in zip(obs, prd))
