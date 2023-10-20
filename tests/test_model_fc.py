import pytest

from geeknn.ordination import GNN, MSN, Euclidean, Mahalanobis, Raw
from geeknn.ordination.utils import Colocation

from .setup import colocation_fc, env_columns, fc, id_field, spp_columns

ESTIMATOR_PARAMETERS = {
    "raw": (Raw, {}),
    "euc": (Euclidean, {}),
    "mah": (Mahalanobis, {}),
    "msn": (MSN, {}),
    "gnn": (GNN, {"spp_transform": "SQRT", "num_cca_axes": 16}),
}


@pytest.fixture()
def training_data():
    return {
        "fc": fc,
        "id_field": id_field,
        "spp_columns": spp_columns,
        "env_columns": env_columns,
    }


@pytest.fixture()
def colocation_obj():
    return Colocation(fc=colocation_fc, location_field="LOC_ID", plot_field="FCID")


@pytest.fixture()
def observed_ids():
    return fc.limit(10).aggregate_array("FCID").getInfo()


def run_method(kls, options, training_data, colocation_obj=None):
    model = kls(**options).train(**training_data)
    return model.predict_fc(
        fc=fc.limit(10),
        colocation_obj=colocation_obj,
    )


@pytest.mark.parametrize(
    "estimator_parameter",
    ESTIMATOR_PARAMETERS.values(),
    ids=ESTIMATOR_PARAMETERS.keys(),
)
@pytest.mark.parametrize("k", [5])
def test_dependent(estimator_parameter, k, training_data, observed_ids):
    """Test that estimators when run without a colocation object return
    itself as the first neighbor"""
    est, options = estimator_parameter
    options["k"] = k
    neighbors = run_method(est, options, training_data)
    prd = [x[0] for x in neighbors.getInfo()]
    assert all(x == y for x, y in zip(observed_ids, prd))


@pytest.mark.parametrize(
    "estimator_parameter",
    ESTIMATOR_PARAMETERS.values(),
    ids=ESTIMATOR_PARAMETERS.keys(),
)
@pytest.mark.parametrize("k", [5])
def test_independent(
    estimator_parameter, k, training_data, colocation_obj, observed_ids
):
    """Test that estimators when run with a colocation object do not
    return itself as the first neighbor"""
    est, options = estimator_parameter
    options["k"] = k
    neighbors = run_method(est, options, training_data, colocation_obj=colocation_obj)
    prd = [x[0] for x in neighbors.getInfo()]
    assert all(x != y for x, y in zip(observed_ids, prd))
