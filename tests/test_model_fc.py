import numpy as np
import pytest

from geeknn.ordination import (
    EuclideanKNNClassifier,
    GNNClassifier,
    MahalanobisKNNClassifier,
    MSNClassifier,
    RawKNNClassifier,
)

ESTIMATOR_PARAMETERS = {
    "raw": (RawKNNClassifier, {}),
    "euc": (EuclideanKNNClassifier, {}),
    "mah": (MahalanobisKNNClassifier, {}),
    "msn": (MSNClassifier, {}),
    "gnn": (GNNClassifier, {"y_transform": np.sqrt, "n_components": 16}),
}


def run_method(kls, options, training_data, colocation_obj=None):
    """Run prediction in feature collection mode for the first 10 features"""
    model = kls(**options).train(**training_data)
    return model.predict(training_data["fc"].limit(10), colocation_obj=colocation_obj)


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
