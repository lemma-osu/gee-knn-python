# geeknn

## What is geeknn?

`geeknn` is a package for running k-nearest neighbor (kNN) imputation using Google Earth Engine as a backend.

## Installation

`geeknn` will be available through PyPI and conda-forge once it is ready for release. Until then, you can install it from source.

```bash
pip install git+https://github.com/lemma-osu/gee-knn-python@main
```

## Dependencies

- Python >= 3.9
- earthengine-api
- joblib
- numpy
- pydantic
- scikit-learn
- sknnr @ git+https://github.com/lemma-osu/scikit-learn-knn-regression@main
