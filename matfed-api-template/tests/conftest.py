import importlib
import os

import pytest


@pytest.fixture(scope="session")
def test_predictor():
    target = os.environ.get(
        "MY_PREDICTOR",
        "example_implementation.my_predictor.RandomForestPredictor",
    )

    module_name, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    predictor_class = getattr(module, class_name)
    return predictor_class()
