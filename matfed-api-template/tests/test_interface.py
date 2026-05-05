from pathlib import Path

import pytest
from pymatgen.core import Structure

from matfed_api import MatFedPredictor, validate_predictions


def load_sample_structures():
    sample_dir = Path(__file__).parent / "sample_structures"
    return [
        Structure.from_file(path)
        for path in sorted(sample_dir.glob("*.cif"))
    ]


def test_predictor_is_abstract():
    with pytest.raises(TypeError):
        MatFedPredictor()


def test_predict_returns_list(test_predictor):
    structures = load_sample_structures()
    outputs = test_predictor.predict(structures)
    assert isinstance(outputs, list)
    assert len(outputs) == len(structures)


def test_predict_required_keys(test_predictor):
    structures = load_sample_structures()
    outputs = test_predictor.predict(structures)
    required_keys = {
        "formation_energy_per_atom",
        "band_gap",
        "model_id",
        "data_sources_used",
    }
    for item in outputs:
        assert required_keys.issubset(item.keys())


def test_describe_required_fields(test_predictor):
    description = test_predictor.describe()
    for key in ["team_name", "model_type", "api_version", "data_sources"]:
        assert key in description


def test_json_schema_valid(test_predictor):
    structures = load_sample_structures()
    outputs = test_predictor.predict(structures)
    validate_predictions(outputs)
