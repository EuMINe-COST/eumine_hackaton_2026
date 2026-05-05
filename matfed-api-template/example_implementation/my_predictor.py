from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Structure

from matfed_api import MatFedPredictor


class RandomForestPredictor(MatFedPredictor):
    def __init__(self) -> None:
        self.featurizer = ElementProperty.from_preset("magpie")
        self.model_ef = None
        self.model_bg = None

    def load_model(self, model_path: str) -> None:
        model_dir = Path(model_path)
        ef_path = model_dir / "baseline_model_ef.joblib"
        bg_path = model_dir / "baseline_model_bg.joblib"

        if ef_path.exists() and bg_path.exists():
            self.model_ef = joblib.load(ef_path)
            self.model_bg = joblib.load(bg_path)

    def _featurize(self, structures: List[Structure]) -> np.ndarray:
        features = []
        for structure in structures:
            features.append(self.featurizer.featurize(structure.composition))
        return np.asarray(features, dtype=float)

    def predict(self, structures: List[Structure]) -> List[Dict]:
        if not structures:
            return []

        x_values = self._featurize(structures)

        if self.model_ef is None or self.model_bg is None:
            ef_preds = np.full(len(structures), -1.0, dtype=float)
            bg_preds = np.full(len(structures), 1.0, dtype=float)
        else:
            ef_preds = self.model_ef.predict(x_values)
            bg_preds = self.model_bg.predict(x_values)

        predictions = []
        for ef_value, bg_value in zip(ef_preds, bg_preds):
            predictions.append(
                {
                    "formation_energy_per_atom": float(ef_value),
                    "band_gap": float(bg_value),
                    "model_id": "rf_magpie_baseline",
                    "data_sources_used": ["Materials Project"],
                }
            )
        return predictions

    def describe(self) -> Dict:
        return {
            "team_name": "EuMINe Organizers",
            "model_type": "RandomForestRegressor + MAGPIE",
            "api_version": "MatFed API v1",
            "data_sources": ["Materials Project"],
            "requires_pretrained_weights": False,
        }
