"""Run with python -m unittest discover -s tests."""

from pathlib import Path
import pickle
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from uab_dl import MARKERS, predict_uab_dl


ROOT = Path(__file__).resolve().parents[1]


class UABDLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "uab_dl_model.pkl").open("rb") as handle:
            cls.artifact = pickle.load(handle)

    def test_full_panel_matches_saved_model(self):
        values = dict(zip(MARKERS, [0.08, 0.01, 0.31, 2100.0, 98.5, 1.45, 14.6]))
        panel = pd.DataFrame([values], columns=MARKERS)
        features = self.artifact["feature_engineer"].transform(panel)
        self.assertEqual(features.shape, (1, 41))
        expected = self.artifact["model"].predict_proba(features)[0]
        result = predict_uab_dl(values, self.artifact)
        by_class = {item["Class"]: item["Confidence"] / 100 for item in result["classifications"]}
        for label, probability in zip(self.artifact["model"].classes_, expected):
            self.assertAlmostEqual(by_class[label], probability)
        self.assertEqual(result["predicted_infection_type"], self.artifact["model"].predict(features)[0])
        self.assertEqual(result["missing_biomarkers"], [])

    def test_partial_panel_preserves_missingness_and_zero_readings(self):
        values = {"PCT": 0.0}
        result = predict_uab_dl(values, self.artifact)
        self.assertEqual(values, {"PCT": 0.0})
        self.assertEqual(result["missing_biomarkers"], [marker for marker in MARKERS if marker != "PCT"])
        panel = pd.DataFrame([{marker: values.get(marker, np.nan) for marker in MARKERS}])
        features = self.artifact["feature_engineer"].transform(panel)
        expected = self.artifact["model"].predict_proba(features)[0]
        self.assertAlmostEqual(result["confidence"], float(expected.max() * 100))

    def test_probabilities_follow_model_class_order(self):
        model = SimpleNamespace(
            classes_=np.array(["VR", "N", "C", "BVR", "BR"]),
            predict_proba=lambda features: np.array([[0.7, 0.1, 0.1, 0.05, 0.05]]),
        )
        artifact = {**self.artifact, "model": model}
        result = predict_uab_dl({"PCT": 1.0}, artifact)
        self.assertEqual(result["predicted_infection_type"], "VR")
        self.assertAlmostEqual(result["confidence"], 70.0)
        self.assertEqual(len(result["classifications"]), 5)

    def test_invalid_panels_are_rejected(self):
        for values in [{}, {"PCT": None}, {"PCT": np.nan}, {"PCT": -1}, {"PCT": np.inf}, {"PCT": -np.inf}, {"unknown": 1}]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                predict_uab_dl(values, self.artifact)

    def test_invalid_model_probabilities_are_rejected(self):
        for probabilities in [[np.nan, 0, 0, 0, 1], [0.5] * 5, [-0.1, 0.2, 0.2, 0.2, 0.5]]:
            model = SimpleNamespace(
                classes_=np.array(["BR", "BVR", "C", "N", "VR"]),
                predict_proba=lambda features, p=probabilities: np.array([p]),
            )
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                predict_uab_dl({"PCT": 1}, {**self.artifact, "model": model})


if __name__ == "__main__":
    unittest.main()
