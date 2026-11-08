"""Switch ML artifacts by changing only model_config.ML_MODEL_FILE."""

from pathlib import Path
import pickle
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

import model_config
from uab_panel import MARKERS, predict_uab_panel


ROOT = Path(__file__).resolve().parents[1]


class MLModelTests(unittest.TestCase):
    def open_ml_page(self):
        app = AppTest.from_file(str(ROOT / "test.py"), default_timeout=30)
        app.session_state.authenticated = True
        app.session_state.current_page = "Machine Learning Approach"
        app.run()
        self.assertFalse(list(app.exception))
        self.assertFalse(list(app.error))
        return app

    def test_default_preserves_existing_ml_pipeline(self):
        self.assertEqual(model_config.ML_MODEL_FILE, "rf1_infection_type.pkl")
        app = self.open_ml_page()
        self.assertEqual(len(app.checkbox), 44)
        app.selectbox(key="rf_rf_biomarker_select_0").set_value("PCT")
        app.number_input(key="rf_rf_threshold_input_0").set_value(1.0)
        app.button(key="rf_rf_add_btn").click().run()
        app.button(key="rf_classify_btn").click().run()
        self.assertFalse(list(app.exception))
        self.assertFalse(list(app.error))
        result = app.session_state.rf_pipeline_result
        self.assertTrue(app.session_state.rf_submitted)
        self.assertEqual(result["final_infection_type"], "VR")
        self.assertAlmostEqual(result["infection_confidence"], 75.98398009373689)
        self.assertEqual(result["final_disease"], "COVID-19")

    def test_filename_alone_selects_uab_rf_with_unit_conversion(self):
        with patch("model_config.ML_MODEL_FILE", "uab_rf_model.pkl"):
            app = self.open_ml_page()
            self.assertEqual(len(app.number_input), 7)
            self.assertEqual(len(app.checkbox), 0)
            self.assertTrue(app.button(key="ml_panel_classify_btn").disabled)
            app.number_input(key="ml_panel_value_0_PCT").set_value(1000.0)
            app.selectbox(key="ml_panel_unit_0_PCT").set_value("pg/mL").run()
            app.button(key="ml_panel_classify_btn").click().run()
            with (ROOT / "uab_rf_model.pkl").open("rb") as handle:
                expected = predict_uab_panel({"PCT": 1.0}, pickle.load(handle))
            self.assertFalse(list(app.exception))
            self.assertFalse(list(app.error))
            self.assertEqual(app.session_state.ml_panel_result, expected)
            self.assertEqual(len(app.dataframe[0].value), 5)
            app.number_input(key="ml_panel_value_0_PCT").set_value(2000.0).run()
            self.assertIsNone(app.session_state.ml_panel_result)

    def test_full_rf_panel_matches_raw_saved_model(self):
        values = dict(zip(MARKERS, [0.08, 0.01, 0.31, 2100.0, 98.5, 1.45, 14.6]))
        with (ROOT / "uab_rf_model.pkl").open("rb") as handle:
            artifact = pickle.load(handle)
        panel = pd.DataFrame([values], columns=MARKERS)
        features = artifact["feature_engineer"].transform(panel)
        expected = artifact["model"].predict_proba(features)[0]
        result = predict_uab_panel(values, artifact)
        by_class = {item["Class"]: item["Confidence"] / 100 for item in result["classifications"]}
        for label, probability in zip(artifact["model"].classes_, expected):
            self.assertAlmostEqual(by_class[label], probability)
        self.assertEqual(result["predicted_infection_type"], artifact["model"].predict(features)[0])

    def test_rf_and_dl_inputs_results_and_reset_are_independent(self):
        with patch("model_config.ML_MODEL_FILE", "uab_rf_model.pkl"):
            app = self.open_ml_page()
            app.number_input(key="ml_panel_value_0_PCT").set_value(1.0).run()
            app.button(key="ml_panel_classify_btn").click().run()
            rf_result = app.session_state.ml_panel_result
            next(button for button in app.sidebar.button if "Deep Learning" in button.label).click().run()
            self.assertIsNone(app.number_input(key="dl_value_0_PCT").value)
            self.assertIsNone(app.session_state.dl_result)
            app.number_input(key="dl_value_0_PCT").set_value(2.0).run()
            app.button(key="dl_classify_btn").click().run()
            dl_result = app.session_state.dl_result
            self.assertIsNotNone(dl_result)
            next(button for button in app.sidebar.button if "Machine Learning" in button.label).click().run()
            self.assertEqual(app.number_input(key="ml_panel_value_0_PCT").value, 1.0)
            self.assertEqual(app.session_state.ml_panel_result, rf_result)
            app.button(key="ml_panel_new_btn").click().run()
            self.assertIsNone(app.session_state.ml_panel_result)
            self.assertTrue(all(widget.value is None for widget in app.number_input))
            self.assertEqual(app.session_state.dl_result, dl_result)
            next(button for button in app.sidebar.button if "Logout" in button.label).click().run()
            self.assertIsNone(app.session_state.ml_panel_result)
            self.assertIsNone(app.session_state.dl_result)
            for inputs in [app.session_state.ml_panel_panel_inputs, app.session_state.dl_panel_inputs]:
                self.assertTrue(all(data["value"] is None for data in inputs.values()))

    def test_switching_model_clears_results_from_previous_model(self):
        with patch("model_config.ML_MODEL_FILE", "uab_rf_model.pkl"):
            app = self.open_ml_page()
            app.number_input(key="ml_panel_value_0_PCT").set_value(1.0).run()
            app.button(key="ml_panel_classify_btn").click().run()
            self.assertIsNotNone(app.session_state.ml_panel_result)
        app.run()
        self.assertEqual(len(app.checkbox), 44)
        self.assertIsNone(app.session_state.ml_panel_result)
        self.assertIsNone(app.session_state.rf_pipeline_result)
        self.assertFalse(app.session_state.rf_submitted)
        with patch("model_config.ML_MODEL_FILE", "uab_rf_model.pkl"):
            app.run()
            self.assertEqual(app.number_input(key="ml_panel_value_0_PCT").value, 1.0)
            self.assertIsNone(app.session_state.ml_panel_result)
            self.assertFalse(list(app.exception))

    def test_missing_configured_model_shows_error_without_fallback(self):
        with patch("model_config.ML_MODEL_FILE", "missing_ml_model.pkl"):
            app = AppTest.from_file(str(ROOT / "test.py"), default_timeout=30)
            app.session_state.authenticated = True
            app.session_state.current_page = "Machine Learning Approach"
            app.run()
            self.assertFalse(list(app.exception))
            self.assertEqual(len(app.error), 1)
            self.assertIn("missing_ml_model.pkl", app.error[0].value)
            self.assertEqual(len(app.checkbox), 0)


if __name__ == "__main__":
    unittest.main()
