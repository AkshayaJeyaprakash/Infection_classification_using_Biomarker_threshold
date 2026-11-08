"""Streamlit interactions for the UAB Deep Learning page."""

from pathlib import Path
import pickle
import unittest

from streamlit.testing.v1 import AppTest

from uab_dl import predict_uab_dl


ROOT = Path(__file__).resolve().parents[1]


class DLPageTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(ROOT / "test.py"), default_timeout=30)
        self.app.session_state.authenticated = True
        self.app.session_state.current_page = "Deep Learning Approach"
        self.app.run()
        self.assertFalse(list(self.app.exception))

    def test_navigation_order_and_empty_panel(self):
        labels = [button.label for button in self.app.sidebar.button]
        ml = labels.index("🌲 Machine Learning Approach")
        dl = labels.index("🧠 Deep Learning Approach")
        llm = labels.index("🤖 LLM-Based Approach")
        self.assertEqual(dl, ml + 1)
        self.assertEqual(llm, dl + 1)
        self.assertEqual(len(self.app.number_input), 7)
        self.assertTrue(self.app.button(key="dl_classify_btn").disabled)
        next(button for button in self.app.sidebar.button if button.label == "🏠 Home").click().run()
        cards = [button.label for button in self.app.button if button.label.startswith("Open ")]
        self.assertEqual(cards, [
            "Open Statistical Approach →", "Open Machine Learning Approach →",
            "Open Deep Learning Approach →", "Open LLM-Based Approach →",
        ])
        self.app.button(key="home_Deep Learning Approach").click().run()
        self.assertEqual(self.app.session_state.current_page, "Deep Learning Approach")

    def test_unit_conversion_prediction_and_edit_invalidation(self):
        self.app.number_input(key="dl_value_0_PCT").set_value(1000.0)
        self.app.selectbox(key="dl_unit_0_PCT").set_value("pg/mL").run()
        self.app.button(key="dl_classify_btn").click().run()
        self.assertFalse(list(self.app.exception))
        self.assertFalse(list(self.app.error))
        with (ROOT / "uab_dl_model.pkl").open("rb") as handle:
            expected = predict_uab_dl({"PCT": 1.0}, pickle.load(handle))
        self.assertEqual(self.app.session_state.dl_result, expected)
        self.assertEqual(len(self.app.dataframe[0].value), 5)
        self.app.selectbox(key="dl_unit_0_PCT").set_value("ng/mL").run()
        self.assertIsNone(self.app.session_state.dl_result)

    def test_navigation_persistence_reset_and_logout(self):
        self.app.number_input(key="dl_value_0_CRP").set_value(200.0).run()
        self.app.button(key="dl_classify_btn").click().run()
        result = self.app.session_state.dl_result
        self.assertIsNotNone(result)
        next(button for button in self.app.sidebar.button if button.label == "🏠 Home").click().run()
        self.app.button(key="home_Deep Learning Approach").click().run()
        self.assertEqual(self.app.number_input(key="dl_value_0_CRP").value, 200.0)
        self.assertEqual(self.app.session_state.dl_result, result)
        self.app.button(key="dl_new_btn").click().run()
        self.assertIsNone(self.app.session_state.dl_result)
        self.assertTrue(all(widget.value is None for widget in self.app.number_input))
        self.assertTrue(self.app.button(key="dl_classify_btn").disabled)
        self.app.number_input(key="dl_value_1_CRP").set_value(300.0).run()
        next(button for button in self.app.sidebar.button if button.label == "🚪 Logout").click().run()
        self.assertFalse(self.app.session_state.authenticated)
        self.assertIsNone(self.app.session_state.dl_result)
        self.assertTrue(all(data['value'] is None for data in self.app.session_state.dl_panel_inputs.values()))


if __name__ == "__main__":
    unittest.main()
