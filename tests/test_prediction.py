import unittest
from pathlib import Path

import joblib
import sklearn

from src.predict import predict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "sleep_disorder_pipeline.joblib"


class PredictionTests(unittest.TestCase):
    def test_saved_pipeline_matches_pinned_sklearn_version(self) -> None:
        pipeline = joblib.load(MODEL_PATH)
        expected_model = max(
            pipeline.cv_results_,
            key=lambda name: (
                pipeline.cv_results_[name]["f1_macro"],
                pipeline.cv_results_[name]["accuracy"],
            ),
        )

        self.assertEqual(
            pipeline.training_metadata_["scikit_learn_version"],
            sklearn.__version__,
        )
        self.assertEqual(pipeline.model_name_, expected_model)

    def test_saved_pipeline_predicts_known_shape(self) -> None:
        patient = {
            "Gender": "Male",
            "Age": 31,
            "Occupation": "Software Engineer",
            "Sleep Duration": 7.2,
            "Quality of Sleep": 8,
            "Physical Activity Level": 60,
            "Stress Level": 4,
            "BMI Category": "Normal",
            "Blood Pressure": "120/80",
            "Heart Rate": 70,
            "Daily Steps": 8_000,
        }

        result = predict(patient, MODEL_PATH)

        self.assertIn(result, {"Healthy", "Insomnia", "Sleep Apnea"})

    def test_saved_pipeline_handles_unseen_categories(self) -> None:
        patient = {
            "Gender": "Non-binary",
            "Age": 29,
            "Occupation": "Astronaut",
            "Sleep Duration": 6.5,
            "Quality of Sleep": 6,
            "Physical Activity Level": 35,
            "Stress Level": 7,
            "BMI Category": "Normal",
            "Blood Pressure": "128/78",
            "Heart Rate": 72,
            "Daily Steps": 6_500,
        }

        result = predict(patient, MODEL_PATH)

        self.assertIn(result, {"Healthy", "Insomnia", "Sleep Apnea"})

    def test_rejects_out_of_range_patient_values(self) -> None:
        patient = {
            "Gender": "Male",
            "Age": 31,
            "Occupation": "Engineer",
            "Sleep Duration": 7.2,
            "Quality of Sleep": 1_000,
            "Physical Activity Level": 60,
            "Stress Level": 4,
            "BMI Category": "Normal",
            "Blood Pressure": "120/80",
            "Heart Rate": 70,
            "Daily Steps": 8_000,
        }

        with self.assertRaisesRegex(ValueError, "Quality of Sleep"):
            predict(patient, MODEL_PATH)

    def test_rejects_impossible_blood_pressure(self) -> None:
        patient = {
            "Gender": "Male",
            "Age": 31,
            "Occupation": "Engineer",
            "Sleep Duration": 7.2,
            "Quality of Sleep": 8,
            "Physical Activity Level": 60,
            "Stress Level": 4,
            "BMI Category": "Normal",
            "Blood Pressure": "80/120",
            "Heart Rate": 70,
            "Daily Steps": 8_000,
        }

        with self.assertRaisesRegex(ValueError, "Systolic"):
            predict(patient, MODEL_PATH)


if __name__ == "__main__":
    unittest.main()
