import unittest

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.preprocessing import build_pipeline
from src.train import make_feature_groups


class TrainingTests(unittest.TestCase):
    def test_duplicate_features_share_a_validation_group(self) -> None:
        features = pd.DataFrame(
            [
                {"Person ID": 1, "Age": 30, "Gender": "Female"},
                {"Person ID": 2, "Age": 30, "Gender": "Female"},
                {"Person ID": 3, "Age": 45, "Gender": "Male"},
            ]
        )

        groups = make_feature_groups(features)

        self.assertEqual(groups.iloc[0], groups.iloc[1])
        self.assertNotEqual(groups.iloc[0], groups.iloc[2])

    def test_pipeline_can_train_without_external_preprocessing(self) -> None:
        rows = []
        targets = []
        for index, label in enumerate(
            ["Healthy", "Insomnia", "Sleep Apnea"] * 3
        ):
            rows.append(
                {
                    "Gender": "Female" if index % 2 else "Male",
                    "Age": 25 + index * 3,
                    "Occupation": f"Occupation {index % 4}",
                    "Sleep Duration": 5.5 + index * 0.25,
                    "Quality of Sleep": 4 + index % 6,
                    "Physical Activity Level": 20 + index * 5,
                    "Stress Level": 1 + index % 10,
                    "BMI Category": "Normal",
                    "Blood Pressure": f"{110 + index * 3}/{70 + index}",
                    "Heart Rate": 60 + index,
                    "Daily Steps": 4_000 + index * 500,
                }
            )
            targets.append(label)

        pipeline = build_pipeline(LogisticRegression(max_iter=1_000))
        pipeline.fit(pd.DataFrame(rows), pd.Series(targets))

        self.assertEqual(len(pipeline.predict(pd.DataFrame(rows))), len(rows))


if __name__ == "__main__":
    unittest.main()
