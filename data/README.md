# Dataset

Training uses the **Sleep Health and Lifestyle Dataset** published on Kaggle:

- Source: [Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)
- Publisher: Laksika Tharmalingam
- License: **CC0: Public Domain**

Download and extract the dataset, then place the CSV at:

```text
data/Sleep_health_and_lifestyle_dataset.csv
```

With the Kaggle CLI configured, you can run:

```bash
kaggle datasets download \
  -d uom190346a/sleep-health-and-lifestyle-dataset \
  -p data \
  --unzip
```

CSV files under `data/` are intentionally excluded from Git. The repository's
MIT License applies to the source code, not to third-party datasets.
