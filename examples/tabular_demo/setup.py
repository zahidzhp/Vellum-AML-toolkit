"""Generate the Iris CSV dataset for tabular demo."""

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris

output_dir = Path(__file__).parent
output_dir.mkdir(parents=True, exist_ok=True)

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["label"] = iris.target

csv_path = output_dir / "iris.csv"
df.to_csv(csv_path, index=False)
print(f"Created {csv_path} ({len(df)} samples, {len(iris.feature_names)} features, {len(set(iris.target))} classes)")
