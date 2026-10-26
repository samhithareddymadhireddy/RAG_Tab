from sklearn.datasets import fetch_openml
import pandas as pd

bunch = fetch_openml(name="vehicle", as_frame=True, parser="auto")
df = bunch.frame
print(f"Vehicle shape: {df.shape}")
print(f"Vehicle target column: {bunch.target_names}")
print(f"Vehicle class distribution:\n{df[bunch.target_names[0]].value_counts()}")
