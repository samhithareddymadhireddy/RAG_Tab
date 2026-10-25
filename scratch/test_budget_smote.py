import sys
sys.path.insert(0, ".")
import pandas as pd
from src.baselines.resamplers import apply_resampling

X = pd.DataFrame({'a': [float(i) for i in range(100)], 'b': [float(i*2) for i in range(100)]})
y = pd.Series(['negative'] * 90 + ['positive'] * 10)

X_res, y_res, prov = apply_resampling(X, y, method='smote', scale_features=True, n_synthetic=30)
print(f"Equal budget SMOTE: total rows = {len(X_res)}, positive count = {(y_res == 'positive').sum()}, synthetic provenance count = {(prov['origin'] == 'synthetic').sum()}")

X_res_full, y_res_full, prov_full = apply_resampling(X, y, method='smote', scale_features=True, n_synthetic=None)
print(f"Full SMOTE: total rows = {len(X_res_full)}, positive count = {(y_res_full == 'positive').sum()}, synthetic provenance count = {(prov_full['origin'] == 'synthetic').sum()}")
