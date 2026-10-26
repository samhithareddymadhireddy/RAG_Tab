from imblearn.over_sampling import BorderlineSMOTE, SMOTE
import pandas as pd

X = pd.DataFrame({'a': range(60), 'b': range(60)})
y = pd.Series([0]*30 + [1]*20 + [2]*10)

try:
    sm = SMOTE(random_state=42)
    X_sm, y_sm = sm.fit_resample(X, y)
    print("SMOTE multiclass: SUCCESS")
except Exception as e:
    print(f"SMOTE multiclass: {e}")

try:
    bsm = BorderlineSMOTE(random_state=42)
    X_bsm, y_bsm = bsm.fit_resample(X, y)
    print("BorderlineSMOTE multiclass: SUCCESS")
except Exception as e:
    print(f"BorderlineSMOTE multiclass: {e}")
