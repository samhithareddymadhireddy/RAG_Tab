from sklearn.datasets import fetch_openml

bunch = fetch_openml(name="dermatology", as_frame=True, parser="auto")
df = bunch.frame
print(f"Dermatology shape: {df.shape}")
print(f"Dermatology classes:\n{df['class'].value_counts()}")
