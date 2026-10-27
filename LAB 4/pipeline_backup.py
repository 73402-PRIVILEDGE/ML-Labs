import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# =====================================================
# LOAD DATASET
# =====================================================

print("Loading Adult dataset...")

adult = fetch_openml(
    name="adult",
    version=2,
    as_frame=True
)

df = adult.frame

print("Rows:", len(df))
print("Columns:", len(df.columns))

# =====================================================
# VALIDATION REPORT
# =====================================================

print("\nVALIDATION REPORT")
print("=" * 50)

age_violations = (
    (df["age"] < 17) |
    (df["age"] > 90)
).sum()

print("Age violations:", age_violations)

hours_violations = (
    (df["hours-per-week"] < 1) |
    (df["hours-per-week"] > 99)
).sum()

print("Hours violations:", hours_violations)

gain_violations = (
    (df["capital-gain"] < 0) |
    (df["capital-gain"] > 99999)
).sum()

print("Capital gain violations:", gain_violations)

loss_violations = (
    (df["capital-loss"] < 0) |
    (df["capital-loss"] > 4356)
).sum()

print("Capital loss violations:", loss_violations)

print("\nMissing values per column:")
print(df.isna().sum())

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

print("\nTRAIN TEST SPLIT")
print("=" * 50)

X = df.drop("class", axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))