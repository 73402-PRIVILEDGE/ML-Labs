import pandas as pd
import joblib
import os
import hashlib

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

os.makedirs("results", exist_ok=True)

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
# VALIDATION
# =====================================================

print("\nVALIDATION REPORT")
print("=" * 50)

print(
    "Age violations:",
    ((df["age"] < 17) | (df["age"] > 90)).sum()
)

print(
    "Hours violations:",
    ((df["hours-per-week"] < 1) |
     (df["hours-per-week"] > 99)).sum()
)

print(
    "Capital gain violations:",
    ((df["capital-gain"] < 0) |
     (df["capital-gain"] > 99999)).sum()
)

print(
    "Capital loss violations:",
    ((df["capital-loss"] < 0) |
     (df["capital-loss"] > 4356)).sum()
)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X = df.drop("class", axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# =====================================================
# DUPLICATES TRAIN ONLY
# =====================================================

train_df = X_train.copy()
train_df["class"] = y_train

before = len(train_df)

train_df = train_df.drop_duplicates()

after = len(train_df)

print("\nDuplicates removed:", before - after)

X_train = train_df.drop("class", axis=1)
y_train = train_df["class"]

# =====================================================
# FEATURE ENGINEERING
# =====================================================

for data in [X_train, X_test]:

    data["capital_net"] = (
        data["capital-gain"]
        - data["capital-loss"]
    )

    data["hours_per_age"] = (
        data["hours-per-week"]
        / (data["age"] - 16)
    )

# =====================================================
# NUMERIC / CATEGORICAL
# =====================================================

numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X_train.select_dtypes(
    exclude=["int64", "float64"]
).columns

# =====================================================
# OUTLIER CLIPPING
# =====================================================

for col in numeric_features:

    low = X_train[col].quantile(0.01)
    high = X_train[col].quantile(0.99)

    X_train[col] = X_train[col].clip(low, high)
    X_test[col] = X_test[col].clip(low, high)

# =====================================================
# PREPROCESSOR
# =====================================================

numeric_transformer = Pipeline(
    [
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)

categorical_transformer = Pipeline(
    [
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    [
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)

joblib.dump(
    preprocessor,
    "results/preprocessor.joblib"
)

# =====================================================
# MODEL
# =====================================================

model = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions,
    pos_label=">50K"
)

print("\nMODEL RESULTS")
print("Accuracy:", round(accuracy, 4))
print("F1 Score:", round(f1, 4))

# =====================================================
# SHA256 HASH
# =====================================================

processed = pd.concat(
    [X_train, y_train],
    axis=1
)

hash_value = hashlib.sha256(
    pd.util.hash_pandas_object(
        processed,
        index=True
    ).values
).hexdigest()

with open(
    "results/dataset_hash.txt",
    "w"
) as f:
    f.write(hash_value)

print("\nSHA256 Hash:")
print(hash_value)

print("\nFinished successfully.")