Lab 4: Data Pipeline and Quality Report

Introduction

The purpose of this lab was to build a reproducible and leakage-free data pipeline using the Adult Income dataset available through OpenML. The pipeline performs data ingestion, validation, preprocessing, feature engineering, model training, and reproducibility tasks. The final solution generates validation results, a trained preprocessing pipeline, performance metrics, and a SHA256 hash for tracking processed datasets. 【1-0a4cf3】

Problem Definition
The Adult Income dataset is a binary classification dataset used to predict whether a person's annual income exceeds $50,000.

Input Features
The dataset contains fourteen predictor variables:

age
workclass
fnlwgt
education
education-num
marital-status
occupation
relationship
race
sex
capital-gain
capital-loss
hours-per-week
native-country
Target Variable
Income class (>50K or <=50K)
Success Metrics
The performance of the model is evaluated using:

Accuracy Score
F1 Score
Constraints
To satisfy lab requirements, the pipeline was designed to:

Prevent data leakage
Ensure reproducibility
Apply transformations learned from the training data only
Maintain a clean separation between training and testing data
Dataset Ingestion
The dataset was loaded from OpenML using:

fetch_openml(
    name="adult",
    version=2,
    as_frame=True
)
The ingestion stage reads the raw dataset without altering any values.

Dataset Statistics
Total Rows: 48,842
Total Columns: 15
Data Validation
Several data quality checks were performed before preprocessing.

Age Validation
The age feature was validated against the specified range:

Minimum: 17
Maximum: 90
Any values outside this range were reported as violations.

Hours Per Week Validation
The hours-per-week feature was checked using:

Minimum: 1
Maximum: 99
Violations were counted and reported.

Capital Gain Validation
The capital-gain feature was validated using:

Minimum: 0
Maximum: 99,999
Capital Loss Validation
The capital-loss feature was validated using:

Minimum: 0
Maximum: 4,356
Missing Value Assessment
The dataset was inspected for missing values and null entries in every column. Missing-value counts were generated to identify data quality issues before preprocessing.

Train-Test Split
The dataset was divided prior to preprocessing to eliminate data leakage.

Configuration used:

train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)
Split Details
Training Set: 70%
Test Set: 30%
Random State: 42
Stratification: Enabled
Stratified sampling preserved the class distribution across both datasets.

Duplicate Handling
Duplicate records were removed from the training dataset only.

Process:

Training features and labels were combined.
Exact duplicate rows were identified.
Duplicate records were removed.
The cleaned training set was reconstructed.
The test dataset remained unchanged to preserve unbiased evaluation.

Feature Engineering
Two derived features were created to provide additional predictive information.

Capital Net
capital_net = capital_gain - capital_loss
This feature represents the net financial gain for an individual.

Hours Per Age
hours_per_age = hours_per_week / (age - 16)
This feature normalizes working hours relative to age.

Both engineered features were added to the training and testing datasets.

Outlier Treatment
Numeric features were processed using percentile-based clipping.

For each numeric feature:

Lower bound = 1st percentile
Upper bound = 99th percentile
Values below the lower bound were clipped upward, while values above the upper bound were clipped downward.

Percentiles were calculated using training data and then applied to both training and testing datasets.

Data Preprocessing
A preprocessing pipeline was constructed using Scikit-Learn.

Numeric Processing
Numeric variables were imputed using the median value.

SimpleImputer(strategy="median")
Categorical Processing
Categorical variables were processed using:

SimpleImputer(strategy="most_frequent")
followed by:

OneHotEncoder(handle_unknown="ignore")
Column Transformer
Separate preprocessing pipelines for numeric and categorical data were combined into a single ColumnTransformer.

The fitted preprocessing pipeline was saved as:

results/preprocessor.joblib
Model Training
A Logistic Regression classifier was trained using the processed data.

Configuration:

LogisticRegression(
    max_iter=1000
)
The model was fitted using the training dataset and evaluated on the test dataset.

Evaluation Metrics
The following metrics were calculated:

Accuracy Score
F1 Score
Results were displayed after model prediction on the test set.

Dataset Hashing and Reproducibility
To support reproducibility, a SHA256 hash was generated from the processed training dataset.

The hash was created using:

hashlib.sha256()
and stored in:

results/dataset_hash.txt
This hash can be used to verify that the processed dataset has not changed between executions.

Generated Artifacts
The pipeline produces the following files:

results/preprocessor.joblib
results/dataset_hash.txt
In addition, the pipeline outputs:

Validation statistics
Dataset dimensions
Duplicate counts
Accuracy score
F1 score
SHA256 hash value
Running the Pipeline
The complete pipeline can be executed using:

python pipeline.py
This command performs all stages of the workflow, including ingestion, validation, preprocessing, feature engineering, model training, and hash generation.

Conclusion
A complete data quality and preprocessing pipeline was successfully implemented for the Adult Income dataset. The solution includes validation checks, duplicate detection, feature engineering, outlier treatment, preprocessing, machine learning model training, and reproducibility mechanisms. The pipeline follows a leakage-free design by performing the train-test split before data preparation and using training data to learn preprocessing parameters. The final workflow can be reproduced consistently using a single command.