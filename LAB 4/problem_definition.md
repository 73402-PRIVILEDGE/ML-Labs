Problem Definition

The objective of this project is to predict whether an individual earns more than $50,000 per year using demographic and employment information from the UCI Adult dataset. The dataset contains 14 input features including age, education, occupation, marital status, work hours, and capital gains or losses.

The target variable is income, classified into two categories: greater than $50K or less than or equal to $50K. The primary evaluation metrics are Accuracy and F1 Score. Accuracy measures overall correctness of predictions, while F1 Score evaluates the balance between precision and recall.

A key project requirement is preventing data leakage. Therefore, the dataset is split into training and testing subsets before cleaning and transformation steps. Missing value statistics and outlier thresholds are learned only from the training data and then applied to the test data.

The workflow is designed to be reproducible through configuration files, saved preprocessing objects, and a SHA256 hash of the processed dataset.