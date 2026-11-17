import pandas as pd

results = pd.DataFrame({

    "Model": [
        "Baseline CNN",
        "Global Pruning",
        "Layerwise Pruning",
        "Structured Pruning",
        "Student",
        "Distilled Student"
    ],

    "Accuracy": ["", "", "", "", "", ""],

    "Parameters": ["", "", "", "", "", ""],

    "Size_MB": ["", "", "", "", "", ""],

    "Latency_ms": ["", "", "", "", "", ""]
})

print(results)

results.to_csv(
    "summary_table.csv",
    index=False
)