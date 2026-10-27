import matplotlib.pyplot as plt

# Replace these with YOUR actual results

models = ["MLP", "CNN", "Attention"]

accuracy = [
    84.5,   # MLP
    87.4,   # CNN
    86.2    # Attention
]

latency = [
    0.30,   # MLP
    1.20,   # CNN
    3.50    # Attention
]

plt.figure(figsize=(8, 6))

plt.scatter(latency, accuracy, s=120)

for i, model in enumerate(models):
    plt.annotate(
        model,
        (latency[i], accuracy[i]),
        xytext=(5, 5),
        textcoords="offset points"
    )

plt.xlabel("Latency (ms)")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Latency")

plt.grid(True)

plt.savefig("Accuracy vs Latency plot.png")

plt.show()