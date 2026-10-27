import matplotlib.pyplot as plt

models = ["MLP", "CNN", "Attention"]

accuracy = [84.5, 87.4, 86.2]

parameters = [
    234762,   # replace with your flops.py results
    18058,
    72138
]

plt.figure(figsize=(8, 6))

plt.scatter(parameters, accuracy, s=120)

for i, model in enumerate(models):
    plt.annotate(
        model,
        (parameters[i], accuracy[i]),
        xytext=(5, 5),
        textcoords="offset points"
    )

plt.xlabel("Parameters")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Parameters")

plt.grid(True)

plt.savefig("Accuracy_vs_Parameters.png")

plt.show()