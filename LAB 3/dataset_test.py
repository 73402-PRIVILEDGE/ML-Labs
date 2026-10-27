from torchvision.datasets import FashionMNIST
from torchvision import transforms

dataset = FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

print("Dataset loaded successfully!")
print("Number of images:", len(dataset))