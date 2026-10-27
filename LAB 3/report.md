The MLP model achieved a test accuracy of 87.20% after five epochs of training on the Fashion-MNIST dataset.

The MLP architecture had the largest parameter count (235,146) but the lowest measured FLOPs among the three architectures.

The CNN architecture was the most parameter-efficient model, requiring only 20,490 parameters due to weight sharing through convolution kernels. Although the CNN had the highest measured FLOPs, it is specifically designed to exploit spatial information in image data.

The Attention architecture required 70,794 parameters and 591,488 FLOPs. Self-attention mechanisms allow interactions between image patches and can capture long-range dependencies that may not be easily modeled by traditional convolutional layers.

Overall, the results demonstrate that neural network architectures involve trade-offs between parameter count, computational cost, memory usage, and predictive performance. CNNs provide excellent parameter efficiency, Attention models offer greater flexibility for global feature modeling, and MLPs provide a simple baseline architecture for comparison.