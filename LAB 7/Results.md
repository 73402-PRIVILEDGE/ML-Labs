Results
Baseline CNN

An oversized CNN was implemented and used as the baseline model for all subsequent experiments. The network contained approximately one million parameters and was trained on the Fashion-MNIST dataset. The model achieved high classification performance while maintaining stable training behaviour. As expected, the baseline model required the largest amount of storage and computation among all evaluated models. The measured performance metrics were recorded and later used as the reference point for pruning and distillation experiments. The goal was to establish a strong teacher network before applying compression techniques.

Global Magnitude Pruning

Global magnitude pruning was applied by ranking all model weights according to their absolute values and removing weights with the smallest magnitudes. Sparsity levels of 0%, 25%, 50%, 75%, 90%, and 95% were tested.

The experimental results showed that moderate pruning levels had only a limited impact on classification accuracy. At lower sparsity levels, the network retained most of its predictive capability despite a significant reduction in active parameters. As sparsity increased beyond 75%, a noticeable reduction in accuracy was observed because a large amount of model information was removed. The 95% sparsity level produced the largest degradation in performance and demonstrated the practical limit of aggressive unstructured pruning.

Overall, global pruning proved to be an effective compression method because it removed a substantial number of weights while preserving acceptable accuracy.

Layer-Wise Pruning

Layer-wise pruning was then performed using the same sparsity levels. Instead of comparing all weights globally, pruning decisions were applied independently within each network layer.

The results indicated that layer-wise pruning generally maintained competitive performance but often produced slightly lower accuracy than global pruning at higher sparsity levels. This behaviour occurred because every layer was forced to lose a fixed percentage of weights regardless of the importance of those weights.

The experiment demonstrated that while layer-wise pruning is simple to implement, global pruning usually makes more efficient use of the available parameter budget.

Structured Pruning

Structured pruning was evaluated by removing entire convolutional filters using L2-norm ranking. Three pruning ratios were examined: 10%, 25%, and 50%.

Unlike unstructured pruning, structured pruning modifies the architecture of the neural network by eliminating entire channels rather than individual connections. The results showed that removing 10% and 25% of filters had a relatively small impact on classification performance. However, when 50% of filters were removed, the reduction in model capacity became more visible and accuracy decreased accordingly.

Although structured pruning may reduce accuracy slightly more than unstructured pruning at similar compression levels, it provides additional advantages for deployment because fewer operations must be executed during inference.

Latency Analysis

Latency measurements were collected for the dense baseline model and for the compressed variants. The comparison focused on the relationship between model compression and practical inference speed.

The results demonstrated that unstructured pruning greatly reduced the number of non-zero parameters and reduced storage requirements. However, latency improvements were limited because standard CPU implementations continue to use dense matrix operations. Consequently, many zero-valued weights are still processed during inference.

Structured pruning produced more meaningful latency improvements because entire filters were removed from computation. As a result, the network performed fewer operations and achieved improved execution efficiency.

The experiment confirms that storage reduction and latency reduction are not necessarily equivalent objectives and should be evaluated separately.

Knowledge Distillation

Knowledge distillation was performed using the baseline CNN as the teacher model and a significantly smaller CNN as the student model. The loss function combined cross-entropy loss with Kullback-Leibler divergence using a temperature of 4 and an alpha value of 0.5 as specified in the laboratory instructions.

Two student models were evaluated: one trained independently and another trained using teacher guidance. The distilled student consistently achieved higher accuracy than the standalone student while maintaining a similar model size and computational cost.

These findings demonstrate the effectiveness of knowledge distillation as a compression technique. The student network was able to learn useful information from the teacher's output distributions and therefore achieved improved generalisation performance despite having far fewer parameters.

Comparative Analysis

The collected experimental results illustrate the trade-off between model size, accuracy, computation, and efficiency. The baseline CNN achieved the highest predictive performance but required the greatest computational resources. Pruning techniques reduced parameter counts and storage requirements, with global pruning generally outperforming layer-wise pruning at equivalent sparsity levels. Structured pruning showed the strongest potential for practical deployment because it provided measurable computational savings.

Knowledge distillation achieved one of the most favourable efficiency trade-offs. The distilled student model remained compact while delivering accuracy closer to that of the larger teacher network. This makes knowledge distillation particularly suitable for deployment on resource-constrained devices.

Conclusion

This laboratory investigated several modern model compression techniques, including global pruning, layer-wise pruning, structured pruning, and knowledge distillation. The results showed that neural networks contain substantial redundancy, allowing many parameters to be removed with only a limited effect on accuracy. Global pruning generally provided the best accuracy retention among unstructured approaches, while structured pruning provided better opportunities for reducing real computational cost. Knowledge distillation enabled a compact student network to approach the performance of a much larger teacher model. Overall, the experiments demonstrated how compression methods can improve efficiency across model size, computation, latency, and energy consumption while maintaining acceptable predictive performance.