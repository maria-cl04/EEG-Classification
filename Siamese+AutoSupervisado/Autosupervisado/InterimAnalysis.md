# Interim Analysis - SimCLR Hyperparameter Experiments
## 1. Executive Summary
The first phase of SimCLR experimentation revealed two distinct and opposing training regimes. While the first experiment provided a modest accuracy improvement over the baseline, the internal loss breakdown showed that the self-supervised objective was not actually being solved. The second experiment successfully achieved convergence of the self-supervised loss but severely degraded classification accuracy. These results demonstrate that SimCLR performance on EEG data is highly sensitive to the balance between **convergence** and **regularization dominance**.

## 2. Experiment 1: The "Diffuse Regularizer"
**Configuration**: $\lambda=0.05, \tau=0.5$
- **Mean Test Accuracy (ep. 181-200): 56.74% - 57.53%** (+0.66 to +1.45 pp vs. baseline).
- **Internal Behavior**: The SimCLR loss remained "stuck" near **3.86** for the entire run, dorpping only 0.60 across 200 epochs.
- **Conclusion**: This configuration acted as a gentle, diffuse regularizer. Because the temperature was too soft ($\tau=0.5$), the model never learned to consistently distinguish positive pairs from the 255 negatives. While it provided a small accuracy boost, it did not achieve the structured representation learning intended by the SimCLR framework.

### Is this experiment really SimCLR?
It is SimCLR implemented correctly, but operating in a regime where the contrastive task is too difficult for the model to solve. The result is that the gradient from NT-Xent nudges the projection head without ever shaping it into the structured hypersphere geometry that SimCLR intends. It's worth reporting, but it is a funding about the limits of self-supervised contrastive learning on EEG - not a successful SimCLR run.

## 3. Experiment 2: The "Sharp Competition"
**Configuration**: $\lambda=0.2, \tau=0.1$
- **Mean Test Accuracy (ep181–200): 49.22%** (−6.86 pp vs. baseline).
- **Internal Behavior**: The SimCLR loss converged perfectly, dropping from 1.90 to 0.31.
- **Conclusion**: This was a clear case of destructive competition between loss functions. The sharp temperature ($\tau=0.1$) created aggressive gradients that solved the contrastive task but overwhelmed the classification task. Furthermore, because SimCLR is label-free, it likely suffered from the False Negative Problem: violently pushing apart different trials that actually belonged to the same category, thereby destroying the class clusters needed by the classifier.

## 4. Problem Diagnosis: The Stability-Convergence Gap
Analysis of these runs reveals a "failure mode bookend":
1. **Too Soft ($\tau=0.5$)**: The model is stable but does not learn representations because the contrastive task is too difficult to solve.
2. **Too Aggressive ($\lambda=0.2, \tau=0.1$)**: The model learns contrastive representations but they are counter-productive for classification.The central challenge for this EEG dataset is finding the "Goldilocks Zone"—a configuration where the contrastive loss converges (showing the model is learning) but remains at a weight that doesn't disrupt the Cross-Entropy (CE) classification boundaries.

