# Method 2: Self-Supervised Contrastive (Autosupervisado)


## Recommendation: SimCLR (Simple Framework for Contrastive Learning)

- **Invariance to "EEG Noise"**: EEG signals are notoriously "noisy" (electrode movement, muscle tension). SimCLR teaches the model that a signal with noise and a signal without noise are the same thing. This makes the Transformer's encoder much more robust.
- **No "Negative" Samplig Bias**: Since it doesn0t use labels, it doesn't get "confused" by the 40 different classes during the pre-training/auxilary phase. It just learns what "EEG data" looks like in general.


## Other Options (and why they aren't optimal):

- **MAE (Masked Autoencoders)**: These are *generative*, not *contrastive*. They try to "reconstruct" missing parts of the signal. While powerful, they can often requiere larger datasets than what we have (6 subjects) in order to converge.
- **BYOL (Bootstrap Your Own Latent)**: This avoids using "negative" samples (it only looks at similarities). However, it is extremely sensitive to hyperparameters. If the learning rate is slightly off, the model "collapses" (it starts giving the same output for every input).


### Key References:

- *Chen, T., et al. (2020). "A Simple Framework for Contrastive Learning of Visual REpresentations." ICML.* (The foundation of SimCLR)
- *Mohsenvand, M. N., et al. (2020). "Contrastive Representation Learning for Electroencephalogram Classification." ML4H.* (Specific application of SimCLR logic to EEG).

---

# Method 3: Siamese Network (Red Siamesa)


## Recommendation: Online Hard Triplet Mining

- **The "40-Class Problem"**: With 40 classes, most "negative" pairs are too easy. If you just pick a random sample from Class 1 and Class 2, they look so different that the model learns nothing. **Hard Mining** forces the model to find the "hardest" cases - the samples that *look* similar but are actually different classes.
- **Clustering**: Siamese networks are designed to create "clusters" in digital space. This is perfect beacuse it helps the model group all Subject 1's "Cat" signals together, and all Subject 2's "Cat" signals together, even if their brainwaves look different.


## Other Options (and why they aren't optimal):

- **Standard Contrastive Loss (Pairs)**: This only compares 2 thing at a time (Same or Different). It's very slow to converge when you have 40 classes because the model spends 99% of its time looking at "Easy Negatives".
- **Prototypical Networks**: These are great for "Few-Shot" learning (when you only have 5 examples of a class). Since there is a decent amount of data (300 trials per class), the complexity of mantaining "Prototypes" (class means) isn't necesarry and can add extra computational lag.


### Key References:

- *Schroff, F., et al. (2015). "FaceNet: A Unified Embedding for Face Recognition and Clustering." CVPR.* (Introduces Triplet Loss and Hard Mining)
- *Chopra, S., et al. (2005). "Learning a Similarity Metric Discriminatively, with Application to Face Verification.* (The original "Siamese Network" paper).

