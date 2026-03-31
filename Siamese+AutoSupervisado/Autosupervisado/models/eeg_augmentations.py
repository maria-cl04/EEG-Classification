import torch
import torch.nn as nn


class EEGAugmentation:
    """
    Label-preserving augmentations for EEG signals, designed for SimCLR-style
    self-supervised contrastive learning.

    All methods operate on tensors of shape (B, T, C):
        B = batch size
        T = time samples (440 for the 20-460 window)
        C = EEG channels (128)

    Each augmentation simulates a realistic source of variability that does NOT
    change the underlying neural response — i.e. the class label is preserved.
    This is the key constraint for valid contrastive augmentation.

    Augmentations are applied on the same device as the input tensor.
    No learnable parameters.
    """

    def __init__(self,
                 noise_std=0.1,
                 channel_mask_p=0.1,
                 amplitude_low=0.8,
                 amplitude_high=1.2):
        """
        Args:
            noise_std      (float): Std of additive Gaussian noise. Default 0.1.
                                    Simulates electrode/amplifier noise.
            channel_mask_p (float): Fraction of channels to zero out per sample.
                                    Default 0.1 (≈13 of 128 channels).
                                    Simulates bad/disconnected electrodes.
            amplitude_low  (float): Lower bound of uniform amplitude scale factor.
            amplitude_high (float): Upper bound of uniform amplitude scale factor.
                                    Range [0.8, 1.2] simulates inter-session
                                    impedance and gain variation.
        """
        self.noise_std = noise_std
        self.channel_mask_p = channel_mask_p
        self.amplitude_low = amplitude_low
        self.amplitude_high = amplitude_high

    # ------------------------------------------------------------------
    # Individual augmentations
    # ------------------------------------------------------------------

    def gaussian_noise(self, x):
        """
        Add zero-mean Gaussian noise to the signal.

        Noise is scaled by the configured std and is independent per
        sample, time step, and channel — matching the statistical profile
        of real electrode noise.

        Args:
            x (Tensor): shape (B, T, C)
        Returns:
            Tensor: shape (B, T, C), same device as input
        """
        noise = torch.randn_like(x) * self.noise_std
        return x + noise

    def channel_mask(self, x):
        """
        Zero out a random subset of channels independently per sample.

        A binary mask of shape (B, 1, C) is drawn per call: each channel
        is independently set to zero with probability channel_mask_p.
        Broadcasting across T means the same channels are masked for the
        full time window of each sample, matching the failure mode of a
        truly disconnected electrode.

        Args:
            x (Tensor): shape (B, T, C)
        Returns:
            Tensor: shape (B, T, C), same device as input
        """
        B, T, C = x.shape
        # mask shape (B, 1, C) — broadcast over time dimension
        mask = torch.bernoulli(
            torch.full((B, 1, C), 1.0 - self.channel_mask_p, device=x.device)
        )
        return x * mask

    def amplitude_scale(self, x):
        """
        Multiply each sample by a scalar drawn uniformly from
        [amplitude_low, amplitude_high].

        One scale factor per sample (shape (B, 1, 1)) preserves the
        relative structure of the signal across time and channels, which
        is what changes between recording sessions due to electrode
        impedance and amplifier gain drift.

        Args:
            x (Tensor): shape (B, T, C)
        Returns:
            Tensor: shape (B, T, C), same device as input
        """
        B, T, C = x.shape
        scale = torch.empty(B, 1, 1, device=x.device).uniform_(
            self.amplitude_low, self.amplitude_high
        )
        return x * scale

    # ------------------------------------------------------------------
    # Composed augmentation
    # ------------------------------------------------------------------

    def random_augment(self, x):
        """
        Apply a random combination of the three augmentations.

        Each of the three transforms is applied independently with
        probability 0.5, so the expected number of active augmentations
        per call is 1.5. This matches standard SimCLR practice of
        composing a random subset of augmentations per view.

        The two calls to random_augment on the same batch during SimCLR
        training will produce different random masks and scales, making
        the two views genuinely independent perturbations.

        Args:
            x (Tensor): shape (B, T, C)
        Returns:
            Tensor: shape (B, T, C), augmented view on same device as input
        """
        if torch.rand(1).item() > 0.5:
            x = self.gaussian_noise(x)
        if torch.rand(1).item() > 0.5:
            x = self.channel_mask(x)
        if torch.rand(1).item() > 0.5:
            x = self.amplitude_scale(x)
        return x