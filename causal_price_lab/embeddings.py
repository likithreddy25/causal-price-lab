"""Image -> vector embedding, behind one interface with two
implementations — the same Backend-interface pattern used for AgentGate's
model calls (internal/backend/{mock,anthropic}.go): a real implementation
that needs external infrastructure, and a deterministic one that doesn't,
so the rest of the pipeline never has to know or care which one produced
its embeddings.

Why embeddings matter *causally*, not just as a "multimodal" checkbox:
product photography quality/style plausibly correlates with both price
(premium products get premium photos) and outcome (better photos drive
more engagement/reviews) — a classic confounder. Leaving it out of the
propensity model would leave that confounding uncorrected. Every user of
Embedder should treat its output as a covariate to control for, not as a
feature being "added for AI credit."
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    """Turns a batch of images into fixed-size embedding vectors."""

    dim: int

    @abstractmethod
    def embed_images(self, images: list) -> np.ndarray:
        """Returns an (n, dim) float array, one row per input image."""
        raise NotImplementedError


class SyntheticEmbedder(Embedder):
    """A deterministic, seeded, network-free stand-in for a real vision
    embedder. It does not look at pixels at all — it is only ever used in
    synthetic.py, where the "image" passed in is actually a latent
    confound value the synthetic generator already knows, and this class's
    only job is to turn that scalar into a `dim`-length vector plus noise,
    the same shape a real embedding would have. This exists so the PSM
    pipeline's *consumption* of image-embedding covariates can be tested
    end to end without a GPU, model weights, or network access — see
    tests/test_psm_recovers_true_effect.py.
    """

    def __init__(self, dim: int = 16, noise_std: float = 0.15, seed: int = 0):
        self.dim = dim
        self.noise_std = noise_std
        self._rng = np.random.default_rng(seed)
        # A fixed random direction the latent confound gets projected onto,
        # so the confound is recoverable from the embedding via a linear
        # combination, similar to how a real embedding's first principal
        # component often captures a dominant visual factor.
        self._direction = self._rng.normal(size=dim)
        self._direction /= np.linalg.norm(self._direction)

    def embed_images(self, images: list) -> np.ndarray:
        latent = np.asarray(images, dtype=float).reshape(-1, 1)
        noise = self._rng.normal(scale=self.noise_std, size=(len(images), self.dim))
        return latent * self._direction[None, :] + noise


class CLIPEmbedder(Embedder):
    """Real image embeddings via OpenAI's CLIP (through `transformers`,
    already a project dependency — no separate `open_clip` install
    needed). Requires network access to download model weights from
    huggingface.co on first use, which the sandbox that wrote this file
    does not have — see real_pipeline/ and the README's verification
    status for what that means in practice.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str | None = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.dim = self.model.config.projection_dim

    def embed_images(self, images: list) -> np.ndarray:
        import torch

        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.get_image_features(**inputs)
        # Handle both direct tensor and BaseModelOutputWithPooling object
        if hasattr(output, 'pooler_output'):
            features = output.pooler_output
        elif hasattr(output, 'cpu'):
            features = output
        else:
            # If it's something else, try to get the last hidden state
            features = output.last_hidden_state if hasattr(output, 'last_hidden_state') else output
        return features.cpu().numpy()
