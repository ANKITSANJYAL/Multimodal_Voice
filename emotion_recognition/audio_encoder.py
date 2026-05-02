"""
Wav2Vec2 audio encoder.

Key choices
-----------
* CNN feature extractor is frozen by default (saves ~6 M params of gradient
  memory while preserving the learned spectral filters).
* SpecAugment is enabled via config so it fires automatically in model.train()
  — no extra code needed in the training loop.
* Mean pooling over time frames is robust and avoids the complexity of
  frame-level attention masks.
"""
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class AudioEncoder(nn.Module):
    def __init__(self, model_name: str = "facebook/wav2vec2-base", freeze_feature_extractor: bool = True):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        self.embed_dim: int = self.wav2vec2.config.hidden_size  # 768 for -base

        # SpecAugment fires during model.train() — covers time & freq masking
        # on the CNN output, which is the right place for Wav2Vec2.
        self.wav2vec2.config.apply_spec_augment = True
        self.wav2vec2.config.mask_time_prob = 0.075   # 7.5 % of frames masked
        self.wav2vec2.config.mask_feature_prob = 0.004

        if freeze_feature_extractor:
            # Freeze the 7-layer CNN stack; keep transformer blocks trainable.
            self.wav2vec2.feature_extractor._freeze_parameters()

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        input_values: (B, T)  raw 16 kHz waveform
        returns:      (B, 768) mean-pooled hidden states
        """
        out = self.wav2vec2(input_values=input_values)
        return out.last_hidden_state.mean(dim=1)
