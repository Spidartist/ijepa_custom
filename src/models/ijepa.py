import torch.nn as nn
import torch
import torch.nn.functional as F
from src.masks.utils import apply_masks
from src.utils.tensors import repeat_interleave_batch

class IJEPA(nn.Module):
    def __init__(self, context_encoder, predictor, target_encoder):
        super(IJEPA, self).__init__()
        self.context_encoder = context_encoder
        self.predictor = predictor
        self.target_encoder = target_encoder
    
        for p in self.target_encoder.parameters():
            p.requires_grad = False
    def forward_target(self, imgs, masks_pred, masks_enc):
        with torch.no_grad():
            h = self.target_encoder(imgs)
            h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
            B = len(h)

            # -- create targets (masked regions of h)
            h = apply_masks(h, masks_pred)
            h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
            return h
    
    def forward_context(self, imgs, masks_pred, masks_enc):
        z = self.context_encoder(imgs, masks_enc)
        z = self.predictor(z, masks_enc, masks_pred)
        return z
    
    def forward(self, imgs, masks_pred, masks_enc):
        z = self.forward_context(imgs, masks_pred, masks_enc)
        h = self.forward_target(imgs, masks_pred, masks_enc)
        return z, h