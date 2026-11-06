"""Rehearsal, lookahead and observation-adaptation objectives."""
import torch
from torch import nn
import torch.nn.functional as F
from utils.geometry import lattice_upsample


class ZHSRLoss(nn.Module):
    """Stage weights 2**(k-K); reductions and weights retain the baseline defaults.

    `target` is an internally generated LR/synthetic target, never HR evaluation GT.
    Rehearsal does not include final observation-consistency loss.
    """
    def forward(self, model, outputs, observed, target=None, observation_consistency=False):
        total = observed.new_zeros(())
        upsampled = lattice_upsample(observed, model.scale)
        for index, output in enumerate(outputs):
            abundance = output['abundance']
            stage_loss = (0.1 * F.l1_loss(output['reconstruction'], upsampled)
                          + 0.1 * (abundance.sum(dim=1) - 1).square().mean()
                          + 0.1 * abundance.abs().mean())
            if target is not None:
                if output['sr'].shape != target.shape:
                    raise ValueError('Training target shape mismatch; no silent cropping')
                stage_loss = stage_loss + F.l1_loss(output['sr'], target)
            total = total + 0.5**(len(outputs) - 1 - index) * stage_loss
        endmembers = model.physics.endmembers.weight[:, :, 0, 0]
        if endmembers.shape[0] > 1:
            total = total + 0.4 * (endmembers[1:] - endmembers[:-1]).square().mean()
        if observation_consistency:
            # Use exactly the same learned operator B as the unfolding GS.
            degraded = model.physics.degradation(outputs[-1]['sr'])[..., ::model.scale, ::model.scale]
            total = total + 0.1 * F.l1_loss(degraded, observed)
        return total
