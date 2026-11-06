"""Deep Quasi-Newton Network (DeepQN-Net) for zero-shot blind HSI SR."""
import torch
from torch import nn
from modules import (UMSRModel, ScatteringPredictor, GradientSolver, BlindSRStage,
                     CurvatureBuffer, ReconstructionModule)
from utils.geometry import lattice_upsample
from utils.initialization import extract_spa


class DeepQNNet(nn.Module):
    def __init__(self, bands, scale=2, rank=5, stages=3, memory_size=3):
        super().__init__()
        if min(bands, rank, stages, memory_size) < 1 or rank > bands or scale not in (2, 4, 8):
            raise ValueError('Require positive dimensions, rank <= bands and scale in {2,4,8}')
        self.scale, self.memory_size = scale, memory_size
        self.physics = UMSRModel(bands, rank)
        self.abundance_encoder = nn.Sequential(nn.Conv2d(bands, rank, 3, padding=1), nn.ReLU())
        self.initial_gradient_solver = GradientSolver(rank)
        self.stages = nn.ModuleList([BlindSRStage(rank) for _ in range(stages)])
        self.reconstruction = ReconstructionModule(bands, rank)
        self.scattering_predictor = ScatteringPredictor(bands)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Accept weights from the initial public preparation, before grouping
        # refinement under Reconstruction. Tensor names/shapes otherwise match.
        old_prefix = prefix + 'refinement.'
        new_prefix = prefix + 'reconstruction.refinement.'
        for key in list(state_dict):
            if key.startswith(old_prefix):
                renamed = new_prefix + key[len(old_prefix):]
                if renamed in state_dict:
                    raise RuntimeError('Ambiguous checkpoint: both old and new refinement keys')
                state_dict[renamed] = state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @torch.no_grad()
    def initialize_endmembers(self, observed_lr):
        """SPA uses the single observed cube only; there is no GT/PSF argument."""
        if observed_lr.ndim != 4 or observed_lr.shape[0] != 1:
            raise ValueError('SPA expects a single NCHW observation')
        rank = self.physics.endmembers.weight.shape[1]
        if observed_lr.shape[-2] * observed_lr.shape[-1] < rank:
            raise ValueError('Not enough observed pixels for SPA')
        spectra, _ = extract_spa(observed_lr[0].flatten(1), rank)
        self.physics.endmembers.weight.copy_(spectra[:, :, None, None])

    def forward(self, observed_lr):
        observation = lattice_upsample(observed_lr, self.scale)
        abundance = self.abundance_encoder(observation)
        rho = self.scattering_predictor(observation)
        gradient = self.initial_gradient_solver(abundance, observation, rho, self.physics)
        buffer = CurvatureBuffer(self.memory_size)
        outputs = []
        for index, stage in enumerate(self.stages):
            abundance, gradient = stage(abundance, gradient, buffer, observation, rho, self.physics)
            physical_sr = self.physics.remix(abundance, rho)
            final_abundance = abundance
            if index == len(self.stages) - 1:
                final_abundance, sr = self.reconstruction(
                    observation, physical_sr, abundance, rho, self.physics)
            else:
                sr = self.physics.remix(final_abundance, rho)
            outputs.append({'sr': sr,
                            'reconstruction': self.physics.degradation(physical_sr),
                            'abundance': abundance, 'refined_abundance': final_abundance,
                            'rho': rho})
        return outputs
