"""DeepQN-Net modules: SP, NR/SJ, GS, QNS, AUM and curvature buffer.

All observations use the same stabilized MLMM. No kernel labels, GT, noise
levels, or dataset identities are accepted by these modules.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

EPS = 1e-8


def nonlinear_remixing(z, rho):
    """NR: MLMM on z <= 1, with its C1 tangent continuation for z > 1."""
    u = z.clamp(max=1.0)
    value = (1 - rho) * u / (1 - rho * u + EPS)
    return value + (z - 1).relu() * scattering_jacobian(u, rho)


def scattering_jacobian(z, rho):
    """SJ: exact derivative of nonlinear_remixing with respect to z."""
    return (1 - rho) * (1 + EPS) / (1 - rho * z.clamp(max=1.0) + EPS).square()


class ScatteringPredictor(nn.Module):
    """SP: spatial three-convolution predictor; rho starts at 0.05, rho < 0.5."""
    def __init__(self, bands):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(bands, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, math.log(0.05 / (0.5 - 0.05)))

    def forward(self, observation):
        return 0.5 * torch.sigmoid(self.net(observation))


class DilatedResidualBlock(nn.Module):
    """DRB in Fig. 4(b), with a depthwise dilated convolution."""
    def __init__(self, channels, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation,
                      dilation=dilation, groups=channels), nn.PReLU(),
            nn.Conv2d(channels, channels, 1), nn.PReLU(),
            nn.Conv2d(channels, channels, 1))

    def forward(self, x):
        return x + self.net(x)


class AbundancePrior(nn.Module):
    """Learned regularization gradient f_phi(A), Fig. 4(a)."""
    def __init__(self, rank):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(rank, 64, 3, padding=1), nn.PReLU(),
            DilatedResidualBlock(64, 1), DilatedResidualBlock(64, 2),
            DilatedResidualBlock(64, 4), nn.Conv2d(64, rank, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, abundance):
        if self.training and torch.is_grad_enabled():
            return checkpoint(self.net, abundance, use_reentrant=False)
        return self.net(abundance)


class EndmemberMixing(nn.Module):
    """f_E and its matched adjoint, sharing unrestricted learnable endmembers."""
    def __init__(self, bands, rank):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(bands, rank, 1, 1))

    def forward(self, x, adjoint=False):
        operation = F.conv_transpose2d if adjoint else F.conv2d
        return operation(x, self.weight)


class SpatialDegradation(nn.Module):
    """f_B: one learned 21x21 kernel shared by all bands; periodic boundary."""
    def __init__(self):
        super().__init__()
        axis = torch.linspace(-10, 10, 21)
        xx, yy = torch.meshgrid(axis, axis, indexing='xy')
        kernel = torch.exp(-(xx.square() + yy.square()) / (2 * 0.7**2))
        self.weight = nn.Parameter((kernel / kernel.sum())[None, None])

    def forward(self, x, adjoint=False):
        h, w = x.shape[-2:]
        iy = torch.arange(-10, h + 10, device=x.device).remainder(h)
        ix = torch.arange(-10, w + 10, device=x.device).remainder(w)
        padded = x.index_select(-2, iy).index_select(-1, ix)
        kernel = self.weight.flip(-2, -1) if adjoint else self.weight
        return F.conv2d(padded, kernel.expand(x.shape[1], -1, -1, -1), groups=x.shape[1])


class UMSRModel(nn.Module):
    """UnMixing-inspired SR: shared endmembers, NR, degradation and data gradient."""
    def __init__(self, bands, rank):
        super().__init__()
        self.endmembers = EndmemberMixing(bands, rank)
        self.degradation = SpatialDegradation()

    def remix(self, abundance, rho):
        return nonlinear_remixing(self.endmembers(abundance), rho)

    def forward(self, abundance, rho):
        return self.degradation(self.remix(abundance, rho))

    def data_gradient(self, abundance, upsampled_observation, rho):
        residual = self(abundance, rho) - upsampled_observation
        spatial = self.degradation(residual, adjoint=True)
        spectral = spatial * scattering_jacobian(self.endmembers(abundance), rho)
        return self.endmembers(spectral, adjoint=True)


class GradientSolver(nn.Module):
    """GS = analytical data gradient + learned abundance-prior gradient."""
    def __init__(self, rank):
        super().__init__()
        self.prior = AbundancePrior(rank)

    def forward(self, abundance, observation, rho, physics):
        return physics.data_gradient(abundance, observation, rho) + self.prior(abundance)


def batch_dot(x, y):
    return (x * y).sum(dim=(1, 2, 3), keepdim=True)


class QuasiNewtonSolver(nn.Module):
    """QNS: L-BFGS two-loop recursion, without an explicit Hessian."""
    def __init__(self):
        super().__init__()
        self.initial_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, gradient, history):
        q, coefficients = gradient, []
        for s, y in reversed(history):
            inverse_curvature = 1 / (batch_dot(y, s) + 1e-6)
            alpha = inverse_curvature * batch_dot(s, q)
            coefficients.append((inverse_curvature, alpha))
            q = q - alpha * y
        if history:
            s, y = history[-1]
            r = batch_dot(s, y) / (batch_dot(y, y) + 1e-6) * q
        else:
            r = self.initial_scale * q
        for (s, y), (inverse_curvature, alpha) in zip(history, reversed(coefficients)):
            beta = inverse_curvature * batch_dot(y, r)
            r = r + s * (alpha - beta)
        return -r


class AbundanceUpdateModule(nn.Module):
    """AUM: free abundance update with a learnable scalar step size."""
    def __init__(self):
        super().__init__()
        self.step_size = nn.Parameter(torch.tensor(0.5))

    def forward(self, abundance, direction):
        return abundance + self.step_size * direction


class CurvatureBuffer:
    """Per-forward, limited-memory history. Retain gradients through its pairs."""
    def __init__(self, memory_size):
        self.memory_size, self.pairs = memory_size, []

    def update(self, previous_abundance, abundance, previous_gradient, gradient):
        self.pairs.append((abundance - previous_abundance, gradient - previous_gradient))
        self.pairs = self.pairs[-self.memory_size:]


class BlindSRStage(nn.Module):
    """One BS stage containing QNS, AUM and GS (Fig. 3)."""
    def __init__(self, rank):
        super().__init__()
        self.gradient_solver = GradientSolver(rank)
        self.abundance_update = AbundanceUpdateModule()
        self.quasi_newton_solver = QuasiNewtonSolver()

    def forward(self, abundance, gradient, buffer, observation, rho, physics):
        direction = self.quasi_newton_solver(gradient, buffer.pairs)
        updated = self.abundance_update(abundance, direction)
        next_gradient = self.gradient_solver(updated, observation, rho, physics)
        buffer.update(abundance, updated, gradient, next_gradient)
        return updated, next_gradient


class AbundanceRefinement(nn.Module):
    """Retained multiscale implementation detail: additive final abundance residual."""
    def __init__(self, bands, rank):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3 * bands + rank, 48, 1), nn.PReLU(48))
        self.branches = nn.ModuleList([nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=d, dilation=d, groups=48),
            nn.Conv2d(48, 24, 1), nn.PReLU(24)) for d in (1, 2, 4)])
        self.fusion = nn.Sequential(nn.Conv2d(72, 48, 1), nn.PReLU(48),
                                    nn.Conv2d(48, 32, 3, padding=1), nn.PReLU(32))
        self.head = nn.Conv2d(32, rank, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, observation, physical_sr, abundance):
        feature = self.stem(torch.cat([observation, physical_sr,
                                       physical_sr - observation, abundance], dim=1))
        return self.head(self.fusion(torch.cat([branch(feature) for branch in self.branches], dim=1)))


class ReconstructionModule(nn.Module):
    """Final Reconstruction block: multiscale abundance refinement followed by NR.

    This groups the existing operations without adding layers or parameters.
    The shared UMSR physics is passed in, not copied or registered a second time.
    """
    def __init__(self, bands, rank):
        super().__init__()
        self.refinement = AbundanceRefinement(bands, rank)

    def forward(self, observation, physical_sr, abundance, rho, physics):
        refined_abundance = abundance + self.refinement(observation, physical_sr, abundance)
        return refined_abundance, physics.remix(refined_abundance, rho)
