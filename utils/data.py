"""Explicit cube I/O. No field guessing, silent cropping or per-image normalization."""
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
import scipy.ndimage
from utils.synthesis import multiscale_indices


def load_cube(path, key='data', layout='CHW'):
    path = Path(path)
    if path.suffix.lower() == '.npy':
        array = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == '.mat':
        values = loadmat(path, variable_names=[key])
        if key not in values:
            raise ValueError(f'{path.name} has no field {key!r}')
        array = values[key]
    else:
        raise ValueError('Supported formats are MATLAB v5/v7 .mat and .npy')
    if array.ndim != 3 or min(array.shape) < 1 or array.dtype.kind != 'f':
        raise ValueError('Expected a floating-point 3D cube, already normalized to the intended scale')
    if layout == 'HWC':
        array = array.transpose(2, 0, 1)
    elif layout != 'CHW':
        raise ValueError('layout must be CHW or HWC')
    array = np.array(array, dtype=np.float32, order='C', copy=True)
    if not np.isfinite(array).all():
        raise ValueError('Cube contains NaN/Inf or overflows float32')
    # Signed/out-of-range observed values are valid, e.g. measured noisy data.
    return torch.from_numpy(array)


def validate_reference(observation, reference, scale):
    expected = (observation.shape[0], observation.shape[1]*scale, observation.shape[2]*scale)
    if tuple(reference.shape) != expected:
        raise ValueError(f'GT must have shape {expected}; got {tuple(reference.shape)}. No automatic crop/resize.')


class RandomAnisotropicDegradation:
    """Independent 21x21 Gaussian, circular mathematical convolution, lattice sampling."""
    def __init__(self, scale, workers=8):
        if scale not in (2, 4, 8) or not 1 <= workers <= 64:
            raise ValueError('Invalid scale or worker count')
        self.scale = scale
        self.pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None

    @staticmethod
    def sample_kernel(rng):
        sigma = rng.uniform(0.2, 4.0, size=2)
        while max(sigma) / min(sigma) < 1.1:
            sigma = rng.uniform(0.2, 4.0, size=2)
        angle = rng.uniform(0.0, np.pi)
        yy, xx = np.meshgrid(np.arange(-10, 11), np.arange(-10, 11), indexing='ij')
        u, v = np.cos(angle)*xx + np.sin(angle)*yy, -np.sin(angle)*xx + np.cos(angle)*yy
        kernel = np.exp(-0.5 * ((u/sigma[0])**2 + (v/sigma[1])**2))
        return (kernel / kernel.sum()).astype(np.float32)

    def __call__(self, target, rng):
        kernel = self.sample_kernel(rng)
        array = target.numpy()
        if any(v % self.scale for v in array.shape[-2:]):
            raise ValueError('Target size must be divisible by scale')
        output = np.empty((array.shape[0], array.shape[1]//self.scale, array.shape[2]//self.scale), dtype=np.float32)
        def channel(index):
            output[index] = scipy.ndimage.convolve(array[index], kernel, mode='wrap')[::self.scale, ::self.scale]
        if self.pool:
            list(self.pool.map(channel, range(array.shape[0])))
        else:
            for index in range(array.shape[0]):
                channel(index)
        return torch.from_numpy(output)

    def close(self):
        if self.pool:
            self.pool.shutdown(wait=True)


class InternalPairs(Dataset):
    """12 observed crops + 12 synthetic targets per epoch for every observation.

    No scene label, denoising or additional input noise. Each synthetic pixel
    inherits one complete observed spectrum. Sampling streams are independent
    for target generation and unknown random blur; no true PSF or HR is read.
    """
    def __init__(self, observation, scale, patch=None, seed=101, workers=8):
        if observation.ndim != 3 or observation.device.type != 'cpu':
            raise ValueError('Pair source must be a CPU CHW observation')
        self.source, self.scale, self.seed, self.epoch = observation, scale, seed, 0
        h, w = observation.shape[-2:]
        available = min(h, w)//scale*scale
        patch = min(512//scale, available) if patch is None else patch
        if patch < scale or patch % scale or patch > available:
            raise ValueError(f'patch must be divisible by {scale}, between {scale} and {available}')
        self.patch = patch
        self.degrade = RandomAnisotropicDegradation(scale, workers)

    def __len__(self):
        return 24

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index, 0]))
        blur_rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index, 1]))
        c, h, w = self.source.shape
        patch, scale = self.patch, self.scale
        if index < 12:
            h0, w0 = h//scale*scale, w//scale*scale
            y = int(rng.integers((h0-patch)//scale+1))
            x = int(rng.integers((w0-patch)//scale+1))
            target = self.source[:, y*scale:y*scale+patch, x*scale:x*scale+patch].contiguous()
            # Blur full observation before cropping, preserving the original context.
            degraded = self.degrade(self.source[:, :h0, :w0], blur_rng)
            return degraded[:, y:y+patch//scale, x:x+patch//scale].contiguous(), target
        indices = multiscale_indices((patch, patch), (h*scale, w*scale), h*w, rng)
        target = self.source.flatten(1)[:, torch.from_numpy(indices.ravel())].reshape(c, patch, patch).contiguous()
        return self.degrade(target, blur_rng), target

    def close(self):
        self.degrade.close()
