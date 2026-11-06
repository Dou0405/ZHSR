"""Upsampling on a declared sampling lattice, independent of image ground truth."""
import torch
import torch.nn.functional as F


def lattice_upsample(x, scale):
    """Input sample (i,j) belongs at output (scale*i,scale*j); periodic boundary."""
    scale = int(scale)
    if scale < 1:
        raise ValueError('scale must be positive')
    if scale == 1:
        return x
    h, w = x.shape[-2:]
    p = 2
    # index_select supports even very small inputs, unlike a large circular pad.
    iy = torch.arange(-p, h+p, device=x.device).remainder(h)
    ix = torch.arange(-p, w+p, device=x.device).remainder(w)
    padded = x.index_select(-2, iy).index_select(-1, ix)
    yy = torch.arange(h*scale, device=x.device, dtype=x.dtype)/scale+p
    xx = torch.arange(w*scale, device=x.device, dtype=x.dtype)/scale+p
    gy, gx = torch.meshgrid(2*yy/(h+2*p-1)-1, 2*xx/(w+2*p-1)-1, indexing='ij')
    grid = torch.stack([gx, gy], -1)[None].expand(x.shape[0], -1, -1, -1)
    return F.grid_sample(padded, grid, mode='bicubic', padding_mode='border', align_corners=True)
