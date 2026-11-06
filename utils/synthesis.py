"""Synthetic targets made only from complete observed spectra."""
import math
import numpy as np
import torch

def multiscale_indices(shape, reference_size, source_pixels, rng):
    height, width = shape
    lower = 2.0  # In target pixels, independent of magnification.
    upper = max(lower, min(reference_size) / 3.0)
    margin = upper / math.sqrt(2.0)
    indices = np.full((height, width), -1, dtype=np.int64)
    remaining = height * width
    attempts = 0
    while remaining:
        attempts += 1
        if attempts > 1000000:
            raise RuntimeError('Multiscale dead-leaves coverage did not converge')
        # Each logarithmic size interval has equal probability; one angle for
        # an entire leaf, and one complete observed spectrum for all its pixels.
        a, b = np.exp(rng.uniform(math.log(lower), math.log(upper), size=2))
        angle = rng.uniform(0.0, math.pi)
        cosine, sine = math.cos(angle), math.sin(angle)
        cx = rng.uniform(-margin, width - 1 + margin)
        cy = rng.uniform(-margin, height - 1 + margin)
        ex = (abs(cosine) * a + abs(sine) * b) / 2
        ey = (abs(sine) * a + abs(cosine) * b) / 2
        x0, x1 = max(0, math.ceil(cx - ex)), min(width, math.floor(cx + ex) + 1)
        y0, y1 = max(0, math.ceil(cy - ey)), min(height, math.floor(cy + ey) + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dx, dy = xx - cx, yy - cy
        region = indices[y0:y1, x0:x1]
        visible = ((np.abs(cosine * dx + sine * dy) <= a / 2)
                   & (np.abs(-sine * dx + cosine * dy) <= b / 2)
                   & (region < 0))
        count = int(visible.sum())
        if count:
            region[visible] = rng.integers(source_pixels)
            remaining -= count
    return indices
