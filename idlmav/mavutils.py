from typing import Mapping
from contextlib import contextmanager
import plotly.io as pio
import warnings
import torch

available_renderers = list(pio.renderers)

@contextmanager
def plotly_renderer(renderer):
    default_renderer = pio.renderers.default
    if renderer:
        if renderer in available_renderers:
            pio.renderers.default = renderer
        else:
            warnings.warn(f'Plotly renderer "{renderer}" is not present in the queried list of available renderers')
    yield
    # Restore the default
    pio.renderers.default = default_renderer

def to_device(x, device):
    # From fastai course https://course.fast.ai/Lessons/lesson15.html around 0:36:00
    if isinstance(x, torch.Tensor): return x.to(device)
    if isinstance(x, Mapping): return {k:v.to(device) for k,v in x.items()}
    return type(x)(to_device(o, device) for o in x)
