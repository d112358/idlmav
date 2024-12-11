from contextlib import contextmanager
import plotly.io as pio
import warnings

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
