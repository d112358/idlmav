# IDLMAV
Interactive deep learning model architecture visualization (IDLMAV) is a tool that creates interactive visualizations of model architectures for display in Jupyter notebooks.
* It does not require a successful forward pass: it can also visualize partial models
* It produces three outputs to allow a trade-off between portability and interactivity
  - A portable figure that works on most environments and displays correctly without the need of a running backend/kernel, e.g. in [nbviewer](https://nbviewer.org/) ([example](https://nbviewer.org/github/d112358/idlmav/blob/main/examples.ipynb)) or [nbsanity](https://nbsanity.com/) ([example](https://nbsanity.com/d112358/idlmav/blob/main/examples.ipynb))
  - An interactive widget with synchronized scrolling and interactions between sub-plots
  - Export to a static HTML file

# Use cases
* Incrementally designing a model and viewing activations, parameter counts and FLOPS "so far" before the whole model has been defined
* Documenting a model in a notebook and generating the architecture in such a way that it is viewable without a running kernel, e.g. in [nbviewer](https://nbviewer.org/) ([example](https://nbviewer.org/github/d112358/idlmav/blob/main/examples.ipynb)) or [nbsanity](https://nbsanity.com/) ([example](https://nbsanity.com/d112358/idlmav/blob/main/examples.ipynb))
* Visualizing 3rd party models after importing them into a notebook
* Finding hotspots (parameters or FLOPS) in a model for optimization purposes

# Static HTML examples
These have limited interactivity and synchronization between panels compared to the interactive widgets (see below), but they provide good examples of how models are visualized.

| Model | Basic | Verbose | Basic, scrolling | Verbose, scrolling |
| ----- | ----- | ------- | ---------------- | ------------------ |
| ResNet18 | [View](https://d112358.github.io/idlmav/export_examples/resnet18.html) | [View](https://d112358.github.io/idlmav/export_examples/resnet18_verbose.html) | [View](https://d112358.github.io/idlmav/export_examples/resnet18_slider.html) | [View](https://d112358.github.io/idlmav/export_examples/resnet18_verbose_slider.html)|
| ResNet34 | [View](https://d112358.github.io/idlmav/export_examples/resnet34.html) | [View](https://d112358.github.io/idlmav/export_examples/resnet34_verbose.html) | [View](https://d112358.github.io/idlmav/export_examples/resnet34_slider.html) | [View](https://d112358.github.io/idlmav/export_examples/resnet34_verbose_slider.html)|
| ConvNeXt small | [View](https://d112358.github.io/idlmav/export_examples/convnext_small.html) | [View](https://d112358.github.io/idlmav/export_examples/convnext_small_verbose.html) | [View](https://d112358.github.io/idlmav/export_examples/convnext_small_slider.html) | [View](https://d112358.github.io/idlmav/export_examples/convnext_small_verbose_slider.html)|
| ViT B/16 | [View](https://d112358.github.io/idlmav/export_examples/vit_b_16_small.html) | [View](https://d112358.github.io/idlmav/export_examples/vit_b_16_small_verbose.html) | [View](https://d112358.github.io/idlmav/export_examples/vit_b_16_small_slider.html) | [View](https://d112358.github.io/idlmav/export_examples/vit_b_16_small_verbose_slider.html)|
| HR-Net W18 | [View](https://d112358.github.io/idlmav/export_examples/hrnet_w18.html) | [View](https://d112358.github.io/idlmav/export_examples/hrnet_w18_verbose.html) | [View](https://d112358.github.io/idlmav/export_examples/hrnet_w18_slider.html) | [View](https://d112358.github.io/idlmav/export_examples/hrnet_w18_verbose_slider.html)|
| YOLOv11 Nano | [View](https://d112358.github.io/idlmav/export_examples/yolov11n.html) | [View](https://d112358.github.io/idlmav/export_examples/yolov11n_verbose.html) | [View](https://d112358.github.io/idlmav/export_examples/yolov11n_slider.html) | [View](https://d112358.github.io/idlmav/export_examples/yolov11n_verbose_slider.html)|
| BLIP vision model | [View](https://d112358.github.io/idlmav/export_examples/blip_visionmodel.html) | [View](https://d112358.github.io/idlmav/export_examples/blip_visionmodel_verbose.html) | [View](https://d112358.github.io/idlmav/export_examples/blip_visionmodel_slider.html) | [View](https://d112358.github.io/idlmav/export_examples/blip_visionmodel_verbose_slider.html)|
| Whisper-tiny | [View](https://d112358.github.io/idlmav/export_examples/whisper_tiny.html) | [View](https://d112358.github.io/idlmav/export_examples/whisper_tiny_verbose.html) | [View](https://d112358.github.io/idlmav/export_examples/whisper_tiny_slider.html) | [View](https://d112358.github.io/idlmav/export_examples/whisper_tiny_verbose_slider.html)|
| BERT mini | [View](https://d112358.github.io/idlmav/export_examples/bert_mini.html) | [View](https://d112358.github.io/idlmav/export_examples/bert_mini_verbose.html) | [View](https://d112358.github.io/idlmav/export_examples/bert_mini_slider.html) | [View](https://d112358.github.io/idlmav/export_examples/bert_mini_verbose_slider.html)|
| ModernBERT base | [View](https://d112358.github.io/idlmav/export_examples/ModernBERT.html) | [View](https://d112358.github.io/idlmav/export_examples/ModernBERT_verbose.html) | [View](https://d112358.github.io/idlmav/export_examples/ModernBERT_slider.html) | [View](https://d112358.github.io/idlmav/export_examples/ModernBERT_verbose_slider.html)|


# Installation
## Using Plotly 5
Since version 6, `plotly` are basing their `go.FigureWidget` object on `anywidget`. The interactive widgets in `idlmav` are based on `go.FigureWidget`. `idlmav` has not yet been tested extensively with `plotly` version 6 and/or `anywidget`. Use the installation steps below to use `idlmav` with `plotly` 5
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install "plotly>=5,<6"
pip install idlmav
```

## Using Plotly 6
To use the latest version of `plotly`, `anywidget` must be installed separately.
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install anywidget
pip install idlmav
```

# Usage examples
## Preparation
Run these steps before running 
```python
import torch, torchvision
from idlmav import MAV
model = torchvision.models.resnet18()
x = torch.randn(16,3,160,160)
mav = MAV(model, x, device='cpu')
```

## Portable figure
* Based on [plotly.graph_objects.Figure](https://plotly.com/python/creating-and-updating-figures/#figures-as-graph-objects)
* No dependency on `ipywidgets` or `plotly.graph_objects.FigureWidget` for portability reasons
* Displays correctly without the need of a running backend/kernel, e.g. in [nbviewer](https://nbviewer.org/) ([example](https://nbviewer.org/github/d112358/idlmav/blob/main/examples.ipynb)) or [nbsanity](https://nbsanity.com/) ([example](https://nbsanity.com/d112358/idlmav/blob/main/examples.ipynb))
* Interactions limited to hover, pan and zoom, slider and dropdown menu provided by Plotly
* No synchronization between graph and table
```python
mav.show_figure()
```
![Portable figure](https://github.com/d112358/idlmav/raw/main/images/portable_figure.png)

## Interactive widget
* Based on [ipywidgets](https://ipywidgets.readthedocs.io/en/latest/) and [plotly.graph_objects.FigureWidget](https://plotly.com/python/figurewidget/)
* Synchronizaton between slider, overview panel, main graph and table
  - Includes responsiveness of other components to plotly's built-in pan and zoom actions
* Clicking a node in the main graph highlights it in the table
* Limited portability expected to fluctuate over time on different environments
```python
mav.show_widget(add_slider=True, add_overview=True)
```
![Interactive widget](https://github.com/d112358/idlmav/raw/main/images/interactive_widget.png)

## HTML export
* Most portable option
* Exports the same portable figure shown above to a standalone HTML file
* The `export_for_offline_use` parameter specifies how to include the plotly dependency in the exported HTML
  - `False` (default): The exported HTML is small, but requires a working internet connection to display correctly
  - `True`: The exported HTML is around 4MB in size and displays correctly without a working internet connection
```python
mav.export_static_html('resnet18.html', export_for_offline_use=False)
```

## Specifying colors
* Palettes from plotly [discrete color sequences](https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express) can be specified by name
* User-defined palettes can be specified as a list of `'#RRGGBB'` formatted strings
* The key to `fixed_color_map` may be a string in the **Operation** column or a category as listed [here](https://pytorch.org/docs/stable/nn.html)
```python
mav.show_figure(
    palette='Vivid',
    avoid_palette_idxs=set([10]),
    fixed_color_map={'Convolution':7, 'add()':0, 'nn.MaxPool2d':5}
)
```
![Specifying colors](https://github.com/d112358/idlmav/raw/main/images/specifying_colors.png)

## Adding and removing panels
* This could help with portability or user experience on some environments, e.g.
  - On Colab the slider gets more in the way rather than adding value
  - Wide models are sometimes easier to navigate without the table
  - The custom JS used for table synchronization may not be supported everywhere
```python
mav.show_widget(add_overview=False, add_slider=False, add_table=False)    
```
![Adding and removing panels](https://github.com/d112358/idlmav/raw/main/images/removing_panels.png)

## Modifying merging behaviour
* `merge_threshold<0` does not perform any merging
* `merge_threshold==0` only merges nodes that have zero parameters
* `merge_threshold` between 0 and 1 sorts nodes from the smallest to the largest by number of parameters and merges from the smallest node until just before the combined parameter count of merged nodes exceed the specified fraction of the total parameter count
* The following nodes are never merged:
  - Input and output nodes to the entire network
  - Nodes with multiple input connections
  - Nodes for which the input node has multiple output connections
* The default `merge_threshold` value normally results in nodes without parameters as well as normalization modules being merged
```python
mav = MAV(model, x, device='cpu', merge_threshold=-1)
mav.show_figure(
    palette='Vivid',
    avoid_palette_idxs=set([10]),
    fixed_color_map={'Convolution':7, 'add()':0, 'nn.MaxPool2d':5}
)
```
![Modifying merging behaviour](https://github.com/d112358/idlmav/raw/main/images/modifying_merging_behaviour.png)

## Calling internal components directly
* For users that wish to replace or augment one or more components
* A typical example would be replacing or subclassing the renderer to work on a specific environment
```python
from idlmav import MavTracer, merge_graph_nodes, layout_graph_nodes, color_graph_nodes, WidgetRenderer
from IPython.display import display

tracer = MavTracer(model, x, device='cpu')
merge_graph_nodes(tracer.g)
layout_graph_nodes(tracer.g)
color_graph_nodes(tracer.g)
renderer = WidgetRenderer(tracer.g)
display(renderer.render())
```
![Calling internal components directly](https://github.com/d112358/idlmav/raw/main/images/calling_internal_components_directly.png)

## Reducing notebook file size
* On some environments, plotly will include the entire plotly library (~ 4MB) in the notebook DOM for portable figures (`go.Figure`)
* This is not the case for interactive widgets (`go.FigureWidget`) where the plotly library is served from the backend
* Using a custom plotly renderer can also avoid this for `go.Figure`, importing plotly via a CDN instead
* Custom plotly renderers are made available in `idlmav` via a context manager:
  ```python
  from idlmav import plotly_renderer 
  with plotly_renderer('notebook_connected'):
      mav.show_figure()
  ```
* Available custom plotly renderers may be listed as follows:
  ```python
  import plotly.io as pio
  list(pio.renderers)
  ```
* It is best to experiment with different renderers for your environment. From personal experience, the following may be good starting points:
  * `notebook_connected` or `vscode` with Plotly 5 on VSCode
  * `vscode` with Plotly 6 on VSCode
  * `iframe` with Plotly 5 on Kaggle

# Features
* Works on incomplete models and models without a successful forward pass
* Can provide a portable figure with basic interactivity that does not require a running kernel
* Can provide an interactive widget with synchronization between panels and limited portability
* Customizable color palette and node or category color mappings
* Customizable node merging behaviour
* Interactions (portable figure)
  - Hover over modules to see activation sizes, number of parameters and FLOPS
  - Pan and zoom provided by Plotly (not synchronized)
  - Scrollable table (not synchronized)
  - Horizontal slider provided by Plotly (not synchronized)
  - Overview window showing full model (only synchronized to slider)
  - Dropdown menu to select node coloring and sizing criteria
* Interactions (interactive widget)
  - Hover over modules to see activation sizes, number of parameters and FLOPS
  - Synchronized scrolling between table and graph
  - Clicking on a module highlights that module in the table
  - Clickable overview window showing full model
  - Range slider from ipywidgets with synchronized pan and zoom functionality
  - Table and sliders synchronize with Plotly's built-in pan and zoom functionality
  - Dropdown menu to select node coloring and sizing criteria

# Limitations
* Inherited [limitations of symbolic tracing](https://pytorch.org/docs/stable/fx.html#limitations-of-symbolic-tracing) from torch.fx
  - Models with dynamic control flow can only be traced using `torch.compile`
  - Models containing non-torch functions can only be traced using `torch.compile` and only up to the non-torch function
* Inherited from `torch.compile`
  - In models parsed with `torch.compile`, classes are flattened into functions and learnable parameters are passed as additional inputs
* Inherited from ipywidgets:
  - Interactive widgets require a running kernel to dynamically create DOM elements
* Inherited from plotly
  - Portable figures can only support a horizontal slider
  - On portable figures, overview panels synchronize only to the slider, not to Plotly built-in pan & zoom controls
* Environment-specific limitations
  - Kaggle recently (Dec 2024) seemed to have trouble displaying `go.FigureWidget`, so only the portable figure is available there

# Planned updates
* Make the primary direction (down/right/up/left) configurable
* Allow the user to specify a latent node at which the graph changes direction (e.g. for autoencoder / UNet architectures)

# Contributing
Reports of any issues encountered as most welcome! 
Please provide reproducible example code and a brief description of your environment to simplify the process of reproducing the issue and verifying fixes

Please also make issues easy to categorize by being specific about the category they belong to:
* An error occurred during parsing, layout or MAV object instantiation
* The parsing, layout or MAV object instantiation step took forever to execute
* An error occurred during rendering
* The rendered graph is a poor / inaccurate representation of the model

Any contributions are also welcome and contributions in the following categories will be especially appreciated!
* Custom renderers to improve the user experience on different platforms / environments
* Unit tests

The development environment is described in [setup_vscode_wsl.ipynb](https://github.com/d112358/idlmav/blob/main/environments/setup_vscode_wsl.ipynb)
* This should be easy to get going in native Linux as well, just skipping the WSL parts
* The build is described in and executed from [build_steps.ipynb](https://github.com/d112358/idlmav/blob/main/environments/build_steps.ipynb)
* Unit tests still need to be developed. At the moment, the following notebooks are used for manual / visual testing:
  - [02_test_layout.ipynb](https://github.com/d112358/idlmav/blob/main/nbs/02_test_layout.ipynb)  
  - [06_test_rendering.ipynb](https://github.com/d112358/idlmav/blob/main/nbs/06_test_rendering.ipynb)  
  - [10_test_tracing.ipynb](https://github.com/d112358/idlmav/blob/main/nbs/10_test_tracing.ipynb)  
  - [12_test_idlmav.ipynb](https://github.com/d112358/idlmav/blob/main/nbs/12_test_idlmav.ipynb)
  - [20_test_export_misc_models.ipynb](https://github.com/d112358/idlmav/blob/main/nbs/20_test_export_misc_models.ipynb)

# License
This repository is released under the MIT license. See [LICENSE](https://github.com/d112358/idlmav/blob/main/LICENSE) for additional details.
