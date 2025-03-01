from .mavoptions import RenderOptions
from .mavtypes import MavNode, MavConnection, MavGraph
from typing import Dict, List, Tuple, Union, Set, overload
import plotly.colors as pc
import warnings

class MavColorer():
    """
    Class that performs the coloring step

    All processing is performed upon instantiation.

    This class is not used after this, so users are encouraged to access
    it via `color_graph_nodes`, except if sub-classed.
    """
    def __init__(self, graph:MavGraph, opts:RenderOptions=RenderOptions(), **kwargs):
        """
        Instantiates a `MavColorer` class and performs the coloring step.

        See `color_graph_nodes` for a description of arguments
        """
        for k,v in kwargs.items(): opts.__setattr__(k,v)
        self.g = graph

        self.specified_palette = opts.palette
        self.palette: List[str] = []                 # Updated in `update_palette`
        self.avoid_palette_idxs: Set[int] = set([])  # Updated in `update_palette`
        self.op_cat_map: Dict[str, str] = {}         # Updated in `update_op_cat_mappings`
        self.op_cat_suffix_map: Dict[str, str] = {}  # Updated in `update_op_cat_mappings`
        self.known_op_list: List[str] = []           # Updated in `update_op_cat_mappings`
        self.fixed_color_map: Dict[str, str] = {}    # Updated in `update_base_color_mappings` and `update_user_color_mappings`

        self.update_palette(opts.palette, opts.avoid_palette_idxs)
        self.update_op_cat_mappings()
        self.update_base_color_mappings()
        self.update_user_color_mappings(opts.fixed_color_map)
        self.color_nodes()

    def update_palette(self, palette:str, avoid_palette_idxs:List[int]):
        # From https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express
        if isinstance(palette, str):
            if palette=='large':
                # Dark24
                self.palette = [
                    '#2E91E5','#E15F99','#1CA71C','#FB0D0D','#DA16FF','#222A2A','#B68100','#750D86',
                    '#EB663B','#511CFB','#00A08B','#FB00D1','#FC0080','#B2828D','#6C7C32','#778AAE',
                    '#862A16','#A777F1','#620042','#1616A7','#DA60CA','#6C4516','#0D2A63','#AF0038'
                ]
                # Avoided colors
                # * 1 is perceptually close to 20
                # * 11 is perceptually close to 4, 12 and 20
                # * 3 is close to red, which is used for errors
                # * 5 is close to black, which is used for input and output nodes
                # * 15 is close to gray, which is used for uncategorized nodes
                self.avoid_palette_idxs = set([1,3,5,11,15])
            elif palette=='small':     
                # Bold
                self.palette = ['#7F3C8D','#11A579','#3969AC','#F2B701','#E73F74',
                                '#80BA5A','#E68310','#008695','#CF1C90','#F97B72']
                self.avoid_palette_idxs = set([])  # Already removed last color that resembled gray
            else:
                available_palettes = [attr for attr in dir(pc.qualitative) if not attr.startswith("_")]
                if palette in available_palettes:
                    self.palette = plotly_palette_as_hex(palette)
                    self.avoid_palette_idxs = avoid_palette_idxs
                else:
                    warnings.warn(f'No palette named "{palette}" found in plotly.colors.quantitative. Defaulting to "Vivid" palette.')
                    self.palette = ['#E58606','#5D69B1','#52BCA3','#99C945','#CC61B0',
                                    '#24796C','#DAA51B','#2F8AC4','#764E9F','#ED645A']
                    self.avoid_palette_idxs = set([])  # Already removed last color that resembled gray
        else:
            self.palette = palette
            self.avoid_palette_idxs = avoid_palette_idxs

    def get_operation_category(self, n:MavNode):
        op:str = n.operation
        if op in self.op_cat_map: return self.op_cat_map[op]
        for s in self.known_op_list:
            if op.endswith(s): return self.op_cat_suffix_map[s]
        return 'Uncategorized'

    def update_op_cat_mappings(self):
        """
        `update_op_cat_mappings` creates dictionaries that map 
        `nn.Module` class names to categories as defined in
        https://pytorch.org/docs/stable/nn.html and function
        names to categories as defined in
        https://pytorch.org/docs/stable/nn.functional.html 
        """

        # Input and output
        for m in ['input', 'output']:
            self.op_cat_map[m] = "InOut"

        # Convolution
        for m in ['nn.Conv1d', 'nn.Conv2d', 'nn.Conv3d', 'nn.ConvTranspose1d', 'nn.ConvTranspose2d', 'nn.ConvTranspose3d', 
                  'nn.LazyConv1d', 'nn.LazyConv2d', 'nn.LazyConv3d', 'nn.LazyConvTranspose1d', 'nn.LazyConvTranspose2d', 
                  'nn.LazyConvTranspose3d', 'nn.Unfold', 'nn.Fold']:
            self.op_cat_map[m] = "Convolution"
        for m in ['Conv', 'Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d']:
            self.op_cat_suffix_map[m] = "Convolution"
        for f in ['conv1d()', 'conv2d()', 'conv3d()', 'conv_transpose1d()', 'conv_transpose2d()', 'conv_transpose3d()', 
                  'unfold()', 'fold()']:
            self.op_cat_map[f] = "Convolution"
        for f in ['conv()', 'conv1d()', 'conv2d()', 'conv3d()', 'conv_transpose1d()', 'conv_transpose2d()', 'conv_transpose3d()']:
            self.op_cat_suffix_map[f] = "Convolution"

        # Pooling
        for m in ['nn.MaxPool1d', 'nn.MaxPool2d', 'nn.MaxPool3d', 'nn.MaxUnpool1d', 'nn.MaxUnpool2d', 'nn.MaxUnpool3d', 
                  'nn.AvgPool1d', 'nn.AvgPool2d', 'nn.AvgPool3d', 'nn.FractionalMaxPool2d', 'nn.FractionalMaxPool3d', 
                  'nn.LPPool1d', 'nn.LPPool2d', 'nn.LPPool3d', 'nn.AdaptiveMaxPool1d', 'nn.AdaptiveMaxPool2d', 
                  'nn.AdaptiveMaxPool3d', 'nn.AdaptiveAvgPool1d', 'nn.AdaptiveAvgPool2d', 'nn.AdaptiveAvgPool3d']:
            self.op_cat_map[m] = "Pooling"
        for m in ['Pool', 'Pool1d', 'Pool2d', 'Pool3d', 'Unpool', 'Unpool1d', 'Unpool2d', 'Unpool3d']:
            self.op_cat_suffix_map[m] = "Pooling"
        for f in ['avg_pool1d()', 'avg_pool2d()', 'avg_pool3d()', 'max_pool1d()', 'max_pool2d()', 'max_pool3d()', 
                  'max_unpool1d()', 'max_unpool2d()', 'max_unpool3d()', 'lp_pool1d()', 'lp_pool2d()', 'lp_pool3d()', 
                  'adaptive_max_pool1d()', 'adaptive_max_pool2d()', 'adaptive_max_pool3d()', 'adaptive_avg_pool1d()', 
                  'adaptive_avg_pool2d()', 'adaptive_avg_pool3d()', 'fractional_max_pool2d()', 'fractional_max_pool3d()']:
            self.op_cat_map[f] = "Pooling"
        for f in ['pool()', 'pool1d()', 'pool2d()', 'pool3d()']:
            self.op_cat_suffix_map[f] = "Pooling"

        # Padding
        for m in ['nn.ReflectionPad1d', 'nn.ReflectionPad2d', 'nn.ReflectionPad3d', 'nn.ReplicationPad1d', 'nn.ReplicationPad2d', 
                  'nn.ReplicationPad3d', 'nn.ZeroPad1d', 'nn.ZeroPad2d', 'nn.ZeroPad3d', 'nn.ConstantPad1d', 'nn.ConstantPad2d', 
                  'nn.ConstantPad3d', 'nn.CircularPad1d', 'nn.CircularPad2d', 'nn.CircularPad3d']:
            self.op_cat_map[m] = "Padding"
        for m in ['Pad', 'Pad1d', 'Pad2d', 'Pad3d']:
            self.op_cat_suffix_map[m] = "Padding"
        for f in ['pad()']:
            self.op_cat_map[f] = "Padding"
        for f in ['pad()']:
            self.op_cat_suffix_map[f] = "Padding"

        # Activation
        for m in ['nn.ELU', 'nn.Hardshrink', 'nn.Hardsigmoid', 'nn.Hardtanh', 'nn.Hardswish', 'nn.LeakyReLU', 'nn.LogSigmoid', 
                  'nn.MultiheadAttention', 'nn.PReLU', 'nn.ReLU', 'nn.ReLU6', 'nn.RReLU', 'nn.SELU', 'nn.CELU', 'nn.GELU', 
                  'nn.Sigmoid', 'nn.SiLU', 'nn.Mish', 'nn.Softplus', 'nn.Softshrink', 'nn.Softsign', 'nn.Tanh', 'nn.Tanhshrink', 
                  'nn.Threshold', 'nn.GLU', 'nn.Softmin', 'nn.Softmax', 'nn.Softmax2d', 'nn.LogSoftmax', 'nn.AdaptiveLogSoftmaxWithLoss']:
            self.op_cat_map[m] = "Activation"
        for f in ['threshold()', 'threshold_()', 'relu()', 'relu_()', 'hardtanh()', 'hardtanh_()', 'hardswish()', 
                  'relu6()', 'elu()', 'elu_()', 'selu()', 'celu()', 'leaky_relu()', 'leaky_relu_()', 'prelu()', 
                  'rrelu()', 'rrelu_()', 'glu()', 'gelu()', 'logsigmoid()', 'hardshrink()', 'tanhshrink()', 
                  'softsign()', 'softplus()', 'softmin()', 'softmax()', 'softshrink()', 'gumbel_softmax()', 
                  'log_softmax()', 'tanh()', 'sigmoid()', 'hardsigmoid()', 'silu()', 'mish()']:
            self.op_cat_map[f] = "Activation"

        # Normalization
        for m in ['nn.BatchNorm1d', 'nn.BatchNorm2d', 'nn.BatchNorm3d', 'nn.LazyBatchNorm1d', 'nn.LazyBatchNorm2d', 'nn.LazyBatchNorm3d', 
                  'nn.GroupNorm', 'nn.SyncBatchNorm', 'nn.InstanceNorm1d', 'nn.InstanceNorm2d', 'nn.InstanceNorm3d', 'nn.LazyInstanceNorm1d', 
                  'nn.LazyInstanceNorm2d', 'nn.LazyInstanceNorm3d', 'nn.LayerNorm', 'nn.LocalResponseNorm', 'nn.RMSNorm']:
            self.op_cat_map[m] = "Normalization"
        for m in ['Norm', 'Norm1d', 'Norm2d', 'Norm3d']:
            self.op_cat_suffix_map[m] = "Normalization"
        for f in ['batch_norm()', 'group_norm()', 'instance_norm()', 'layer_norm()', 'local_response_norm()', 
                  'rms_norm()', 'normalize()']:
            self.op_cat_map[f] = "Normalization"
        for f in ['norm()']:
            self.op_cat_suffix_map[f] = "Normalization"

        # Recurrent
        for m in ['nn.RNNBase', 'nn.RNN', 'nn.LSTM', 'nn.GRU', 'nn.RNNCell', 'nn.LSTMCell', 'nn.GRUCell']:
            self.op_cat_map[m] = "Recurrent"
        for m in ['Cell']:
            self.op_cat_suffix_map[m] = "Recurrent"

        # Transformer
        for m in ['nn.Transformer', 'nn.TransformerEncoder', 'nn.TransformerDecoder', 'nn.TransformerEncoderLayer', 'nn.TransformerDecoderLayer']:
            self.op_cat_map[m] = "Transformer"
        for m in ['Transformer', 'TransformerEncoder', 'TransformerDecoder']:
            self.op_cat_suffix_map[m] = "Transformer"
        for f in ['scaled_dot_product_attention()']:
            self.op_cat_map[f] = "Transformer"
        for f in ['attention()']:
            self.op_cat_suffix_map[f] = "Transformer"

        # Linear
        for m in ['nn.Identity', 'nn.Linear', 'nn.Bilinear', 'nn.LazyLinear']:
            self.op_cat_map[m] = "Linear"
        for m in ['Linear']:
            self.op_cat_suffix_map[m] = "Linear"
        for f in ['linear()', 'bilinear()']:
            self.op_cat_map[f] = "Linear"
        for f in ['linear()']:
            self.op_cat_suffix_map[f] = "Linear"

        # Dropout
        for m in ['nn.Dropout', 'nn.Dropout1d', 'nn.Dropout2d', 'nn.Dropout3d', 'nn.AlphaDropout', 'nn.FeatureAlphaDropout']:
            self.op_cat_map[m] = "Dropout"
        for m in ['Dropout', 'Dropout1d', 'Dropout2d', 'Dropout3d']:
            self.op_cat_suffix_map[m] = "Dropout"
        for f in ['dropout()', 'alpha_dropout()', 'feature_alpha_dropout()', 'dropout1d()', 'dropout2d()', 'dropout3d()']:
            self.op_cat_map[f] = "Dropout"
        for f in ['dropout()', 'dropout1d()', 'dropout2d()', 'dropout3d()']:
            self.op_cat_suffix_map[f] = "Dropout"

        # Sparse
        for m in ['nn.Embedding', 'nn.EmbeddingBag']:
            self.op_cat_map[m] = "Sparse"
        for m in ['Embedding']:
            self.op_cat_suffix_map[m] = "Sparse"
        for f in ['embedding()', 'embedding_bag()', 'one_hot()']:
            self.op_cat_map[f] = "Sparse"
        for f in ['embedding()']:
            self.op_cat_suffix_map[f] = "Sparse"

        # Distance
        for m in ['nn.CosineSimilarity', 'nn.PairwiseDistance']:
            self.op_cat_map[m] = "Distance"
        for m in ['Similarity', 'Distance']:
            self.op_cat_suffix_map[m] = "Distance"
        for f in ['pairwise_distance()', 'cosine_similarity()', 'pdist()']:
            self.op_cat_map[f] = "Distance"
        for f in ['distance()', 'similarity()']:
            self.op_cat_suffix_map[f] = "Distance"

        # Loss
        for m in ['nn.L1Loss', 'nn.MSELoss', 'nn.CrossEntropyLoss', 'nn.CTCLoss', 'nn.NLLLoss', 'nn.PoissonNLLLoss', 
                  'nn.GaussianNLLLoss', 'nn.KLDivLoss', 'nn.BCELoss', 'nn.BCEWithLogitsLoss', 'nn.MarginRankingLoss', 
                  'nn.HingeEmbeddingLoss', 'nn.MultiLabelMarginLoss', 'nn.HuberLoss', 'nn.SmoothL1Loss', 'nn.SoftMarginLoss', 
                  'nn.MultiLabelSoftMarginLoss', 'nn.CosineEmbeddingLoss', 'nn.MultiMarginLoss', 'nn.TripletMarginLoss', 
                  'nn.TripletMarginWithDistanceLoss']:
            self.op_cat_map[m] = "Loss"
        for m in ['Loss']:
            self.op_cat_suffix_map[m] = "Loss"
        for f in ['binary_cross_entropy()', 'binary_cross_entropy_with_logits()', 'poisson_nll_loss()', 
                  'cosine_embedding_loss()', 'cross_entropy()', 'ctc_loss()', 'gaussian_nll_loss()', 
                  'hinge_embedding_loss()', 'kl_div()', 'l1_loss()', 'mse_loss()', 'margin_ranking_loss()', 
                  'multilabel_margin_loss()', 'multilabel_soft_margin_loss()', 'multi_margin_loss()', 'nll_loss()', 
                  'huber_loss()', 'smooth_l1_loss()', 'soft_margin_loss()', 'triplet_margin_loss()', 
                  'triplet_margin_with_distance_loss()']:
            self.op_cat_map[f] = "Loss"
        for f in ['entropy()', 'logits()', 'loss()']:
            self.op_cat_suffix_map[f] = "Loss"

        # Vision
        for m in ['nn.PixelShuffle', 'nn.PixelUnshuffle', 'nn.Upsample', 'nn.UpsamplingNearest2d', 
                  'nn.UpsamplingBilinear2d', 'nn.ChannelShuffle']:
            self.op_cat_map[m] = "Vision"
        for f in ['pixel_shuffle()', 'pixel_unshuffle()', 'interpolate()', 'upsample()', 
                  'upsample_nearest()', 'upsample_bilinear()', 'grid_sample()', 'affine_grid()']:
            self.op_cat_map[f] = "Vision"

        self.known_op_list = list(self.op_cat_suffix_map.keys())

    def get_inout_color(self):
        if "InOut" in self.fixed_color_map:
            return self.palette[self.fixed_color_map["InOut"]]
        return '#000000'  # Black

    def get_error_color(self):
        if "Error" in self.fixed_color_map:
            return self.palette[self.fixed_color_map["Error"]]
        return '#FF0000'  # Bright red

    def get_uncategorized_color(self):
        if "Uncategorized" in self.fixed_color_map:
            return self.palette[self.fixed_color_map["Uncategorized"]]
        return '##A5AA99'  # Gray used in Bold and Vivid palettes

    def update_base_color_mappings(self):
        if not isinstance(self.specified_palette, str): return
        if self.specified_palette == 'large':
            # * Unused colors: 1, 11
            # * Assignable to Common nodes from lists: 22, 7, 21, 23, 18, 16
            # * Nodes not categorized: 15
            self.fixed_color_map["InOut"]         = 5
            self.fixed_color_map["Error"]         = 3
            self.fixed_color_map["Uncategorized"] = 15

            self.fixed_color_map["Convolution"]   = 10
            self.fixed_color_map["Linear"]        = 0
            self.fixed_color_map["Transformer"]   = 17
            self.fixed_color_map["Recurrent"]     = 6

            self.fixed_color_map["Activation"]    = 19
            self.fixed_color_map["Normalization"] = 8
            self.fixed_color_map["Pooling"]       = 20
            self.fixed_color_map["Padding"]       = 14
            self.fixed_color_map["Dropout"]       = 2

            self.fixed_color_map["Sparse"]        = 13
            self.fixed_color_map["Vision"]        = 12
            self.fixed_color_map["Distance"]      = 4
            self.fixed_color_map["Loss"]          = 9
        elif self.specified_palette == 'small':
            self.fixed_color_map["Convolution"]   = 7
            self.fixed_color_map["Activation"]    = 2
            self.fixed_color_map["Normalization"] = 6
            self.fixed_color_map["Pooling"]       = 4

    def update_user_color_mappings(self, user_color_map:Dict[str,int]):
        # Override base color mappings
        for k,v in user_color_map.items():
            self.fixed_color_map[k] = v

    def color_nodes(self):
        """
        When rendering, the user can choosed between three criteria
        for node coloring:
        * The type of node (e.g. conv, relu)
        * The number of parameters
        * The number of FLOPS

        The `color_nodes` method assigns type-based colors to each 
        node, which will be used whenever the user selects the 
        first criterion above

        This implementation uses the following rules:
        * The operation category of each node (e.g. "Convolution",
          "Pooling", "Normalization", "Activation") is checked
          against `fixed_color_map` and the corresponding color
          in the map is used if it exists
        * The operation itself (e.g. "nn.Conv2d", "nn.MaxPool2d",
          "max_pool2d()", "relu()") of each node is checked against 
          the same `fixed_color_map` and the corresponding color
          in the map is used if it exists
        * Input nodes, output nodes and error nodes are assigned
          from `fixed_color_map["InOut"]`, fixed_color_map["Error"]
          or default colors if these don't exist in the map
        * For remaining unassigned nodes, the operations (e.g. 
          "nn.Conv2d") are sorted into two descending lists:
          - Most to least occurrences in the model
          - Most to least total number of parameters in the model
        * The remaining colors in the palette are assigned by
          picking operations from both lists, taking turns between 
          lists
          - When an operation is picked, all nodes that perform 
            that operation are assigned the next available color
            from the palette
          - If an operation is present in both lists, it is removed 
            from both upon being picked and the list that picked it 
            gets to pick again
        * When we run out of colors in the palette, remaining nodes
          are assigned using `fixed_color_map["Uncategorized"]` or
          a default color if this color does not exist in the map
        """
        # Operation categories: https://pytorch.org/docs/stable/nn.html
        # Palettes: https://plotly.com/python/discrete-color

        # Initialize list of available indices in palette
        available_color_idxs = set(range(len(self.palette)))
        for idx in self.avoid_palette_idxs:
            available_color_idxs.remove(idx)
        if "InOut" in self.fixed_color_map:
            color_idx = self.fixed_color_map["InOut"]
            if color_idx in available_color_idxs: available_color_idxs.remove(color_idx)
        if "Error" in self.fixed_color_map:
            color_idx = self.fixed_color_map["Error"]
            if color_idx in available_color_idxs: available_color_idxs.remove(color_idx)
        if "Uncategorized" in self.fixed_color_map:
            color_idx = self.fixed_color_map["Uncategorized"]
            if color_idx in available_color_idxs: available_color_idxs.remove(color_idx)

        # Assign colors according to fixed mappings
        uncategorized_nodes: Dict[str, List[MavNode]] = {}  # Indexed by operation
        for n in self.g.nodes:
            operation = n.operation
            op_cat = self.get_operation_category(n)
            if n.error:
                # Node raised an error during forward pass
                n.op_color = self.get_error_color()
            elif operation in ["input", "output"]:
                # Node is an input or output node
                n.op_color = self.get_inout_color()
            elif op_cat in self.fixed_color_map:
                # Operation category (e.g. "Convolution") found in fixed color map
                color_idx = self.fixed_color_map[op_cat]
                n.op_color = self.palette[color_idx]
                if color_idx in available_color_idxs: available_color_idxs.remove(color_idx)
            elif operation in self.fixed_color_map:
                # Operation (e.g. "nn.Conv2d") found in fixed color map
                color_idx = self.fixed_color_map[operation]
                n.op_color = self.palette[color_idx]
                if color_idx in available_color_idxs: available_color_idxs.remove(color_idx)
            else:
                # Node properties not found in fixed color map. 
                # Will be assigned from palette in the next step
                if operation in uncategorized_nodes:
                    uncategorized_nodes[operation].append(n)
                else:
                    uncategorized_nodes[operation] = [n]
        
        # Sort unassigned operations by most occurrences and most total parameters
        num_occurrences     = {k: len(v) for k, v in uncategorized_nodes.items()}
        total_params        = {k: sum([n.params for n in v]) for k, v in uncategorized_nodes.items()}
        list_by_occurrences = sorted(uncategorized_nodes, key=lambda k: num_occurrences[k], reverse=True)
        list_by_params      = sorted(uncategorized_nodes, key=lambda k: total_params[k], reverse=True)

        # Assign the remaining colors
        while uncategorized_nodes and available_color_idxs:
            # Decide which list to pick from
            # * If one list is longer than the other, pick from the longer list
            pick_from_occurrences = False
            if len(list_by_occurrences) > len(list_by_params):
                pick_from_occurrences = True
            elif len(list_by_params) > len(list_by_occurrences):
                pick_from_occurrences = False
            elif len(list_by_occurrences) == 0 and len(list_by_params) == 0:
                break

            # Pick an operation and assign color to node that perform that operation
            if pick_from_occurrences:
                operation = list_by_occurrences.pop(0)
                nodes = uncategorized_nodes.pop(operation)
                color_idx = available_color_idxs.pop()
                for n in nodes:
                    n.op_color = self.palette[color_idx]
                if operation in list_by_params:
                    list_by_params.remove(operation)
            else:
                operation = list_by_params.pop(0)
                nodes = uncategorized_nodes.pop(operation)
                color_idx = available_color_idxs.pop()
                for n in nodes:
                    n.op_color = self.palette[color_idx]
                if operation in list_by_occurrences:
                    list_by_occurrences.remove(operation)
        
        # No more colors to assign. If uncategorized nodes remain, they receive the uncategorized color
        for node_list in uncategorized_nodes.values():
            for n in node_list:
                n.op_color = self.get_uncategorized_color()

def rgb_to_hex(colors):
    """
    Converts a color of format "rgb(red,green,blue)" to a color of format 
    "#RRGGBB". The input and output colors are both strings.

    Also accepts lists, tuples and sets of color strings, in which case 
    the resurn value will be an iterator of the same type.

    Plotly color sequences (e.g. `plotly.colors.quantitative.Bold`)
    don't all return their values in the same format. Plotly
    provides `plotly.colors.convert_colors_to_same_type` to convert
    all colors to RGB format, but nothing currently to convert
    all colors to hex format. That is where this function comes in.
    """
    def rgb_to_hex_1color(rgb_color_str):
        if isinstance(rgb_color_str, str) and rgb_color_str.startswith("rgb"):
            r, g, b = map(int, rgb_color_str[4:-1].split(","))
            return f"#{r:02X}{g:02X}{b:02X}"
        else:
            return rgb_color_str

    # Check the type of the input
    if isinstance(colors, (list, tuple, set)):
        return type(colors)(rgb_to_hex_1color(c) for c in colors)
    elif hasattr(colors, "__iter__") and not isinstance(colors, str):
        return [rgb_to_hex_1color(c) for c in colors]
    else:
        return rgb_to_hex_1color(colors)
    
def colors_to_hex(colors):
    return rgb_to_hex(pc.convert_colors_to_same_type(colors, colortype="rgb")[0])

def plotly_palette_as_hex(palette):
    return colors_to_hex(getattr(pc.qualitative, palette))

def color_graph_nodes(g:MavGraph, opts:RenderOptions=RenderOptions(), **kwargs):
    """
    Performs the coloring step
    
    Keyword arguments may be passed either via a `RenderOptions` object or
    as-is. Using a `RenderOptions` object provides better intellisense, 
    but plain keyword arguments results in more concise code.

    The following two lines are equivalent:
    ```
    color_graph_nodes(g, RenderOptions(palette='vivid', avoid_palette_idxs={10}))  
    color_graph_nodes(g, palette='vivid', avoid_palette_idxs={10})  
    ```

    Parameters
    ----------
    g: MavGraph
        Graph object produced by tracing step

    palette: string or list of strings
        A discrete color palette to use for node marker colors when coloring
        by operation. The value may be in any of the following formats:
        * A named palette from https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express, e.g. 'Vivid'
        * A list of strings in '#RRGGBB' format
        * One of the strings 'large' or 'small'

    avoid_palette_idxs: set[int]
        Indices in the specified `palette` that should not be used for marker colors. 
        This is useful when specifying a named palette from https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express

    fixed_color_map: dict[str,int]
        Force specific node operations to specific colors to ensure consistency
        across models visualized.
        
        Keys may take on any of the following formats:
        * Any value in the "Operations" column of the table produced by IDLMAV
        * Any category listed at https://pytorch.org/docs/stable/nn.html

        Example:
        ```
        fixed_color_map={'Convolution':7, 'add()':0, 'nn.MaxPool2d':5}
        ```   
    """
    for k,v in kwargs.items(): opts.__setattr__(k,v)
    colorer = MavColorer(g, opts)