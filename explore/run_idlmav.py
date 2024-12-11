"""
This script provides an entry point to run and step through the MAV
workflow for various models
"""

# Notes:
# * In VSCode, remember to add `"justMyCode": false` to "launch.json"
# * In VSCode, hit Alt+Z on terminal to toggle line wrapping

# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------
DEVICE               = 'cpu'  # ['cpu','cuda']
MODEL_NAME           = 'cnn_mnist'  # ['cnn_mnist', 'cnn_mnist_broken', 'tv_resnet18', 'timm_resnet18', 'miniai_resnet', 'miniai_autoenc', 'miniai_unet']
INTERACTIVE          = False
ADD_TABLE            = True
ADD_SLIDER           = True
ADD_OVERVIEW         = True
NUM_LEVELS_DISPLAYED = 25

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
import sys
import os
workspace_path = os.path.abspath('')
sys.path.append(workspace_path)

from idlmav import MAV, plotly_renderer

import numpy as np
import torch
from torch import nn, fx, Tensor
import torch.nn.functional as F
import torchvision
import timm

import torchinfo
import torchview

from miniai.init import clean_mem
from miniai.resnet import ResBlock, act_gr, conv
from IPython.display import display
from ipywidgets.embed import embed_minimal_html

# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
def main():
    model, input_size = get_model_and_input_size(MODEL_NAME, DEVICE)
    x_input = torch.randn(input_size).to(DEVICE)
    v = MAV(model, x_input, device=DEVICE)
    if INTERACTIVE:
        with plotly_renderer('browser'):            
            container = v.render_widget(add_table=ADD_TABLE, add_slider=ADD_SLIDER, add_overview=ADD_OVERVIEW,
                                                 num_levels_displayed=NUM_LEVELS_DISPLAYED)
            embed_minimal_html('idlmav.html', views=[container], title='MAV')
    else:
        with plotly_renderer('browser'):
            v.show_figure(add_table=ADD_TABLE, add_slider=ADD_SLIDER,
                          num_levels_displayed=NUM_LEVELS_DISPLAYED)

# ------------------------------------------------------------------------------
# Models to analyze
# ------------------------------------------------------------------------------
class MnistCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
def get_model_cnn_mnist():
    return MnistCnn()

def get_model_cnn_mnist_broken():
    model = MnistCnn()
    model.fc1 = nn.Linear(9216, 120)
    return model

def get_model_tv_resnet18():
    return torchvision.models.resnet.resnet18()

def get_model_timm_resnet18():
    return timm.create_model('resnet18', in_chans=3, num_classes=10)

def get_model_miniai_resnet(act=nn.ReLU, nfs=(16,32,64,128,256), norm=nn.BatchNorm2d):
    class GlobalAvgPool(nn.Module):
        def forward(self, x:Tensor): return x.mean((-2,-1))
    layers = [conv(1, 16, ks=5, stride=1, act=act, norm=norm)]
    layers += [ResBlock(nfs[i], nfs[i+1], act=act, norm=norm, stride=2) for i in range(len(nfs)-1)]
    layers += [GlobalAvgPool(), nn.Linear(256, 10, bias=False), nn.BatchNorm1d(10)]
    return nn.Sequential(*layers)

def get_model_miniai_autoenc(act=act_gr, nfs=(32,64,128,256,512), norm=nn.BatchNorm2d, drop=0.1):
    def up_block(ni, nf, ks=3, act=act_gr, norm=None):
        return nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                            ResBlock(ni, nf, ks=ks, act=act, norm=norm))
    layers = [ResBlock(3, nfs[0], ks=5, stride=1, act=act, norm=norm)]
    layers += [ResBlock(nfs[i], nfs[i+1], act=act, norm=norm, stride=2) for i in range(len(nfs)-1)]
    layers += [up_block(nfs[i], nfs[i-1], act=act, norm=norm) for i in range(len(nfs)-1,0,-1)]
    layers += [ResBlock(nfs[0], 3, act=nn.Identity, norm=norm)]
    return nn.Sequential(*layers)
def up_block(ni, nf, ks=3, act=act_gr, norm=None):
    return nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                        ResBlock(ni, nf, ks=ks, act=act, norm=norm))

class TinyUnet(nn.Module):
    def __init__(self, act=act_gr, nfs=(32,64,128,256,512), norm=nn.BatchNorm2d):
        super().__init__()
        self.start = ResBlock(3, nfs[0], stride=1, act=act, norm=norm)
        self.dn = nn.ModuleList([ResBlock(nfs[i], nfs[i+1], act=act, norm=norm, stride=2)
                                 for i in range(len(nfs)-1)])
        self.up = nn.ModuleList([up_block(nfs[i], nfs[i-1], act=act, norm=norm)
                                 for i in range(len(nfs)-1,0,-1)])
        self.up += [ResBlock(nfs[0], 3, act=act, norm=norm)]
        self.end = ResBlock(3, 3, act=nn.Identity, norm=norm)

    def forward(self, x):
        layers = []
        layers.append(x)
        x = self.start(x)
        for l in self.dn:
            layers.append(x)
            x = l(x)
        n = len(layers)
        for i,l in enumerate(self.up):
            if i!=0: x += layers[n-i]
            x = l(x)
        return self.end(x+layers[0])
    
def get_model_miniai_unet():
    return TinyUnet()

# ------------------------------------------------------------------------------
# Model preparation helpers
# ------------------------------------------------------------------------------
def get_model_and_input_size(name, device):
    match name:
        case 'cnn_mnist':
            return get_model_cnn_mnist().to(device), (16,1,28,28)
        case 'cnn_mnist_broken':
            return get_model_cnn_mnist_broken().to(device), (16,1,28,28)
        case 'tv_resnet18':
            return get_model_tv_resnet18().to(device), (16,3,160,160)
        case 'timm_resnet18':
            return get_model_timm_resnet18().to(device), (16,3,160,160)
        case 'miniai_resnet':
            return get_model_miniai_resnet().to(device), (16,1,28,28)
        case 'miniai_autoenc':
            return get_model_miniai_autoenc().to(device), (16,3,160,160)
        case 'miniai_unet':
            return get_model_miniai_unet().to(device), (16,3,160,160)
        case _:
            return None
        
# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()