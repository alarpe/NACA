import collections
from math import ceil
from collections import OrderedDict
import functools
import torch
# import torch_npu  # 如需使用华为NPU请取消注释
#from torch_npu.contrib import transfer_to_npu  #
from torch import feature_alpha_dropout, nn as nn
#import torch_scatter
#from torch_scatter.composite import scatter_softmax
import torch.nn.functional as F
import os
import sys
sys.path.insert(0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
)
from torch_geometric.nn import knn_interpolate
from torch_geometric.nn.conv import GCNConv,GATConv,SAGEConv
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.models import MLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.cuda.set_device(device)

"""
class LazyMLP(nn.Module):
    def __init__(self, output_sizes):
        super().__init__()
        num_layers = len(output_sizes)
        self._layers_ordered_dict = OrderedDict()
        for index, output_size in enumerate(output_sizes):
            self._layers_ordered_dict["linear_" + str(index)] = nn.LazyLinear(output_size)
            if index < (num_layers - 1):
                self._layers_ordered_dict["relu_" + str(index)] = nn.ReLU()
        self.layers = nn.Sequential(self._layers_ordered_dict)

    def forward(self, input):
        input = input.to(device)
        y = self.layers(input)
        return y
"""

class Encoder(nn.Module):
    """Encodes node and edge features into latent features."""

    # input_sizes: [input_node, input_edge]
    def __init__(self, make_mlp, input_sizes, latent_size):
        super().__init__()
        self._make_mlp = make_mlp
        self._latent_size = latent_size
        self.node_model = self._make_mlp(input_sizes[0], latent_size)
        self.mesh_edge_model = self._make_mlp(input_sizes[1], latent_size)
        '''
        for _ in graph.edge_sets:
          edge_model = make_mlp(latent_size)
          self.edge_models.append(edge_model)
        '''

    def forward(self, graph):
        node_latents = self.node_model(graph.x)
        edge_latent = self.mesh_edge_model(graph.edge_attr)
        graph.x=node_latents
        graph.edge_attr=edge_latent
        return graph

class Decoder(nn.Module):
    """Decodes node features from graph."""
    # decoder = self._make_mlp(self._output_size, layer_norm=False)
    # return decoder(graph.node_features)

    """Encodes node and edge features into latent features."""

    def __init__(self, make_mlp, input_size, output_size):
        super().__init__()
        self.model = make_mlp(input_size, output_size)

    def forward(self,node_features):
        return self.model(node_features)

class EncodeProcessDecode(nn.Module):
    #Encode-Process-Decode model.
    def __init__(self,
                 input_sizes,
                 output_size,
                 latent_size,
                 num_layers,
                 proc_layers,
                 hidden_channels,
                 modelname,
                 mp,
                 lambda_gnn
                 ):
        super().__init__()
        self._latent_size = latent_size
        self._output_size = output_size
        self._num_layers = num_layers
        self._proc_layers = proc_layers
        self._hidden_channels = hidden_channels
        self.encoder = Encoder(make_mlp=self._make_mlp, input_sizes = input_sizes, latent_size=self._latent_size)
        if modelname=='GCN':
            self.processor = MeshGCN(in_channels=self._latent_size,
                                     hidden_channels=self._hidden_channels,
                                     out_channels=self._latent_size,
                                     num_layers=self._proc_layers,
                                     improved=mp['improved'], cached=mp['cached'],
                                     bias=mp['bias'])
        elif modelname=='GAT':
            self.processor = MeshGAT(in_channels=self._latent_size,
                                 hidden_channels=self._hidden_channels,
                                 out_channels=self._latent_size,
                                 num_layers=self._proc_layers,
                                 heads=mp['heads'], slope=mp['slope'], dropout=mp['dropout'],
                                 lambda_gnn=lambda_gnn)
        else:
            modelname = 'SAGE'
            self.processor = MeshSAGE(in_channels=self._latent_size,
                                     hidden_channels=self._hidden_channels,
                                     out_channels=self._latent_size,
                                     num_layers=self._proc_layers,
                                     aggr=mp['aggr'], root_weight=mp['root_weight'],
                                     project=mp['project'],
                                     lambda_gnn=lambda_gnn)
        self.decoder = Decoder(make_mlp=functools.partial(self._make_mlp, layer_norm=False),
                               input_size=self._latent_size, output_size=self._output_size)

    def _make_mlp(self, input_size, output_size, layer_norm=True):
        widths = [input_size] + [self._latent_size] * self._num_layers + [output_size]
        network = MLP(widths,norm=None)  #
        if layer_norm:
            network = nn.Sequential(network, nn.LayerNorm(normalized_shape=widths[-1]))
        return network

    def forward(self, graph):
        latent_graph = self.encoder(graph)
        x= self.processor(latent_graph)
        return self.decoder(x)

class MeshGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=6, improved=False,
                 cached=False, bias=True):
        super().__init__()
        self.sdf = None

        channels = [in_channels]
        channels += [hidden_channels] * (num_layers - 1)
        channels.append(out_channels)

        convs = []
        for i in range(num_layers):
            convs.append(GCNConv(channels[i], channels[i+1], improved=improved,
                                 cached=cached, bias=bias))
        self.convs = nn.ModuleList(convs)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index)
        return x

class MeshGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=6, heads=1, slope=0.2, dropout=0.0,
                 lambda_gnn=0.8):
        super().__init__()
        self.sdf = None
        self.lambda_gnn = lambda_gnn

        channels = [in_channels]
        channels += [hidden_channels] * (num_layers - 1)
        channels.append(out_channels)

        convs = []
        for i in range(num_layers):
            if i%2==0 or i==num_layers-1:
                convs.append(GATConv(channels[i], channels[i+1],
                    heads = 1, negative_slope = slope, dropout = dropout))
            else:
                convs.append(GATConv(channels[i], channels[i + 1]//heads,
                    heads=heads, negative_slope=slope, dropout=dropout))

        self.convs = nn.ModuleList(convs)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        for i, conv in enumerate(self.convs[:-1]):
            if i>0:
                x = conv(x, edge_index, edge_attr) + self.lambda_gnn*x
            else:
                x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index, edge_attr)
        return x

class MeshSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=6, aggr='mean',
                 root_weight = True, project = False, bias=True,lambda_gnn=0.8):
        super().__init__()
        self.sdf = None
        self.lambda_gnn = lambda_gnn

        channels = [in_channels]
        channels += [hidden_channels] * (num_layers - 1)
        channels.append(out_channels)

        convs = []
        for i in range(num_layers):
            convs.append(SAGEConv(channels[i], channels[i+1], aggr=aggr, root_weight = root_weight,
                                 project = project, bias=bias))
        self.convs = nn.ModuleList(convs)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        for i, conv in enumerate(self.convs[:-1]):
            if i > 0:
                x = conv(x, edge_index) + self.lambda_gnn*x
            else:
                x = conv(x, edge_index)
            x = F.relu(x)

        x = self.convs[-1](x, edge_index)
        return x
