import torch
import torch.nn as nn
import dgl
import torch
import torch.nn as nn
import dgl.function as fn
import numpy as np
import torch.nn.functional as F

class GraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 jump=True, 
                 norm=True,
                 bias=True,
                 activation=None):
        super(GraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._jump = jump
        
        if jump:
            self.weight = nn.Parameter(torch.Tensor(2*in_feats, out_feats))
        else:
            self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self._activation = activation
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)
            
    def forward(self, feat, graph, mask = None):
        if self._jump:
            _feat = feat

        if self._norm:
            if mask is None:
                norm = torch.pow(graph.in_degrees().float(), -0.5)
                norm.masked_fill_(graph.in_degrees() == 0, 1.0)
                shp = norm.shape + (1,) * (feat.dim() - 1)
                norm = torch.reshape(norm, shp).to(feat.device)
                feat = feat * norm.unsqueeze(1)
            else:
                graph.ndata['h'] = mask.float()
                graph.update_all(
                    fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='h')
                    )
                masked_deg = graph.ndata.pop('h')
                norm = torch.pow(masked_deg, -0.5)
                norm.masked_fill_(masked_deg == 0, 1.0)
                feat = feat * norm.unsqueeze(-1)

        if mask is not None:
            feat = mask.float().unsqueeze(-1) * feat            
        
        graph.ndata['h'] = feat
        graph.update_all(
            fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='h')
            )
        rst = graph.ndata.pop('h')

        if self._norm:
            rst = rst * norm.unsqueeze(-1)
        
        if self._jump:
            rst = torch.cat([rst, _feat], dim = -1)
        
        rst = torch.matmul(rst, self.weight)

        if self.bias is not None:
            rst = rst + self.bias

        if self._activation is not None:
            rst = self._activation(rst)
            
        return rst

class PolicyGraphConvNet(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers
        ):
        super(PolicyGraphConvNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            GraphConv(input_dim, hidden_dim, activation=F.relu)
            )
        for i in range(num_layers - 1):
            self.layers.append(
                GraphConv(hidden_dim, hidden_dim, activation=F.relu))
        
        self.layers.append(
            GraphConv(hidden_dim, output_dim, activation=None)
            )

        with torch.no_grad():
            self.layers[-1].bias[2].add_(3.0)
        
    def forward(self, h, g, mask = None):
        for i, layer in enumerate(self.layers):
            h = layer(h, g, mask = mask)
            
        return h

class ValueGraphConvNet(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers
        ):
        super(ValueGraphConvNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            GraphConv(input_dim, hidden_dim, activation=F.relu)
            )
        for i in range(num_layers - 1):
            self.layers.append(
                GraphConv(hidden_dim, hidden_dim, activation=F.relu))

        self.layers.append(
            GraphConv(hidden_dim, output_dim, activation=None)
            )
            
    def forward(self, h, g, mask = None):
        for i, layer in enumerate(self.layers):
            h = layer(h, g, mask = mask)

        return h