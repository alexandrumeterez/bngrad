import torch
from torch import nn
from math import sqrt
import numpy as np

class SinAct(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.sin(x)

class CustomBatchNorm1d(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.bn = nn.BatchNorm1d(d, affine=False)

    def forward(self, x):
        return self.bn(x)

class CustomNormalization(nn.Module):
    def __init__(self, norm_type, mean_reduction, force_factor=None):
        super().__init__()
        self.mean_reduction = mean_reduction
        self.norm_type = norm_type
        self.force_factor = force_factor

        # Expects (batch size, feature dim)
        if norm_type == 'bn':
            self.dim = 0 #normalize across batch size (columns)
        elif norm_type == 'ln':
            self.dim = 1 #normalize across feature dim (rows)
        elif norm_type == 'id':
            self.dim = -1
        else:
            raise ValueError("No such normalization.")
    def forward(self, X):
        if self.dim == -1:
            return X
        
        if self.mean_reduction:
            X = X - X.mean(dim=self.dim, keepdim=True)

        norm = X.norm(dim=self.dim, keepdim=True)
        factor = sqrt(X.shape[self.dim]) # n if BN, d if LN

        if self.force_factor is not None:
            factor = self.force_factor
        X = X / (norm / factor)
        return X

class GainedActivation(nn.Module):
    def __init__(self, activation, gain):
        super().__init__()
        self.activation = activation()
        self.gain = nn.Parameter(torch.tensor([gain], requires_grad=True))

    def forward(self, x):
        return self.activation(self.gain * x)


class MLPWithBatchNorm(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 num_layers, 
                 hidden_dim, 
                 norm_type, 
                 mean_reduction, 
                 activation, 
                 save_hidden, 
                 exponent, 
                 order='act_norm', 
                 force_factor=None, 
                 bias=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.hiddens = {}
        self.initialized = False
        self.exponent = exponent 
        self.save_hidden = save_hidden
        self.order = order
        if self.order not in ['act_norm', 'norm_act']:
            raise ValueError("Unknown order")
        # print(f"Using order: {self.order}", flush=True)
        
        # Create layers with batch normalization
        self.layers = nn.ModuleDict()
        self.layers[f'fc_0'] = nn.Linear(input_dim, hidden_dim, bias=bias)
        if norm_type == 'torch_bn':
            self.layers[f'norm_0'] = nn.BatchNorm1d(hidden_dim)
        else:
            self.layers[f'norm_0'] = CustomNormalization(norm_type, mean_reduction, force_factor=force_factor)
        self.layers[f'act_0'] = activation()

        for l in range(1, num_layers):
            self.layers[f'fc_{l}'] = nn.Linear(hidden_dim, hidden_dim, bias=bias)
            if norm_type == 'torch_bn':
                self.layers[f'norm_{l}'] = nn.BatchNorm1d(hidden_dim)
            else: 
                self.layers[f'norm_{l}'] = CustomNormalization(norm_type, mean_reduction, force_factor=force_factor)
            self.layers[f'act_{l}'] = activation()
        self.layers[f'fc_{num_layers}'] = nn.Linear(hidden_dim, output_dim, bias=bias)

    def forward(self, x):
        assert self.initialized is True

        x = x.view(-1, self.input_dim)  # Flatten the input tensor
        for l in range(self.num_layers):
            layer_gain = ((l+1)**self.exponent) # start at l+1 because l starts at 0
            
            x = self.layers[f'fc_{l}'](x)
            if self.save_hidden:
                self.hiddens[f'fc_{l}'] = x.clone().detach()

            if self.order == 'norm_act':
                x = self.layers[f'norm_{l}'](x)
                if self.save_hidden:
                    self.hiddens[f'norm_{l}'] = x.clone().detach()

                x = self.layers[f'act_{l}'](x * layer_gain)
                if self.save_hidden:
                    self.hiddens[f'act_{l}'] = x.clone().detach()
            
            elif self.order == 'act_norm':
                x = self.layers[f'act_{l}'](x * layer_gain)
                if self.save_hidden:
                    self.hiddens[f'act_{l}'] = x.clone().detach() 

                x = self.layers[f'norm_{l}'](x)
                if self.save_hidden:
                    self.hiddens[f'norm_{l}'] = x.clone().detach()

        x = self.layers[f'fc_{self.num_layers}'](x)
        if self.save_hidden:
            self.hiddens[f'fc_{self.num_layers}'] = x.clone().detach()
        return x

    def set_save_hidden(self, state):
        if state is True:
            self.save_hidden = True
            self.hiddens.clear()
        elif state is False:
            self.save_hidden = False
            self.hiddens.clear()

    def reset_parameters(self, init_type, gain=1.0):
        for name, p in self.named_modules():
            if isinstance(p, nn.Linear):
                if init_type == 'xavier_normal':
                    nn.init.xavier_normal_(p.weight, gain=gain)
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(p.weight)
                else:
                    raise ValueError("No such initialization scheme.")
        self.initialized = True
