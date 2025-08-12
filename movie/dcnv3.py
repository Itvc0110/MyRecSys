import pandas as pd
import numpy as np
import torch
from torch import nn
from processing import process_data
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


class ExponentialCrossNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 num_cross_layers=3,
                 layer_norm=True,
                 batch_norm=False,
                 net_dropout=0.1):
        super(ExponentialCrossNetwork, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_cross_layers = num_cross_layers
        self.layer_norm = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.w = nn.ModuleList()
        self.b = nn.ParameterList()
        
        # compute split size
        self.input_dim = input_dim
        self.half = input_dim // 2

        for i in range(num_cross_layers):
            w_layer = nn.Linear(input_dim, self.half, bias=False).to(self.device)
            self.w.append(w_layer)
            self.b.append(nn.Parameter(torch.zeros((input_dim,), device=self.device)))
            if layer_norm:
                self.layer_norm.append(nn.LayerNorm(self.half).to(self.device))
            if batch_norm:
                self.batch_norm.append(nn.BatchNorm1d(self.half).to(self.device))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout).to(self.device))
            nn.init.uniform_(self.b[i].data)
            
        self.masker = nn.ReLU().to(self.device)
        self.dfc = nn.Linear(input_dim, 1).to(self.device)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        x = x.to(self.device)
        x0 = x

        for i in range(self.num_cross_layers):
            H = self.w[i](x)  

            if len(self.batch_norm) > i:
                H = self.batch_norm[i](H)
            if len(self.layer_norm) > i:
                H = self.layer_norm[i](H)

            mask = self.masker(H)

            H = torch.cat([H, H * mask], dim=-1) 

            if H.shape[-1] != self.input_dim:
                pad_size = self.input_dim - H.shape[-1]
                pad = H.new_zeros(H.size(0), pad_size)
                H = torch.cat([H, pad], dim=-1)

            x = x0 * (H + self.b[i]) + x

            x = torch.clamp(x, min=-100, max=100)
            x = x / (torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-12)

            if len(self.dropout) > i:
                x = self.dropout[i](x)

        logit = self.dfc(x)
        return logit


class LinearCrossNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 num_cross_layers=3,
                 layer_norm=True,
                 batch_norm=True,
                 net_dropout=0.1):
        super(LinearCrossNetwork, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_cross_layers = num_cross_layers
        self.layer_norm = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.w = nn.ModuleList()
        self.b = nn.ParameterList()
        
        # compute split size
        self.input_dim = input_dim
        self.half = input_dim // 2
        
        for i in range(num_cross_layers):
            w_layer = nn.Linear(input_dim, self.half, bias=False).to(self.device)
            self.w.append(w_layer)
            self.b.append(nn.Parameter(torch.zeros((input_dim,), device=self.device)))
            
            if layer_norm:
                self.layer_norm.append(nn.LayerNorm(self.half).to(self.device))
            if batch_norm:
                self.batch_norm.append(nn.BatchNorm1d(self.half).to(self.device))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout).to(self.device))
            nn.init.uniform_(self.b[i].data)
            
        self.masker = nn.ReLU().to(self.device)
        self.sfc = nn.Linear(input_dim, 1).to(self.device)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        x = x.to(self.device)
        x0 = x

        for i in range(self.num_cross_layers):
            H = self.w[i](x)  

            if len(self.batch_norm) > i:
                H = self.batch_norm[i](H)

            if len(self.layer_norm) > i:
                H = self.layer_norm[i](H)

            mask = self.masker(H)

            H = torch.cat([H, H * mask], dim=-1)
            if H.shape[-1] != self.input_dim:
                pad_size = self.input_dim - H.shape[-1]
                pad = H.new_zeros(H.size(0), pad_size)
                H = torch.cat([H, pad], dim=-1)

            x = x0 * (H + self.b[i]) + x

            # Stabilize
            x = torch.clamp(x, min=-100, max=100)
            x = x / (torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-12)

            if len(self.dropout) > i:
                x = self.dropout[i](x)

        logit = self.sfc(x)
        return logit


class DCNv3(nn.Module):
    def __init__(self,
                 input_dim,
                 num_deep_cross_layers=1,
                 num_shallow_cross_layers=1,
                 deep_net_dropout=0.05,
                 shallow_net_dropout=0.05,
                 layer_norm=True,
                 batch_norm=False,
                 deep_tower_units=[1024, 512],
                 activation=nn.ReLU 
                 ):
        super(DCNv3, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.ECN = ExponentialCrossNetwork(
            input_dim=input_dim,
            num_cross_layers=num_deep_cross_layers,
            net_dropout=deep_net_dropout,
            layer_norm=layer_norm,
            batch_norm=batch_norm
        ).to(self.device)
        
        self.LCN = LinearCrossNetwork(
            input_dim=input_dim,
            num_cross_layers=num_shallow_cross_layers,
            net_dropout=shallow_net_dropout,
            layer_norm=layer_norm,
            batch_norm=batch_norm
        ).to(self.device)

######################################### 
        # Deep tower after combining ECN & LCN
        tower_layers = []
        
        # For deep tower, we will concatenate the last hidden features from ECN and LCN
        tower_input_dim = input_dim * 2
        for units in deep_tower_units:
            tower_layers.append(nn.Linear(tower_input_dim, units))
            tower_layers.append(activation())
            tower_layers.append(nn.Dropout(0.1))
            tower_input_dim = units
        tower_layers.append(nn.Linear(tower_input_dim, 1))
        self.deep_tower = nn.Sequential(*tower_layers)

        # Remove final projection from ECN/LCN for feature extraction
        self.ECN_proj = self.ECN.dfc
        self.LCN_proj = self.LCN.sfc
        self.ECN.dfc = nn.Identity()
        self.LCN.sfc = nn.Identity()
##################################################################       
        self.apply(self._init_weights)
        self.output_activation = torch.sigmoid

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight)

    def forward(self, inputs):
        #inputs = inputs.to(self.device)
        
        #feature_emb = inputs
        #dlogit = self.ECN(feature_emb).mean(dim=1)
        #slogit = self.LCN(feature_emb).mean(dim=1)
        #logit = (dlogit + slogit) * 0.5
    
        #y_pred = self.output_activation(logit)
        #y_d = self.output_activation(dlogit)
        #y_s = self.output_activation(slogit)
        #return {
        #    "y_pred": y_pred,
        #    "y_d": y_d,
        #    "y_s": y_s
        #}
####################################
        inputs = inputs.to(self.device)
        
        # Get ECN and LCN hidden features (no final projection)
        d_feat = self.ECN(inputs)  # [batch, input_dim]
        s_feat = self.LCN(inputs)  # [batch, input_dim]

        # Concatenate features for deep tower
        concat_feat = torch.cat([d_feat, s_feat], dim=1)  # [batch, 2*input_dim]
        logit = self.deep_tower(concat_feat)

        # Get original ECN & LCN logits by applying saved projections
        dlogit = self.ECN_proj(d_feat)
        slogit = self.LCN_proj(s_feat)

        y_pred = self.output_activation(logit)
        y_d = self.output_activation(dlogit)
        y_s = self.output_activation(slogit)

        return {
            "y_pred": y_pred,
            "y_d": y_d,
            "y_s": y_s
        }       
####################################        

class TriBCE_Loss(nn.Module):
    def __init__(self):
        super(TriBCE_Loss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, y_pred, y_true, y_d, y_s):
        loss = self.bce_loss(y_pred.squeeze(), y_true)
        loss_d = self.bce_loss(y_d.squeeze(), y_true)
        loss_s = self.bce_loss(y_s.squeeze(), y_true)
        weight_d = loss_d - loss
        weight_s = loss_s - loss
        weight_d = torch.where(weight_d > 0, weight_d, torch.zeros(1).to(weight_d.device))
        weight_s = torch.where(weight_s > 0, weight_s, torch.zeros(1).to(weight_s.device))
        loss = loss + loss_d * weight_d + loss_s * weight_s
        return loss

class Weighted_TriBCE_Loss(nn.Module):
    def __init__(self, pos_weight=1.0):
        super(Weighted_TriBCE_Loss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, y_pred, y_true, y_d, y_s):
        sample_weight = torch.where(y_true == 1, self.pos_weight, 1.0)
        
        # define BCE loss with sample weights
        bce_loss = nn.BCELoss(weight=sample_weight)
        
        # compute losses
        loss = bce_loss(y_pred.squeeze(), y_true)
        loss_d = bce_loss(y_d.squeeze(), y_true)
        loss_s = bce_loss(y_s.squeeze(), y_true)
        
        # compute weights for deep and shallow losses
        weight_d = loss_d - loss
        weight_s = loss_s - loss
        weight_d = torch.where(weight_d > 0, weight_d, torch.zeros_like(weight_d))
        weight_s = torch.where(weight_s > 0, weight_s, torch.zeros_like(weight_s))
        
        # combine losses
        loss = loss + loss_d * weight_d + loss_s * weight_s
        return loss