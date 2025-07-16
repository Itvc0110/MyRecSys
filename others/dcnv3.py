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
                 net_dropout=0.1,
                 num_heads=1):
        super(ExponentialCrossNetwork, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_cross_layers = num_cross_layers
        self.layer_norm = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.w = nn.ModuleList()
        self.b = nn.ParameterList()
        
        # Compute split size
        self.input_dim = input_dim
        self.half = input_dim // 2

        # Initialize layers
        for i in range(num_cross_layers):
            w_layer = nn.Linear(input_dim, self.half, bias=False).to(self.device)
            self.w.append(w_layer)
            self.b.append(nn.Parameter(torch.zeros((input_dim,), device=self.device)))
            if layer_norm:
                self.layer_norm.append(nn.LayerNorm(self.half).to(self.device))
            if batch_norm:
                self.batch_norm.append(nn.BatchNorm1d(num_heads).to(self.device))
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
        # Ensure input is on the correct device
        x = x.to(self.device)
        x0 = x

        for i in range(self.num_cross_layers):
            H = self.w[i](x)

            if len(self.batch_norm) > i:
                H = self.batch_norm[i](H)
            if len(self.layer_norm) > i:
                norm_H = self.layer_norm[i](H)
                mask = self.masker(norm_H)
            else:
                mask = self.masker(H)
            # Concatenate and pad if necessary
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
                 net_dropout=0.1,
                 num_heads=1):
        super(LinearCrossNetwork, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_cross_layers = num_cross_layers
        self.layer_norm = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.w = nn.ModuleList()
        self.b = nn.ParameterList()
        
        # Compute split size
        self.input_dim = input_dim
        self.half = input_dim // 2
        
        # Initialize layers
        for i in range(num_cross_layers):
            w_layer = nn.Linear(input_dim, self.half, bias=False).to(self.device)
            self.w.append(w_layer)
            self.b.append(nn.Parameter(torch.zeros((input_dim,), device=self.device)))
            
            if layer_norm:
                self.layer_norm.append(nn.LayerNorm(self.half).to(self.device))
            if batch_norm:
                self.batch_norm.append(nn.BatchNorm1d(num_heads).to(self.device))
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
        # Ensure input is on the correct device
        x = x.to(self.device)
        x0 = x

        for i in range(self.num_cross_layers):
            H = self.w[i](x)
    
            if len(self.batch_norm) > i:
                H = self.batch_norm[i](H)
            if len(self.layer_norm) > i:
                norm_H = self.layer_norm[i](H)
                mask = self.masker(norm_H)
            else:
                mask = self.masker(H)

            # Concatenate and pad if necessary
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
                
        logit = self.sfc(x)
        return logit


class DCNv3(nn.Module):
    def __init__(self,
                 input_dim,
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 num_deep_cross_layers=3,
                 num_shallow_cross_layers=3,
                 deep_net_dropout=0.1,
                 shallow_net_dropout=0.1,
                 layer_norm=True,
                 batch_norm=False,
                 num_heads=1):
        super(DCNv3, self).__init__()
        # Move device initialization to the beginning
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize networks with device placement
        self.ECN = ExponentialCrossNetwork(
            input_dim=input_dim,
            num_cross_layers=num_deep_cross_layers,
            net_dropout=deep_net_dropout,
            layer_norm=layer_norm,
            batch_norm=batch_norm,
            num_heads=num_heads
        ).to(self.device)
        
        self.LCN = LinearCrossNetwork(
            input_dim=input_dim,
            num_cross_layers=num_shallow_cross_layers,
            net_dropout=shallow_net_dropout,
            layer_norm=layer_norm,
            batch_norm=batch_norm,
            num_heads=num_heads
        ).to(self.device)
        
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
        # Ensure inputs are on the correct device
        inputs = inputs.to(self.device)
        
        feature_emb = inputs
        dlogit = self.ECN(feature_emb).mean(dim=1)
        slogit = self.LCN(feature_emb).mean(dim=1)
        logit = (dlogit + slogit) * 0.5
    
        y_pred = self.output_activation(logit)
        y_d = self.output_activation(dlogit)
        y_s = self.output_activation(slogit)
        return {
            "y_pred": y_pred,
            "y_d": y_d,
            "y_s": y_s
        }

class TriBCE_Loss(nn.Module):
    def __init__(self):
        super(TriBCE_Loss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, y_pred, y_true, y_d, y_s):
        loss = self.bce_loss(y_pred, y_true)
        loss_d = self.bce_loss(y_d, y_true)
        loss_s = self.bce_loss(y_s, y_true)
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
        # Compute sample weights based on true labels
        sample_weight = torch.where(y_true == 1, self.pos_weight, 1.0)
        
        # Define BCE loss with sample weights
        bce_loss = nn.BCELoss(weight=sample_weight)
        
        # Compute losses
        loss = bce_loss(y_pred, y_true)
        loss_d = bce_loss(y_d, y_true)
        loss_s = bce_loss(y_s, y_true)
        
        # Compute weights for deep and shallow losses
        weight_d = loss_d - loss
        weight_s = loss_s - loss
        weight_d = torch.where(weight_d > 0, weight_d, torch.zeros_like(weight_d))
        weight_s = torch.where(weight_s > 0, weight_s, torch.zeros_like(weight_s))
        
        # Combine losses
        loss = loss + loss_d * weight_d + loss_s * weight_s
        return loss