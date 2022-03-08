import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from utils import to_tensor

# local block, adapted with n-beats basic block
class LocalBlock(nn.Module):
    def __init__(self, input_size, theta_size, basis_function, layers, layer_size):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_features=input_size, out_features=layer_size)] +
                                      [nn.Linear(in_features=layer_size, out_features=layer_size)
                                       for _ in range(layers - 1)])
        self.basis_parameters = nn.Linear(in_features=layer_size, out_features=theta_size)
        self.basis_function = basis_function

    def forward(self, x):
        # batch * 2 * input_size
        block_input = x
        for layer in self.layers:
            block_input = torch.relu(layer(block_input))
        basis_parameters = self.basis_parameters(block_input)
        return self.basis_function(basis_parameters)

# generic basis function for local block
class GenericBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: torch.Tensor):
        return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]


# Periodic Block
class PeriodBlock(nn.Module):
    def __init__(self, layer_num: int, input_size, layer_size):
        super().__init__()
        if layer_num == 0:
            self.layers = nn.ModuleList( [nn.Linear(input_size, input_size)] )
        else:
            layers = [nn.Linear(input_size, layer_size), ]
            for _ in range(layer_num-1):
                layers.append(nn.Linear(layer_size, layer_size))
            layers.append(nn.Linear(layer_size, input_size))
            self.layers = nn.ModuleList(layers)

    def forward(self, x, x_z, y_z):
        split_dim = x_z.shape[-1]
        x_g = torch.cat([x_z, y_z], dim=-1)
        for i,layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                x_g = layer(x_g)
            else:
                x_g = torch.relu(layer(x_g))
        return x_g[:, :split_dim], x_g[:, split_dim:]


class DEPTS_EM(nn.Module):
    def __init__(self, local_blocks: nn.ModuleList, period_blocks: nn.ModuleList, num_series: int):
        super().__init__()
        assert len(local_blocks) == len(period_blocks)
        self.local_blocks = local_blocks
        self.global_blocks = period_blocks # call periodic blocks as global blocks
        # Use alpha to optimize each series in fine-grained: each series has its an alpha
        self.alpha = nn.Parameter(torch.ones(num_series))

    def forward(self, x, x_z, input_mask, y_z, ids):
        residuals = x.flip(dims=(1,))
        global_x_residuals, global_y_residuals = x_z.flip(dims=(1,)), y_z # call 'z' as global part
        input_mask = input_mask.flip(dims=(1,))
        forecast = 0
        global_part, local_part = 0, 0
        for i in range(len(self.local_blocks)):

            # get global backcast, forecast
            local_block, global_block = self.local_blocks[i], self.global_blocks[i]
            global_backcast, global_forecast = global_block(residuals, global_x_residuals, global_y_residuals)

            # alpha strategy
            ids_alpha = self.alpha.index_select(0, ids.long().view(-1)).reshape(-1, 1)
            global_backcast = global_backcast * ids_alpha
            global_forecast = global_forecast * ids_alpha
            global_x_residuals = global_x_residuals - global_backcast
            global_y_residuals = global_y_residuals - global_forecast
            residuals = (residuals - global_backcast) * input_mask

            # get local backcast, forecast
            local_backcast, local_forecast = local_block(residuals)
            residuals = (residuals - local_backcast) * input_mask
            # get different parts
            local_part = local_part + local_forecast
            global_part = global_part + global_forecast
            forecast = global_part + local_part

        return forecast, global_part, local_part


def depts_expansion_general(input_size: int, output_size: int, layer_size: int, stacks: int, local_layers: int, period_layers: int, num_series: int):
    return DEPTS_EM(
        nn.ModuleList([
            LocalBlock(input_size=input_size,
                       theta_size=input_size + output_size,
                       basis_function=GenericBasis(backcast_size=input_size, forecast_size=output_size),
                       layers=local_layers,
                       layer_size=layer_size) for _ in range(stacks)
        ]),
        nn.ModuleList([
            PeriodBlock(layer_num=period_layers,
                                 input_size=input_size+output_size,
                                 layer_size=layer_size) for _ in range(stacks)
        ]),
        num_series
    )


# Periodicity Module
class PeriodicityModule(nn.Module):
    def __init__(self, all_a, all_p, all_f, all_mean):
        super(PeriodicityModule, self).__init__()
        # parameters
        all_a, all_p, all_f, all_mean = map(to_tensor, (all_a, all_p, all_f, all_mean))
        self.num_series, self.K = all_a.shape # num_series * K
        # inner cos
        self.layers1_weight = torch.nn.Parameter(all_f.unsqueeze(1))  # num_series * 1 * K
        self.layers1_bias = torch.nn.Parameter(all_p.unsqueeze(1)) # num_series * 1 * K
        # outside cos
        self.layers2_weight = torch.nn.Parameter(all_a.unsqueeze(-1)) # num_series * K * 1
        self.layers2_bias = torch.nn.Parameter(all_mean.unsqueeze(-1).unsqueeze(-1)) # num_series * 1 * 1
    def forward(self, x, series_id, inputKmask=None):
        # x: batch * backcast_length,  series_id: batch
        # output: batch * backcast_length
        batch_layer1_weight = self.layers1_weight.index_select(0, series_id.view(-1))  # batch * 1 * K
        batch_layer1_bias = self.layers1_bias.index_select(0, series_id.view(-1)) # batch * 1 * K
        batch_layer2_weight = self.layers2_weight.index_select(0, series_id.view(-1))  # batch * K * 1
        batch_layer2_bias = self.layers2_bias.index_select(0, series_id.view(-1)) # batch * 1 * 1
        # forward function
        # batch * backcast * K
        x = 2 * np.pi * torch.bmm(x.unsqueeze(-1), batch_layer1_weight) + batch_layer1_bias
        if inputKmask is None:
            x = torch.cos(x)
        else: # inputKmask: batch * K
            x = torch.mul(torch.cos(x), inputKmask.unsqueeze(1).repeat(1,x.shape[1],1))
        x = torch.bmm(x, batch_layer2_weight) + batch_layer2_bias
        return x.squeeze(-1)