import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn import init
import math
import utils
from net import MLP, StateTransition


class LPGNN(nn.Module):

    def __init__(self, config, state_net=None, out_net=None, nodewise_lagrangian_flag=True,
                 dimwise_lagrangian_flag=False, ):
        super(LPGNN, self).__init__()

        self.config = config

        # hyperparameters and general properties
        self.n_nodes = config.n_nodes
        self.state_dim = config.state_dim
        self.label_dim = config.label_dim
        self.output_dim = config.output_dim
        self.state_transition_hidden_dims = config.state_transition_hidden_dims
        self.output_function_hidden_dims = config.output_function_hidden_dims

        # TODO add dims of layered LPGNN
        self.λ_n = self.n_nodes if nodewise_lagrangian_flag else 1
        self.λ_d = self.state_dim if dimwise_lagrangian_flag else 1

        self.λ_list = nn.ParameterList()
        self.state_variable_list = nn.ParameterList()
        self.diffusion_constraint_list = []
        self.node_state_list = []
        self.state_transition_function_list = nn.ModuleList()  # to be visible for parameters and gpu
        # state constraints

        self.state_constraint_function = utils.get_G_function(descr=self.config.state_constraint_function,
                                                              eps=self.config.eps)

        for l in range(self.config.layers):  # loop over layers
            # defining lagrangian multipliers
            self.λ_list.append(nn.Parameter(
                torch.zeros(*[self.λ_n, self.λ_d]), requires_grad=True))  # (n,1) by default ,

            self.state_variable_list.append(nn.Parameter(
                torch.zeros(*[self.n_nodes, self.state_dim[l]]), requires_grad=True))  # (n,d_state)

            # adding to lists

            # node state initialization
            # self.node_state_list.append(
            #     torch.zeros(*[self.n_nodes, self.state_dim[l]]).to(self.config.device))  # (n,d_n)
            # state and output transition functions
            if state_net is None:
                # torch.manual_seed(self.config.seed)
                if l == 0:
                    input_dim = self.state_dim[0] + 2 * self.label_dim  # arc state computation f(l_v, l_n, x_n)

                else:

                    ## f(x_v_l, x_v_l-1, x_n_l-1) 
                    input_dim = self.state_dim[l] + 2 * self.state_dim[l - 1]  # + 2 * self.label_dim ##

                output_dim = self.state_dim[l]
                self.state_transition_function_list.append(StateTransition(input_dim=input_dim,
                                                                           output_dim=output_dim,
                                                                           mlp_hidden_dim=self.state_transition_hidden_dims,
                                                                           activation_function=config.activation))

        if state_net is not None:  # only once, give as input the list TODO
            self.state_transition_function_list = state_net

        if out_net is None:

            self.output_function = MLP(self.state_dim[-1], self.output_function_hidden_dims, self.output_dim)
        else:
            self.output_function = out_net

        self.graph_based = self.config.graph_based

    def reset_parameters(self):
        with torch.no_grad():
            for l in range(self.config.layers):
                self.state_transition_function_list[l].mlp.init()
                nn.init.constant_(self.state_variable_list[l], 0)
                nn.init.constant_(self.λ_list[l], 0)
            self.output_function.init()

    def lagrangian_composition(self, new_state_list):

        # loss definition  TODO add for , add tensorboard
        convergence_loss_list = []
        for l in range(self.config.layers):
            constraint = self.state_constraint_function(self.state_variable_list[l] - new_state_list[l])
            convergence_loss_list.append(torch.mean(torch.mean(self.λ_list[l] * constraint, -1)))
        # self.diffusion_constraint_list.append(torch.mean(torch.mean(self.λ * constraint, -1))) TODO why not working
        # convergence_loss = torch.mean(torch.mean(self.λ * constraint, -1))

        return torch.sum(torch.stack(convergence_loss_list), dim=0)
        # if use_energy_constraint:
        #     total_loss += lpgnn.energy_loss(lpgnn.state_variable, new_state)

    def forward(self,
                edges,
                agg_matrix,
                node_labels,
                node_states=None,
                graph_agg=None
                ):
        # convergence loop
        # state initialization
        node_states = self.state_variable_list[0] if node_states is None else node_states  #
        new_state_list = []

        for l in range(self.config.layers):
            if l == 0:
                new_layer_state = self.state_transition_function_list[l](node_states, node_labels, edges, agg_matrix, l)
                new_state_list.append(new_layer_state)
            else:
                new_layer_state = self.state_transition_function_list[l](self.state_variable_list[l], new_layer_state,
                                                                         edges, agg_matrix, l)
                new_state_list.append(new_layer_state)
        new_state = new_layer_state
        output = self.output_function(new_state)

        return new_state_list, output
