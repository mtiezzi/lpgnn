import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
import utils
import dataloader
import torch.optim as optim
from lpgnn_wrapper import LPGNNWrapper


#
# # fix random seeds for reproducibility
# SEED = 123
# torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(SEED)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch')
    parser.add_argument('--epochs', type=int, default=100000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--cuda_dev', type=int, default=0,
                        help='select specific CUDA device for training')
    parser.add_argument('--n_gpu_use', type=int, default=1,
                        help='select number of CUDA device for training')
    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='logging training status cadency')
    parser.add_argument('--tensorboard', action='store_true', default=True,
                        help='For logging the model in tensorboard')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if not use_cuda:
        args.n_gpu_use = 0

    device = utils.prepare_device(n_gpu_use=args.n_gpu_use, gpu_id=args.cuda_dev)
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # torch.manual_seed(args.seed)
    # # fix random seeds for reproducibility
    # SEED = 123
    # torch.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(SEED)

    # configugations
    cfg = LPGNNWrapper.Config()
    cfg.use_cuda = use_cuda
    cfg.device = device

    cfg.log_interval = args.log_interval
    cfg.tensorboard = args.tensorboard

    # cfg.batch_size = args.batch_size
    # cfg.test_batch_size = args.test_batch_size
    # cfg.momentum = args.momentum

    cfg.dataset_path = './data'
    cfg.epochs = args.epochs
    cfg.lrw = args.lr
    cfg.activation = nn.Tanh()
    cfg.state_transition_hidden_dims = [15, ]
    cfg.output_function_hidden_dims = [5, ]
    cfg.state_dim = [10, ]
    # cfg.state_dim = 10
    cfg.graph_based = False
    cfg.log_interval = 10
    cfg.lrw = 0.01
    cfg.lrx = 0.01
    cfg.lrλ = 0.001
    cfg.task_type = "multiclass"
    cfg.layers = len(cfg.state_dim) if type(
        cfg.state_dim) is list else 1  # getting number of LPGNN layers from state_dim list

    # LPGNN
    cfg.eps = 1e-6
    cfg.state_constraint_function = "eps"
    cfg.optimizer = optim.SGD

    # model creation
    model_tr = LPGNNWrapper(cfg)
    model_val = LPGNNWrapper(cfg)
    model_tst = LPGNNWrapper(cfg)
    # dataset creation
    dset = dataloader.get_subgraph(set="sub_30_15_200", aggregation_type="sum",
                                   sparse_matrix=True)  # generate the dataset
    model_tr(dset["train"])  # dataset initalization into the GNN
    model_val(dset["validation"], state_net=model_tr.lpgnn.state_transition_function_list,
              out_net=model_tr.lpgnn.output_function)  # dataset initalization into the GNN
    model_tst(dset["test"], state_net=model_tr.lpgnn.state_transition_function_list,
              out_net=model_tr.lpgnn.output_function)  # dataset initalization into the GNN

    # training code
    for epoch in range(1, args.epochs + 1):
        model_tr.train_step(epoch)

        model_tst.test_step(epoch)
        model_val.valid_step(epoch)
    # model.test_step()

    # if args.save_model:
    #     torch.save(model.gnn.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
