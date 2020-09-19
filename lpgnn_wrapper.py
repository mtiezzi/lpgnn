import torch
import torch.nn as nn
import torch.nn.functional as F
import dataloader
import torch.optim as optim
from abc import ABCMeta, abstractmethod
from utils import Accuracy
from torch.utils.tensorboard import SummaryWriter
import torchvision
from utils import matplotlib_imshow
import utils
from pylpgnn import LPGNN
import time
import itertools


class LPGNNWrapper:
    class Config:
        def __init__(self):
            self.device = None
            self.use_cuda = None
            self.dataset_path = None
            self.log_interval = None
            self.tensorboard = None
            self.task_type = None

            # hyperparams
            self.lrw = None
            self.lrx = None
            self.lrλ = None
            self.loss_f = None
            self.layers = None
            self.n_nodes = None
            self.state_dim = None
            self.label_dim = None
            self.output_dim = None
            self.graph_based = False
            self.activation = torch.nn.Tanh()
            self.state_transition_hidden_dims = None
            self.output_function_hidden_dims = None
            self.epochs = None
            self.optimizer = optim.Adam
            self.eps = None
            self.state_constraint_function = None
            # optional
            self.loss_w = 1.
            self.energy_weight = 0.
            self.l2_weight = 0.

    def __init__(self, config: Config):
        self.config = config

        # to be populated
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.test_loader = None

        if self.config.tensorboard:
            self.writer = SummaryWriter('logs/tensorboard')
        self.first_flag_writer = True

    def __call__(self, dset, state_net=None, out_net=None):
        # handle the dataset info
        self._data_loader(dset)
        self.lpgnn = LPGNN(self.config, state_net, out_net).to(self.config.device)
        self._criterion()
        self._optimizer()
        self._accuracy()
        self.lpgnn.reset_parameters()

    def _data_loader(self, dset):  # handle dataset data and metadata
        self.dset = dset.to(self.config.device)
        self.config.label_dim = self.dset.node_label_dim
        self.config.n_nodes = self.dset.num_nodes
        self.config.output_dim = self.dset.num_classes

    def _optimizer(self):
        # list_parasms = [p for p in self.lpgnn.parameters()]
        par_st = []
        for i in range(len(self.lpgnn.state_transition_function_list)):
            for p in self.lpgnn.state_transition_function_list[i].parameters():
                par_st.append(p)
        par_out = [p for p in self.lpgnn.output_function.parameters()]

        self.w_optimizer = self.config.optimizer(
            par_st + par_out,
            lr=self.config.lrw)

        self.x_optimizer = self.config.optimizer(self.lpgnn.state_variable_list, lr=self.config.lrx)
        self.λ_optimizer = self.config.optimizer(self.lpgnn.λ_list, lr=self.config.lrλ)

    def _criterion(self):
        self.criterion = nn.CrossEntropyLoss()

    def _accuracy(self):
        self.TrainAccuracy = Accuracy(type=self.config.task_type)
        self.ValidAccuracy = Accuracy(type=self.config.task_type)
        self.TestAccuracy = Accuracy(type=self.config.task_type)

    def train_step(self, epoch):
        self.lpgnn.train()
        data = self.dset
        self.lpgnn.zero_grad()
        self.TrainAccuracy.reset()
        # output computation
        new_state, output = self.lpgnn(data.edges, data.agg_matrix, data.node_labels)

        # outloss
        output_loss = self.criterion(output, data.targets)
        # loss computation - semisupervised
        convergence_loss = self.lpgnn.lagrangian_composition(new_state)
        loss = self.config.loss_w * output_loss + convergence_loss  # TODO add more layers

        if self.config.tensorboard:
            self.writer.add_scalar('Lagrangian',
                                   loss,
                                   epoch)
            self.writer.add_scalar('Output Loss',
                                   output_loss,
                                   epoch)
            self.writer.add_scalar('Constraint Loss',
                                   convergence_loss,
                                   epoch)

        loss.backward()

        with torch.no_grad():
            # Do gradient ascent on lagrangian multipliers
            for l in range(self.config.layers):
                self.lpgnn.λ_list[l].grad.mul_(-1)

        self.w_optimizer.step()
        self.x_optimizer.step()
        self.λ_optimizer.step()

        # # updating accuracy
        # batch_acc = self.TrainAccuracy.update((output, target), batch_compute=True)
        with torch.no_grad():  # Accuracy computation
            # accuracy_train = torch.mean(
            #     (torch.argmax(output[data.idx_train], dim=-1) == data.targets[data.idx_train]).float())
            self.TrainAccuracy.update(output, data.targets)
            accuracy_train = self.TrainAccuracy.compute()

            if epoch % self.config.log_interval == 0:
                print(
                    f'Train Epoch: {epoch} \t  Loss: {loss:.4f}\t  OutLoss: {output_loss:.4f}\t  ConstrLoss: {convergence_loss:.4f}\t  Acc.: {accuracy_train:.4f} ({self.TrainAccuracy.get_best():.4f})')

                if self.config.tensorboard:
                    self.writer.add_scalar('Training Accuracy',
                                           accuracy_train,
                                           epoch)
                    self.writer.add_scalar('Training Loss',
                                           loss,
                                           epoch)
                    # self.writer.add_scalar('Training Iterations',
                    #                        iterations,
                    #                        epoch)

                    for name, param in self.lpgnn.named_parameters():
                        self.writer.add_histogram(name, param, epoch)
        # self.TrainAccuracy.reset()

    def predict(self, edges, agg_matrix, node_labels):
        return self.lpgnn(edges, agg_matrix, node_labels)

    def test_step(self, epoch):
        ####  TEST
        self.lpgnn.eval()
        data = self.dset
        self.lpgnn.zero_grad()
        self.TestAccuracy.reset()

        new_state, output = self.lpgnn(data.edges, data.agg_matrix, data.node_labels)
        test_loss = self.criterion(output, data.targets)
        convergence_loss = self.lpgnn.lagrangian_composition(new_state)  # TODO check
        self.TestAccuracy.update(output, data.targets)
        acc_test = self.TestAccuracy.compute()

        # inference optimization

        loss = convergence_loss

        if self.config.tensorboard:
            self.writer.add_scalar('Test Lagrangian',
                                   loss,
                                   epoch)
            self.writer.add_scalar('Test Output Loss',
                                   test_loss,
                                   epoch)
            self.writer.add_scalar('Test Constraint Loss',
                                   convergence_loss,
                                   epoch)
        loss.backward()

        with torch.no_grad():
            # Do gradient ascent on lagrangian multipliers
            for l in range(self.config.layers):
                self.lpgnn.λ_list[l].grad.mul_(-1)

        # self.w_optimizer.step() WEIGHTS NOT UPDATED
        self.x_optimizer.step()
        self.λ_optimizer.step()

        if epoch % self.config.log_interval == 0:
            print(
                f'Test Epoch: {epoch} \t  Loss: {test_loss:.4f}\t   Acc.Test: {acc_test:.4f} ({self.TestAccuracy.get_best():.4f})')

            if self.config.tensorboard:
                self.writer.add_scalar('Test Accuracy',
                                       acc_test,
                                       epoch)
                self.writer.add_scalar('Test Loss',
                                       test_loss,
                                       epoch)
                # self.writer.add_scalar('Test Iterations',

    def valid_step(self, epoch):
        ####  TEST
        self.lpgnn.eval()
        data = self.dset
        self.lpgnn.zero_grad()
        self.ValidAccuracy.reset()

        new_state, output = self.lpgnn(data.edges, data.agg_matrix, data.node_labels)
        valid_loss = self.criterion(output, data.targets)
        convergence_loss = self.lpgnn.lagrangian_composition(new_state)  # TODO check

        self.ValidAccuracy.update(output, data.targets)
        acc_valid = self.ValidAccuracy.compute()
        # inference optimization

        loss = convergence_loss

        if self.config.tensorboard:
            self.writer.add_scalar('Valid Lagrangian',
                                   loss,
                                   epoch)
            self.writer.add_scalar('Valid Output Loss',
                                   valid_loss,
                                   epoch)
            self.writer.add_scalar('Valid Constraint Loss',
                                   convergence_loss,
                                   epoch)
        loss.backward()

        with torch.no_grad():
            # Do gradient ascent on lagrangian multipliers
            for l in range(self.config.layers):
                self.lpgnn.λ_list[l].grad.mul_(-1)

        # self.w_optimizer.step() WEIGHTS NOT UPDATED
        self.x_optimizer.step()
        self.λ_optimizer.step()

        if epoch % self.config.log_interval == 0:
            # print('Valid set: Average loss: {:.4f}, Accuracy:  ({:.4f}) , Best Accuracy:  ({:.4f})'.format(
            #     test_loss, acc_valid, self.ValidAccuracy.get_best()))
            print(
                f'Valid Epoch: {epoch} \t  Loss: {valid_loss:.4f}\t   Acc.Test: {acc_valid:.4f} ({self.ValidAccuracy.get_best():.4f})')

            if self.config.tensorboard:
                self.writer.add_scalar('Valid Accuracy',
                                       acc_valid,
                                       epoch)
                self.writer.add_scalar('Valid Loss',
                                       valid_loss,
                                       epoch)
                # self.writer.add_scalar('Valid Iterations',
                #                        iterations,
                #                        epoch)
            print("---------------------------------")


class SemiSupLPGNNWrapper(LPGNNWrapper):

    def __init__(self, config: LPGNNWrapper.Config):
        super().__init__(config)
        self.PRINT_DEBUG = False

    def __call__(self, dset):
        # handle the dataset info
        self._data_loader(dset)
        self.lpgnn = LPGNN(self.config).to(self.config.device)
        self._criterion()
        self._optimizer()
        self._accuracy()
        self.lpgnn.reset_parameters()

    def _data_loader(self, dset):  # handle dataset data and metadata
        self.dset = dset.to(self.config.device)
        self.config.label_dim = self.dset.node_label_dim
        self.config.n_nodes = self.dset.num_nodes
        self.config.output_dim = self.dset.num_classes

    def _accuracy(self):
        self.TrainAccuracy = Accuracy(type="semisupervised")
        self.ValidAccuracy = Accuracy(type="semisupervised")
        self.TestAccuracy = Accuracy(type="semisupervised")

    def global_step(self, epoch, start_get=None):
        self.lpgnn.train()
        data = self.dset
        self.lpgnn.zero_grad()
        self.TrainAccuracy.reset()
        # output computation
        new_state, output = self.lpgnn(data.edges, data.agg_matrix, data.node_labels)
        # loss computation - semisupervised
        output_loss = self.criterion(output[data.idx_train], data.targets[data.idx_train])
        convergence_loss = self.lpgnn.lagrangian_composition(new_state)  # on all the constraints, all indexes
        loss = self.config.loss_w * output_loss + convergence_loss

        if self.PRINT_DEBUG:
            step_get = time.time() - start_get
            with open('logger_MIO_init_ON.txt', "a") as f:
                f.write(f"Epoch {epoch} Time from start: {step_get}  \n  State Transition  Weight_0:\n ")
                f.write(f"{self.lpgnn.state_transition_function.mlp.mlp[0].weight.data.to('cpu').numpy()}\n\n")
                f.write(f"State Transition Weight_1:\n ")
                f.write(f"{self.lpgnn.state_transition_function.mlp.mlp[2].weight.data.to('cpu').numpy()}\n\n")
                f.write(f"\n###################\n\n Output function \n Weight_0:\n ")
                f.write(f"{self.lpgnn.output_function.mlp[0].weight.data.to('cpu').numpy()}\n\n")
                f.write(f"\n Output function \n Weight_1:\n ")
                f.write(f"{self.lpgnn.output_function.mlp[2].weight.data.to('cpu').numpy()}\n\n")
                f.write(f"###################\n\n New State\n")
                f.write(f"{new_state.detach().to('cpu').numpy()[:20]}\n\n")
                f.write(f"###################\n\n Output\n")
                f.write(f'{output.detach().to("cpu").numpy()[:20]}\n\n')
                f.write(f"###################\n\n State variable\n")
                f.write(f"{self.lpgnn.state_variable.detach().to('cpu').numpy()[:20]}\n\n")
                f.write(f"###################\n\n lambda \n")
                f.write(f"{self.lpgnn.λ.detach().to('cpu').numpy()[:20]}\n\n")
                f.write(
                    f"\n\nTotal loss: {loss.detach().to('cpu').numpy()} \t Output_loss: {output_loss.detach().to('cpu').numpy()} \t Constr_loss: {convergence_loss.to('cpu').detach().numpy()}")
                f.write(
                    "=============================================\n=============================================\n\n\n")

        if self.config.tensorboard:
            self.writer.add_scalar('Lagrangian',
                                   loss,
                                   epoch)
            self.writer.add_scalar('Output Loss',
                                   output_loss,
                                   epoch)
            self.writer.add_scalar('Constraint Loss',
                                   convergence_loss,
                                   epoch)
        loss.backward()

        with torch.no_grad():
            # Do gradient ascent on lagrangian multipliers
            for l in range(self.config.layers):
                self.lpgnn.λ_list[l].grad.mul_(-1)

        self.w_optimizer.step()
        self.x_optimizer.step()
        self.λ_optimizer.step()

        # # updating accuracy
        # batch_acc = self.TrainAccuracy.update((output, target), batch_compute=True)
        with torch.no_grad():  # Accuracy computation
            # accuracy_train = torch.mean(
            #     (torch.argmax(output[data.idx_train], dim=-1) == data.targets[data.idx_train]).float())
            self.TrainAccuracy.update(output, data.targets, idx=data.idx_train)
            accuracy_train = self.TrainAccuracy.compute()

            self.ValidAccuracy.update(output, data.targets, idx=data.idx_valid)
            acc_valid = self.ValidAccuracy.compute()

            # self.TestAccuracy.update(output, data.targets, idx=data.idx_test)
            # acc_test = self.TestAccuracy.compute()

            if epoch % self.config.log_interval == 0:
                print(
                    f'Epoch: {epoch:05d} \t  Loss: {loss:.4f}\t  OutLoss: {output_loss:.4f}\t '
                    f' ConstrLoss: {convergence_loss:.4f}\t '
                    f' Acc.Tr.: {accuracy_train:.4f} ({self.TrainAccuracy.get_best():.4f}) '
                    f' Acc.Vl.: {acc_valid:.4f} ({self.ValidAccuracy.get_best():.4f})')

                if self.config.tensorboard:
                    self.writer.add_scalar('Training Accuracy',
                                           accuracy_train,
                                           epoch)
                    self.writer.add_scalar('Training Loss',
                                           loss,
                                           epoch)
                    # self.writer.add_scalar('Training Iterations',
                    #                        iterations,
                    #                        epoch)

                    for name, param in self.lpgnn.named_parameters():
                        self.writer.add_histogram(name, param, epoch)
            # self.TrainAccuracy.reset()
        return output  # used for plotting

    def predict(self, edges, agg_matrix, node_labels):
        return self.lpgnn(edges, agg_matrix, node_labels)

    # def test_step(self, epoch):
    #     ####  TEST
    #     self.lpgnn.eval()
    #     data = self.dset
    #     self.TestAccuracy.reset()
    #     new_state, output = self.lpgnn(data.edges, data.agg_matrix, data.node_labels)
    #     test_loss = self.criterion(output[data.idx_test], data.targets[data.idx_test])
    #     self.TestAccuracy.update(output, data.targets, idx=data.idx_test)
    #     acc_test = self.TestAccuracy.compute()
    #
    #     # inference optimization
    #
    #     if self.config.tensorboard:
    #         self.writer.add_scalar('Test Output Loss',
    #                                test_loss,
    #                                epoch)
    #
    #     if epoch % self.config.log_interval == 0:
    #         print(
    #             f'Test Epoch: {epoch:05d} \t  Loss: {test_loss:.4f}\t   Acc.Test: {acc_test:.4f} ({self.TestAccuracy.get_best():.4f})')
    #
    #         if self.config.tensorboard:
    #             self.writer.add_scalar('Test Accuracy',
    #                                    acc_test,
    #                                    epoch)
    #             self.writer.add_scalar('Test Loss',
    #                                    test_loss,
    #                                    epoch)
    #             # self.writer.add_scalar('Test Iterations',
    #
    # def valid_step(self, epoch):
    #     ####  TEST
    #     self.lpgnn.eval()
    #     data = self.dset
    #     self.ValidAccuracy.reset()
    #
    #     new_state, output = self.lpgnn(data.edges, data.agg_matrix, data.node_labels)
    #     valid_loss = self.criterion(output[data.idx_valid], data.targets[data.idx_valid])
    #
    #     self.ValidAccuracy.update(output, data.targets, idx=data.idx_valid)
    #     acc_valid = self.ValidAccuracy.compute()
    #     # inference optimization
    #
    #     if epoch % self.config.log_interval == 0:
    #         print(
    #             f'Valid Epoch: {epoch} \t  Loss: {valid_loss:.4f}\t   Acc.Test: {acc_valid:.4f} ({self.ValidAccuracy.get_best():.4f})')
    #
    #         if self.config.tensorboard:
    #             self.writer.add_scalar('Valid Accuracy',
    #                                    acc_valid,
    #                                    epoch)
    #             self.writer.add_scalar('Valid Loss',
    #                                    valid_loss,
    #                                    epoch)
    #             # self.writer.add_scalar('Valid Iterations',
    #             #                        iterations,
    #             #                        epoch)
    #         print("---------------------------------")
