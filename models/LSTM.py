"""
    This file contains LSTM based models for intensity prediction
"""

# imports
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np


# modules
class ScaledSoftplus(nn.Module):
    def __init__(self):
        """
           input:
                  None
           model parameters:
                  s - softplus scaling coefficient, trainable
        """
        super().__init__()
        self.s = Parameter(torch.ones(1))

    def forward(self, x):
        """
           forward pass

           input:
                  x - torch.Tensor

           output:
                  scaled_softplus(x) - torch.Tensor, shape = x.shape
        """
        return self.s * torch.log(1 + torch.exp(x / self.s))


class LSTMSinglePointProcess(nn.Module):
    """
        Single Point Process Model
    """

    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 batch_first=True, dropout=0, bidirectional=False):
        """
           input:
                  input_size - int, input size of the data for LSTM
                  hidden_size - int, LSTM hidden state size
                  num_layers - int, number of LSTM layers
                  num_classes - int, number of types of events that can occur
                  batch_first - bool, whether the batch should go first in LSTM
                  dropout - float (>=0,<1), dropout probability for all LSTM layers but the last one
                  bidirectional - bool, bidirectional LSTM or not

           model parameters:
                  lstm - torch.nn.Module, LSTM model
                  hidden0 - torch.nn.parameter.Parameter, initial hidden state
                  cell0 - torch.nn.parameter.Parameter, initial cell state
                  W - torch.nn.parameter.Parameter, weighs for mapping hidden state to lambda
                  f - torch.nn.Module, Scaled Softplus
                  num_classes - int, number of types of events that can occur
                  bidir - bool, bidirectional LSTM or not
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=batch_first, dropout=dropout,
                            bidirectional=bidirectional)
        if bidirectional:
            self.hidden0 = Parameter(torch.randn(num_layers * 2, hidden_size))
            self.cell0 = Parameter(torch.randn(num_layers * 2, hidden_size))
            self.W = Parameter(torch.randn(hidden_size * 2, num_classes))
        else:
            self.hidden0 = Parameter(torch.randn(num_layers, hidden_size))
            self.cell0 = Parameter(torch.randn(num_layers, hidden_size))
            self.W = Parameter(torch.randn(hidden_size, num_classes))
        self.f = ScaledSoftplus()
        self.num_classes = num_classes
        self.bidir = bidirectional

    def forward(self, s, provide_states=False):
        """
           forward pass of the model

           input:
                  s - torch.Tensor, size = (batch size, sequence length, input size)
                  provide_states - bool, if True, model returns states also

           output:
                  lambdas - torch.Tensor, size = (batch size, sequence length, num_classes)
                  if provide_states:
                     hidden - torch.Tensor, hidden state of LSTM
                     cell - torch.Tensor, cell state of LSTM
        """
        bs, seq_len, _ = s.shape

        # preparing hidden and cell states
        hidden0 = self.hidden0[:, None, :].repeat(1, bs, 1)
        cell0 = self.cell0[:, None, :].repeat(1, bs, 1)

        # processing input with LSTM
        out, (hidden, cell) = self.lstm(s, (hidden0, cell0))

        # finding lambdas
        lambdas = torch.zeros(bs, seq_len, self.num_classes)
        if self.bidir:
            h0 = hidden0[-3:, 0, :].reshape(-1)
        else:
            h0 = hidden0[-1, 0, :].reshape(-1)
        # initial lambdas
        lambdas[:, 0, :] = self.f(h0 @ self.W)[None, :].repeat(bs, 1)
        # lambdas during the process
        lambdas[:, 1:, :] = self.f(out[:, :-1, :] @ self.W)

        if not provide_states:
            return lambdas
        return lambdas, hidden, cell

    def simulate(self, batch_size, dt, seq_len, verbose=False):
        """
           conducts simulation of the process with model parameters

           input:
                  batch_size - int, number of sequences to generate
                  dt - Tensor like, size = (batch_size), delta time during the generation (Poisson = Poisson(lambda*dt))
                  seq_len - int, sequence length
                  verbose - bool, if True, print the info during generation

           output:
                  sequences - torch.Tensor, size = (batch_size, seq_len, num_classes), simulated data
        """
        with torch.no_grad():
            self.eval()

            # result template
            res = torch.zeros(batch_size, seq_len, 1 + self.num_classes)

            # initial hidden and cell state preprocessing
            hidden0 = self.hidden0[:, None, :]
            cell0 = self.cell0[:, None, :]

            # iterations over batch
            for b in range(batch_size):
                if verbose:
                    print('Generating batch {}/{}'.format(b + 1, batch_size))
                # iteration over sequence
                for s in range(seq_len):
                    if verbose and s % 100 == 0:
                        print('>>> Generating sequence step {}/{}'.format(s + 1, seq_len))

                    # first iteration doesn't depend on the history and should be processed independently
                    if s == 0:
                        # computing lambda
                        if self.bidir:
                            h0 = hidden0[-3:, :, :].reshape(-1)
                        else:
                            h0 = hidden0[-1, :, :].reshape(-1)
                        lambdas = self.f(h0 @ self.W)
                        # simulation
                        res[b, s, 1:] = torch.poisson(lambdas * dt[b])
                        res[b, s, 0] = dt[b]
                        hidden = hidden0.clone()
                        cell = cell0.clone()
                    else:
                        # computing lambda
                        o, (hidden, cell) = self.lstm(res[b, s - 1][None, None, :], (hidden, cell))
                        lambdas = self.f(o[0, -1, :] @ self.W)
                        # simulation
                        res[b, s, 1:] = torch.poisson(lambdas * dt[b])
                        res[b, s, 0] = dt[b]
            return res


class LSTMMultiplePointProcesses(nn.Module):
    """
        Multiple Point Processes Model, Point Processes are distinguished with different initial hidden states
    """

    def __init__(self, input_size, hidden_size, num_layers, num_classes, num_clusters, n_steps,
                 batch_first=True, dropout=0, bidirectional=False):
        """
           input:
                  input_size - int, input size of the data for LSTM
                  hidden_size - int, LSTM hidden state size
                  num_layers - int, number of LSTM layers
                  num_classes - int, number of types of events that can occur
                  num_clusters - int, number of different point processes
                  n_steps - int, sequence length (used for batch normalization)
                  batch_first - bool, whether the batch should go first in LSTM
                  dropout - float (>=0,<1), dropout probability for all LSTM layers but the last one
                  bidirectional - bool, bidirectional LSTM or not

           model parameters:
                  hidden_size - int, LSTM hidden state size
                  lstm - torch.nn.Module, LSTM model
                  bn - torch.nn.Module, Batch Normalization
                  hidden0 - torch.nn.parameter.Parameter, initial hidden states, size[0] = num_clusters
                  cell0 - torch.nn.parameter.Parameter, initial cell states, size[0] = num_clusters
                  W - torch.nn.parameter.Parameter, weighs for mapping hidden state to lambda
                  f_{k} - torch.nn.Module, Scaled Softplus, k - number of point process, 0<=k<num_clusters
                  num_classes - int, number of types of events that can occur
                  num_clusters - int, number of different point processes
                  bidir - bool, bidirectional LSTM or not
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=batch_first, dropout=dropout,
                            bidirectional=bidirectional)
        self.bn = nn.BatchNorm1d(n_steps)
        if bidirectional:
            self.hidden0 = Parameter(self.init_weigh(num_clusters, num_layers * 2, hidden_size))
            self.cell0 = Parameter(self.init_weigh(num_clusters, num_layers * 2, hidden_size))
            self.W = Parameter(self.init_weigh(hidden_size * 2, num_classes))
        else:
            self.hidden0 = Parameter(self.init_weigh(num_clusters, num_layers, hidden_size))
            self.cell0 = Parameter(self.init_weigh(num_clusters, num_layers, hidden_size))
            self.W = Parameter(self.init_weigh(hidden_size, num_classes))
        for k in range(num_clusters):
            setattr(self, 'f_{}'.format(k), ScaledSoftplus())
        self.num_classes = num_classes
        self.num_clusters = num_clusters
        self.bidir = bidirectional

    def random_permutation(self):
        """
            Conducts random permutation of clusters

            input:
                   None
            output:
                   None
        """
        new_order = np.random.permutation(self.num_clusters)
        self.hidden0 = Parameter(self.hidden0[new_order, :, :])
        self.cell0 = Parameter(self.cell0[new_order, :, :])

    def init_weigh(self, *args):
        """
           Used for weight initialization, output ~ U(-1/sqrt(hidden_size),1/sqrt(hidden_size))

           input:
                  args - arguments, used for torch.rand, should be desired size of Tensor
           output:
                  weighs - torch.Tensor, size = args
        """
        tmp = self.hidden_size
        return torch.rand(*args) * 2 / tmp - 1 / tmp

    def forward(self, s):
        """
            forward pass of the model

            input:
                   s - torch.Tensor, size = (batch size, sequence length, input size)

            output:
                   lambdas - torch.Tensor, size = (batch size, sequence length, num_classes)
        """
        bs, seq_len, _ = s.shape

        # preparing lambdas template
        lambdas = torch.zeros(self.num_clusters, bs, seq_len, self.num_classes)

        # iterating over point processes
        for k in range(self.num_clusters):
            # hidden and cell state preprocessing
            hidden0 = self.hidden0[k, :, None, :].repeat(1, bs, 1)
            cell0 = self.cell0[k, :, None, :].repeat(1, bs, 1)

            # LSTM forward pass
            out, (hidden, cell) = self.lstm(s, (hidden0, cell0))
            out = self.bn(out)

            # finding lambdas
            if self.bidir:
                h0 = hidden0[-3:, 0, :].reshape(-1)
            else:
                h0 = hidden0[-1, 0, :].reshape(-1)

            # first lambda doesn't depend on history and depends only on the initial hidden state
            lambdas[k, :, 0, :] = getattr(self, 'f_{}'.format(k))(h0 @ self.W)[None, :].repeat(bs, 1)
            lambdas[k, :, 1:, :] = getattr(self, 'f_{}'.format(k))(out[:, :-1, :] @ self.W)
        return lambdas
