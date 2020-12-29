import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def log_gaussian_prob(x, mu, sigma):
    element_wise_log_prob = -0.5 * torch.Tensor([np.log(2 * np.pi)]).to(mu.device) - torch.log(sigma) - 0.5 * (
                x - mu) ** 2 / sigma ** 2
    return element_wise_log_prob.sum()


class BayesianLayer(nn.Module):
    def __init__(self, n_input, n_output, sigma_prior):
        super(BayesianLayer, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.sigma_prior = sigma_prior
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.w_mu = nn.Parameter(torch.Tensor(n_input, n_output).normal_(0, sigma_prior))
        self.w_rho = nn.Parameter(torch.Tensor(n_input, n_output).normal_(0, sigma_prior))
        self.b_mu = nn.Parameter(torch.Tensor(n_output).normal_(0, sigma_prior))
        self.b_rho = nn.Parameter(torch.Tensor(n_output).normal_(0, sigma_prior))

        self.log_q = 0
        self.log_p = 0

    def forward(self, x, sample=True):
        if sample:

            # reparameterization trick
            w = self.w_mu + F.softplus(self.w_rho) * torch.Tensor(self.in_dim, self.out_dim).to(self.device).normal_(0,
                                                                                                                     self.stddev_prior)
            b = self.b_mu + F.softplus(self.b_rho) * torch.Tensor(self.out_dim).to(self.device).normal_(0,
                                                                                                        self.stddev_prior)

            self.log_q = log_gaussian_prob(w, self.w_mu, self.w_rho)
            self.log_p = log_gaussian_prob(w, torch.zeros_like(self.w_mu, device=self.device),
                                           self.stddev_prior * torch.ones_like(F.softplus(self.w_rho),
                                                                               device=self.device))

            self.q_w += log_gaussian_prob(b, self.b_mu, self.b_rho)
            self.p_w += log_gaussian_prob(b, torch.zeros_like(self.b_mu, device=self.device),
                                          self.stddev_prior * torch.ones_like(F.softplus(self.w_rho),
                                                                              device=self.device))
        else:
            w = self.w_mu
            b = self.b_mu
        return x @ w + b


print(nn.Parameter(torch.Tensor(2, 2).normal_(0, 0.01)))
