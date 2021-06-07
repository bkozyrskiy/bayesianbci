import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def acc(outputs, targets):
    return np.mean(outputs.cpu().numpy().argmax(axis=1) == targets.data.cpu().numpy())


def softmax_log_likelihood(logits, targets):
    return (torch.sum(targets*logits, dim=-1) - torch.logsumexp(logits, dim=-1)).squeeze()

def elbo(logits, targets, kl, beta, beta_type, datasize):
        '''
        Takes log-probability  as input!
        '''
        assert not targets.requires_grad
        if beta_type == "Blundell":
            return -softmax_log_likelihood(logits, targets).sum(dim=0) + beta * kl
        elif beta_type is None:
            return -softmax_log_likelihood(logits, targets).mean(dim=0) * datasize + beta * kl


def KL_DIV(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl

def get_beta(batch_idx, n_batches, beta_type, epoch, num_epochs):
    if type(beta_type) is float:
        return beta_type

    if beta_type == "Blundell":
        beta = 2 ** (n_batches - (batch_idx + 1)) / (2 ** n_batches - 1)
    elif beta_type == "Soenderby":
        if epoch is None or num_epochs is None:
            raise ValueError('Soenderby method requires both epoch and num_epochs to be passed.')
        beta = min(epoch / (num_epochs // 4), 1)
    elif beta_type == "Standard":
        beta = 1 / n_batches
    else:
        beta = 0
    return beta


# def uncertainty_estimation(model, data):
    