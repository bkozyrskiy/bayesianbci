import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os,sys

from contextlib import contextmanager



def logmeanexp(x, dim=None, keepdim=False):
    """Stable computation of log(mean(exp(x))"""
    if dim is None:
        x, dim = x.view(-1), 0
    x_max, _ = torch.max(x, dim, keepdim=True)
    x = x_max + torch.log(torch.mean(torch.exp(x - x_max), dim, keepdim=True))
    return x if keepdim else x.squeeze(dim)

class Expression(torch.nn.Module):
    """Compute given expression on forward pass.
    Parameters
    ----------
    expression_fn : callable
        Should accept variable number of objects of type
        `torch.autograd.Variable` to compute its output.
    """

    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)

    def __repr__(self):
        if hasattr(self.expression_fn, "func") and hasattr(
            self.expression_fn, "kwargs"
        ):
            expression_str = "{:s} {:s}".format(
                self.expression_fn.func.__name__, str(self.expression_fn.kwargs)
            )
        elif hasattr(self.expression_fn, "__name__"):
            expression_str = self.expression_fn.__name__
        else:
            expression_str = repr(self.expression_fn)
        return (
            self.__class__.__name__ +
            "(expression=%s) " % expression_str
        )
        
def reliability_plot(model, data_loader, device):
    '''
    Reliability plot for 2 class problem
    '''
    model.eval()
    x, y = [],[]
    
    for train_data in iter(data_loader):
        x_batch, y_batch = train_data[0], train_data[1]
        x.append(x_batch), y.append(y_batch)
    x,y = torch.cat(x), torch.cat(y)
    probs = F.softmax(model(x[:,None,:,:].permute(0,1,3,2).to(device)),dim=1)
    
    for cl in range(probs.shape[1]):
        clastered_indx = []
        for thrs in np.arange(0, 0.9, 0.1):
            clastered_indx.append(((probs[:,cl] >= thrs) * (probs[:,cl] < (thrs+0.1))).nonzero())
        clastered_indx.append((probs[:,cl] >= 0.9).nonzero())
        

        acc_bins = []
        for bin in clastered_indx:
            acc_bins.append((y[bin] == cl).float().mean())
        plt.figure()
        plt.plot(np.arange(0.05, 1, 0.1), acc_bins, marker='s')
        plt.plot(range(0,2), range(0,2),linestyle='--')
        plt.title('class {}'.format(cl))
        plt.xlabel('confidence')
        plt.ylabel('accuracy')
        plt.savefig(os.path.join('res','Class_{}.png'.format(cl)))

def test_on_ood(model, ood_dataloader, device, subject):
    for x, _ in iter(ood_dataloader):
        x = x.to(device)
        predictions = F.softmax(model(x[:,None,:,:].permute(0,1,3,2)), dim=1).cpu().numpy()
    plt.hist(predictions[0,:])
    plt.savefig('ood_predictions_{}.png'.format(subject))
        
class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
    
    