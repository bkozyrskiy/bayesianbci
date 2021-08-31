import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score 


def OOD_detection_auc(in_entropies, ood_entropies):
    entropies = np.hstack(in_entropies, ood_entropies)
    labels = np.hstack(np.zeros(len(in_entropies)), np.ones(len(ood_entropies)))
    auc = roc_auc_score(labels, entropies)
    return auc

def get_entropy_uncertainties(p_hat):
    p_hat[p_hat==0]=1e-20
    pred_entropy = -(p_hat.mean(axis=0) * np.log(p_hat.mean(axis=0))).sum(axis=-1)
    aleatoric_unc = -((p_hat*np.log(p_hat)).sum(axis=-1)).mean(axis=0) 
    epistemic_unc = pred_entropy - aleatoric_unc
    confidence  = p_hat.mean(axis=0).max(axis=-1)
    return pred_entropy, epistemic_unc, aleatoric_unc, confidence

# def get_variance_uncertainties(logits):
#     '''
#     Metric is used in https://arxiv.org/pdf/1901.02731.pdf (https://github.com/kumar-shridhar/PyTorch-BayesianCNN)
#     and derivation in https://openreview.net/pdf?id=Sk_P2Q9sG
#     '''
#     l_bar = logits.mean(axis=0)
    
#     epistemic = np.std(l_bar, axis=0)**2

def get_logits_variance(logits):
    return np.std(logits)**2

def kl_disagreement(p_hat):
    '''
    Metric is used to evaluate disagreement in "Simple and Scalable Predictive Uncertainty
    Estimation using Deep Ensembles"
    https://arxiv.org/pdf/1612.01474.pdf
    '''
    if p_hat.dim()==3:
        return p_hat * np.log(p_hat/p_hat.mean(axis=0)).sum(axis=0).sum(axis=-1).mean()
    elif p_hat.dim()==2:
        return p_hat * np.log(p_hat/p_hat.mean(axis=0)).sum(axis=0).sum(axis=-1)
    
    
def get_model_predictions(model, dataloader, n_classes, device):
    model.to(device)
    model.eval()
    predictions = torch.empty((len(dataloader.dataset), n_classes),device=device)
    start = 0
    y_total = np.empty(len(dataloader.dataset))
    for data in iter(dataloader):
        x = data[0]
        y_total[start:start+x.shape[0]] = data[1].numpy()
        x = x[:,None,:,:].permute(0,1,3,2).to(device)
        predictions[start:start+x.shape[0], :] = F.softmax(model(x),dim=-1)
        start = start+x.shape[0]
    return predictions.detach().cpu().numpy(), y_total

def get_error(predictions, y):
    error_per_trial = np.argmax(predictions, axis=-1) != y
    incorrect_clfed_idx = error_per_trial.nonzero()
    correct_clfed_idx = (error_per_trial == False).nonzero()
    return error_per_trial.mean(), correct_clfed_idx, incorrect_clfed_idx 
     

# def plot_historgrams()