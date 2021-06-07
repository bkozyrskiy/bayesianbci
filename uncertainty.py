import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from models.bayesian_model import BayesianShallowModel
from dataloaders.moabb_dataset import get_dataloaders
from models.layers import square, safe_log
import numpy as np
import matplotlib.pyplot as plt
from utils import suppress_stdout


def get_variance_uncertainty(model, dataloader,n_samples, n_classes, normalize, device):
    model.to(device)
    model.train()
    model.sample(True)
    
    p_hat = np.zeros((n_samples,len(dataloader.dataset), n_classes))
    start=0
    for data in iter(dataloader):
        x, y = data[0],data[1]
        x = x[:,None,:,:].permute(0,1,3,2).to(device)
        
        for sample in range(n_samples):
            if normalize:
                p_hat[sample,start:start+x.shape[0],:] = F.softmax(model.forward(x)[0],dim=1).detach().cpu().numpy()
            else:
                p_hat[sample,start:start+x.shape[0],:] = model.forward(x)[0].detach().cpu().numpy()
        start = start+x.shape[0]
    epistemic = np.zeros((len(dataloader.dataset),2))
    p_bar = p_hat.mean(axis=0)
    for idx in range(len(dataloader.dataset)):
        temp = p_hat[:,idx,:] - np.expand_dims(p_bar[idx,:],0)
        epistemic[idx,:] = np.diag(np.dot(temp.T,temp)/n_samples)
    
    aleatoric = np.zeros((len(dataloader.dataset),2))
    for idx in range(len(dataloader.dataset)):
        aleatoric[idx,:] = np.diag(np.diag(p_bar[idx,:]) - (np.dot(p_hat[:,idx,:].T, p_hat[:,idx,:])/n_samples))
    return epistemic, aleatoric

def get_entropy_uncertainty(model, dataloader,n_samples, n_classes, device):
    model.to(device)
    model.train()
    model.sample(True)
    p_hat = np.zeros((n_samples,len(dataloader.dataset), n_classes))
    start = 0
    for data in iter(dataloader):
        x, y = data[0],data[1]
        x = x[:,None,:,:].permute(0,1,3,2).to(device)
        for sample in range(n_samples):
            p_hat[sample,start:start+x.shape[0],:] = F.softmax(model.forward(x)[0],dim=1).detach().cpu().numpy()  
        start = start+x.shape[0]
    pred_entropy = -(p_hat.mean(axis=0) * np.log(p_hat.mean(axis=0))).sum(axis=-1)
    aleatoric_unc = -((p_hat*np.log(p_hat)).sum(axis=-1)).mean(axis=0) 
    epistemic_unc = pred_entropy - aleatoric_unc
    confidence  = p_hat.mean(axis=0).max(axis=-1)
    return pred_entropy, epistemic_unc, aleatoric_unc, confidence

def bcic2a_uncertainty(): 
    device='cuda:0'
    entropies, epistemics, confidence = [], [], []
    for subject in range(1,10):
        with suppress_stdout():
            train_dataloader, test_dataloader, n_chans, input_time_length, ood_dataloader \
                = get_dataloaders(subjects=[subject], batch_size=None, events=[0,1])
            others_train_dataloader, others_test_dataloader, n_chans, input_time_length, others_ood_dataloader \
                = get_dataloaders(subjects=[subj for subj in range(1,10) if subj != subject], batch_size=None, events=[0,1])
                
            others_dataloader_same_events = DataLoader(ConcatDataset([others_train_dataloader.dataset, others_test_dataloader.dataset]),
                                           batch_size=others_train_dataloader.batch_size, shuffle=False,
                                           num_workers=0,
                                           pin_memory=False)
        model = BayesianShallowModel(in_chans=n_chans, input_time_length=input_time_length, n_classes=2, conv_nonlin=square, pool_nonlin=safe_log)
        model.load_state_dict(torch.load('checkpoints/subj{}.pth'.format(subject)))
        entropy_train, epistemic_train, aleatoric_train, confidence_train = get_entropy_uncertainty(model, dataloader=train_dataloader,n_samples=100, n_classes=2, device=device)
        entropy_test, epistemic_test, aleatoric_test, confidence_test = get_entropy_uncertainty(model, dataloader=test_dataloader,n_samples=100, n_classes=2, device=device)
        entropy_ood, epistemic_ood, aleatoric_ood, confidence_ood = get_entropy_uncertainty(model, dataloader=ood_dataloader,n_samples=100, n_classes=2, device=device)
        entropy_others, epistemic_others, aleatoric_others, confidence_others = get_entropy_uncertainty(model, dataloader=others_dataloader_same_events,n_samples=100, n_classes=2, device=device)
        entropy_others_ood, epistemic_others_ood, aleatoric_others_ood, confidence_others_ood = get_entropy_uncertainty(model, dataloader=others_ood_dataloader,n_samples=100, n_classes=2, device=device)
        
        entropies.append((entropy_train.mean(axis=0), entropy_test.mean(axis=0), entropy_ood.mean(axis=0), entropy_others.mean(axis=0), entropy_others_ood.mean(axis=0)))
        epistemics.append((epistemic_train.mean(axis=0), epistemic_test.mean(axis=0), epistemic_ood.mean(axis=0), epistemic_others.mean(axis=0), epistemic_others_ood.mean(axis=0)))
        confidence.append((confidence_train.mean(axis=0), confidence_test.mean(axis=0), confidence_ood.mean(axis=0), confidence_others.mean(axis=0), confidence_others_ood.mean(axis=0) ))

    for subj in range(1,10):
        print('subject {}, entropy train {:.2f}, test {:.2f}, ood {:.2f}, other subj (same events) {:.2f}, other subj (anther events) {:.2f}'.format(subj, *entropies[subj-1]))
    
    for subj in range(1,10):
        print('subject {}, epistemic train {:.2f}, test {:.2f}, ood {:.2f}, other subj (same events) {:.2f}, other subj (anther events) {:.2f}'.format(subj, *epistemics[subj-1]))
    
    for subj in range(1,10):
        print('Subject {}, confidence train {:.2f}, test {:.2f}, ood {:.2f}, other subj (same events) {:.2f}, other subj (anther events) {:.2f}'.format(subj, *confidence[subj-1]))
    
    pass

if __name__ == '__main__':
    