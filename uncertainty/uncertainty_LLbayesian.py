import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

import numpy as np
import matplotlib.pyplot as plt


import os, sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from uncertainty import get_entropy_uncertainties
from dataloaders.bcic2a_dataset import get_dataloaders
from models.bayesian_model import LLBayesianShallowModel
from utils import suppress_stdout
from models.layers import square, safe_log



def get_variance_uncertainty(model, dataloader,n_samples, n_classes, normalize, device):
    model.to(device)
    model.train()
    model.sample(True)
    
    p_hat = np.zeros((n_samples,len(dataloader.dataset), n_classes))
    start = 0
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

# def get_entropy_uncertainty(model, dataloader,n_samples, n_classes, device, option=None):
#     '''
#     @param option: str "all", "correct", "incorrect" calculates entropy for all/coorectly/incorrectly classified samples
#     '''
#     model.to(device)
#     model.train()
#     model.sample(True)
#     p_hat = np.zeros((n_samples,len(dataloader.dataset), n_classes))
#     start = 0
#     for data in iter(dataloader):
#         x, y = data[0],data[1]
#         x = x[:,None,:,:].permute(0,1,3,2).to(device)
#         for sample in range(n_samples):
#             p_hat[sample,start:start+x.shape[0],:] = F.softmax(model.forward(x)[0],dim=1).detach().cpu().numpy()  
#         start = start+x.shape[0]
    
#     return get_entropy_uncertainties(p_hat, option)

def get_model_predictions(model,n_samples, dataloader, n_classes, device):
    model.to(device)
    model.train()
    model.sample(True)
    predictions = np.zeros((n_samples,len(dataloader.dataset), n_classes))
    start = 0
    y_total = np.empty(len(dataloader.dataset))
    for data in iter(dataloader):
        x = data[0]
        y_total[start:start+x.shape[0]] = data[1].numpy()
        x = x[:,None,:,:].permute(0,1,3,2).to(device)
        for sample in range(n_samples):
            predictions[sample,start:start+x.shape[0],:] = F.softmax(model.forward(x)[0],dim=1).detach().cpu().numpy()  
        start = start+x.shape[0] 
    return predictions, y_total

def get_error(predicions, y):
    error_per_trial = (np.argmax(predicions.mean(axis=0), axis=-1) != y)
    incorrect_clfed_idx = error_per_trial.nonzero()
    correct_clfed_idx = (error_per_trial == False).nonzero()
    return error_per_trial.mean(), correct_clfed_idx, incorrect_clfed_idx 


def get_llbayesian_uncertaintes(subject, test_dataloader, ood_dataloader, events, n_chans, input_time_length, path_to_checkpoints, device='cuda:0'):
    n_samples = 200
    model = LLBayesianShallowModel(in_chans=n_chans, input_time_length=input_time_length, n_classes=len(events), conv_nonlin=square, pool_nonlin=safe_log)
    model.load_state_dict(torch.load(os.path.join(path_to_checkpoints ,'subj{}.pth'.format(subject))))
    # test_predictions = np.empty((n_samples, len(test_dataloader.dataset), len(events)))
    # ood_predictions = np.empty((n_samples, len(ood_dataloader.dataset), len(events)))
    test_predictions, y_true = get_model_predictions(model, n_samples, test_dataloader, n_classes=len(events), device=device)
    ood_predictions, _ = get_model_predictions(model, n_samples, ood_dataloader, n_classes=len(events), device=device)
    
    test_entropy, test_epistemic, test_aleatoric, test_confidence = get_entropy_uncertainties(test_predictions)
    ood_entropy, ood_epistemic, ood_aleatoric, ood_confidence = get_entropy_uncertainties(ood_predictions)
    error = get_error(test_predictions, y_true)
    return (test_entropy, test_epistemic, test_aleatoric, test_confidence), (ood_entropy, ood_epistemic, ood_aleatoric, ood_confidence), error
        

def bcic2a_uncertainty(): 
    device='cuda:0'
    entropies, epistemics, confidence = [], [], []
    path_to_checkpoints = './checkpoints/LLbcic2a_2cl/'
    for subject in range(1,10):
        with suppress_stdout():
            train_dataloader, test_dataloader, ood_dataloader, n_chans, input_time_length \
                = get_dataloaders(subjects=[subject], batch_size=None, events=[0,1])
            # others_train_dataloader, others_test_dataloader, others_ood_dataloader, n_chans, input_time_length \
            #     = get_dataloaders(subjects=[subj for subj in range(1,10) if subj != subject], batch_size=None, events=[0,1])
                
            # others_dataloader_same_events = DataLoader(ConcatDataset([others_train_dataloader.dataset, others_test_dataloader.dataset]),
            #                                batch_size=others_train_dataloader.batch_size, shuffle=False,
            #                                num_workers=0,
            #                                pin_memory=False)
        # model = LLBayesianShallowModel(in_chans=n_chans, input_time_length=input_time_length, n_classes=2, conv_nonlin=square, pool_nonlin=safe_log)
        # model.load_state_dict(torch.load('checkpoints/LLbcic2a_2cl/subj{}.pth'.format(subject)))
       
        (test_entropy, test_epistemic, test_aleatoric, test_confidence), (ood_entropy, ood_epistemic, ood_aleatoric, ood_confidence), error =\
            get_llbayesian_uncertaintes(subject, test_dataloader, 
                                        ood_dataloader, events=[0,1], 
                                        n_chans=n_chans, input_time_length=input_time_length, path_to_checkpoints=path_to_checkpoints, device='cuda:0')        
        # entropies.append((entropy_train.mean(axis=0), entropy_test.mean(axis=0), entropy_ood.mean(axis=0), entropy_others.mean(axis=0), entropy_others_ood.mean(axis=0)))
        # epistemics.append((epistemic_train.mean(axis=0), epistemic_test.mean(axis=0), epistemic_ood.mean(axis=0), epistemic_others.mean(axis=0), epistemic_others_ood.mean(axis=0)))
        # confidence.append((confidence_train.mean(axis=0), confidence_test.mean(axis=0), confidence_ood.mean(axis=0), confidence_others.mean(axis=0), confidence_others_ood.mean(axis=0) ))

    for subj in range(1,10):
        print('subject {}, entropy train {:.2f}, test {:.2f}, ood {:.2f}, other subj (same events) {:.2f}, other subj (anther events) {:.2f}'.format(subj, *entropies[subj-1]))
    
    for subj in range(1,10):
        print('subject {}, epistemic train {:.2f}, test {:.2f}, ood {:.2f}, other subj (same events) {:.2f}, other subj (anther events) {:.2f}'.format(subj, *epistemics[subj-1]))
    
    for subj in range(1,10):
        print('Subject {}, confidence train {:.2f}, test {:.2f}, ood {:.2f}, other subj (same events) {:.2f}, other subj (anther events) {:.2f}'.format(subj, *confidence[subj-1]))
    
    pass

if __name__ == '__main__':
    bcic2a_uncertainty()