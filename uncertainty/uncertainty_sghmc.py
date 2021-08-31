import torch
import torch.nn.functional as F
import numpy as np


import os, sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from uncertainty_utils import get_entropy_uncertainties
from models.baseline_shallow_model import ShallowModel
from SGHMC.SGHMC_trainer import SGHMCBayesianNet
from SGHMC.priors import PriorGaussian
from SGHMC.likelihoods import LikMultinomial 
from dataloaders.bcic2a_dataset import get_dataloaders
from models.layers import square, safe_log
from utils import suppress_stdout
from uncertainty_utils import get_model_predictions, get_error

def get_sghmc_predictions(subject, test_dataloader, ood_dataloader, events, n_chans, input_time_length, checkpoint_dir, device):
    subj_checkpoint_dir = os.path.join(checkpoint_dir,'subj{}'.format(subject))
    model = ShallowModel(in_chans=n_chans, input_time_length=input_time_length, n_classes=len(events), conv_nonlin=square, pool_nonlin=safe_log)
    prior = PriorGaussian(sw2=1)
    likelihood = LikMultinomial()
    net = SGHMCBayesianNet(model, likelihood, prior, subj_checkpoint_dir, weights_format='state_dict', device=device)
    mean_test_logits, test_logits =\
        net.evaluate(test_dataloader, return_individual_predictions=True, all_sampled_weights=True)
    mean_ood_predictions, ood_logits =\
        net.evaluate(ood_dataloader, return_individual_predictions=True, all_sampled_weights=True)
    return test_logits, ood_logits

def get_sghmc_uncertainties(test_logits, ood_logits, test_dataloader): 
    test_predictions =  F.softmax(test_logits,dim=-1).detach().cpu().numpy()
    ood_predictions = F.softmax(ood_logits,dim=-1).detach().cpu().numpy()
    error,corrctly_clfed, incorrctly_clfed = get_error(test_predictions.mean(axis=0), test_dataloader.dataset.tensors[1].numpy())
    test_entropy, test_epistemic, test_aleatoric, test_confidence = get_entropy_uncertainties(test_predictions)
    ood_entropy, ood_epistemic, ood_aleatoric, ood_confidence = get_entropy_uncertainties(ood_predictions)
    
    return (test_entropy, test_epistemic, test_aleatoric, test_confidence),\
        (ood_entropy, ood_epistemic, ood_aleatoric, ood_confidence), (error, corrctly_clfed, incorrctly_clfed)

if __name__ == "__main__":
    device = "cuda:1"
    checkpoint_dir = 'checkpoints/bcic_shallow_SGHMC/'
    subjects = [7,8]
    events = [0,1]
    for subject in subjects:
        with suppress_stdout():
            train_dataloader, test_dataloader, ood_dataloader, n_chans, input_time_length =\
                get_dataloaders(subjects=[subject], batch_size=64, events=events, permute=(0,2,1), insert_rgb_dim=True)
        test_logits, ood_logits = get_sghmc_predictions(subject, test_dataloader, ood_dataloader, events, n_chans,
                                                        input_time_length, checkpoint_dir, device=device)
        
        test_info, ood_info, error_info = get_sghmc_uncertainties(test_logits, ood_logits, test_dataloader)
            
        test_entropy, test_epistemic, test_aleatoric, test_confidence = test_info
        ood_entropy, ood_epistemic, ood_aleatoric, ood_confidence = ood_info
        test_error, correctly_clfed_idx, incorrectly_clfed_idx = error_info
        
        print('Subject {}, test error {:.2f}, entropy {:.2f}, epistemic {:.2f}, aleatoric {:.2f}'.format(subject, 
                                                                                                         test_error,
                                                                                                         test_entropy.mean(),
                                                                                                         test_epistemic.mean(),
                                                                                                         test_aleatoric.mean()))
        print('Correctly classified entropy {:.2f}, epistemic {:.2f}, aleatoric {:.2f}'.format(test_entropy[correctly_clfed_idx].mean(),
                                                                                               test_epistemic[correctly_clfed_idx].mean(),
                                                                                               test_aleatoric[correctly_clfed_idx].mean()))
        print('OOD entropy {:.2f}, epistemic {:.2f}, aleatoric {:.2f}'.format(ood_entropy.mean(),
                                                                              ood_epistemic.mean(),
                                                                              ood_aleatoric.mean()))
        print(' ')